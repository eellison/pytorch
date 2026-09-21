"""Compose host input maps with a device output map and owned global scratch."""

import json
import math
import struct
import sys
from unittest import mock

import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectHost, DirectTriton, InputContract, IntegerRange, IntExpr, TensorInput,
)
from torch._inductor.runtime._cudagraph.frontend import DirectPhysicalCall
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource, ParameterSource, PointerSource
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram, _PhysicalCall
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._triton import has_triton_stable_tma_api, has_triton_tensor_descriptor_host_tma


@triton.jit(do_not_specialize=["M", "N"], do_not_specialize_on_alignment=["M", "N"])
def mixed_descriptor_gemm(a_desc, b_desc, c_ptr, M, N, K: tl.constexpr):
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [32, 32])
    offset_m = tl.program_id(0) * 32
    offset_n = tl.program_id(1) * 32
    accumulator = tl.zeros((32, 32), tl.float32)
    for offset_k in range(0, K, 32):
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        accumulator = tl.dot(a, b.T, accumulator)
    c_desc.store([offset_m, offset_n], accumulator.to(tl.bfloat16))


GEMM = None


def host(box):
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, n, a, b = box
    box.clear()
    output = torch.empty_strided((m, n), (n, 1), dtype=a.dtype, device=a.device)
    a_desc = TensorDescriptor.from_tensor(a, [32, 32])
    b_desc = TensorDescriptor.from_tensor(b, [32, 32])
    GEMM[(triton.cdiv(m, 32), triton.cdiv(n, 32))](a_desc, b_desc, output, M=m, N=n, K=32)
    return (output,)


class TestMixedTmaReplay(TestCase):
    def test_host_maps_and_device_scratch_in_one_kernel(self, device):
        if torch.version.hip or torch.cuda.get_device_capability(device) < (9, 0):
            self.skipTest("requires CUDA SM90 or newer for TMA")
        with torch.cuda.device(device):
            if not (has_triton_stable_tma_api() and has_triton_tensor_descriptor_host_tma()):
                self.skipTest("requires Triton host and device tensor descriptors")

        from cuda.bindings import driver
        from triton.backends.nvidia.driver import make_tensordesc_arg
        from triton.runtime._allocation import _allocator
        from triton.tools.tensor_descriptor import TensorDescriptor

        self.enterContext(torch.cuda.device(device))
        allocator_requests = []

        def allocate(size, alignment, stream):
            tensor = torch.empty(size, dtype=torch.uint8, device=device)
            self.assertEqual(tensor.data_ptr() % alignment, 0)
            allocator_requests.append((size, alignment))
            return tensor

        self.addCleanup(triton.set_allocator, _allocator.get())
        triton.set_allocator(allocate)
        adapter = DirectTriton(mixed_descriptor_gemm)
        self.addCleanup(adapter.close)
        self.enterContext(mock.patch.dict(globals(), {"GEMM": adapter}))
        m, n = (IntExpr("boxed", index) for index in range(2))
        contract = InputContract(
            ("integer", "integer", "tensor", "tensor"),
            (TensorInput(2, torch.bfloat16, (m, 32), (32, 1)),
             TensorInput(3, torch.bfloat16, (n, 32), (32, 1))),
            (IntegerRange(0, 32, 128), IntegerRange(1, 32, 128)),
            device_index=torch.cuda.current_device(),
        )
        runtime = DirectHost(host, contract)
        self.addCleanup(runtime.close)
        observations = self.enterContext(mock.patch.object(
            direct_host, "_observe_direct", wraps=direct_host._observe_direct,
        ))
        preparations = self.enterContext(mock.patch.object(
            direct_host, "_prepare_observed", wraps=direct_host._prepare_observed,
        ))
        captures = []
        make_replay = replay._make_replay

        def inspect_capture(*args, **kwargs):
            _, input_count, allocations, _, copies, calls, launches, buffers, stream = args
            self.assertEqual(input_count, 4)
            self.assertEqual(copies, ())
            self.assertEqual(len(allocations), 2)
            call, = calls
            launch, = launches
            self.assertIs(type(call), _PhysicalCall)
            owner = call.module.owner
            module = owner.module
            formals = {row.formal: row for row in owner.formals}
            self.assertTrue(module._has_tensordesc)
            self.assertGreater(module.global_scratch_size, 0)
            self.assertLessEqual(module.global_scratch_align, 256)
            self.assertEqual(len(owner.descriptors), 2)
            self.assertEqual(tuple(field.parameter for field in call.tensor_maps),
                             (formals["a_desc"].abi_index, formals["b_desc"].abi_index))
            self.assertEqual(tuple(field.pointer.root for field in call.tensor_maps),
                             (InputSource(2), InputSource(3)))
            fields = {field.parameter: field for field in call.fields}
            output_source = fields[formals["c_ptr"].abi_index].source
            self.assertIs(type(output_source), PointerSource)
            self.assertIs(type(output_source.root), BufferSource)
            scratch_slot = len(module.arg_tys)
            scratch = fields[scratch_slot].source
            self.assertIs(type(scratch), PointerSource)
            self.assertIs(type(scratch.root), BufferSource)
            self.assertNotEqual(scratch.root, output_source.root)
            self.assertEqual({layout.source for layout in allocations}, {scratch.root, output_source.root})
            numeric = kwargs["numeric"]
            m, n, a, b = kwargs["capture_inputs"]
            grid = ((m + 31) // 32, (n + 31) // 32, 1)
            tensor = buffers[scratch.root]
            size = math.prod(grid) * module.global_scratch_size
            self.assertEqual((tensor.dtype, tuple(tensor.shape), tensor.stride()),
                             (torch.uint8, (size,), (1,)))
            self.assertEqual(tensor.data_ptr() % module.global_scratch_align, 0)
            self.assertEqual(numeric.values[numeric.add(scratch.byte_offset)], 0)
            self.assertEqual(launch.argument_bytes[scratch_slot], struct.pack("P", tensor.data_ptr()))
            for index in owner.scratch_abi_indices[1:]:
                self.assertEqual(fields[index].source, ParameterSource("constant", 64, 0))
                self.assertEqual(launch.argument_bytes[index], bytes(8))
            for name, value in (("M", m), ("N", n)):
                row = formals[name]
                self.assertIsNotNone(row.abi_index)
                self.assertEqual(fields[row.abi_index].kind, row.triton_type)
                code = {"i32": "i", "i64": "q"}[row.triton_type]
                self.assertEqual(launch.argument_bytes[row.abi_index], struct.pack("<" + code, value))
            expanded = []
            for source, metadata in zip((a, b), owner.binary.metadata.tensordesc_meta, strict=True):
                expanded.extend(make_tensordesc_arg(TensorDescriptor.from_tensor(source, [32, 32]), metadata, None))
            expanded.extend((buffers[output_source.root], m, n))
            expanded.extend(buffers[fields[index].source.root]
                            if type(fields[index].source) is PointerSource else None
                            for index in owner.scratch_abi_indices)
            reference_graph = torch.cuda.CUDAGraph(keep_graph=True)
            try:
                # The reference uses Triton's own map objects and the same supplied scratch.
                with torch.cuda.graph(reference_graph, stream=stream):
                    module.C_impl._launch_kernel(
                        module.function, *grid, module.num_warps, module.shared,
                        module.arg_tys + "O" * len(owner.scratch_abi_indices), tuple(expanded), stream.cuda_stream,
                    )
                    frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                node, = (node for node, _ in frontier[3])
                reference, = reference_graph._inspect_captured_kernel_nodes((node,))[3]
                self.assertEqual(tuple(payload for _, _, payload in reference[8]), launch.argument_bytes)
            finally:
                reference_graph.reset()
            for index, payload in enumerate(launch.argument_bytes):
                _, width = _check_cuda_bindings(driver.cuFuncGetParamInfo(module.function, index))
                self.assertEqual(width, len(payload))
            captures.append({"map_slots": [field.parameter for field in call.tensor_maps],
                             "scratch_slot": scratch_slot, "scratch_bytes": size,
                             "parameter_widths": [len(payload) for payload in launch.argument_bytes]})
            return make_replay(*args, **kwargs)

        self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
        cases = ((64, 64, 0), (64, 64, 8), (96, 64, 16), (64, 96, 24), (96, 96, 8))
        samples = []
        for m, n, offset in cases:
            inputs = []
            for rows in (m, n):
                storage = torch.randn(rows * 32 + offset, dtype=torch.bfloat16, device=device)
                tensor = storage[offset:].view(rows, 32)
                self.assertEqual(tensor.storage_offset(), offset)
                self.assertEqual(tensor.data_ptr(), storage.data_ptr() + offset * tensor.element_size())
                self.assertEqual(tensor.data_ptr() % 16, 0)
                inputs.append(tensor)
            samples.append((m, n, *inputs))
        self.assertEqual(len({value.data_ptr() for inputs in samples for value in inputs[2:]}), 10)
        held, scratch_sizes = [], []
        for index, inputs in enumerate(samples):
            box, frames = list(inputs), []
            allocator_count = len(allocator_requests)

            def profile(frame, event, result):
                if event == "call":
                    frames.append(frame.f_code)

            if index == 0:
                actual, = runtime(box)
            else:
                try:
                    sys.setprofile(profile)
                    actual, = runtime.entry(box)
                finally:
                    sys.setprofile(None)
                self.assertEqual(frames, [])
                self.assertEqual(len(allocator_requests), allocator_count)
            self.assertEqual(box, [])
            self.assertEqual(observations.call_count, 1)
            self.assertEqual(preparations.call_count, 1)
            self.assertEqual(len(runtime.variants), 1)
            ordinary, = host(list(inputs))
            m, n, a, b = inputs
            expected = a @ b.T
            self.assertEqual(actual, ordinary, atol=0, rtol=0)
            self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
            self.assertEqual((tuple(actual.shape), actual.stride()), ((m, n), (n, 1)))
            self.assertNotEqual(actual.data_ptr(), ordinary.data_ptr())
            held.append((actual, expected))
            program = runtime.variants[0].program
            event, = [event for event in program.events if type(event) is DirectPhysicalCall]
            fields = {field.parameter: field for field in event.bound.fields}
            scratch = fields[len(event.owner.module.arg_tys)].source
            layout, = [layout for layout in program.allocations if layout.source == scratch.root]
            numeric = _NumericProgram(program, list(inputs))
            size, = (numeric.values[numeric.add(axis)] if type(axis) is IntExpr else axis for axis in layout.size)
            self.assertEqual(size, ((m + 31) // 32) * ((n + 31) // 32)
                             * event.owner.binary.metadata.global_scratch_size)
            scratch_sizes.append(size)
        self.assertEqual(len(captures), 1)
        self.assertEqual(len(adapter._owners), 1)
        self.assertEqual(len(set(scratch_sizes)), 3)
        self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(samples))
        module, = [owner.module for owner in adapter._owners]
        runtime.close()
        self.assertEqual(module._graph_borrows, 0)
        adapter.close()
        for actual, expected in held:
            self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        print("MIXED_TMA_REPLAY_RESULT=" + json.dumps({
            "samples": len(samples), "variants": 1, "native_hits": len(samples) - 1,
            "ordinary_references": len(samples), "fresh_input_addresses": 10,
            "shapes_and_element_offsets": cases, "captures": captures,
            "scratch_bytes_per_sample": scratch_sizes, "held_outputs_after_close": len(held),
        }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestMixedTmaReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
