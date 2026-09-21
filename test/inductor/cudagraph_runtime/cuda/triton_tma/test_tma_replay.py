"""Replay device TMA descriptors with compiler-sized owned scratch."""

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
from torch._inductor.runtime._cudagraph.frontend import DirectKernelCall
from torch._inductor.runtime.cudagraph_arg_mapping import ParameterSource, PointerSource
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._triton import has_triton_stable_tma_api


@triton.jit(do_not_specialize=["M", "N"], do_not_specialize_on_alignment=["M", "N"])
def device_descriptor_gemm(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [32, 32])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [32, 32])
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
    m, n, k, a, b = box
    box.clear()
    output = torch.empty_strided((m, n), (n, 1), dtype=a.dtype, device=a.device)
    GEMM[(triton.cdiv(m, 32), triton.cdiv(n, 32))](a, b, output, M=m, N=n, K=k)
    return (output,)


class TestTritonTmaReplay(TestCase):
    def test_device_descriptors_and_owned_scratch(self, device):
        if torch.version.hip or torch.cuda.get_device_capability(device) < (9, 0):
            self.skipTest("requires CUDA SM90 or newer for TMA")
        with torch.cuda.device(device):
            if not has_triton_stable_tma_api():
                self.skipTest("requires CUDA Triton device-side TMA")

        from cuda.bindings import driver
        from triton.runtime._allocation import _allocator

        self.enterContext(torch.cuda.device(device))
        allocator_requests = []

        def allocate(size, alignment, stream):
            tensor = torch.empty(size, dtype=torch.uint8, device=device)
            self.assertEqual(tensor.data_ptr() % alignment, 0)
            allocator_requests.append((size, alignment))
            return tensor

        self.addCleanup(triton.set_allocator, _allocator.get())
        triton.set_allocator(allocate)
        adapter = DirectTriton(device_descriptor_gemm)
        self.addCleanup(adapter.close)
        self.enterContext(mock.patch.dict(globals(), {"GEMM": adapter}))
        m, n, k = (IntExpr("boxed", index) for index in range(3))
        contract = InputContract(
            ("integer", "integer", "integer", "tensor", "tensor"),
            (TensorInput(3, torch.bfloat16, (m, k), (k, 1)),
             TensorInput(4, torch.bfloat16, (n, k), (k, 1))),
            tuple(IntegerRange(index, 32, 128) for index in range(3)),
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
            _, input_count, allocations, _, copies, calls, launches, buffers, _ = args
            self.assertEqual(input_count, 5)
            self.assertEqual(copies, ())
            self.assertEqual(len(allocations), 2)
            call, = calls
            launch, = launches
            module = call.module
            self.assertFalse(module._has_tensordesc)
            self.assertGreater(module.global_scratch_size, 0)
            self.assertGreater(module.global_scratch_align, 0)
            self.assertLessEqual(module.global_scratch_align, 256)
            self.assertEqual(len(call.scratch), int(module.has_global_scratch) + int(module.has_profile_scratch))
            scratch = call.scratch[0]
            self.assertIs(type(scratch), PointerSource)
            layout, = [layout for layout in allocations if layout.source == scratch.root]
            tensor = buffers[scratch.root]
            numeric = kwargs["numeric"]
            m, n, _ = kwargs["capture_inputs"][:3]
            grid = ((m + 31) // 32, (n + 31) // 32, 1)
            size = math.prod(grid) * module.global_scratch_size
            self.assertEqual(layout.dtype, torch.uint8)
            self.assertEqual(layout.stride, (1,))
            self.assertEqual(tuple(numeric.values[numeric.add(value)] if type(value) is IntExpr else value
                                   for value in layout.size), (size,))
            self.assertEqual((tuple(tensor.shape), tensor.stride()), ((size,), (1,)))
            self.assertEqual(tensor.data_ptr() % module.global_scratch_align, 0)
            self.assertEqual(numeric.values[numeric.add(scratch.byte_offset)], 0)
            scratch_slot = len(call.arguments)
            self.assertEqual(launch.argument_bytes[scratch_slot], struct.pack("P", tensor.data_ptr()))
            for index, source in enumerate(call.scratch[1:], scratch_slot + 1):
                self.assertEqual(source, ParameterSource("constant", 64, 0))
                self.assertEqual(launch.argument_bytes[index], bytes(8))
            self.assertEqual(len(launch.argument_bytes), scratch_slot + len(call.scratch))
            for index, payload in enumerate(launch.argument_bytes):
                _, width = _check_cuda_bindings(driver.cuFuncGetParamInfo(module.function, index))
                self.assertEqual(width, len(payload))
                if index >= scratch_slot:
                    self.assertEqual(width, 8)
            captures.append({"grid": grid, "bytes": size, "per_cta_bytes": module.global_scratch_size,
                             "alignment": module.global_scratch_align, "scratch_slot": scratch_slot})
            return make_replay(*args, **kwargs)

        self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
        cases = ((64, 64, 32, 0, True), (96, 64, 32, 8, False), (64, 96, 32, 16, False),
                 (96, 96, 64, 8, True), (64, 64, 64, 16, False), (64, 64, 32, 24, False),
                 (128, 96, 32, 32, False))
        samples = []
        for m, n, k, offset, miss in cases:
            tensors = []
            for rows in (m, n):
                storage = torch.randn(rows * k + offset, dtype=torch.bfloat16, device=device)
                tensor = storage[offset:].view(rows, k)
                self.assertEqual(tensor.storage_offset(), offset)
                self.assertEqual(tensor.data_ptr(), storage.data_ptr() + offset * tensor.element_size())
                self.assertEqual(tensor.data_ptr() % 16, 0)
                tensors.append(tensor)
            samples.append(((m, n, k, *tensors), miss))
        self.assertEqual(len({tensor.data_ptr() for inputs, _ in samples for tensor in inputs[3:]}), 14)
        held, scratch_sizes = [], []
        misses = 0
        for index, (inputs, miss) in enumerate(samples):
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
                self.assertEqual(bool(frames), miss)
            if not miss:
                self.assertEqual(len(allocator_requests), allocator_count)
            misses += int(miss)
            self.assertEqual(box, [])
            self.assertEqual(observations.call_count, misses)
            self.assertEqual(preparations.call_count, misses)
            self.assertEqual(len(runtime.variants), misses)
            m, n, k, a, b = inputs
            ordinary, = host(list(inputs))
            expected = a @ b.T
            self.assertEqual(actual, ordinary, atol=0, rtol=0)
            self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
            self.assertEqual((tuple(actual.shape), actual.stride()), ((m, n), (n, 1)))
            self.assertNotEqual(actual.data_ptr(), ordinary.data_ptr())
            held.append((actual, expected))
            variant = runtime.variants[int(k == 64)]
            call, = [event for event in variant.program.events if type(event) is DirectKernelCall]
            self.assertIs(type(call.scratch[0]), PointerSource)
            layout, = [layout for layout in variant.program.allocations if layout.source == call.scratch[0].root]
            numeric = _NumericProgram(variant.program, list(inputs))
            size, = (numeric.values[numeric.add(value)] if type(value) is IntExpr else value for value in layout.size)
            selected = call.owner.binary.metadata
            self.assertEqual(size, ((m + 31) // 32) * ((n + 31) // 32) * selected.global_scratch_size)
            scratch_sizes.append(size)
            for name in ("M", "N"):
                formal, = [formal for formal in call.owner.formals if formal.formal == name]
                self.assertIsNotNone(formal.abi_index)
            constant, = [formal for formal in call.owner.formals if formal.formal == "K"]
            self.assertIsNone(constant.abi_index)
            self.assertEqual(constant.constant, k)
        self.assertEqual(misses, 2)
        self.assertEqual(len(captures), 2)
        self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(samples))
        modules = [owner.module for owner in adapter._owners]
        runtime.close()
        self.assertTrue(all(module._graph_borrows == 0 for module in modules))
        adapter.close()
        for actual, expected in held:
            self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        print("TRITON_TMA_REPLAY_RESULT=" + json.dumps({
            "mechanism": "device", "samples": len(samples), "variants": misses, "native_hits": 5,
            "ordinary_references": len(samples), "fresh_input_addresses": 14,
            "shapes_and_element_offsets": [case[:4] for case in cases], "captures": captures,
            "scratch_bytes_per_sample": scratch_sizes, "held_outputs_after_close": len(held),
        }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTritonTmaReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
