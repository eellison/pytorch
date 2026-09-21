"""Replay host tensor maps and compare their full ABI with Triton's encoder."""

import ctypes
import json
import sys
from dataclasses import replace
from unittest import mock

import torch
import triton
import triton.language as tl
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectHost, DirectTriton, InputContract, IntegerRange, IntExpr, TensorInput,
)
from torch._inductor.runtime._cudagraph.frontend import DirectPhysicalCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource
from torch._inductor.runtime.cudagraph_boxed_replay import _PhysicalCall
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._triton import has_triton_tensor_descriptor_host_tma


@triton.jit
def host_descriptor_gemm(a_desc, b_desc, c_desc, K: tl.constexpr):
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

    m, n, k, a, b = box
    box.clear()
    output = torch.empty_strided((m, n), (n, 1), dtype=a.dtype, device=a.device)
    descriptors = [TensorDescriptor.from_tensor(tensor, [32, 32]) for tensor in (a, b, output)]
    GEMM[(triton.cdiv(m, 32), triton.cdiv(n, 32))](*descriptors, K=k)
    return (output,)


class TestHostTmaReplay(TestCase):
    def test_host_tensor_maps_and_full_compiler_abi(self, device):
        if torch.version.hip or torch.cuda.get_device_capability(device) < (9, 0):
            self.skipTest("requires CUDA SM90 or newer for TMA")
        with torch.cuda.device(device):
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires CUDA Triton host-side TMA")

        from triton.backends.nvidia.driver import make_tensordesc_arg
        from triton.tools.tensor_descriptor import TensorDescriptor

        self.enterContext(torch.cuda.device(device))
        adapter = DirectTriton(host_descriptor_gemm)
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
            _, input_count, allocations, _, copies, calls, launches, buffers, stream = args
            self.assertEqual(input_count, 5)
            self.assertEqual(copies, ())
            output_layout, = allocations
            call, = calls
            launch, = launches
            self.assertIs(type(call), _PhysicalCall)
            self.assertEqual(len(call.tensor_maps), 3)
            self.assertEqual({field.pointer.root for field in call.tensor_maps},
                             {InputSource(3), InputSource(4), output_layout.source})
            numeric = kwargs["numeric"]
            for field in call.tensor_maps:
                self.assertEqual(numeric.values[numeric.add(field.pointer.byte_offset)], 0)
                self.assertEqual(len(launch.argument_bytes[field.parameter]), 128)
            owner, = [owner for owner in adapter._owners if owner.module.function == call.module.function]
            module = owner.module
            self.assertTrue(module._has_tensordesc)
            self.assertFalse(module.global_scratch_size)
            self.assertFalse(module.profile_scratch_size)
            m, n, _, a, b = kwargs["capture_inputs"]
            output = buffers[output_layout.source]
            expanded, map_slots = [], []
            for tensor, metadata in zip((a, b, output), owner.binary.metadata.tensordesc_meta):
                self.assertIsNotNone(metadata)
                descriptor = TensorDescriptor.from_tensor(tensor, [32, 32])
                map_slots.append(len(expanded))
                expanded.extend(make_tensordesc_arg(descriptor, metadata, None))
            self.assertEqual(tuple(field.parameter for field in call.tensor_maps), tuple(map_slots))
            expanded.extend([None] * (int(module.has_global_scratch) + int(module.has_profile_scratch)))
            reference_graph = torch.cuda.CUDAGraph(keep_graph=True)
            try:
                # Capture Triton's actual map objects; inspect bytes without replaying this reference graph.
                with torch.cuda.graph(reference_graph, stream=stream):
                    module.C_impl._launch_kernel(
                        module.function, (m + 31) // 32, (n + 31) // 32, 1,
                        module.num_warps, module.shared,
                        module.arg_tys + "O" * (int(module.has_global_scratch) + int(module.has_profile_scratch)),
                        tuple(expanded), stream.cuda_stream,
                    )
                    frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                node, = (node for node, _ in frontier[3])
                reference, = reference_graph._inspect_captured_kernel_nodes((node,))[3]
                self.assertEqual(tuple(payload for _, _, payload in reference[8]), launch.argument_bytes)
                for slot in map_slots:
                    self.assertEqual(tuple(width for _, width, _ in reference[8][slot:slot + 5]),
                                     (128, 4, 4, 8, 8))
                reference_graph.instantiate()
                field = call.tensor_maps[0]
                buffer_indices = {output_layout.source: input_count}
                binding = field.binding(numeric, input_count, buffer_indices, node)
                wrong_width = replace(field, parameter=field.parameter + 1).binding(
                    numeric, input_count, buffer_indices, node,
                )
                with self.assertRaisesRegex(ValueError, "width differs from the selected ABI"):
                    reference_graph._prepare_kernel_replay_updates(
                        (), input_count + 1, (), (), len(numeric.values),
                        tensor_map_bindings=(wrong_width,),
                    )
                with self.assertRaisesRegex(ValueError, "bindings overlap"):
                    reference_graph._prepare_kernel_replay_updates(
                        (), input_count + 1, (), (), len(numeric.values),
                        tensor_map_bindings=(binding, binding),
                    )
                captures.append({"map_slots": map_slots, "map_width": 128,
                                 "native_binding_rejections": 2,
                                 "parameter_widths": [width for _, width, _ in reference[8]]})
            finally:
                reference_graph.reset()
            return make_replay(*args, **kwargs)

        self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
        cases = ((64, 64, 32, 0, True), (96, 64, 32, 8, False), (64, 96, 32, 16, False),
                 (96, 96, 64, 8, True), (64, 64, 64, 16, False), (64, 64, 32, 24, False),
                 (128, 96, 32, 32, False))
        samples = []
        for m, n, k, offset, miss in cases:
            tensors = []
            for rows in (m, n):
                storage = torch.randn(rows * k + offset + 1, dtype=torch.bfloat16, device=device)
                tensor = storage[offset:offset + rows * k].view(rows, k)
                self.assertEqual(tensor.storage_offset(), offset)
                self.assertEqual(tensor.data_ptr(), storage.data_ptr() + offset * tensor.element_size())
                self.assertEqual(tensor.data_ptr() % 16, 0)
                tensors.append(tensor)
            samples.append(((m, n, k, *tensors), miss))
        self.assertEqual(len({tensor.data_ptr() for inputs, _ in samples for tensor in inputs[3:]}), 14)
        held = []
        misses = rejected = 0
        for index, (inputs, miss) in enumerate(samples):
            box, frames = list(inputs), []

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
            event, = [event for event in variant.program.events if type(event) is DirectPhysicalCall]
            self.assertEqual(len(event.bound.tensor_maps), 3)
            constant, = [formal for formal in event.owner.formals if formal.formal == "K"]
            self.assertIsNone(constant.abi_index)
            self.assertEqual(constant.constant, k)
            guard = variant.guard
            self.assertEqual(set(guard.boxed_pointer_indices), {3, 4})
            integers = [inputs[slot] for slot in guard.boxed_integer_indices]
            pointers = [inputs[slot].data_ptr() for slot in guard.boxed_pointer_indices]
            offsets = [inputs[slot].storage_offset() for slot in guard.boxed_storage_offset_indices]
            values = integers + pointers + offsets
            predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_double))(guard.function_address)
            self.assertEqual(predicate((ctypes.c_int64 * len(values))(*values), None), 1)
            for ordinal, slot in enumerate(guard.boxed_pointer_indices, len(integers)):
                tensor = inputs[slot]
                probe = values.copy()
                probe[ordinal] += tensor.element_size()
                self.assertLessEqual((tensor.storage_offset() + tensor.numel() + 1) * tensor.element_size(),
                                     tensor.untyped_storage().nbytes())
                if slot in guard.boxed_storage_offset_indices:
                    offset_slot = len(integers) + len(pointers) + guard.boxed_storage_offset_indices.index(slot)
                    probe[offset_slot] += 1
                self.assertEqual(predicate((ctypes.c_int64 * len(probe))(*probe), None), 0)
                rejected += 1
        self.assertEqual(misses, 2)
        self.assertEqual(len(captures), 2)
        self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(samples))
        modules = [owner.module for owner in adapter._owners]
        runtime.close()
        self.assertTrue(all(module._graph_borrows == 0 for module in modules))
        adapter.close()
        for actual, expected in held:
            self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        print("HOST_TMA_REPLAY_RESULT=" + json.dumps({
            "samples": len(samples), "variants": misses, "native_hits": 5,
            "ordinary_references": len(samples), "fresh_input_addresses": 14,
            "shapes_and_element_offsets": [case[:4] for case in cases], "captures": captures,
            "reference_abi_captures": len(captures), "alignment_guard_rejections": rejected,
            "native_binding_rejections": sum(capture["native_binding_rejections"] for capture in captures),
            "held_outputs_after_close": len(held),
        }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestHostTmaReplay, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
