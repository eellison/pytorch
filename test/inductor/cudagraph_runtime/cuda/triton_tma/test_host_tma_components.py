"""Host tensor maps preserve rank, dtype, padded strides and returned views."""

import ctypes
import json
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
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram, _PhysicalCall
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.utils._triton import has_triton_tensor_descriptor_host_tma


@triton.jit
def transform_rank1(source, target):
    offset = tl.program_id(0) * 128
    value = source.load([offset])
    target.store([offset], value + 1)


@triton.jit
def transform_rank3(source, target):
    first = tl.program_id(0) * 2
    second = tl.program_id(1) * 4
    third = tl.program_id(2) * 32
    value = source.load([first, second, third])
    target.store([first, second, third], value + 1)


TRANSFORM = None


def host_rank1(box):
    from triton.tools.tensor_descriptor import TensorDescriptor

    count, source = box
    box.clear()
    output = torch.empty_strided((count,), (1,), dtype=source.dtype, device=source.device)
    TRANSFORM[(triton.cdiv(count, 128),)](
        TensorDescriptor.from_tensor(source, [128]),
        TensorDescriptor.from_tensor(output, [128]),
    )
    return output, source, source[4:]


def host_rank3(box):
    from triton.tools.tensor_descriptor import TensorDescriptor

    first, second, third, _plane_stride, _row_stride, source = box
    box.clear()
    output = torch.empty_strided(
        (first, second, third), (second * third, third, 1), dtype=source.dtype, device=source.device,
    )
    TRANSFORM[(triton.cdiv(first, 2), triton.cdiv(second, 4), triton.cdiv(third, 32))](
        TensorDescriptor.from_tensor(source, [2, 4, 32]),
        TensorDescriptor.from_tensor(output, [2, 4, 32]),
    )
    return output, source, source[..., 8:]


class TestHostTmaComponents(TestCase):
    @parametrize("rank,dtype", ((1, torch.float32), (3, torch.bfloat16)))
    def test_rank_dtype_and_symbolic_strides(self, device, rank, dtype):
        if torch.version.hip or torch.cuda.get_device_capability(device) < (9, 0):
            self.skipTest("requires CUDA SM90 or newer for TMA")
        with torch.cuda.device(device):
            if not has_triton_tensor_descriptor_host_tma():
                self.skipTest("requires Triton host tensor descriptors")

        from triton.backends.nvidia.driver import make_tensordesc_arg
        from triton.tools.tensor_descriptor import TensorDescriptor

        self.enterContext(torch.cuda.device(device))
        reference_stream = torch.cuda.Stream(device=device)
        if rank == 1:
            host, kernel, block = host_rank1, transform_rank1, (128,)
            count = IntExpr("boxed", 0)
            contract = InputContract(
                ("integer", "tensor"), (TensorInput(1, dtype, (count,), (1,)),),
                (IntegerRange(0, 8, 512),), device_index=torch.cuda.current_device(),
            )
            cases = (((128,), (1,), 0), ((192,), (1,), 0), ((257,), (1,), 0),
                     ((257,), (1,), 12), ((128,), (1,), 16))
        else:
            host, kernel, block = host_rank3, transform_rank3, (2, 4, 32)
            first, second, third, plane, row = (IntExpr("boxed", index) for index in range(5))
            contract = InputContract(
                ("integer",) * 5 + ("tensor",),
                (TensorInput(5, dtype, (first, second, third), (plane, row, 1)),),
                (IntegerRange(0, 2, 8), IntegerRange(1, 4, 16), IntegerRange(2, 32, 96),
                 IntegerRange(3, 128, 4096), IntegerRange(4, 32, 160)),
                device_index=torch.cuda.current_device(),
            )
            cases = (((2, 8, 64), (768, 80, 1), 0), ((2, 8, 64), (1024, 80, 1), 0),
                     ((2, 8, 64), (1024, 96, 1), 0), ((2, 8, 64), (1024, 96, 1), 24),
                     ((3, 8, 64), (1024, 96, 1), 8), ((3, 5, 48), (1024, 96, 1), 16))
        adapter = DirectTriton(kernel)
        self.addCleanup(adapter.close)
        self.enterContext(mock.patch.dict(globals(), {"TRANSFORM": adapter}))
        runtime = DirectHost(host, contract)
        self.addCleanup(runtime.close)
        observations = self.enterContext(mock.patch.object(
            direct_host, "_observe_direct", wraps=direct_host._observe_direct,
        ))
        preparations = self.enterContext(mock.patch.object(
            direct_host, "_prepare_observed", wraps=direct_host._prepare_observed,
        ))
        captures = []

        def ordinary_parameters(owner, source, output, stream):
            module = owner.module
            expanded = []
            for tensor, metadata in zip((source, output), owner.binary.metadata.tensordesc_meta, strict=True):
                expanded.extend(make_tensordesc_arg(TensorDescriptor.from_tensor(tensor, list(block)), metadata, None))
            expanded.extend([None] * len(owner.scratch_abi_indices))
            grid = tuple((extent + tile - 1) // tile for extent, tile in zip(source.shape, block))
            grid = (*grid, *(1 for _ in range(3 - len(grid))))
            reference = torch.cuda.CUDAGraph(keep_graph=True)
            try:
                with torch.cuda.graph(reference, stream=stream):
                    module.C_impl._launch_kernel(
                        module.function, *grid, module.num_warps, module.shared,
                        module.arg_tys + "O" * len(owner.scratch_abi_indices), tuple(expanded), stream.cuda_stream,
                    )
                    frontier = torch._C._cuda_get_capture_frontier(stream.cuda_stream)
                node, = (node for node, _ in frontier[3])
                captured, = reference._inspect_captured_kernel_nodes((node,))[3]
                return tuple(payload for _, _, payload in captured[8])
            finally:
                reference.reset()

        make_replay = replay._make_replay

        def inspect_capture(*args, **kwargs):
            _, input_count, allocations, _, copies, calls, launches, buffers, stream = args
            output_layout, = allocations
            call, = calls
            launch, = launches
            self.assertIs(type(call), _PhysicalCall)
            self.assertEqual(copies, ())
            self.assertEqual(len(call.tensor_maps), 2)
            self.assertEqual(tuple(field.pointer.root for field in call.tensor_maps),
                             (InputSource(input_count - 1), output_layout.source))
            input_map, output_map = call.tensor_maps
            dimensions = tuple(IntExpr("boxed", index) for index in reversed(range(rank)))
            self.assertEqual(input_map.dimensions, dimensions)
            self.assertEqual(output_map.dimensions, dimensions)
            stride_sources = (() if rank == 1 else tuple(
                IntExpr("multiply", None, (IntExpr("boxed", index), IntExpr("constant", dtype.itemsize)))
                for index in (4, 3)
            ))
            self.assertEqual(input_map.strides, stride_sources)
            owner = call.module.owner
            self.assertFalse(owner.module.global_scratch_size)
            self.assertFalse(owner.module.profile_scratch_size)
            self.assertEqual(launch.argument_bytes, ordinary_parameters(
                owner, kwargs["capture_inputs"][-1], buffers[output_layout.source], stream,
            ))
            for field in call.tensor_maps:
                widths = tuple(len(payload) for payload in launch.argument_bytes[
                    field.parameter:field.parameter + 1 + 2 * rank
                ])
                self.assertEqual(widths, (128,) + (4,) * rank + (8,) * rank)
            captures.append(tuple(field.parameter for field in call.tensor_maps))
            return make_replay(*args, **kwargs)

        self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=inspect_capture))
        samples = []
        capacity = max(1 + sum((extent - 1) * step for extent, step in zip(shape, stride)) + offset
                       for shape, stride, offset in cases)
        shared_storage = torch.randn(capacity + 32, dtype=dtype, device=device)
        for index, (shape, stride, offset) in enumerate(cases):
            span = 1 + sum((extent - 1) * step for extent, step in zip(shape, stride))
            storage = shared_storage if index < 3 else torch.randn(span + offset + 32, dtype=dtype, device=device)
            source = torch.as_strided(storage, shape, stride, offset)
            self.assertEqual(source.data_ptr(), storage.data_ptr() + offset * source.element_size())
            self.assertEqual(source.data_ptr() % 16, 0)
            inputs = (*shape, source) if rank == 1 else (*shape, *stride[:2], source)
            samples.append(inputs)
        self.assertEqual(len({inputs[-1].data_ptr() for inputs in samples[:3]}), 1)
        self.assertEqual(len({inputs[-1].data_ptr() for inputs in samples}), len(samples) - 2)
        if rank == 1:
            self.assertEqual(len({tuple(inputs[-1].shape) for inputs in samples[:3]}), 3)
        else:
            self.assertEqual(len({tuple(inputs[-1].shape) for inputs in samples[:3]}), 1)
            self.assertEqual(samples[0][-1].stride()[1:], samples[1][-1].stride()[1:])
            self.assertEqual(samples[1][-1].stride()[0], samples[2][-1].stride()[0])
            self.assertNotEqual(samples[0][-1].stride()[0], samples[1][-1].stride()[0])
            self.assertNotEqual(samples[1][-1].stride()[1], samples[2][-1].stride()[1])
        held, aliases, encoded_maps, stride_rejections = [], [], 0, 0
        for index, inputs in enumerate(samples):
            box, frames = list(inputs), []

            def profile(frame, event, result):
                if event == "call":
                    frames.append(frame.f_code)

            if index == 0:
                actual, returned_source, view = runtime(box)
            else:
                try:
                    sys.setprofile(profile)
                    actual, returned_source, view = runtime.entry(box)
                finally:
                    sys.setprofile(None)
                self.assertEqual(frames, [])
            self.assertEqual(box, [])
            self.assertEqual(observations.call_count, 1)
            self.assertEqual(preparations.call_count, 1)
            self.assertEqual(len(runtime.variants), 1)
            source = inputs[-1]
            ordinary, _, _ = host(list(inputs))
            expected = source + 1
            self.assertEqual(actual, ordinary, atol=0, rtol=0)
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertTrue(actual.is_contiguous())
            self.assertIs(returned_source, source)
            start = 4 if rank == 1 else 8
            self.assertTrue(torch._C._is_alias_of(view, source))
            self.assertEqual(view, source[..., start:])
            self.assertEqual(view.storage_offset(), source.storage_offset() + start)
            self.assertEqual(view.stride(), source.stride())
            self.assertNotEqual(actual.data_ptr(), ordinary.data_ptr())
            program = runtime.variants[0].program
            event, = [event for event in program.events if type(event) is DirectPhysicalCall]
            call = event.bound
            output_layout, = program.allocations
            numeric = _NumericProgram(program, list(inputs))
            buffers = {output_layout.source: actual}
            buffer_indices = {output_layout.source: len(inputs)}
            reference_stream.wait_stream(torch.cuda.current_stream())
            parameters = ordinary_parameters(event.owner, source, actual, reference_stream)
            for field, tensor in zip(call.tensor_maps, (source, actual), strict=True):
                self.assertEqual(field.encode(numeric, inputs, buffers, buffer_indices), parameters[field.parameter])
                self.assertEqual(tuple(numeric.values[numeric.add(value)] for value in field.dimensions),
                                 tuple(reversed(tensor.shape)))
                self.assertEqual(tuple(numeric.values[numeric.add(value)] for value in field.strides),
                                 tuple(step * tensor.element_size() for step in reversed(tensor.stride()[:-1])))
                encoded_maps += 1
            if rank == 3:
                guard = runtime.variants[0].guard
                self.assertIsNotNone(guard)
                integers = [inputs[slot] for slot in guard.boxed_integer_indices]
                pointers = [inputs[slot].data_ptr() for slot in guard.boxed_pointer_indices]
                offsets = [inputs[slot].storage_offset() for slot in guard.boxed_storage_offset_indices]
                values = integers + pointers + offsets
                predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                            ctypes.POINTER(ctypes.c_double))(guard.function_address)
                self.assertEqual(predicate((ctypes.c_int64 * len(values))(*values), None), 1)
                for slot in (3, 4):
                    self.assertIn(slot, guard.boxed_integer_indices)
                    probe = values.copy()
                    probe[guard.boxed_integer_indices.index(slot)] += 1
                    stride = list(source.stride())
                    stride[slot - 3] += 1
                    span = 1 + sum((extent - 1) * step for extent, step in zip(source.shape, stride))
                    self.assertLessEqual((source.storage_offset() + span) * source.element_size(),
                                         source.untyped_storage().nbytes())
                    self.assertEqual(predicate((ctypes.c_int64 * len(probe))(*probe), None), 0)
                    stride_rejections += 1
            held.append((actual, expected))
            aliases.append((returned_source, view, source))
        self.assertEqual(len(captures), 1)
        self.assertEqual(len(adapter._owners), 1)
        self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(held))
        module, = [owner.module for owner in adapter._owners]
        runtime.close()
        self.assertEqual(module._graph_borrows, 0)
        adapter.close()
        for actual, expected in held:
            self.assertEqual(actual, expected, atol=0, rtol=0)
        for returned_source, view, source in aliases:
            self.assertIs(returned_source, source)
            self.assertEqual(view, source[..., 4 if rank == 1 else 8:])
        print("HOST_TMA_COMPONENT_RESULT=" + json.dumps({
            "rank": rank, "dtype": str(dtype), "samples": len(samples), "variants": 1,
            "native_hits": len(samples) - 1, "ordinary_references": len(samples),
            "same_pointer_metadata_samples": 3, "distinct_input_addresses": len(samples) - 2,
            "map_slots": captures[0], "native_map_byte_comparisons": encoded_maps,
            "stride_alignment_guard_rejections": stride_rejections,
            "shapes_strides_offsets": cases, "held_outputs_after_close": len(held),
        }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestHostTmaComponents, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
