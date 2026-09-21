"""Replay upstream TMA descriptors built from shifted input views."""

import ctypes
import json
from pathlib import Path
import struct
import sys
from unittest import mock

from torch._inductor.runtime._cudagraph import _sdk

_sdk.activate()

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import get_leading_dim
from cuda.bindings import driver
import torch
from torch._inductor.runtime._cudagraph import direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe,
    DirectHost,
    InputContract,
    IntegerRange,
    ObservedOrdinaryEntry,
    PythonEntry,
    SignaturePolicy,
    TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr, ParameterSource
from torch._inductor.runtime.cudagraph_boxed_replay import _ParameterProgram, _PhysicalCall
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import recover_orig_fp32_precision, run_tests, TestCase

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from fixture_support import load_source, REPO_ROOT

UPSTREAM = REPO_ROOT / "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py"
upstream = load_source("_descriptor_view_upstream_gemm", UPSTREAM)
GEMM = None
TMA = None


@cute.jit
def invocation(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: driver.CUstream):
    GEMM(a, b, c, stream)


def convert_arguments(a, b, c):
    return tuple(
        from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=get_leading_dim(value))
        for value in (a, b, c)
    )


def host(box):
    rows, a, b = box
    box.clear()
    a = a[7:].view(1, rows, 128).permute(1, 2, 0)
    b = b[5:].view(1, 128, 128).permute(1, 2, 0)
    output = torch.empty_strided((rows, 128, 1), (128, 1, rows * 128), dtype=torch.float32, device=a.device)
    TMA(a, b, output)
    return (output,)


class TestTmaDescriptorViews(TestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    def test_addresses_shapes_and_effective_alignment(self, device):
        if torch.version.hip or torch.cuda.get_device_capability(device)[0] != 10:
            self.skipTest("The unchanged upstream GEMM requires SM100-family tcgen05")
        with torch.cuda.device(device):
            gemm = upstream.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
            self.enterContext(mock.patch.dict(globals(), {"GEMM": gemm}))
            owner = ObservedOrdinaryEntry(
                PythonEntry(invocation), gemm.kernel,
                policy=SignaturePolicy(32, 64, 16, "stream"), conversion=convert_arguments,
            )
            self.addCleanup(owner.close)
            self.enterContext(mock.patch.dict(globals(), {"TMA": DirectCuTe(owner)}))
            rows = IntExpr("boxed", 0)
            length = IntExpr("add", None, (
                IntExpr("multiply", None, (rows, IntExpr("constant", 128))), IntExpr("constant", 7),
            ))
            contract = InputContract(
                ("integer", "tensor", "tensor"),
                (TensorInput(1, torch.float16, (length,), (1,)),
                 TensorInput(2, torch.float16, (16389,), (1,))),
                (IntegerRange(0, 128, 256),), device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            captures = []
            make_replay = replay._make_replay

            def capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
                self.assertEqual(input_count, 3)
                self.assertEqual(len(allocations), 1)
                self.assertEqual(copies, ())
                call, = calls
                launch, = launches
                self.assertIs(type(call), _PhysicalCall)
                self.assertEqual(call.module.cluster, (1, 1, 1))
                numeric = kwargs["numeric"]
                parameters = _ParameterProgram(numeric, input_count, {allocations[0].source: input_count})
                late = [(field, parameters.add(field.source)) for field in call.fields
                        if type(field.source) is ParameterSource]
                descriptor_fields = [field for field, _ in late if field.source.pointers]
                self.assertTrue(descriptor_fields)
                pointers = {pointer for field in descriptor_fields for pointer in field.source.pointers}
                offsets = {(pointer.root, numeric.values[numeric.add(pointer.byte_offset)]) for pointer in pointers}
                self.assertEqual(offsets, {
                    (InputSource(1), 14), (InputSource(2), 10), (allocations[0].source, 0),
                })
                words = parameters.evaluate(kwargs["capture_inputs"], buffers)
                for field, index in late:
                    payload = struct.pack("i" if field.source.width == 32 else "q", words[index])
                    actual = launch.argument_bytes[field.parameter][field.byte_offset:field.byte_offset + len(payload)]
                    self.assertEqual(actual, payload)
                captures.append(len(descriptor_fields))
                return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            observe = self.enterContext(mock.patch.object(direct_host, "_observe_direct", wraps=direct_host._observe_direct))
            trace = self.enterContext(mock.patch.object(direct_host, "trace_host", wraps=direct_host.trace_host))
            samples = []
            for step, (m, a_offset, b_offset) in enumerate(((128, 1, 3), (128, 1, 3), (128, 9, 11),
                                                         (256, 17, 19), (128, 25, 27))):
                # Reserve one element for the predicate-only forward-shifted view.
                a_storage = torch.randn(m * 128 + 8 + a_offset, dtype=torch.float16, device=device) * 0.125 + step * 0.25
                b_storage = torch.randn(16390 + b_offset, dtype=torch.float16, device=device) * 0.125 - step * 0.125
                a = a_storage[a_offset:a_offset + m * 128 + 7]
                b = b_storage[b_offset:b_offset + 16389]
                self.assertEqual((a.storage_offset(), b.storage_offset()), (a_offset, b_offset))
                self.assertEqual((a.data_ptr() % 16, b.data_ptr() % 16), (2, 6))
                self.assertEqual((a[7:].data_ptr() % 16, b[5:].data_ptr() % 16), (0, 0))
                samples.append((m, a, b))
            self.assertEqual(len({a.data_ptr() for _, a, _ in samples}), len(samples))
            self.assertEqual(len({b.data_ptr() for _, _, b in samples}), len(samples))
            held = []
            for step, inputs in enumerate(samples):
                box, frames = list(inputs), []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                if step == 0:
                    actual, = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual, = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                ordinary, = host(list(inputs))
                m, a, b = inputs
                expected = a[7:].view(m, 128).float() @ b[5:].view(128, 128).float().T
                self.assertEqual(actual, ordinary, atol=0, rtol=0)
                self.assertEqual(actual[..., 0], expected, atol=2e-3, rtol=2e-4)
                self.assertNotEqual(actual.data_ptr(), ordinary.data_ptr())
                self.assertEqual((tuple(actual.shape), actual.stride()), ((m, 128, 1), (128, 1, m * 128)))
                held.append((actual, actual.clone()))
            self.assertEqual(observe.call_count, 1)
            self.assertEqual(trace.call_count, 1)
            self.assertEqual(len(captures), 1)
            self.assertEqual(owner._capture.calls, 1)
            variant, = runtime.variants
            call, = [event for event in variant.program.events if type(event) is CuTeCall]
            self.assertIs(call.receipt.invocation.compilation.selected, owner.selected)
            alignments = tuple(formal.data_alignment for formal in call.bound.module.artifact.formals if formal.kind == "Tensor")
            self.assertEqual(alignments, (16, 16, 16))
            guard = variant.guard
            self.assertEqual(set(guard.boxed_pointer_indices), {1, 2})
            predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_double))(guard.function_address)
            rejected = 0
            for inputs in samples:
                integers = [inputs[index] for index in guard.boxed_integer_indices]
                pointers = [inputs[index].data_ptr() for index in guard.boxed_pointer_indices]
                offsets = [inputs[index].storage_offset() for index in guard.boxed_storage_offset_indices]
                values = integers + pointers + offsets
                self.assertEqual(predicate((ctypes.c_int64 * len(values))(*values), None), 1)
                # Predicate-only probes never pass an invalid alignment to the TMA kernel.
                for ordinal, pointer in enumerate(pointers, start=len(integers)):
                    tensor_index = guard.boxed_pointer_indices[ordinal - len(integers)]
                    tensor = inputs[tensor_index]
                    for invalid in (pointer + 2, pointer - pointer % 16):
                        probe = values.copy()
                        probe[ordinal] = invalid
                        delta, remainder = divmod(invalid - pointer, tensor.element_size())
                        self.assertEqual(remainder, 0)
                        storage_offset = tensor.storage_offset() + delta
                        self.assertGreaterEqual(storage_offset, 0)
                        self.assertLessEqual((storage_offset + tensor.numel()) * tensor.element_size(),
                                             tensor.untyped_storage().nbytes())
                        if tensor_index in guard.boxed_storage_offset_indices:
                            slot = len(integers) + len(pointers) + guard.boxed_storage_offset_indices.index(tensor_index)
                            probe[slot] = storage_offset
                        self.assertEqual(predicate((ctypes.c_int64 * len(probe))(*probe), None), 0)
                        rejected += 1
            self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(samples))
            runtime.close()
            self.assertTrue(call.receipt.closed)
            self.assertEqual(owner._native_borrows, set())
            owner.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("TMA_DESCRIPTOR_VIEWS_RESULT=" + json.dumps({
                "accepted": True, "samples": len(samples), "rows": [inputs[0] for inputs in samples],
                "storage_offsets": [[a.storage_offset(), b.storage_offset()] for _, a, b in samples],
                "view_byte_offsets": [14, 10], "captures": len(captures), "variants": 1,
                "ordinary_compilations": 1, "native_hits": len(samples) - 1,
                "ordinary_references": len(samples), "alignment_guard_rejections": rejected,
                "guard_storage_offset_indices": list(guard.boxed_storage_offset_indices),
                "held_outputs_after_close": len(held),
            }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTmaDescriptorViews, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
