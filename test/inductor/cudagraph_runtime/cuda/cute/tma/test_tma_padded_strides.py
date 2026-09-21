"""Replay CuTe TMA with independent padded strides and fp16/bf16 inputs."""

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
from torch._inductor.runtime._cudagraph import cute_adapter, direct_host, replay
from torch._inductor.runtime._cudagraph.api import (
    DirectCuTe, DirectHost, InputContract, IntegerRange, ObservedOrdinaryEntry,
    PythonEntry, SignaturePolicy, TensorInput,
)
from torch._inductor.runtime._cudagraph.cute_types import CuTeCall
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import TensorProperties
from torch._inductor.runtime.cudagraph_arg_mapping import (
    ExpressionSource, InputSource, IntExpr, ParameterSource, PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _ParameterProgram, _PhysicalCall
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, recover_orig_fp32_precision, run_tests, TestCase


WORKTREE = next(parent for parent in Path(__file__).resolve().parents if (parent / "torch/_inductor").is_dir())
SUPPORT = WORKTREE / "test/inductor/cudagraph_runtime/cuda/cute"
sys.path.insert(0, str(SUPPORT))
from fixture_support import load_source

UPSTREAM = WORKTREE / "third_party/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py"
upstream = load_source("_padded_stride_upstream_gemm", UPSTREAM)
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
    lda, ldb, a, b = box
    box.clear()
    output = torch.empty_strided((128, 128, 1), (128, 1, 16384), dtype=torch.float32, device=a.device)
    TMA(a, b, output)
    return (output,)


def boxed_origins(values):
    pending, result = list(values), set()
    while pending:
        value = pending.pop()
        if type(value) is IntExpr:
            if value.op == "boxed":
                result.add(value.value)
            pending.extend(value.args)
        elif type(value) is ParameterSource:
            pending.extend((*value.args, value.value))
        elif type(value) is PointerSource:
            pending.append(value.byte_offset)
        elif type(value) is ExpressionSource:
            pending.append(value.expression)
    return result


class TestTmaPaddedStrides(TestCase):
    @recover_orig_fp32_precision
    @tf32_off()
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_independent_leading_strides(self, device, dtype):
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
            lda, ldb = (IntExpr("boxed", index) for index in (0, 1))
            batch_a = IntExpr("multiply", None, (IntExpr("constant", 128), lda))
            batch_b = IntExpr("multiply", None, (IntExpr("constant", 128), ldb))
            contract = InputContract(
                ("integer", "integer", "tensor", "tensor"),
                (TensorInput(2, dtype, (128, 128, 1), (lda, 1, batch_a)),
                 TensorInput(3, dtype, (128, 128, 1), (ldb, 1, batch_b))),
                (IntegerRange(0, 128, 256), IntegerRange(1, 128, 256)),
                device_index=torch.cuda.current_device(),
            )
            runtime = DirectHost(host, contract)
            self.addCleanup(runtime.close)
            observations = self.enterContext(mock.patch.object(
                direct_host, "_observe_direct", wraps=direct_host._observe_direct,
            ))
            traces = self.enterContext(mock.patch.object(direct_host, "trace_host", wraps=direct_host.trace_host))
            operands_seen, captures = [], []
            original_operands, make_replay = cute_adapter._Operands, replay._make_replay

            def observe_operands(*args, **kwargs):
                operands = original_operands(*args, **kwargs)
                operands_seen.append(operands)
                return operands

            def capture(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs):
                self.assertEqual(input_count, 4)
                self.assertEqual(copies, ())
                self.assertEqual(len(allocations), 1)
                call, = calls
                launch, = launches
                self.assertIs(type(call), _PhysicalCall)
                self.assertEqual(call.module.cluster, (1, 1, 1))
                self.assertEqual(boxed_origins(call.grid), set())
                self.assertEqual(boxed_origins(field.source for field in call.fields), {0, 1})
                numeric = kwargs["numeric"]
                parameters = _ParameterProgram(numeric, input_count, {allocations[0].source: input_count})
                late = [(field, parameters.add(field.source)) for field in call.fields
                        if type(field.source) is ParameterSource]
                self.assertTrue(late)
                words = parameters.evaluate(kwargs["capture_inputs"], buffers)
                for field, index in late:
                    payload = struct.pack("i" if field.source.width == 32 else "q", words[index])
                    image = launch.argument_bytes[field.parameter]
                    self.assertEqual(image[field.byte_offset:field.byte_offset + len(payload)], payload)
                self.assertEqual(set(parameters.roots.values()), {InputSource(2), InputSource(3), allocations[0].source})
                captures.append(len(late))
                return make_replay(graph, input_count, allocations, outputs, copies, calls, launches, buffers, stream, **kwargs)

            self.enterContext(mock.patch.object(cute_adapter, "_Operands", side_effect=observe_operands))
            self.enterContext(mock.patch.object(replay, "_make_replay", side_effect=capture))
            storage = [(torch.randn(128 * 256, dtype=dtype, device=device) * 0.125,
                        torch.randn(128 * 256, dtype=dtype, device=device) * 0.125) for _ in range(3)]
            specs = ((0, 136, 144), (0, 160, 144), (0, 160, 176),
                     (1, 160, 176), (2, 192, 224), (0, 136, 144))
            samples = []
            for bank, left_stride, right_stride in specs:
                a_storage, b_storage = storage[bank]
                a = a_storage.as_strided((128, 128, 1), (left_stride, 1, 128 * left_stride))
                b = b_storage.as_strided((128, 128, 1), (right_stride, 1, 128 * right_stride))
                self.assertEqual((a.data_ptr() % 16, b.data_ptr() % 16), (0, 0))
                self.assertEqual((a.stride()[0], b.stride()[0]), (left_stride, right_stride))
                samples.append((left_stride, right_stride, a, b))
            pairs = [(a.data_ptr(), b.data_ptr()) for _, _, a, b in samples]
            self.assertEqual(pairs[:3], [pairs[0]] * 3)
            self.assertEqual(len(set(pairs)), 3)
            held = []
            for index, inputs in enumerate(samples):
                box, frames = list(inputs), []

                def profile(frame, event, result):
                    if event == "call":
                        frames.append((frame.f_code.co_filename, frame.f_code.co_name))

                ordinary, = host(list(inputs))
                expected = inputs[2][..., 0].float() @ inputs[3][..., 0].float().T
                self.assertEqual(ordinary[..., 0], expected, atol=2e-3, rtol=2e-4)
                if index == 0:
                    actual, = runtime(box)
                else:
                    try:
                        sys.setprofile(profile)
                        actual, = runtime.entry(box)
                    finally:
                        sys.setprofile(None)
                    self.assertEqual(frames, [])
                self.assertEqual(box, [])
                self.assertEqual(actual, ordinary, atol=0, rtol=0)
                self.assertEqual(actual[..., 0], expected, atol=2e-3, rtol=2e-4)
                self.assertEqual((tuple(actual.shape), actual.stride()), ((128, 128, 1), (128, 1, 16384)))
                held.append((actual, actual.clone()))
            self.assertEqual(observations.call_count, 1)
            self.assertEqual(traces.call_count, 1)
            self.assertEqual(len(captures), 1)
            self.assertEqual(owner._capture.calls, 1)
            variant, = runtime.variants
            call, = [event for event in variant.program.events if type(event) is CuTeCall]
            self.assertIs(call.receipt.invocation.compilation.selected, owner.selected)
            operands, = operands_seen
            self.assertIs(operands.artifact, call.bound.module.artifact)
            imports = set()
            pending = [field.source.expression for field in
                       (*call.bound.module.site.fields.pointers, *call.bound.module.site.fields.integers)
                       if field.source.kind == "compiler_expression"]
            while pending:
                expression = pending.pop()
                if expression.kind == "argument":
                    imports.add((expression.source_arg_index, expression.path))
                pending.extend(expression.operands)
            formals = {formal.name: formal for formal in operands.artifact.formals}
            for name, index in (("a", 0), ("b", 1)):
                formal = formals[name]
                leaf, = [leaf for leaf in formal.leaves if leaf.property == "stride" and leaf.property_path == (0,)]
                self.assertIn((formal.source_arg_index, leaf.path), imports)
                self.assertEqual(operands.numeric(formal.source_arg_index, leaf.path).expression, IntExpr("boxed", index))
                self.assertEqual(operands.numeric_value(operands.property(formal, "stride", (1,))).expression,
                                 IntExpr("constant", 1))
            self.assertEqual(gemm.a_dtype, cutlass.Float16 if dtype is torch.float16 else cutlass.BFloat16)
            self.assertEqual(gemm.b_dtype, gemm.a_dtype)
            artifact = operands.artifact
            site = call.bound.module.site
            self.assertTrue(site.tma_stride_domains)
            self.assertEqual(sorted(index for domain in site.tma_stride_domains for index in domain.indices),
                             list(range(len(site.tma_stride_divisors))))
            self.assertTrue(site.tma_dimension_domains)
            shape_indices = tuple(index for domain in site.tma_dimension_domains for index in domain.indices)
            self.assertEqual(shape_indices, tuple(range(len(shape_indices))))
            self.assertEqual(shape_indices, tuple(consumer.index for consumer in artifact.consumers
                                                  if consumer.site_id == site.site_id and consumer.role == "tma_shape"))
            stride_indices = tuple(index for domain in site.tma_dimension_domains if domain.grouped
                                   for index in domain.indices)
            self.assertEqual(stride_indices, tuple(consumer.index for consumer in artifact.consumers
                                                   if consumer.site_id == site.site_id
                                                   and consumer.role == "tma_dimension_stride"))
            tensors = {"a": samples[0][2], "b": samples[0][3], "c": held[0][0]}
            properties = {formals[name].source_arg_index: TensorProperties(tuple(value.shape), value.stride())
                          for name, value in tensors.items()}
            selected, = artifact.select_sites(artifact.bind_properties(properties))
            self.assertIs(selected.site, site)
            for name in ("a", "b", "c"):
                tensor = tensors[name]
                self.assertEqual(tensor.size(2), 1)
                for delta in (-16, 0):
                    strides = list(tensor.stride())
                    strides[2] = ((1 << 40) + delta) // tensor.element_size()
                    view = tensor.as_strided(tensor.shape, strides)
                    self.assertEqual(view.data_ptr(), tensor.data_ptr())
                    self.assertEqual(view.untyped_storage().nbytes(), tensor.untyped_storage().nbytes())
                    probe = dict(properties)
                    probe[formals[name].source_arg_index] = TensorProperties(tuple(view.shape), view.stride())
                    metadata = artifact.bind_properties(probe)
                    if delta < 0:
                        selected, = artifact.select_sites(metadata)
                        self.assertIs(selected.site, site)
                    else:
                        # Selection evaluates metadata only; it never encodes or launches this descriptor.
                        with self.assertRaisesRegex(ValueError, "outer byte stride must be below"):
                            artifact.select_sites(metadata)
            guard = variant.guard
            self.assertIsNotNone(guard)
            predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_double))(guard.function_address)

            def accepts(inputs):
                values = [inputs[index] for index in guard.boxed_integer_indices]
                values.extend(inputs[index].data_ptr() for index in guard.boxed_pointer_indices)
                values.extend(inputs[index].storage_offset() for index in guard.boxed_storage_offset_indices)
                return predicate((ctypes.c_int64 * len(values))(*values), None)

            valid = samples[0]
            self.assertEqual(accepts(valid), 1)
            for index in (0, 1):
                probe = list(valid)
                probe[index] += 1
                source = probe[index + 2]
                stride = probe[index]
                probe[index + 2] = source.as_strided((128, 128, 1), (stride, 1, 128 * stride))
                self.assertEqual(probe[index + 2].data_ptr(), source.data_ptr())
                self.assertLessEqual((127 * stride + 128) * source.element_size(), source.untyped_storage().nbytes())
                self.assertNotEqual(stride * source.element_size() % 16, 0)
                # Only the predicate sees this invalid TMA stride; no kernel is launched.
                self.assertEqual(accepts(probe), 0, f"Missing byte-stride guard for operand {index}, dtype {dtype}")
            self.assertEqual(len({actual.data_ptr() for actual, _ in held}), len(samples))
            runtime.close()
            self.assertTrue(call.receipt.closed)
            self.assertEqual(owner._native_borrows, set())
            owner.close()
            for actual, expected in held:
                self.assertEqual(actual, expected)
            print("TMA_PADDED_STRIDES_RESULT=" + json.dumps({
                "accepted": True, "dtype": str(dtype), "samples": len(samples),
                "leading_strides": [[left, right] for _, left, right in specs],
                "descriptor_stride_axes": {"a": [0], "b": [0]}, "boxed_stride_sources": [0, 1],
                "constant_grid": True, "same_pointer_stride_updates": 2, "distinct_pointer_pairs": len(set(pairs)),
                "invalid_byte_stride_guard_rejections": 2,
                "cold_stride_domain_checks": 6,
                "captures": len(captures), "variants": 1, "ordinary_compilations": 1,
                "native_hits": len(samples) - 1, "ordinary_references": len(samples),
                "held_outputs_after_close": len(held),
            }, sort_keys=True), flush=True)


instantiate_device_type_tests(TestTmaPaddedStrides, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
