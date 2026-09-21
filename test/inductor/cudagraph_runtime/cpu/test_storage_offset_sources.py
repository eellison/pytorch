"""Original storage metadata survives tracing, signed views and local guards."""

from contextlib import nullcontext
import ctypes
from dataclasses import replace
from unittest import mock


import sympy
import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import FXTraceDeclined
from torch._inductor.runtime._cudagraph.api import InputContract, TensorInput
from torch._inductor.runtime._cudagraph.direct_host import _direct_origin
from torch._inductor.runtime._cudagraph.extraction import trace_host
from torch._inductor.runtime._cudagraph.frontend import lower_terminal
from torch._inductor.runtime._cudagraph.guard_export import export_guards, GuardExportDeclined, prepare_guard
from torch._dynamo.source import TensorProperty, TensorPropertySource
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntegerOutput, TensorViewOutput
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges


copy_if_misaligned = torch._C._dynamo.guards.copy_if_misaligned


def views(box):
    tensor, = box
    box.clear()
    return tensor.as_strided((4,), (1,), 0), tensor[1:5], tensor.storage_offset()


def normalized_relative(box):
    tensor, = box
    box.clear()
    tensor = copy_if_misaligned(tensor)
    return (tensor[1:5],)


def normalized_absolute(box):
    tensor, = box
    box.clear()
    tensor = copy_if_misaligned(tensor)
    return (tensor.as_strided((4,), (1,), 0),)


def offset_before_normalize(box):
    tensor, = box
    box.clear()
    offset = tensor.storage_offset()
    tensor = copy_if_misaligned(tensor)
    return tensor, offset


def normalized_branch(box):
    tensor, = box
    box.clear()
    tensor = copy_if_misaligned(tensor)
    return (tensor[1:5] if tensor.storage_offset() == 0 else tensor[2:6],)


def offset_branch(box):
    tensor, = box
    box.clear()
    return (tensor[1:5] if tensor.storage_offset() % 2 == 0 else tensor[2:6],)


def offset_equality_branch(box):
    tensor, = box
    box.clear()
    return (tensor[1:5] if tensor.storage_offset() == 3 else tensor[2:6],)


def extract(host, offset):
    example = torch.empty(32)[offset:offset + 8]
    contract = InputContract(("tensor",), (TensorInput(0, torch.float32, (8,), (1,)),), (), device_index=0)
    origin, _ = _direct_origin(host, contract)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        trace = trace_host(host, contract, [example], (), None,
                           direct=True, context_factory=lambda state: nullcontext())
    return replace(trace, compiler_binding=origin), example


@instantiate_parametrized_tests
class TestStorageOffsetSources(TestCase):
    @parametrize("offset", (0, 1, 3))
    def test_original_metadata_and_signed_view_displacements(self, offset):
        trace, example = extract(views, offset)
        binding, = trace.storage_offset_bindings
        self.assertIs(type(binding.source), TensorPropertySource)
        self.assertIs(binding.source.prop, TensorProperty.STORAGE_OFFSET)
        self.assertEqual(binding.source.base.local_name, "boxed_0")
        self.assertEqual(trace.placeholders[0].storage_offset().node.expr, binding.symbol)
        self.assertEqual(trace.tensor_roots.inputs[0].root, InputSource(0))
        self.assertEqual(trace.tensor_roots.inputs[0].byte_offset, 0)
        program = lower_terminal(trace, ())
        self.addCleanup(program.close)
        absolute, relative, original = program.outputs
        self.assertIs(type(absolute), TensorViewOutput)
        self.assertIs(type(relative), TensorViewOutput)
        self.assertIs(type(original), IntegerOutput)
        self.assertEqual(relative.offset, 1)
        numeric = _NumericProgram(program, [example])
        delta = numeric.add(absolute.offset)
        current = numeric.add(original.value)
        self.assertEqual(numeric.values[delta], -offset)
        self.assertEqual(numeric.values[current], offset)
        self.assertIn(("storage_offset", 0), numeric.instructions)
        self.assertEqual(program.guards.expressions, ())


    def test_normalized_relative_displacement_cancels(self):
        trace, _ = extract(normalized_relative, 3)
        program = lower_terminal(trace, ())
        self.addCleanup(program.close)
        output, = program.outputs
        self.assertEqual(output.offset, 1)
        symbol = trace.storage_offset_bindings[0].symbol
        guards = export_guards(trace, (sympy.Ge(symbol, 0),))
        self.assertEqual(guards.expressions, ())

    @parametrize("host", (normalized_absolute, offset_before_normalize, normalized_branch))
    def test_unrepresented_normalization_generation_declines(self, host):
        trace, _ = extract(host, 3)
        with self.assertRaisesRegex(FXTraceDeclined, "[Ss]torage-offset|storage-offset"):
            lower_terminal(trace, ())

    @parametrize("change", ("range", "replacement", "predicate"))
    def test_normalization_preflight_uses_original_domain(self, change):
        trace, _ = extract(normalized_relative, 3)
        symbol = trace.storage_offset_bindings[0].symbol
        environment = trace.shape_env
        before = environment.var_to_range[symbol]
        replacements = dict(environment.replacements)
        try:
            extra = ()
            if change == "range":
                environment.var_to_range[symbol] = ValueRanges(0, 8)
            elif change == "replacement":
                environment.replacements[symbol] = sympy.Integer(3)
            else:
                extra = (sympy.Eq(symbol, 3),)
            with self.assertRaisesRegex(GuardExportDeclined, "[Ss]torage-offset|storage-offset"):
                export_guards(trace, extra)
        finally:
            environment.var_to_range[symbol] = before
            environment.replacements.clear()
            environment.replacements.update(replacements)

    def test_original_offset_branch_uses_metadata_guard_slot(self):
        trace, example = extract(offset_branch, 2)
        program = lower_terminal(trace, ())
        self.addCleanup(program.close)
        guard = prepare_guard(program, [example])
        self.assertIsNotNone(guard)
        self.assertEqual(guard.boxed_integer_indices, ())
        self.assertEqual(guard.boxed_pointer_indices, ())
        self.assertEqual(guard.boxed_storage_offset_indices, (0,))
        self.assertEqual(len(guard.registration), 5)
        predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                    ctypes.POINTER(ctypes.c_double))(guard.function_address)
        for offset in (0, 1, 2, 3, (1 << 40) + 1):
            self.assertEqual(predicate((ctypes.c_int64 * 1)(offset), None), int(offset % 2 == 0))


    @parametrize("trace_offset", (3, 7))
    def test_offset_equality_preserves_both_branch_predicates(self, trace_offset):
        trace, example = extract(offset_equality_branch, trace_offset)
        program = lower_terminal(trace, ())
        self.addCleanup(program.close)
        guard = prepare_guard(program, [example])
        self.assertIsNotNone(guard)
        self.assertEqual(guard.boxed_integer_indices, ())
        self.assertEqual(guard.boxed_pointer_indices, ())
        self.assertEqual(guard.boxed_storage_offset_indices, (0,))
        self.assertEqual(len(guard.registration), 5)
        predicate = ctypes.CFUNCTYPE(ctypes.c_int8, ctypes.POINTER(ctypes.c_int64),
                                    ctypes.POINTER(ctypes.c_double))(guard.function_address)
        for offset in (0, 1, 2, 3, 7, (1 << 40) + 1, (1 << 63) - 1):
            self.assertEqual(predicate((ctypes.c_int64 * 1)(offset), None),
                             int((offset == 3) == (trace_offset == 3)))


if __name__ == "__main__":
    run_tests()
