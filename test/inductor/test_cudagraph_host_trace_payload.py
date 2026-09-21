# Owner(s): ["module: inductor"]

import ctypes
import unittest
from types import SimpleNamespace

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph.host_trace import (
    _HostIntegers,
    lower_host_trace,
)
from torch._inductor.runtime._cudagraph.host_trace_guards import (
    compile_host_trace_guard,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntExpr,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    integer_payload_contract,
    integer_payload_contract_from_guards,
    simplify_integer_payload,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_cuda import _get_torch_cuda_version
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import (
    CeilToInt,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    Min,
    Mod,
)


def evaluate_guard(guard, tensors):
    metadata = torch._C._cuda_boxed_tensor_metadata
    values = [tensors[index].data_ptr() for index in guard.boxed_pointer_indices]
    values.extend(
        tensors[index].storage_offset() for index in guard.boxed_storage_offset_indices
    )
    values.extend(
        metadata(tensors[index], kind)
        if dimension is None
        else metadata(tensors[index], kind, dimension)
        for kind, index, dimension in guard.metadata_bindings
    )
    bits = (ctypes.c_uint64 * len(values))(*(value % (1 << 64) for value in values))
    function = ctypes.CFUNCTYPE(
        ctypes.c_int8, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)
    )(guard.function_address)
    return function(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None)


@unittest.skipUnless(
    hasattr(torch._C, "_cuda_boxed_tensor_metadata"),
    "requires native graph Tensor metadata bindings",
)
@instantiate_parametrized_tests
class TestHostTracePayload(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.tensor = torch.ones(9)
        self.record = SimpleNamespace(
            position=0,
            name="arg0",
            dtype=torch.float32,
            device=self.tensor.device,
            pinned=False,
            sizes=[self.symbol("size", 9)],
            strides=[self.symbol("stride", 1)],
            offset=self.symbol("offset", 0),
            root=SimpleNamespace(
                name="input",
                itemsize=4,
                sym=self.symbol("base", self.tensor.data_ptr()),
            ),
        )
        self.output = SimpleNamespace(
            root=self.record.root,
            dtype=torch.float32,
            sizes=[self.record.sizes[0]],
            strides=[1],
            offset=self.record.offset,
        )
        self.tape = SimpleNamespace(
            shape_env=self.environment,
            inputs=[self.record],
            allocs=[],
            launches=[],
            outputs=[self.output],
            guards=[],
            nargs=1,
            device=SimpleNamespace(index=-1),
            rng_increment=None,
            opaque=[],
        )

    def symbol(self, name, hint):
        source = LocalSource(name)
        expression = self.environment.create_unspecified_symbol(
            hint, source, DimDynamic.DYNAMIC
        )
        return self.environment.create_symintnode(expression, hint=hint, source=source)

    def lower(self):
        self.tape.guards = list(
            dict.fromkeys(
                (*self.tape.guards, *(guard.expr for guard in self.environment.guards))
            )
        )
        return lower_host_trace(self.tape, [self.tensor])

    @parametrize("rows,expected", ((8, 1), (16, 1), (7, 0), (17, 0)))
    def test_range_fold_keeps_original_guards(self, rows, expected):
        original = self.record.sizes[0].node._expr
        self.tape.guards = [sympy.Ge(original, 8), sympy.Le(original, 16)]
        self.output.sizes = [Min(original, 4)]
        program, mapping, extra = self.lower()
        self.assertEqual(program.outputs[0].size, (IntExpr("constant", 4),))
        self.assertIn(original, mapping.metadata_symbols)
        self.assertEqual(
            mapping.translate(sympy.Ge(original, 8)), sympy.Ge(original, 8)
        )
        guard = compile_host_trace_guard(self.tape, mapping, [self.tensor], extra)
        self.assertEqual(evaluate_guard(guard, [torch.ones(rows)]), expected)

    @parametrize("rows,expected", ((9, 1), (10, 0)))
    def test_equality_fold_keeps_original_binding(self, rows, expected):
        size = self.record.sizes[0]
        original = size.node._expr
        self.output.sizes = [FloorDiv(original, 3)]
        self.assertTrue(size == 9)
        program, mapping, extra = self.lower()
        self.assertEqual(program.outputs[0].size, (IntExpr("constant", 3),))
        self.assertIn(original, mapping.metadata_symbols)
        self.assertEqual(mapping.metadata_symbols[original].property, "size")
        self.assertEqual(
            mapping.translate(sympy.Eq(original, 9)), sympy.Eq(original, 9)
        )
        guard = compile_host_trace_guard(self.tape, mapping, [self.tensor], extra)
        self.assertEqual(evaluate_guard(guard, [torch.ones(rows)]), expected)

    def test_exported_divisibility_guard_is_not_duplicated(self):
        size = self.record.sizes[0].node._expr
        self.tape.guards = [sympy.Eq(Mod(size, 3), 0, evaluate=False)]
        self.output.sizes = [FloorDiv(size, 3)]
        program, mapping, extra = self.lower()
        obligation = sympy.Eq(Mod(size, 3), 0, evaluate=False)
        predicates = (*self.tape.guards, *extra)
        self.assertEqual(predicates.count(obligation), 1)
        self.assertEqual(predicates[0], obligation)
        guard = compile_host_trace_guard(self.tape, mapping, [self.tensor], extra)
        self.assertEqual(evaluate_guard(guard, [torch.ones(9)]), 1)
        self.assertEqual(evaluate_guard(guard, [torch.ones(10)]), 0)
        self.assertEqual(program.outputs[0].source, InputSource(0))

    def test_input_pointer_and_offset_mapping_is_unchanged(self):
        self.tensor = torch.ones(20)[3:12]
        self.output.offset = self.record.offset + 2
        size = self.record.sizes[0].node._expr
        self.tape.guards = [sympy.Ge(size, 8), sympy.Le(size, 16)]
        self.output.sizes = [Min(size, 4)]
        program, mapping, _ = self.lower()
        (address,) = mapping.address_symbols
        self.assertEqual(
            mapping.translate(self.record.root.sym),
            address - 4 * self.record.offset.node._expr,
        )
        self.assertEqual(mapping.root_alignments, {InputSource(0): 1})
        self.assertEqual(program.outputs[0].source, InputSource(0))
        self.assertEqual(program.outputs[0].offset, 2)

    def test_float_barrier_is_untouched(self):
        mapping = HostTraceSymbolMapping(self.tape)
        expression = Identity(sympy.Float(0.5) * self.record.sizes[0].node._expr)
        contract = integer_payload_contract(self.environment)
        lower = _HostIntegers(mapping, contract)
        self.assertEqual(mapping.translate(expression), expression)
        with self.assertRaisesRegex(UnsupportedCapture, "integer-only"):
            lower(expression)
        self.assertEqual(mapping.translate(expression), expression)

    def test_explicit_snapshot_ignores_unexported_and_later_facts(self):
        original = self.record.sizes[0].node._expr
        guards = []
        contract = integer_payload_contract_from_guards(guards, (original,))
        self.assertTrue(self.record.sizes[0] % 3 == 0)
        guards.append(sympy.Eq(original, 9))
        expression = FloorDiv(original, 3)
        actual, _ = simplify_integer_payload(expression, contract.shape_env, contract)
        self.assertEqual(actual, expression)
        self.assertFalse(contract.shape_env.backed_var_to_val)
        current = integer_payload_contract_from_guards(guards, (original,))
        actual, _ = simplify_integer_payload(expression, current.shape_env, current)
        self.assertEqual(actual, 3)
        actual, _ = simplify_integer_payload(expression, contract.shape_env, contract)
        self.assertEqual(actual, expression)

    @parametrize("equality", (False, True))
    def test_transient_specialization_remains_a_runtime_guard(self, equality):
        self.tensor = torch.ones(12)
        size = self.record.sizes[0].node._expr
        predicate = sympy.Eq(size, 12) if equality else sympy.Ge(size, 10)
        with self.environment.patch_source_specialization(
            LocalSource("size"),
            lambda value: value == 12 if equality else value >= 10,
        ):
            self.tape.guards = [predicate]
            program, mapping, extra = self.lower()
        self.assertEqual(self.tape.guards, [predicate])
        self.assertTrue((*self.tape.guards, *extra))
        guard = compile_host_trace_guard(self.tape, mapping, [self.tensor], extra)
        self.assertEqual(evaluate_guard(guard, [torch.ones(12)]), 1)
        self.assertEqual(evaluate_guard(guard, [torch.ones(9)]), 0)
        size = program.outputs[0].size[0]
        if equality:
            self.assertEqual(size, 12)
        else:
            numeric = _NumericProgram(program, [self.tensor])
            slot = numeric.add(size)
            self.assertEqual(numeric.values[slot], 12)

    @parametrize("equality", (False, True))
    def test_stale_payload_contract_declines_after_transient_specialization(
        self, equality
    ):
        mapping = HostTraceSymbolMapping(self.tape)
        contract = integer_payload_contract(self.environment)
        lower = _HostIntegers(mapping, contract)
        with self.environment.patch_source_specialization(
            LocalSource("size"),
            lambda value: value == 12 if equality else value >= 10,
        ):
            with self.assertRaisesRegex(UnsupportedCapture, "cannot be simplified"):
                lower(self.record.sizes[0].node._expr + 1)

    def test_low_level_syntax_lowering_remains_contract_free(self):
        mapping = HostTraceSymbolMapping(self.tape)
        size = self.record.sizes[0].node._expr
        stride = self.record.strides[0].node._expr
        expression = FloorDiv(size, stride)
        lower = _HostIntegers(mapping)
        result = lower(expression)
        self.assertEqual(result.op, "floordiv")
        self.assertIn(sympy.Ge(size, 0), lower.guards)
        self.assertIn(sympy.Gt(stride, 0), lower.guards)
        guard = compile_host_trace_guard(
            self.tape, mapping, [self.tensor], lower.guards
        )
        zero_stride = torch.ones(1).expand(9)
        self.assertEqual(evaluate_guard(guard, [zero_stride]), 0)
        self.assertEqual(evaluate_guard(guard, [self.tensor]), 1)

    @parametrize("offset,stride,expected", ((0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)))
    def test_explicit_facts_keep_conditional_division_domain(
        self, offset, stride, expected
    ):
        mapping = HostTraceSymbolMapping(self.tape)
        size = self.record.sizes[0].node._expr
        step = self.record.strides[0].node._expr
        start = self.record.offset.node._expr
        self.tape.guards = [sympy.Ge(size, 1), sympy.Ge(step, 0), sympy.Ge(start, 0)]
        contract = integer_payload_contract_from_guards(
            self.tape.guards, mapping.metadata_symbols
        )
        lower = _HostIntegers(mapping, contract)
        lower(0)
        expression = Min(
            sympy.Piecewise((FloorDiv(size, step), sympy.Eq(start, 0)), (0, True)),
            0,
        )
        self.assertEqual(lower(expression), IntExpr("constant", 0))
        guard = compile_host_trace_guard(
            self.tape, mapping, [self.tensor], lower.guards
        )
        tensor = torch.ones(32).as_strided((9,), (stride,), offset)
        self.assertEqual(evaluate_guard(guard, [tensor]), expected)

    @parametrize("operation,expected", (("identity", 9), ("ceil", 42), ("floor", 41)))
    def test_tape_keeps_integer_identity_and_arange_syntax(self, operation, expected):
        size = self.record.sizes[0].node._expr
        expression = {
            "identity": Identity(size),
            "ceil": CeilToInt(IntTrueDiv(size**2 + 2, 2)),
            "floor": FloorToInt(IntTrueDiv(size**2 + 2, 2)),
        }[operation]
        self.tape.guards = [sympy.Ge(size, 3)]
        self.output.sizes = [expression]
        program, mapping, _ = self.lower()
        numeric = _NumericProgram(program, [self.tensor])
        slot = numeric.add(program.outputs[0].size[0])
        self.assertEqual(numeric.values[slot], expected)
        contract = integer_payload_contract(self.environment)
        with self.assertRaisesRegex(UnsupportedCapture, "integer-only"):
            _HostIntegers(mapping, contract)(expression)

    @parametrize("barrier", ("float", "rational", "identity"))
    def test_explicit_context_does_not_learn_noninteger_facts(self, barrier):
        size = self.record.sizes[0].node._expr
        bound = {
            "float": sympy.Float(8.5),
            "rational": sympy.Rational(17, 2),
            "identity": Identity(sympy.Integer(9)),
        }[barrier]
        guard = sympy.Ge(size, bound, evaluate=False)
        guards = [guard]
        contract = integer_payload_contract_from_guards(guards, (size,))
        expression = Min(size, 4)
        actual, _ = simplify_integer_payload(expression, contract.shape_env, contract)
        self.assertEqual(actual, expression)
        self.assertEqual(guards, [guard])
        self.assertEqual(contract.guards, ())


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and _get_torch_cuda_version() >= (12, 8)
    and hasattr(torch._C, "_HostTraceRecorder"),
    "requires NVIDIA CUDA >= 12.8 and host tracing",
)
class TestHostTracePayloadDevice(TestCase):
    @unittest.skipUnless(
        hasattr(torch._C, "_CUDAGraphCompiledEvaluation")
        and hasattr(torch._C, "_CUDAGraphBoxedDispatch"),
        "requires native host-trace replay",
    )
    def test_guarded_view_offset_retraces_and_reuses(self, device):
        from torch._inductor.runtime._cudagraph.host_trace import HostTraceReplay

        width = 4096

        def fn(x):
            rows = x.size(0)
            if 8 <= rows <= 16:
                offset = (rows // 32) * width
            else:
                offset = width
            output = torch.ops.aten.native_layer_norm.default(
                x, [width], None, None, 1e-5
            )[0]
            return output.as_strided((4, width), (width, 1), offset)

        inputs = [
            torch.randn(rows, width, device=device) for rows in (9, 12, 40, 10, 41)
        ]
        self.assertEqual(len({x.data_ptr() for x in inputs}), len(inputs))
        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        held = []
        for x, variants in zip(inputs, (1, 1, 2, 2, 2)):
            expected = fn(x)
            box = [x]
            (actual,) = replay(box)
            self.assertEqual(box, [])
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(len(replay.variants), variants)
            held.append((actual, expected))

        replay.close()
        for actual, expected in held:
            self.assertEqual(actual, expected, atol=0, rtol=0)

    def test_real_tape_range_reduces_numeric_dag(self, device):
        from torch.cuda import _host_trace as ht

        def fn(x):
            rows = x.size(0)
            if not 8 <= rows <= 16:
                raise RuntimeError("rows outside the traced branch")
            result = torch.ops.aten.native_layer_norm.default(
                x, [4096], None, None, 1e-5
            )[0]
            return result.as_strided((4, 4096), (4096, 1), (rows // 32) * 4096)

        example = torch.ones(9, 4096, device=device)
        tape = ht.trace(fn, (example,))
        raw_guards = tuple(tape.guards)
        baseline_mapping = HostTraceSymbolMapping(tape)
        raw_offset = _HostIntegers(baseline_mapping).value(tape.outputs[0].offset)
        program, mapping, extra = lower_host_trace(tape, [example])
        self.assertEqual(tuple(tape.guards), raw_guards)
        self.assertEqual(mapping.metadata_symbols, baseline_mapping.metadata_symbols)
        self.assertEqual(mapping.root_alignments, baseline_mapping.root_alignments)
        self.assertIsInstance(program.outputs[0].source, BufferSource)
        old_numeric = _NumericProgram(program, [example])
        new_numeric = _NumericProgram(program, [example])
        old_slot = old_numeric.add(raw_offset)
        new_slot = new_numeric.add(program.outputs[0].offset)
        self.assertEqual(new_numeric.values[new_slot], 0)
        self.assertEqual(old_numeric.values[old_slot], new_numeric.values[new_slot])
        self.assertGreater(len(old_numeric.instructions), len(new_numeric.instructions))
        guard = compile_host_trace_guard(tape, mapping, [example], extra)
        self.assertEqual(evaluate_guard(guard, [example]), 1)
        self.assertEqual(evaluate_guard(guard, [torch.ones(7, 4096, device=device)]), 0)
        self.assertEqual(
            evaluate_guard(guard, [torch.ones(17, 4096, device=device)]), 0
        )


instantiate_device_type_tests(TestHostTracePayloadDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
