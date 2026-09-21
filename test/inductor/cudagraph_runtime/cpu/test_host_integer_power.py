# Owner(s): ["module: inductor"]

import ctypes
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph.host_trace import _HostIntegers
from torch._inductor.runtime._cudagraph.host_trace_guards import (
    _OrderedPrinter,
    compile_host_trace_guard,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_numeric,
    EarlyStatus,
    integer_payload_contract,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.value_ranges import ValueRanges


@instantiate_parametrized_tests
class TestHostIntegerPower(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.tensor = torch.empty((1, 1))
        self.record = SimpleNamespace(
            position=0,
            name="arg0",
            dtype=self.tensor.dtype,
            device=self.tensor.device,
            pinned=False,
            sizes=[self.symbol("rows", 1), self.symbol("columns", 1)],
            strides=[self.symbol("row_stride", 1), self.symbol("column_stride", 1)],
            offset=self.symbol("offset", 0),
            root=SimpleNamespace(
                name="p0",
                itemsize=4,
                sym=self.symbol("base", self.tensor.untyped_storage().data_ptr()),
            ),
        )
        self.tape = SimpleNamespace(
            shape_env=self.environment,
            inputs=[self.record],
            allocs=[],
            opaque=[],
            nargs=1,
            guards=[],
            device=SimpleNamespace(index=-1),
        )
        self.rows, self.columns = (value.node._expr for value in self.record.sizes)

    def symbol(self, name, hint):
        source = LocalSource(name)
        expression = self.environment.create_unspecified_symbol(
            hint, source, DimDynamic.DYNAMIC
        )
        return self.environment.create_symintnode(expression, hint=hint, source=source)

    def lower(self, expression):
        contract = integer_payload_contract(self.environment)
        mapping = HostTraceSymbolMapping(self.tape)
        integers = _HostIntegers(mapping, contract)
        value = integers(expression)
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        return value, integers, mapping

    def compile_numeric(self, value):
        records = SimpleNamespace(input_names=("arg0",), integer_inputs=())
        program = _NumericProgram(records, (self.tensor,))
        output = program.add(value)
        compiled = compile_numeric(program)
        self.assertNotIn("std::pow", compiled.cpp_source)
        return compiled, output

    def evaluate_numeric(self, compiled, rows, columns=1):
        sizes = (rows, columns)
        leaves = tuple(
            sizes[dimension] for kind, index, dimension in compiled.leaf_bindings
        )
        return compiled.evaluate_leaves(leaves)

    def compile_guard(self, integers, mapping):
        return compile_host_trace_guard(
            self.tape, mapping, [self.tensor], tuple(integers.guards)
        )

    def evaluate_guard(self, guard, rows, columns=1):
        sizes = (rows, columns)
        values = [self.tensor.data_ptr() for _ in guard.boxed_pointer_indices]
        values.extend(
            self.tensor.storage_offset() for _ in guard.boxed_storage_offset_indices
        )
        values.extend(
            sizes[dimension]
            if kind == "size"
            else torch._C._cuda_boxed_tensor_metadata(self.tensor, kind, dimension)
            for kind, index, dimension in guard.metadata_bindings
        )
        bits = (ctypes.c_uint64 * len(values))(*(value % (1 << 64) for value in values))
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        return predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None)

    @parametrize("exponent", (0, 1, 2, 3, 5, 8, 13, 63))
    @parametrize("base", (-2, -1, 0, 1, 2, 17))
    def test_exact_native_integer_values(self, exponent, base):
        value, _, _ = self.lower(self.rows**exponent)
        compiled, output = self.compile_numeric(value)
        expected = base**exponent
        status, actual = self.evaluate_numeric(compiled, base)
        if -(2**63) <= expected < 2**63:
            self.assertEqual(status, EarlyStatus.SUCCESS)
            self.assertEqual(actual[output], expected)
        else:
            self.assertEqual(status, EarlyStatus.MULTIPLY_OVERFLOW)

    @parametrize(
        "exponent,base",
        (
            (2, 3037000499),
            (2, 3037000500),
            (3, 2097151),
            (3, 2097152),
            (63, -2),
            (63, 2),
        ),
    )
    def test_native_int64_boundary(self, exponent, base):
        value, _, _ = self.lower(self.rows**exponent)
        compiled, output = self.compile_numeric(value)
        status, actual = self.evaluate_numeric(compiled, base)
        expected = base**exponent
        if -(2**63) <= expected < 2**63:
            self.assertEqual(status, EarlyStatus.SUCCESS)
            self.assertEqual(actual[output], expected)
        else:
            self.assertEqual(status, EarlyStatus.MULTIPLY_OVERFLOW)

    @parametrize("base,exponent", ((0, 0), (0, 3), (1, 63), (-1, 63), (-1, 64)))
    def test_constant_folding(self, base, exponent):
        value, _, _ = self.lower(sympy.Pow(base, exponent))
        compiled, output = self.compile_numeric(value)
        status, actual = compiled.evaluate_leaves(())
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(actual[output], base**exponent)

    @parametrize("kind", ("symbolic", "negative", "fractional", "floating"))
    def test_unsupported_exponent_declines(self, kind):
        exponent = {
            "symbolic": self.columns,
            "negative": sympy.Integer(-1),
            "fractional": sympy.Rational(1, 2),
            "floating": sympy.Float(2),
        }[kind]
        with self.assertRaises(UnsupportedCapture):
            self.lower(sympy.Pow(self.rows, exponent, evaluate=False))

    @parametrize(
        "exponent,rows,expected",
        (
            (2, 3037000499, 1),
            (2, 3037000500, 0),
            (3, 2097151, 1),
            (3, 2097152, 0),
            (63, 1, 1),
            (63, 2, 0),
            (63, 1000, 0),
        ),
    )
    def test_reuse_guard_checks_integer_range(self, exponent, rows, expected):
        _, integers, mapping = self.lower(self.rows**exponent)
        guard = self.compile_guard(integers, mapping)
        expressions = "\n".join(guard.expressions)
        self.assertIn("guard_power(", expressions)
        self.assertNotIn("guard_float_power(", expressions)
        self.assertEqual(self.evaluate_guard(guard, rows), expected)

    @parametrize(
        "exponent,rows,expected",
        ((3, 2097154, 1), (3, 2097155, 0), (63, 4, 1), (63, 5, 0), (63, 1002, 0)),
    )
    def test_guard_power_preserves_negative_base(self, exponent, rows, expected):
        mapping = HostTraceSymbolMapping(self.tape)
        power = (2 - self.rows) ** exponent
        bounds = (sympy.Ge(power, -(2**63)), sympy.Lt(power, 2**63))
        guard = compile_host_trace_guard(self.tape, mapping, [self.tensor], bounds)
        self.assertEqual(self.evaluate_guard(guard, rows), expected)

    @parametrize("exponent,rows", ((3, 2097152), (63, 2)))
    def test_canonical_positive_intermediate_remains_guarded(self, exponent, rows):
        _, integers, mapping = self.lower((-self.rows) ** exponent)
        guard = self.compile_guard(integers, mapping)
        self.assertEqual((-rows) ** exponent, -(2**63))
        self.assertEqual(self.evaluate_guard(guard, rows), 0)

    @parametrize("rows,expected", ((1, 1), (0, 0), (2**62, 1)))
    def test_unselected_power_guard_does_not_overflow(self, rows, expected):
        mapping = HostTraceSymbolMapping(self.tape)
        condition = sympy.Or(self.rows >= 2, sympy.Eq(self.rows**3, 1))
        guard = compile_host_trace_guard(
            self.tape, mapping, [self.tensor], (condition,)
        )
        self.assertEqual(self.evaluate_guard(guard, rows), expected)

    def test_replacement_product_keeps_raw_equality(self):
        self.environment._set_replacement(self.columns, self.rows, "test")
        value, integers, mapping = self.lower(self.rows * self.columns)
        compiled, output = self.compile_numeric(value)
        status, actual = self.evaluate_numeric(compiled, 4, 4)
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(actual[output], 16)
        guard = self.compile_guard(integers, mapping)
        self.assertEqual(self.evaluate_guard(guard, 4, 4), 1)
        self.assertEqual(self.evaluate_guard(guard, 4, 5), 0)

    def test_eliminated_power_keeps_range_obligation(self):
        self.environment.constrain_symbol_range(self.rows, 1, 16)
        value, integers, mapping = self.lower(self.rows**0)
        compiled, output = self.compile_numeric(value)
        status, actual = compiled.evaluate_leaves(())
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(actual[output], 1)
        guard = self.compile_guard(integers, mapping)
        self.assertEqual(self.evaluate_guard(guard, 8), 1)
        self.assertEqual(self.evaluate_guard(guard, 17), 0)

    def test_new_specialization_rejects_finalized_contract(self):
        integers = _HostIntegers(
            HostTraceSymbolMapping(self.tape),
            integer_payload_contract(self.environment),
        )
        self.environment._set_replacement(self.rows, sympy.Integer(1), "test")
        with self.assertRaisesRegex(UnsupportedCapture, "finalized"):
            integers(self.rows**2)

    def test_integer_power_does_not_request_hints(self):
        integers = _HostIntegers(
            HostTraceSymbolMapping(self.tape),
            integer_payload_contract(self.environment),
        )
        with (
            mock.patch.object(
                self.environment,
                "size_hint",
                side_effect=AssertionError("hint requested"),
            ),
            mock.patch.object(
                self.environment,
                "evaluate_expr",
                side_effect=AssertionError("guard hint requested"),
            ),
        ):
            value = integers(self.rows**2)
        compiled, output = self.compile_numeric(value)
        status, actual = self.evaluate_numeric(compiled, 17)
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(actual[output], 289)

    @parametrize("exponent", (1, 2, 3))
    @parametrize("sign", ("positive", "negative", "signed"))
    def test_power_fact_preserves_integer_domains_and_sign(self, exponent, sign):
        assumptions = {} if sign == "signed" else {sign: True}
        rows = sympy.Symbol("fact_rows", integer=True, **assumptions)
        lower, upper = {"positive": (1, 4), "negative": (-4, -1), "signed": (-4, 4)}[
            sign
        ]
        source = LocalSource("rows")
        sources = {rows: [source]}
        printer = _OrderedPrinter(
            sources,
            lambda source: source.local_name,
            sources,
            domains={rows: ValueRanges(lower, upper)},
            sources=[source],
        )
        printer.assume(sympy.Ge(rows**2, 0, evaluate=False))
        endpoints = (lower**exponent, upper**exponent)
        expected = (
            (0, 16) if sign == "signed" and exponent == 2 else tuple(sorted(endpoints))
        )
        bounds = printer.bounds(rows**exponent)
        self.assertTrue(bounds.is_int)
        self.assertIsInstance(bounds.lower, sympy.Integer)
        self.assertIsInstance(bounds.upper, sympy.Integer)
        self.assertEqual((bounds.lower, bounds.upper), expected)
        negative = printer.bounds(-rows)
        self.assertTrue(negative.is_int)
        self.assertEqual((negative.lower, negative.upper), (-upper, -lower))


if __name__ == "__main__":
    run_tests()
