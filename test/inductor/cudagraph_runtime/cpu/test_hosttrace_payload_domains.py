# Owner(s): ["module: inductor"]
"""The domains of a payload's partial operations on this line: the lowering keeps the
original payload (no value simplification), so a division's domain is an obligation
whenever the division is lowered, a nested division's domain precedes the relation over
its value, and a conditional payload declines by name. The runtime team's R22 scalar
cases (cleanup_r22/guards) re-pinned to this behaviour."""

import ctypes
import unittest.mock
from types import SimpleNamespace

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph import direct_hosttrace
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import (
    FXTraceDeclined,
)
from torch._inductor.runtime._cudagraph.address_scalars import symbolic_integer
from torch._inductor.runtime._cudagraph.direct_hosttrace import (
    _HostIntegers,
    _Lowering,
    _PAYLOAD_SIMPLIFY_SYMBOLS,
    HostTraceLoweringDeclined,
)
from torch._inductor.runtime._cudagraph.host_trace_guards import (
    compile_host_trace_guard,
)
from torch._inductor.runtime.cudagraph_arg_mapping import IntExpr
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    integer_payload_contract_from_guards,
    simplify_integer_payload,
)
from torch._inductor.runtime.cudagraph_host_trace_mapping import HostTraceSymbolMapping
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv, Mod


@instantiate_parametrized_tests
class TestHostTracePayloadDomains(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.tensor = torch.empty(9)
        self.record = SimpleNamespace(
            position=0,
            name="arg0",
            dtype=self.tensor.dtype,
            device=self.tensor.device,
            pinned=False,
            sizes=[self.symbol("size", 9)],
            strides=[self.symbol("stride", 1)],
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
        size, step = self.record.sizes[0], self.record.strides[0]
        self.assertTrue(size >= 0)
        self.assertTrue(step >= 0)
        self.size, self.step = size.node._expr, step.node._expr

    def symbol(self, name, hint):
        source = LocalSource(name)
        expression = self.environment.create_unspecified_symbol(
            hint, source, DimDynamic.DYNAMIC
        )
        return self.environment.create_symintnode(expression, hint=hint, source=source)

    def integers(self):
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        mapping = HostTraceSymbolMapping(self.tape)
        return _HostIntegers(mapping), mapping

    def evaluate(self, guard, tensor):
        values = [tensor.data_ptr() for _ in guard.boxed_pointer_indices]
        values.extend(
            tensor.storage_offset() for _ in guard.boxed_storage_offset_indices
        )
        values.extend(
            torch._C._cuda_boxed_tensor_metadata(tensor, kind, dimension)
            for kind, _, dimension in guard.metadata_bindings
        )
        bits = (ctypes.c_uint64 * len(values))(*(value % (1 << 64) for value in values))
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(guard.function_address)
        return predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None)

    @parametrize("stride", (0, 1, 2))
    def test_division_domain_is_an_obligation_of_the_original_payload(self, stride):
        expression = torch.sym_min(self.record.sizes[0] // self.record.strides[0], 0)
        self.assertTrue(expression.node._expr.has(FloorDiv))
        integers, mapping = self.integers()
        zero = integers(sympy.Integer(0))
        lowered = integers(expression)
        # the payload is lowered as written: the division stays in the plan
        self.assertIsNot(lowered, zero)
        self.assertEqual(lowered.op, "min")
        self.assertIn("floordiv", [arg.op for arg in lowered.args])
        self.assertEqual(
            integers.guards, [sympy.Ge(self.size, 0), sympy.Gt(self.step, 0)]
        )
        self.assertIs(integers(expression), lowered)
        self.assertEqual(len(integers.guards), 2)
        guard = compile_host_trace_guard(
            self.tape, mapping, [self.tensor], tuple(integers.guards)
        )
        tensor = torch.empty(8 * stride + 1).as_strided((9,), (stride,))
        if stride:
            self.assertEqual(min(tensor.size(0) // tensor.stride(0), 0), 0)
        else:
            with self.assertRaises(ZeroDivisionError):
                min(tensor.size(0) // tensor.stride(0), 0)
        self.assertEqual(self.evaluate(guard, tensor), int(stride != 0))

    @parametrize("stride,offset", ((0, 0), (0, 1), (1, 0), (1, 1)))
    def test_conditional_division_declines_by_name(self, stride, offset):
        expression = sympy.Piecewise(
            (
                torch.sym_min(
                    self.record.sizes[0] // self.record.strides[0], 0
                ).node._expr,
                sympy.Eq(self.record.offset.node._expr, 0),
            ),
            (0, True),
        )
        integers, _ = self.integers()
        # no Piecewise lowering: the shared plan evaluates every instruction, so an
        # unselected division could not stay unselected; the tape declines by name
        with self.assertRaisesRegex(HostTraceLoweringDeclined, "Piecewise"):
            integers(expression)
        self.assertEqual(integers.guards, [])
        tensor = torch.empty(8 * stride + offset + 1).as_strided(
            (9,), (stride,), offset
        )
        if offset or stride:
            self.assertEqual(
                0 if offset else min(tensor.size(0) // tensor.stride(0), 0), 0
            )
        else:
            with self.assertRaises(ZeroDivisionError):
                min(tensor.size(0) // tensor.stride(0), 0)

    def test_a_nested_division_domain_precedes_the_relation_over_its_value(self):
        inner = FloorDiv(self.size, self.step)
        outer = FloorDiv(inner + 1, 2)  # (a // b) // c would fold to a // (b * c)
        integers, _ = self.integers()
        integers(outer)
        # the divisor 2 folds its own domain away; the inner division's precedes the
        # relation over its value; the payload's sum carries its signed-range
        # obligation last (every sum, product and power of a payload, A299 / 6a949808db7)
        self.assertEqual(
            integers.guards,
            [
                sympy.Ge(self.size, 0),
                sympy.Gt(self.step, 0),
                sympy.Ge(inner + 1, 0),
                sympy.Le(inner + 1, 2**63 - 1, evaluate=False),
            ],
        )

    def test_a_frozen_contract_survives_the_compression_of_a_decided_divisibility(self):
        # the recorded order puts the divisibility before the equality that decides it
        # (the symm tp layer: Eq(Mod(2*d*b, 16), 0) then Eq(d, 4096)); the frozen
        # ShapeEnv drops the decided fact from `divisible` at its first FloorDiv
        # simplification, which must not read as a changed contract
        d, b = sympy.Symbol("d", integer=True), sympy.Symbol("b", integer=True)
        guards = (sympy.Eq(Mod(2 * d * b, 16), 0), sympy.Eq(d, 4096))
        contract = integer_payload_contract_from_guards(guards, (d, b))
        expression, env = 512 * FloorDiv(4 * b + 511, 512), contract.shape_env
        simplified, _ = simplify_integer_payload(expression, env, contract)
        self.assertEqual(simplified.free_symbols, {b})
        again = simplify_integer_payload(expression, env, contract)[0]
        self.assertEqual(again, simplified)

    def test_the_tape_lowering_keeps_a_wide_payload_as_written(self):
        # a Min / Max over many terms costs the simplifier a static evaluation per
        # pair of arguments (a wide cat's output length: 8-11 s each); the tape
        # lowering simplifies up to _PAYLOAD_SIMPLIFY_SYMBOLS free symbols and lowers
        # a wider payload as written
        symbols = sympy.symbols(f"a0:{_PAYLOAD_SIMPLIFY_SYMBOLS + 1}", integer=True)
        contract = integer_payload_contract_from_guards((), symbols)
        calls = []

        def counting(expression, shape_env, contract=None):
            calls.append(expression)
            return simplify_integer_payload(expression, shape_env, contract)

        # the plan node itself is not the point: _lower is stubbed, only the
        # simplification decision is observed
        with (
            unittest.mock.patch.object(
                direct_hosttrace, "simplify_integer_payload", counting
            ),
            unittest.mock.patch.object(_Lowering, "_lower", return_value=None),
        ):
            lowering = _Lowering(SimpleNamespace(), {}, contract, strict_payload=False)
            lowering.lower(sympy.Max(*symbols[:-1]))
            lowering.lower(sympy.Max(*symbols))
        self.assertEqual(calls, [sympy.Max(*symbols[:-1])])

    @parametrize("columns", (0, 128, 129, 1024))
    def test_dynamic_floor_divisor_survives_guard_translation(self, columns):
        size = IntExpr("size", 0, (IntExpr("constant", 0),))
        chunks = IntExpr(
            "floordiv",
            args=(
                IntExpr("add", args=(IntExpr("constant", 127), size)),
                IntExpr("constant", 128),
            ),
        )
        recipe = IntExpr(
            "floordiv",
            args=(
                IntExpr(
                    "add",
                    args=(
                        IntExpr("constant", -1),
                        IntExpr("multiply", args=(IntExpr("constant", 2), chunks)),
                    ),
                ),
                chunks,
            ),
        )
        expression = symbolic_integer(recipe, {("size", 0, 0): self.size})
        self.assertTrue(expression.has(FloorDiv))
        _, mapping = self.integers()
        guard = compile_host_trace_guard(
            self.tape,
            mapping,
            [self.tensor],
            (sympy.Eq(expression, 1, evaluate=False),),
        )
        self.assertEqual(self.evaluate(guard, torch.empty(columns)), int(columns > 0))
        if columns:
            count = (columns + 127) // 128
            self.assertEqual(
                expression.subs(self.size, columns), (2 * count - 1) // count
            )

    @parametrize("numerator", ("zero", "same"))
    @parametrize("stride", (0, 2))
    def test_translated_floor_keeps_cancelled_denominator_domain(
        self, numerator, stride
    ):
        divisor = IntExpr("stride", 0, (IntExpr("constant", 0),))
        dividend = IntExpr("constant", 0) if numerator == "zero" else divisor
        recipe = IntExpr("floordiv", args=(dividend, divisor))
        expression = symbolic_integer(recipe, {("stride", 0, 0): self.step})
        self.assertIsInstance(expression, FloorDiv)
        self.assertEqual(expression.args[1], self.step)
        _, mapping = self.integers()
        guard = compile_host_trace_guard(
            self.tape,
            mapping,
            [self.tensor],
            (sympy.Eq(expression, int(numerator == "same"), evaluate=False),),
        )
        tensor = torch.empty(9 * stride + 1).as_strided((9,), (stride,))
        self.assertEqual(self.evaluate(guard, tensor), int(stride > 0))

    @parametrize("divisor", (0, -1))
    def test_translated_floor_still_refuses_nonpositive_literal(self, divisor):
        recipe = IntExpr(
            "floordiv", args=(IntExpr("constant", 1), IntExpr("constant", divisor))
        )
        with self.assertRaisesRegex(FXTraceDeclined, "Guard exceeds"):
            symbolic_integer(recipe, {})


if __name__ == "__main__":
    run_tests()
