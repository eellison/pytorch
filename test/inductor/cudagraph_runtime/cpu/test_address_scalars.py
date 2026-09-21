"""Check structural address arithmetic against the shared native evaluator."""

from types import SimpleNamespace



import sympy
import torch
from torch._inductor.runtime._cudagraph._compiler.fx_adapter.contract import FXTraceDeclined, NormalizeEvent
from torch._inductor.runtime._cudagraph.address_scalars import lower_address_scalar
from torch._inductor.runtime.cudagraph_arg_mapping import BufferSource, InputSource, IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram, _ParameterProgram
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase
from torch.utils._sympy.functions import FloorDiv


UINT64_MAX = (1 << 64) - 1


def make_trace(root, lower=0, upper=UINT64_MAX, generation=0, normalized=False):
    environment = ShapeEnv()
    pointer = environment.create_unbacked_symint()
    symbol = pointer.node.expr
    environment.constrain_symbol_range(symbol, lower, upper)
    count = environment.create_unbacked_symint().node.expr
    environment.constrain_symbol_range(count, 1, 1024)
    trace = SimpleNamespace(shape_env=environment, symbol_sources={count: IntExpr("boxed", 0)},
        address_bindings=(SimpleNamespace(root=root, generation=generation, symbol=symbol),),
        storage_offset_bindings=(),
        events=(NormalizeEvent(None),) if normalized else (),
        tensor_roots=SimpleNamespace(event=lambda event: SimpleNamespace(root=root)))
    return trace, pointer, count


def lower(trace, expression, abi="i64"):
    def early(value):
        return int(value) if isinstance(value, sympy.Integer) else trace.symbol_sources[value]

    return lower_address_scalar(expression, abi, trace, early)


class TestAddressScalars(TestCase):
    @parametrize("case", ("owned_bits", "composition", "signed_result", "early_operand", "bounded_offset",
                         "canonicalized_dividend"))
    def test_exact_native_arithmetic(self, case):
        root = InputSource(1) if case == "bounded_offset" else BufferSource("owned")
        trace, pointer, count = make_trace(root, 8 if case == "bounded_offset" else 0,
                                          (1 << 63) - 1 if case == "bounded_offset" else UINT64_MAX)
        symbol = pointer.node.expr
        low_bits = sympy.Mod(symbol, 1 << 31)
        expression, abi = {
            "owned_bits": (low_bits, "i32"),
            "composition": (FloorDiv(low_bits * 3 + 7, 2), "i64"),
            "signed_result": (low_bits - 10, "i32"),
            "early_operand": (low_bits + count, "i64"),
            "bounded_offset": (symbol - 8, "i64"),
            "canonicalized_dividend": (FloorDiv(sympy.Mod(symbol, 16) - 16, 2), "i64"),
        }[case]
        if case == "canonicalized_dividend":
            self.assertEqual(expression, FloorDiv(sympy.Mod(symbol, 16), 2) - 8)
        source, guards = lower(trace, expression, abi)
        self.assertEqual(guards, ())
        self.assertEqual({item.root for item in source.pointers}, {root})
        records = SimpleNamespace(input_names=("count", "tensor"), integer_inputs=(IntegerInput("count", 0),))
        numeric = _NumericProgram(records, (7, None))
        program = _ParameterProgram(numeric, 2, {root: 2} if type(root) is BufferSource else {})
        program.add(source)
        samples = (8, 100, (1 << 63) - 1) if type(root) is InputSource else (0, 255, (1 << 63) + 19, UINT64_MAX)
        for address in samples:
            pointers = (0, address) if type(root) is InputSource else (0, 0, address)
            actual, = torch._C._cuda_evaluate_parameter_program(program.plan(), tuple(numeric.values), pointers)
            expected = ((address % 16 - 16) // 2 if case == "canonicalized_dividend" else
                        int(expression.subs({symbol: address, count: 7})))
            self.assertEqual(actual, expected)

    def test_full_input_address_retains_signed_abi_guard(self):
        trace, pointer, _ = make_trace(InputSource(1))
        source, guards = lower(trace, pointer)
        self.assertEqual(source.op, "pointer")
        self.assertEqual(source.width, 64)
        self.assertTrue(all(bool(guard.subs(pointer.node.expr, (1 << 63) - 1)) for guard in guards))
        self.assertFalse(all(bool(guard.subs(pointer.node.expr, 1 << 63)) for guard in guards))

    @parametrize("case", ("full_owned", "owned_i32_overflow", "intermediate_overflow", "negative_dividend"))
    def test_unproven_arithmetic_declines(self, case):
        trace, pointer, _ = make_trace(BufferSource("owned"))
        symbol = pointer.node.expr
        expression, abi = {
            "full_owned": (symbol, "i64"),
            "owned_i32_overflow": (sympy.Mod(symbol, 1 << 32), "i32"),
            "intermediate_overflow": (sympy.Mod(symbol + 8, 1 << 31), "i32"),
            "negative_dividend": (FloorDiv(sympy.Mod(symbol, 16) - 16, 2, evaluate=False), "i64"),
        }[case]
        with self.assertRaises(FXTraceDeclined):
            lower(trace, expression, abi)

    @parametrize("generation,normalized", ((1, False), (0, True)))
    def test_normalized_address_generation_declines(self, generation, normalized):
        trace, pointer, _ = make_trace(InputSource(1), generation=generation, normalized=normalized)
        with self.assertRaisesRegex(FXTraceDeclined, "normalization"):
            lower(trace, pointer)

    def test_foreign_environment_declines(self):
        trace, _, _ = make_trace(InputSource(1))
        _, other, _ = make_trace(InputSource(1))
        with self.assertRaisesRegex(FXTraceDeclined, "another tracing environment"):
            lower(trace, other)


instantiate_parametrized_tests(TestAddressScalars)

if __name__ == "__main__":
    run_tests()
