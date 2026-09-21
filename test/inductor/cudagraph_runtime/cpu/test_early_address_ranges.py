"""Check guards for the emitted early numeric tape inside address scalars."""

from types import SimpleNamespace



import sympy
from torch._inductor.runtime._cudagraph.address_scalars import lower_address_scalar
from torch._inductor.runtime.cudagraph_arg_mapping import InputSource, IntExpr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase
from torch.utils._sympy.functions import FloorDiv


I64_MAX = (1 << 63) - 1


class TestEarlyAddressRanges(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv()
        self.pointer = self.environment.create_unbacked_symint().node.expr
        self.environment.constrain_symbol_range(self.pointer, 0, (1 << 64) - 1)
        self.a = self.environment.create_unbacked_symint().node.expr
        self.b = self.environment.create_unbacked_symint().node.expr
        self.offset = self.environment.create_unbacked_symint().node.expr
        for symbol in (self.a, self.b, self.offset):
            self.environment.constrain_symbol_range(symbol, 0, I64_MAX)
        self.trace = SimpleNamespace(shape_env=self.environment,
            symbol_sources={self.a: IntExpr("boxed", 1), self.b: IntExpr("boxed", 2)},
            address_bindings=(SimpleNamespace(root=InputSource(0), generation=0, symbol=self.pointer),),
            storage_offset_bindings=(SimpleNamespace(index=0, symbol=self.offset),), events=())

    def guards(self, term, recipe):
        def early(value):
            self.assertEqual(value, term)
            return recipe

        expression = sympy.Mod(self.pointer, 2) * term
        source, guards = lower_address_scalar(expression, "i64", self.trace, early)
        self.assertEqual({pointer.root for pointer in source.pointers}, {InputSource(0)})
        self.assertTrue(guards)
        return guards

    def test_overflowing_prefix_rejects_even_when_final_value_fits(self):
        term = self.a + self.b - I64_MAX
        recipe = IntExpr("add", args=(
            IntExpr("add", args=(IntExpr("boxed", 1), IntExpr("boxed", 2))),
            IntExpr("constant", -I64_MAX)))
        guards = self.guards(term, recipe)
        valid = {self.pointer: 1, self.a: I64_MAX - 1, self.b: 1}
        overflow = {self.pointer: 1, self.a: I64_MAX, self.b: 1}
        self.assertEqual(term.subs(valid), 0)
        self.assertEqual(term.subs(overflow), 1)
        self.assertTrue(all(bool(guard.subs(valid)) for guard in guards))
        self.assertFalse(all(bool(guard.subs(overflow)) for guard in guards))

    def test_emitted_ceildiv_requires_nonnegative_numerator(self):
        term = FloorDiv(self.offset, 2)
        recipe = IntExpr("ceildiv", args=(
            IntExpr("add", args=(IntExpr("storage_offset", 0), IntExpr("constant", -1))),
            IntExpr("constant", 2)))
        guards = self.guards(term, recipe)
        self.assertTrue(all(bool(guard.subs({self.pointer: 1, self.offset: 1})) for guard in guards))
        self.assertTrue(all(bool(guard.subs({self.pointer: 1, self.offset: 2})) for guard in guards))
        self.assertFalse(all(bool(guard.subs({self.pointer: 1, self.offset: 0})) for guard in guards))

    @parametrize("op", ("and", "select"))
    def test_emitted_boolean_operand_requires_zero_or_one(self, op):
        term = sympy.Piecewise((1, sympy.Eq(self.offset, 1)), (0, True))
        offset = IntExpr("storage_offset", 0)
        one, zero = IntExpr("constant", 1), IntExpr("constant", 0)
        recipe = IntExpr(op, args=(offset, one) if op == "and" else (offset, one, zero))
        guards = self.guards(term, recipe)
        self.assertTrue(all(bool(guard.subs({self.pointer: 1, self.offset: 0})) for guard in guards))
        self.assertTrue(all(bool(guard.subs({self.pointer: 1, self.offset: 1})) for guard in guards))
        self.assertFalse(all(bool(guard.subs({self.pointer: 1, self.offset: 2})) for guard in guards))


instantiate_parametrized_tests(TestEarlyAddressRanges)

if __name__ == "__main__":
    run_tests()
