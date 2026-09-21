# Owner(s): ["module: inductor"]
"""Integer simplification preserves native guards and temporary ShapeEnv facts."""

import ctypes
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.source import LocalSource
from torch._inductor.runtime._cudagraph.host_trace import _HostIntegers
from torch._inductor.runtime._cudagraph.host_trace_guards import (
    compile_host_trace_guard,
)
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    _contract_is_current,
    integer_payload_contract,
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
from torch.utils._sympy.functions import (
    CleanDiv,
    FloorDiv,
    Identity,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    TruncToInt,
)
from torch.utils._sympy.value_ranges import ValueRanges


@instantiate_parametrized_tests
class TestIntegerPayloadSimplification(TestCase):
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
        # A synthetic CPU metadata contract exercises the predicate without a CUDA allocation.
        self.tape = SimpleNamespace(
            shape_env=self.environment,
            inputs=[self.record],
            allocs=[],
            opaque=[],
            nargs=1,
            guards=[],
            device=SimpleNamespace(index=-1),
        )

    def symbol(self, name, hint):
        source = LocalSource(name)
        expression = self.environment.create_unspecified_symbol(
            hint, source, DimDynamic.DYNAMIC
        )
        return self.environment.create_symintnode(expression, hint=hint, source=source)

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

    def compile_contract(self, contract):
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        mapping = HostTraceSymbolMapping(self.tape)
        return compile_host_trace_guard(
            self.tape,
            mapping,
            [self.tensor],
            tuple(mapping.translate(guard) for guard in contract.additional_guards),
        )

    def compile_simplified_zero(self, expression):
        self.tape.guards = [guard.expr for guard in self.environment.guards]
        mapping = HostTraceSymbolMapping(self.tape)
        integers = _HostIntegers(mapping, integer_payload_contract(self.environment))
        zero = integers(sympy.Integer(0))
        self.assertIs(integers(expression), zero)
        guard_count = len(integers.guards)
        self.assertIs(integers(expression), zero)
        self.assertEqual(len(integers.guards), guard_count)
        return compile_host_trace_guard(
            self.tape, mapping, [self.tensor], tuple(integers.guards)
        )

    @parametrize("stride", (0, 1, 2))
    def test_eliminated_division_domain(self, stride):
        size, step = self.record.sizes[0], self.record.strides[0]
        self.assertTrue(size >= 0)
        self.assertTrue(step >= 0)
        expression = torch.sym_min(size // step, 0)
        self.assertTrue(expression.node._expr.has(FloorDiv))
        guard = self.compile_simplified_zero(expression)
        tensor = torch.empty(8 * stride + 1).as_strided((9,), (stride,))
        if stride:
            self.assertEqual(min(tensor.size(0) // tensor.stride(0), 0), 0)
        else:
            with self.assertRaises(ZeroDivisionError):
                min(tensor.size(0) // tensor.stride(0), 0)
        self.assertEqual(self.evaluate(guard, tensor), int(stride != 0))

    @parametrize("stride,offset", ((0, 0), (0, 1), (1, 0), (1, 1)))
    def test_eliminated_conditional_division_domain(self, stride, offset):
        size, step = self.record.sizes[0], self.record.strides[0]
        self.assertTrue(size >= 0)
        self.assertTrue(step >= 0)
        expression = sympy.Piecewise(
            (
                torch.sym_min(size // step, 0).node._expr,
                sympy.Eq(self.record.offset.node._expr, 0),
            ),
            (0, True),
        )
        guard = self.compile_simplified_zero(expression)
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
        self.assertEqual(self.evaluate(guard, tensor), int(bool(offset or stride)))

    def test_payload_replacement_keeps_raw_guard(self):
        size = self.record.sizes[0]
        original = size.node._expr
        self.assertTrue(size == 9)
        result, contract = simplify_integer_payload(3 * original + 2, self.environment)
        self.assertEqual(result, 29)
        self.assertIn(sympy.Eq(original, 9, evaluate=False), contract.obligations)
        guard = self.compile_contract(contract)
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(10)), 0)

    @parametrize("value,expected", ((8, 1), (16, 1), (7, 0), (17, 0)))
    def test_payload_range_keeps_both_bounds(self, value, expected):
        size = self.record.sizes[0].node._expr
        self.environment.constrain_symbol_range(size, 8, 16)
        result, contract = simplify_integer_payload(Min(size, 4), self.environment)
        self.assertEqual(result, 4)
        self.assertIn(sympy.Ge(size, 8, evaluate=False), contract.obligations)
        self.assertIn(sympy.Le(size, 16, evaluate=False), contract.obligations)
        guard = self.compile_contract(contract)
        self.assertEqual(self.evaluate(guard, torch.empty(value)), expected)

    def test_payload_singleton_range_has_no_hint_dependency(self):
        size = self.record.sizes[0].node._expr
        self.environment.constrain_symbol_range(size, 9, 9)
        result, contract = simplify_integer_payload(size + 1, self.environment)
        self.assertEqual(result, 10)
        guard = self.compile_contract(contract)
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(10)), 0)

    @parametrize("value,expected", ((9, 1), (12, 1), (10, 0)))
    def test_payload_python_mod_preserves_actual_shapeenv_behavior(
        self, value, expected
    ):
        size = self.record.sizes[0]
        original = size.node._expr
        self.assertTrue(size % 3 == 0)
        result, contract = simplify_integer_payload(
            FloorDiv(original, 3), self.environment
        )
        self.assertIsInstance(result, FloorDiv)
        self.assertEqual(result.args, (original, sympy.Integer(3)))
        self.assertEqual(self.environment.divisible, set())
        guard = self.compile_contract(contract)
        self.assertEqual(self.evaluate(guard, torch.empty(value)), expected)

    @parametrize("value,expected", ((9, 1), (12, 1), (10, 0)))
    def test_payload_divisibility_marker_keeps_original_mod(self, value, expected):
        size = self.record.sizes[0].node._expr
        obligation = sympy.Eq(Mod(size, 3), 0, evaluate=False)
        self.assertTrue(self.environment.evaluate_expr(obligation))
        self.assertIn(Mod(size, 3), self.environment.divisible)
        result, contract = simplify_integer_payload(
            3 * FloorDiv(size, 3), self.environment
        )
        self.assertEqual(result, 3 * CleanDiv(size, 3))
        self.assertIn(obligation, contract.obligations)
        self.assertEqual(contract.additional_guards, ())
        guard = self.compile_contract(contract)
        self.assertEqual(self.evaluate(guard, torch.empty(value)), expected)

    def test_payload_unspecialized_size_never_uses_example_hint(self):
        size = self.record.sizes[0].node._expr
        with (
            mock.patch.object(
                self.environment,
                "size_hint",
                side_effect=AssertionError("hint requested"),
            ),
            mock.patch.object(
                self.environment,
                "evaluate_expr",
                side_effect=AssertionError("hint guard requested"),
            ),
        ):
            result, contract = simplify_integer_payload(size + 1, self.environment)
        self.assertEqual(result, size + 1)
        guard = self.compile_contract(contract)
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(10)), 1)

    def test_payload_reuses_one_immutable_contract(self):
        size = self.record.sizes[0].node._expr
        contract = integer_payload_contract(self.environment)
        first, first_contract = simplify_integer_payload(
            size + 1, self.environment, contract
        )
        second, second_contract = simplify_integer_payload(
            2 * size, self.environment, contract
        )
        self.assertEqual((first, second), (size + 1, 2 * size))
        self.assertIs(first_contract, contract)
        self.assertIs(second_contract, contract)
        self.assertIs(first_contract.obligations, second_contract.obligations)

    def test_payload_exports_unrecorded_axioms(self):
        size = self.record.sizes[0].node._expr
        self.environment.axioms[sympy.Eq(size, 9)] = sympy.true
        result, contract = simplify_integer_payload(size + 1, self.environment)
        self.assertEqual(result.subs(size, 9), 10)
        self.assertIn(sympy.Eq(size, 9, evaluate=False), contract.additional_guards)
        guard = self.compile_contract(contract)
        self.assertEqual(self.evaluate(guard, torch.empty(9)), 1)
        self.assertEqual(self.evaluate(guard, torch.empty(10)), 0)

    def test_payload_rejects_contract_after_new_specialization(self):
        size = self.record.sizes[0]
        original = size.node._expr
        contract = integer_payload_contract(self.environment)
        self.assertTrue(size == 9)
        with self.assertRaisesRegex(ValueError, "finalized"):
            simplify_integer_payload(original + 1, self.environment, contract)


@instantiate_parametrized_tests
class TestPayloadDomains(TestCase):
    @parametrize("kind", ("float", "identity", "float_cast", "foreign", "boolean"))
    def test_payload_rejects_other_domains(self, kind):
        environment = ShapeEnv()
        size = environment.create_unbacked_symint().node._expr
        expression = {
            "float": sympy.Float(1.5),
            "identity": Identity(size),
            "float_cast": TruncToInt(IntTrueDiv(size, 2)),
            "foreign": sympy.Symbol("foreign", integer=True),
            "boolean": sympy.Eq(size, 9),
        }[kind]
        with self.assertRaises(ValueError):
            simplify_integer_payload(expression, environment)


@instantiate_parametrized_tests
class TestPayloadContractState(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.source = LocalSource("x")
        self.x = self.environment.create_unspecified_symbol(
            2, self.source, DimDynamic.DYNAMIC
        )
        self.y = self.environment.create_unspecified_symbol(
            2, LocalSource("y"), DimDynamic.DYNAMIC
        )
        self.assertTrue(
            self.environment.evaluate_expr(sympy.Ge(Max(0, self.x), self.y))
        )
        self.expression = sympy.Piecewise((1, self.x >= self.y), (0, True))

    @parametrize("field", ("axioms", "replacements", "ranges", "divisible", "guards"))
    def test_direct_mutation_rejects_stale_contract(self, field):
        contract = integer_payload_contract(self.environment)
        versions = (
            self.environment._version_counter,
            self.environment._replacements_version_counter,
        )
        if field == "axioms":
            expression = next(iter(self.environment.axioms))
            self.environment.axioms[expression] = sympy.false
        elif field == "replacements":
            self.environment.replacements[self.x] = sympy.Integer(2)
        elif field == "ranges":
            self.environment.var_to_range[self.x] = ValueRanges(0, 8)
        elif field == "divisible":
            self.environment.divisible.add(Mod(self.x, 2))
        else:
            self.environment.guards[0] = mock.Mock(expr=sympy.Ge(self.x, self.y))
        self.assertEqual(
            versions,
            (
                self.environment._version_counter,
                self.environment._replacements_version_counter,
            ),
        )
        with self.assertRaisesRegex(ValueError, "finalized"):
            simplify_integer_payload(self.expression, self.environment, contract)

    def test_false_axiom_retains_raw_negation(self):
        expression = sympy.Lt(self.x, 0, evaluate=False)
        self.environment.axioms[expression] = sympy.false
        contract = integer_payload_contract(self.environment)
        self.assertIn(sympy.Not(expression), contract.additional_guards)
        invalid, valid = {self.x: -1, self.y: 0}, {self.x: 3, self.y: 2}
        self.assertFalse(
            all(bool(guard.subs(invalid)) for guard in contract.additional_guards)
        )
        result, _ = simplify_integer_payload(
            self.expression, self.environment, contract
        )
        self.assertEqual(result.subs(valid), self.expression.subs(valid))

    def test_unrecorded_replacement_retains_raw_equality(self):
        self.environment._set_replacement(self.x, sympy.Integer(2), "test")
        contract = integer_payload_contract(self.environment)
        self.assertIn(sympy.Eq(self.x, 2, evaluate=False), contract.additional_guards)
        result, _ = simplify_integer_payload(self.x + 1, self.environment, contract)
        self.assertEqual(result, 3)

    def test_represented_implications_add_no_guards(self):
        contract = integer_payload_contract(self.environment)
        self.assertEqual(contract.additional_guards, ())

    @parametrize("equality", (False, True))
    def test_contract_created_before_context_rejects_temporary_fact(self, equality):
        contract = integer_payload_contract(self.environment)
        specialization = (
            (lambda value: value == 2) if equality else (lambda value: value >= 0)
        )
        with self.environment.patch_source_specialization(self.source, specialization):
            with self.assertRaisesRegex(ValueError, "finalized"):
                simplify_integer_payload(self.expression, self.environment, contract)

    @parametrize("equality", (False, True))
    def test_contract_created_inside_context_exports_temporary_fact(self, equality):
        specialization = (
            (lambda value: value == 2) if equality else (lambda value: value >= 0)
        )
        with self.environment.patch_source_specialization(self.source, specialization):
            contract = integer_payload_contract(self.environment)
            result, _ = simplify_integer_payload(
                self.expression, self.environment, contract
            )
            values = {self.x: 2, self.y: 1}
            self.assertEqual(result.subs(values), self.expression.subs(values))
        invalid = {self.x: -1, self.y: 0}
        self.assertFalse(
            all(bool(guard.subs(invalid)) for guard in contract.additional_guards)
        )
        with self.assertRaisesRegex(ValueError, "finalized"):
            simplify_integer_payload(self.expression, self.environment, contract)

    @parametrize("reuse", (False, True))
    def test_restored_context_does_not_reuse_cached_temporary_fact(self, reuse):
        original = integer_payload_contract(self.environment)
        with self.environment.patch_source_specialization(
            self.source, lambda value: value >= 0
        ):
            specialized = integer_payload_contract(self.environment)
            simplify_integer_payload(self.expression, self.environment, specialized)
        contract = original if reuse else integer_payload_contract(self.environment)
        result, _ = simplify_integer_payload(
            self.expression, self.environment, contract
        )
        invalid = {self.x: -1, self.y: 0}
        self.assertEqual(result.subs(invalid), self.expression.subs(invalid))

    def test_simplification_mutation_rejects_the_result(self):
        contract = integer_payload_contract(self.environment)
        original = self.environment.simplify

        def mutate(expression, *args, **options):
            result = original(expression, *args, **options)
            self.environment.axioms[sympy.Ge(self.x, 0)] = sympy.true
            return result

        with mock.patch.object(self.environment, "simplify", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed its finalized"):
                simplify_integer_payload(self.expression, self.environment, contract)


@instantiate_parametrized_tests
class TestPayloadReplacementChains(TestCase):
    def setUp(self):
        super().setUp()
        self.environment = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        self.symbols = tuple(
            self.environment.create_unspecified_symbol(
                9, LocalSource(name), DimDynamic.DYNAMIC
            )
            for name in ("a", "b", "c")
        )
        self.value = self.environment.create_symintnode(self.symbols[0], hint=9)

    @parametrize("frozen", (False, True))
    def test_lazy_expression_preserves_finalized_contract(self, frozen):
        a, b, c = self.symbols
        self.environment._set_replacement(a, b, "test")
        self.environment._set_replacement(b, c, "test")
        self.assertEqual(self.environment.replacements[a], b)
        self.environment.frozen = frozen
        contract = integer_payload_contract(self.environment)
        self.assertEqual(contract.replacements[a], b if frozen else c)
        self.assertEqual(self.value.node._expr, a)
        self.assertEqual(self.value.node.expr, c)
        self.assertTrue(_contract_is_current(contract, self.environment))
        for expression, expected in ((a + 1, c + 1), (2 * b, 2 * c)):
            actual, reused = simplify_integer_payload(
                expression, self.environment, contract
            )
            self.assertEqual(actual, expected)
            self.assertIs(reused, contract)
        self.assertTrue(_contract_is_current(contract, self.environment))
        invalid = {a: 10, b: 9, c: 9}
        self.assertFalse(all(bool(g.subs(invalid)) for g in contract.obligations))

    def test_compression_range_refinement_precedes_snapshot(self):
        a, b, c = self.symbols
        self.environment._set_replacement(a, b, "test")
        self.environment.constrain_symbol_range(c, 8, 16)
        self.environment._set_replacement(b, c, "test")
        self.assertNotEqual(self.environment.var_to_range[a], ValueRanges(8, 16))
        contract = integer_payload_contract(self.environment)
        self.assertEqual(contract.range_map[a], ValueRanges(8, 16))
        self.assertEqual(contract.range_map, self.environment.var_to_range)
        self.assertIn(sympy.Ge(a, 8, evaluate=False), contract.obligations)
        self.assertIn(sympy.Le(a, 16, evaluate=False), contract.obligations)
        self.assertEqual(self.value.node.expr, c)
        result, _ = simplify_integer_payload(Min(a, 4), self.environment, contract)
        self.assertEqual(result, 4)
        self.assertTrue(_contract_is_current(contract, self.environment))
        for value in (7, 17):
            values = dict.fromkeys(self.symbols, value)
            self.assertFalse(all(bool(g.subs(values)) for g in contract.obligations))

    @parametrize("fact", ("replacement", "range", "guard"))
    def test_new_assumption_still_rejects_canonical_contract(self, fact):
        a, b, c = self.symbols
        self.environment._set_replacement(a, b, "test")
        self.environment._set_replacement(b, c, "test")
        contract = integer_payload_contract(self.environment)
        self.assertEqual(self.value.node.expr, c)
        self.assertTrue(_contract_is_current(contract, self.environment))
        if fact == "replacement":
            self.environment._set_replacement(c, sympy.Integer(9), "test")
        elif fact == "range":
            self.environment.constrain_symbol_range(c, 8, 16)
        else:
            self.assertTrue(self.environment.evaluate_expr(sympy.Ge(c, 8)))
        self.assertFalse(_contract_is_current(contract, self.environment))
        with self.assertRaisesRegex(ValueError, "finalized"):
            simplify_integer_payload(a + 1, self.environment, contract)


if __name__ == "__main__":
    run_tests()
