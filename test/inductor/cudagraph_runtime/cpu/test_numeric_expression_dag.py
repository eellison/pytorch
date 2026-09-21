# Owner(s): ["module: inductor"]

import gc
import weakref
from types import SimpleNamespace
from unittest import mock

from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_numeric,
    EarlyStatus,
)
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestNumericExpressionDAG(TestCase):
    def program(self, *values):
        names = tuple(f"arg{index}" for index in range(len(values)))
        records = SimpleNamespace(
            input_names=names,
            integer_inputs=tuple(
                IntegerInput(name, index) for index, name in enumerate(names)
            ),
        )
        return _NumericProgram(records, values)

    def power(self, exponent):
        result = IntExpr("constant", 1)
        power = IntExpr("boxed", 0)
        while exponent:
            if exponent & 1:
                result = IntExpr("multiply", args=(result, power))
            exponent >>= 1
            if exponent:
                power = IntExpr("multiply", args=(power, power))
        return result

    @parametrize("exponent", (0, 1, 2, 63, 1 << 63, (1 << 64) - 1))
    @parametrize("base", (-1, 0, 1))
    def test_full_uint64_exponents_use_shared_integer_dag(self, exponent, base):
        expression = self.power(exponent)
        program = self.program() if exponent == 0 else self.program(1)
        with mock.patch.object(
            IntExpr, "__hash__", side_effect=AssertionError("recursive hash")
        ):
            output = program.add(expression)
        self.assertLessEqual(len(program.instructions), 129)
        compiled = compile_numeric(program)
        inputs = () if exponent == 0 else (base,)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs(inputs))
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual(values[output], pow(base, exponent))

    def test_distinct_shared_dags_preserve_structural_deduplication(self):
        first = self.power((1 << 64) - 1)
        second = self.power((1 << 64) - 1)
        program = self.program(1)
        with (
            mock.patch.object(
                IntExpr, "__hash__", side_effect=AssertionError("recursive hash")
            ),
            mock.patch.object(
                IntExpr, "__eq__", side_effect=AssertionError("recursive equality")
            ),
        ):
            original = program.add(first)
            count = len(program.instructions)
            self.assertEqual(program.add(second), original)
            self.assertEqual(program.add(first), original)
        self.assertEqual(len(program.instructions), count)

    def test_same_preparation_values_keep_distinct_input_sources(self):
        program = self.program(1, 1)
        left, right = IntExpr("boxed", 0), IntExpr("boxed", 1)
        total = program.add(IntExpr("add", args=(left, right)))
        product = program.add(IntExpr("multiply", args=(left, right)))
        compiled = compile_numeric(program)
        status, values = compiled.evaluate_leaves(compiled.bind_inputs((2, 3)))
        self.assertEqual(status, EarlyStatus.SUCCESS)
        self.assertEqual((values[total], values[product]), (5, 6))
        self.assertEqual(compiled.leaf_bindings, (("boxed", 0), ("boxed", 1)))

    def test_identity_cache_retains_expression_owners(self):
        program = self.program(1)
        first, second = self.power(63), self.power(63)
        first_ref, second_ref = weakref.ref(first), weakref.ref(second)
        self.assertEqual(program.add(first), program.add(second))
        del first, second
        gc.collect()
        self.assertIsNotNone(first_ref())
        self.assertIsNotNone(second_ref())
        del program
        gc.collect()
        self.assertIsNone(first_ref())
        self.assertIsNone(second_ref())

    @parametrize(
        "kind",
        (
            "boolean_constant",
            "bad_payload",
            "bad_arity",
            "bad_args",
            "unknown",
            "overflow",
            "zero_divisor",
            "negative_dividend",
            "nonboolean_and",
            "nonboolean_select",
            "unselected_division",
        ),
    )
    def test_new_expression_validation_precedes_deduplication(self, kind):
        program = self.program(1)
        one, zero = IntExpr("constant", 1), IntExpr("constant", 0)
        program.add(one)
        program.add(IntExpr("multiply", args=(one, one)))
        bad_division = IntExpr("floordiv", args=(one, zero))
        cases = {
            "boolean_constant": IntExpr("constant", True),
            "bad_payload": IntExpr("multiply", 1, (one, one)),
            "bad_arity": IntExpr("add", args=(one,)),
            "bad_args": IntExpr("add", args=[one, one]),
            "unknown": IntExpr("mystery"),
            "overflow": IntExpr(
                "multiply",
                args=(IntExpr("constant", (1 << 63) - 1), IntExpr("constant", 2)),
            ),
            "zero_divisor": bad_division,
            "negative_dividend": IntExpr(
                "floordiv", args=(IntExpr("constant", -1), one)
            ),
            "nonboolean_and": IntExpr("and", args=(one, IntExpr("constant", 2))),
            "nonboolean_select": IntExpr(
                "select", args=(IntExpr("constant", 2), one, zero)
            ),
            "unselected_division": IntExpr("select", args=(one, one, bad_division)),
        }
        with self.assertRaises(UnsupportedCapture):
            program.add(cases[kind])

    @parametrize("value", (True, 1 << 63, "invalid"))
    def test_invalid_callback_result_is_not_cached(self, value):
        owner = mock.Mock(side_effect=(value, 3))
        expression = IntExpr("call", (1, owner))
        program = self.program()
        with self.assertRaises(UnsupportedCapture):
            program.add(expression)
        index = program.add(expression)
        self.assertEqual(program.values[index], 3)
        self.assertEqual(owner.call_count, 2)

    def test_distinct_callback_owners_are_not_merged(self):
        first, second = mock.Mock(return_value=2), mock.Mock(return_value=3)
        program = self.program()
        left = program.add(IntExpr("call", (1, first)))
        right = program.add(IntExpr("call", (1, second)))
        self.assertNotEqual(left, right)
        self.assertEqual((program.values[left], program.values[right]), (2, 3))

    def test_shared_callback_is_prepared_once(self):
        owner = mock.Mock(return_value=2)
        shared = IntExpr("call", (1, owner))
        program = self.program()
        output = program.add(IntExpr("multiply", args=(shared, shared)))
        self.assertEqual(program.values[output], 4)
        self.assertEqual(owner.call_count, 1)
        self.assertEqual(len(program.instructions), 2)

    def test_prepared_value_is_read_only(self):
        expression = self.power((1 << 64) - 1)
        program = self.program(1)
        program.add(expression)
        count = len(program.instructions)
        with (
            mock.patch.object(
                program, "add", side_effect=AssertionError("late registration")
            ),
            mock.patch.object(
                IntExpr, "__hash__", side_effect=AssertionError("recursive hash")
            ),
        ):
            self.assertEqual(program.prepared_value(expression), 1)
            with self.assertRaisesRegex(
                UnsupportedCapture, "registered before capture"
            ):
                program.prepared_value(IntExpr("constant", 42))
        self.assertEqual(len(program.instructions), count)


if __name__ == "__main__":
    run_tests()
