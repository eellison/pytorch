# Owner(s): ["module: inductor"]

import gc
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.runtime.cudagraph_arg_mapping import (
    BufferSource,
    InputSource,
    IntegerInput,
    IntExpr,
    ParameterSource,
    PointerSource,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _NumericProgram,
    _ParameterProgram,
)
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestParameterExpressionDAG(TestCase):
    def program(self, values=(1,), buffers=None):
        names = tuple(f"arg{index}" for index in range(len(values)))
        records = SimpleNamespace(
            input_names=names,
            integer_inputs=tuple(
                IntegerInput(name, index)
                for index, (name, value) in enumerate(zip(names, values))
                if type(value) is int
            ),
        )
        numeric = _NumericProgram(records, values)
        return _ParameterProgram(numeric, len(values), buffers or {})

    def power(self, index=0, depth=64):
        value = IntExpr("boxed", index)
        for _ in range(depth):
            value = IntExpr("multiply", args=(value, value))
        return value

    def forbid_tree_hashing(self):
        stack = ExitStack()
        for cls in (ParameterSource, PointerSource):
            for method in ("__hash__", "__eq__"):
                stack.enter_context(
                    mock.patch.object(
                        cls,
                        method,
                        side_effect=AssertionError("recursive tree operation"),
                    )
                )
        return stack

    @parametrize("kind", ("value", "pointer", "parameter"))
    def test_shared_dags_prepare_without_recursive_tree_hashing(self, kind):
        if kind == "pointer":
            program = self.program((torch.empty(1), 1))
            source = ParameterSource(
                "pointer", 64, PointerSource(InputSource(0), self.power(1))
            )
            roots = (4096, 0)
            expected = 4097
        else:
            program = self.program()
            source = ParameterSource("value", 64, self.power())
            expected = 1
            if kind == "parameter":
                for _ in range(64):
                    source = ParameterSource("add", 64, args=(source, source))
                expected = 0
            roots = (0,)
        with self.forbid_tree_hashing():
            output = program.add(source)
            self.assertEqual(program.prepared_index(source), output)
        actual = torch._C._cuda_evaluate_parameter_program(
            program.plan(), tuple(program.numeric.values), roots
        )
        self.assertEqual(actual[output], expected)
        self.assertLessEqual(len(program.rows), 65)
        self.assertLessEqual(len(program.numeric.instructions), 65)

    def test_distinct_equal_dags_deduplicate_in_first_use_order(self):
        program = self.program((1, 1))
        first = ParameterSource("value", 64, self.power(0))
        equal = ParameterSource("value", 64, self.power(0))
        second = ParameterSource("value", 64, self.power(1))
        narrowed = ParameterSource("trunc", 32, args=(first,))
        with self.forbid_tree_hashing():
            self.assertEqual(program.add(second), 0)
            self.assertEqual(program.add(first), 1)
            self.assertEqual(program.add(equal), 1)
            self.assertEqual(program.add(narrowed), 2)
            self.assertEqual(program.add(first), 1)
        self.assertEqual(
            program.rows, [("value", 64, 64), ("value", 64, 129), ("trunc", 32, 1)]
        )
        self.assertEqual(program.outputs, [0, 1, 2])
        self.assertEqual(program.numeric.instructions[0], ("boxed", 1))
        self.assertEqual(program.numeric.instructions[65], ("boxed", 0))

    def test_pointer_roots_and_offsets_remain_distinct(self):
        owner = BufferSource("temporary")
        program = self.program((torch.empty(1), 3, 5), {owner: 3})
        sources = [
            ParameterSource(
                "pointer", 64, PointerSource(root, IntExpr("boxed", scalar))
            )
            for root, scalar in ((InputSource(0), 1), (owner, 1), (InputSource(0), 2))
        ]
        with self.forbid_tree_hashing():
            self.assertEqual([program.add(source) for source in sources], [0, 1, 2])
        self.assertEqual(program.roots, {0: InputSource(0), 3: owner})
        self.assertEqual(program.numeric.instructions, [("boxed", 1), ("boxed", 2)])
        self.assertEqual(
            program.rows,
            [("pointer", 64, 0, 0), ("pointer", 64, 3, 0), ("pointer", 64, 0, 1)],
        )
        self.assertEqual(
            torch._C._cuda_evaluate_parameter_program(
                program.plan(), (7, 11), (4096, 0, 0, 8192)
            ),
            (4103, 8199, 4107),
        )

    def test_prepared_lookup_is_read_only_and_requires_registered_output(self):
        program = self.program()
        child = ParameterSource("constant", 64, 1)
        source = ParameterSource("add", 64, args=(child, child))
        program.add(source)
        before = program.plan(), tuple(program.numeric.instructions)
        with (
            mock.patch.object(
                program, "_node", side_effect=AssertionError("late registration")
            ),
            mock.patch.object(
                program.numeric, "add", side_effect=AssertionError("early registration")
            ),
        ):
            self.assertEqual(program.prepared_index(source), 0)
            for unprepared in (child, ParameterSource("add", 64, args=(child, child))):
                with self.assertRaisesRegex(
                    UnsupportedCapture, "not registered before capture"
                ):
                    program.prepared_index(unprepared)
        self.assertEqual((program.plan(), tuple(program.numeric.instructions)), before)

    def test_identity_cache_retains_equal_source_owners(self):
        program = self.program()
        first = ParameterSource("value", 64, self.power())
        second = ParameterSource("value", 64, self.power())
        refs = weakref.ref(first), weakref.ref(second)
        self.assertEqual(program.add(first), program.add(second))
        del first, second
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        del program
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_callback_owners_and_float_bits_are_not_merged(self):
        program = self.program()
        calls = []

        def first(args):
            calls.append("first")
            return args[0]

        def second(args):
            calls.append("second")
            return args[0]

        zero = IntExpr("fconst", 0)
        negative_zero = IntExpr("fconst", -(1 << 63))
        sources = [
            ParameterSource("value", 64, expression)
            for expression in (
                zero,
                negative_zero,
                IntExpr("call", (16, first), (zero,)),
                IntExpr("call", (16, second), (zero,)),
            )
        ]
        self.assertEqual([program.add(source) for source in sources], [0, 1, 2, 3])
        self.assertEqual(program.add(sources[2]), 2)
        self.assertEqual(calls, ["first", "second"])
        self.assertIs(program.numeric.instructions[2][2], first)
        self.assertIs(program.numeric.instructions[3][2], second)
        self.assertEqual(
            torch._C._cuda_evaluate_parameter_program(
                program.plan(), tuple(program.numeric.values), (0,)
            ),
            (0, -(1 << 63), 0, 0),
        )

    def test_fixed_width_flags_preserve_overflow_behavior(self):
        program = self.program()
        maximum = ParameterSource("constant", 64, (1 << 63) - 1)
        one = ParameterSource("constant", 64, 1)
        wrapping = ParameterSource("add", 64, args=(maximum, one))
        flagged = ParameterSource("add", 64, args=(maximum, one), flags=("nsw",))
        program.add(wrapping)
        self.assertEqual(
            torch._C._cuda_evaluate_parameter_program(program.plan(), (), (0,)),
            (-(1 << 63),),
        )
        program.add(flagged)
        self.assertEqual(len(program.rows), 4)
        with self.assertRaisesRegex((RuntimeError, ValueError), "poison|overflow"):
            torch._C._cuda_evaluate_parameter_program(program.plan(), (), (0,))

    def test_invalid_equal_payload_is_validated_before_deduplication(self):
        program = self.program()
        program.add(ParameterSource("constant", 64, 1))
        with self.assertRaisesRegex(
            UnsupportedCapture, "Unsupported typed late parameter"
        ):
            program.add(ParameterSource("constant", 64, True))
        with self.assertRaisesRegex(UnsupportedCapture, "exceeds int64"):
            program.add(
                ParameterSource(
                    "value",
                    64,
                    IntExpr(
                        "multiply",
                        args=(IntExpr("constant", (1 << 62)), IntExpr("constant", 4)),
                    ),
                )
            )


if __name__ == "__main__":
    run_tests()
