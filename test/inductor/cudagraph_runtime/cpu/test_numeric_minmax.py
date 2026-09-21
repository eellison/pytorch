# Owner(s): ["module: inductor"]
"""The numeric plan's n-ary max and min: one instruction over every operand slot."""

import random
import time
from types import SimpleNamespace

import sympy

from torch._inductor.runtime._cudagraph.address_scalars import symbolic_integer
from torch._inductor.runtime.cudagraph_arg_mapping import (
    grid_expression_inputs,
    IntegerInput,
    IntExpr,
)
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    compile_numeric,
    EarlyStatus,
    MAX_I64,
    MIN_I64,
)
from torch._inductor.runtime.cudagraph_launch_association import UnsupportedCapture
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


RECORDS = SimpleNamespace(
    input_names=("a", "b"),
    integer_inputs=(IntegerInput("a", 0), IntegerInput("b", 1)),
)
A = IntExpr("boxed", 0)
B = IntExpr("boxed", 1)


def const(value):
    return IntExpr("constant", value)


@instantiate_parametrized_tests
class TestNumericMinMax(TestCase):
    @parametrize("op", ("max", "min"))
    @parametrize("count", (2, 3, 17, 300))
    def test_mixed_constants_and_boxed_operands(self, op, count):
        rng = random.Random(count)
        constants = [rng.randint(-1000, 1000) for _ in range(count - 2)]
        operands = (A, B, *(const(c) for c in constants))
        for a, b in ((7, -3), (-(1 << 62), 1 << 62), (0, 0)):
            program = _NumericProgram(RECORDS, [a, b])
            slot = program.add(IntExpr(op, args=operands))
            reducer = max if op == "max" else min
            self.assertEqual(program.values[slot], reducer([a, b, *constants]))
            instruction = program.instructions[slot]
            self.assertEqual(instruction[0], op)
            self.assertEqual(len(instruction), count + 1)
            # every operand slot precedes the result
            self.assertTrue(all(operand < slot for operand in instruction[1:]))

    def test_nested_and_memoized(self):
        inner = IntExpr("max", args=(A, const(5)))
        outer = IntExpr("min", args=(inner, B, IntExpr("add", args=(inner, const(1)))))
        program = _NumericProgram(RECORDS, [3, 4])
        slot = program.add(outer)
        self.assertEqual(program.values[slot], 4)  # min(max(3, 5), 4, 6)
        count = len(program.instructions)
        # the same DAG again: the memo returns the slot, no new instructions
        self.assertEqual(program.add(outer), slot)
        self.assertEqual(len(program.instructions), count)
        # inner appears once even though outer refers to it twice
        self.assertEqual(sum(1 for row in program.instructions if row[0] == "max"), 1)

    def test_a_wide_max_is_one_node_and_fast(self):
        # a 300-input cat's per-launch grid: max over every input's size
        leaves = [A, B] + [const(i) for i in range(298)]
        started = time.perf_counter()
        program = _NumericProgram(RECORDS, [1, 2])
        slot = program.add(IntExpr("max", args=tuple(leaves)))
        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertEqual(program.values[slot], 297)
        # the operands plus the one max instruction
        self.assertEqual(len(program.instructions), len(leaves) + 1)

    @parametrize("op", ("max", "min"))
    def test_one_operand_is_rejected(self, op):
        program = _NumericProgram(RECORDS, [1, 2])
        with self.assertRaisesRegex(UnsupportedCapture, "Unresolved"):
            program.add(IntExpr(op, args=(A,)))

    def test_grid_expression_inputs_accept_the_n_ary_form(self):
        size = IntExpr("size", 0, (const(0),))
        stride = IntExpr("stride", 2, (const(1),))
        expression = IntExpr("max", args=(size, stride, const(4)))
        self.assertEqual(grid_expression_inputs(expression), (0, 2))
        self.assertIsNone(grid_expression_inputs(IntExpr("max", args=(size,))))

    def test_symbolic_integer_renders_sympy_max_and_min(self):
        a, b = sympy.symbols("a b", integer=True)
        symbols = {0: a, 1: b}
        expression = IntExpr("max", args=(A, B, const(9)))
        self.assertEqual(symbolic_integer(expression, symbols), sympy.Max(a, b, 9))
        expression = IntExpr("min", args=(A, const(9)))
        self.assertEqual(symbolic_integer(expression, symbols), sympy.Min(a, 9))

    @parametrize("op", ("max", "min"))
    def test_compiled_replay_reads_every_operand(self, op):
        constants = tuple(range(-149, 150))
        program = _NumericProgram(RECORDS, [1, 2])
        expression = IntExpr(op, args=(A, *(const(c) for c in constants), B))
        output = program.add(expression)
        compiled = compile_numeric(program)
        reducer = max if op == "max" else min
        for operands in (
            (MIN_I64, MAX_I64),
            (MAX_I64, MIN_I64),
            (0, 0),
            (-1000, -999),
            (999, 1000),
        ):
            status, values = compiled.evaluate_leaves(compiled.bind_inputs(operands))
            self.assertEqual(status, EarlyStatus.SUCCESS)
            self.assertEqual(values[output], reducer((*operands, *constants)))


if __name__ == "__main__":
    run_tests()
