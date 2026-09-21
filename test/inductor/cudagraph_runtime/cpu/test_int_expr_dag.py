# Owner(s): ["module: inductor"]
"""IntExpr equality and the numeric program's memo on shared-chain DAGs."""

import time
from types import SimpleNamespace

from torch._inductor.runtime.cudagraph_arg_mapping import IntegerInput, IntExpr
from torch._inductor.runtime.cudagraph_boxed_replay import _NumericProgram
from torch.testing._internal.common_utils import run_tests, TestCase


RECORDS = SimpleNamespace(
    input_names=("scalar",), integer_inputs=(IntegerInput("scalar", 0),)
)


def select_chain(depth, leaves):
    # an n-ary min folded pairwise: every step refers to its predecessor twice
    result = leaves[0]
    for leaf in leaves[1:depth]:
        cond = IntExpr("lt", args=(result, leaf))
        result = IntExpr("select", args=(cond, result, leaf))
    return result


class TestIntExprDag(TestCase):
    def test_distinct_but_equal_chains_compare_in_linear_time(self):
        leaves = [IntExpr("boxed", 0)] + [IntExpr("constant", i) for i in range(1, 400)]
        a = select_chain(400, leaves)
        b = select_chain(400, leaves)
        self.assertIsNot(a, b)
        started = time.perf_counter()
        self.assertEqual(hash(a), hash(b))
        self.assertTrue(a == b)
        self.assertLess(time.perf_counter() - started, 1.0)
        other = select_chain(400, leaves[:-1] + [IntExpr("constant", 1000)])
        self.assertFalse(a == other)
        self.assertFalse(a == IntExpr("select", args=a.args[:2]))
        self.assertFalse(a == "select")

    def test_program_memo_reuses_a_structurally_equal_chain(self):
        # add() recurses into the operands: a 200-deep chain stays inside the default
        # recursion limit (cat's per-launch chain is at most 128 steps)
        leaves = [IntExpr("boxed", 0)] + [IntExpr("constant", i) for i in range(1, 200)]
        program = _NumericProgram(RECORDS, [7])
        started = time.perf_counter()
        first = program.add(select_chain(200, leaves))
        count = len(program.instructions)
        second = program.add(select_chain(200, leaves))
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertEqual(first, second)
        self.assertEqual(len(program.instructions), count)
        self.assertEqual(program.values[first], 1)  # min over boxed 7 and 1..199

    def test_deep_chains_neither_recurse_nor_hang(self):
        # a 3000-input cat's n-ary max and its prefix sum of offsets, well past the
        # interpreter's recursion limit
        leaves = [IntExpr("boxed", 0)] + [
            IntExpr("constant", i) for i in range(1, 3000)
        ]
        chain = select_chain(3000, leaves)
        total = leaves[0]
        for leaf in leaves[1:]:
            total = IntExpr("add", args=(total, leaf))
        started = time.perf_counter()
        hash(chain)
        hash(total)
        self.assertTrue(chain == select_chain(3000, leaves))
        program = _NumericProgram(RECORDS, [5])
        slot = program.add(chain)
        self.assertEqual(program.values[slot], 1)
        slot = program.add(total)
        self.assertEqual(program.values[slot], 5 + sum(range(1, 3000)))
        self.assertLess(time.perf_counter() - started, 5.0)
        # the same left-to-right post-order as before: operands precede their node
        self.assertEqual(program.instructions[0], ("boxed", 0))
        self.assertEqual(program.instructions[1], ("constant", 1))
        self.assertEqual(program.instructions[2][0], "lt")


if __name__ == "__main__":
    run_tests()
