# Owner(s): ["module: cuda graphs"]

import random

import sympy
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

import torch
from torch.cuda._host_trace import _TraceShapeEnv, BitLength, Declined, F32Div
from torch.cuda._host_trace_lower import Lowering
from torch.cuda._host_trace_program import (
    compile_program,
    IntegerProgram,
    MAX_I64,
    MIN_I64,
    Status,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import (
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    CeilToInt,
    FloorDiv,
    FloorToInt,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    PowByNatural,
    PythonMod,
)


P, Q = sympy.symbols("p q", integer=True, positive=True)
S, T = sympy.symbols("s t", integer=True)
SYMBOLS = (P, Q, S, T)
HINT = (8, 6, 7, -3)

EXPRESSIONS = [
    FloorDiv(S, T),
    FloorDiv(S, 4),
    FloorDiv(S, -4),
    FloorDiv(P, T),
    FloorDiv(-P, 16) + 14,  # sympy's form of (224 - p) // 16
    FloorDiv(P, Q),
    FloorDiv(S + P, 2 * T),
    CeilToInt(IntTrueDiv(S, T)),  # how math.ceil(s / t) traces
    CeilToInt(IntTrueDiv(P, 32)),
    FloorToInt(IntTrueDiv(-P, Q)),
    PythonMod(S, T),
    PythonMod(S, 3),
    PythonMod(P * Q, T),
    PythonMod(-P, Q),
    Mod(P, Q),
    Min(S, T, P),
    Max(S, -T),
    S**2,
    PowByNatural(S, 3),
    3 * S - 2 * T + 7,
    BitwiseFn_bitwise_and(S, T),
    BitwiseFn_bitwise_or(S - 1, FloorDiv(S - 1, 2)),
    BitwiseFn_bitwise_xor(P, -T),
    BitLength(S),
    PowByNatural(2, P),
    F32Div(P, Q),
    # IntDivider's magic number
    FloorDiv(2**32 * (PowByNatural(2, BitLength(P - 1)) - P), P) + 1,
    sympy.Eq(S, T),
    sympy.Ne(PythonMod(S, T), 0),
    sympy.Lt(S, T),
    sympy.Le(S * T, 12),
    sympy.Gt(S, 0),
    sympy.Ge(FloorDiv(S, T), -1),
    sympy.And(sympy.Lt(S, T), sympy.Gt(P, 3)),
    sympy.Or(sympy.Eq(S, 1), sympy.Eq(T, 8)),
    sympy.Not(sympy.Eq(S, T)),
]

LO, HI = MIN_I64, MAX_I64
BOUNDS = [(LO, -1), (LO, 1), (HI, -1), (HI, 2), (LO, LO), (LO + 1, -1), (-1, HI)]


def _lowered(expressions, hint=HINT):
    program = IntegerProgram(list(hint))
    lowering = Lowering(program, {x: ("boxed", i) for i, x in enumerate(SYMBOLS)})
    rows = [lowering.lower(e) for e in expressions]
    return compile_program(program), rows


def _python(e, point):
    # the host's value: sympy folds each function on integers as Python does
    try:
        v = e.xreplace(dict(zip(SYMBOLS, map(sympy.Integer, point))))
    except ZeroDivisionError:
        return None
    if isinstance(v, (BooleanTrue, BooleanFalse)):
        return int(bool(v))
    return int(v) if isinstance(v, sympy.Integer) else None


class TestLowering(TestCase):
    def test_matches_python_at_small_points(self):
        # every sign of numerator and divisor, and zero divisors: the compiled
        # values are Python's, and a division by zero is a failing status
        compiled, rows = _lowered(EXPRESSIONS)
        rng = random.Random(0)
        points = [HINT, (1, 1, 0, 1), (5, 5, -5, 5), (3, 4, -7, -2)]
        size, value = (lambda: rng.randint(1, 12)), (lambda: rng.randint(-12, 12))
        points += [(size(), size(), value(), value()) for _ in range(400)]
        for point in points:
            want = [_python(e, point) for e in EXPRESSIONS]
            status, outputs = compiled.evaluate_inputs(point)
            if None in want:
                self.assertNotEqual(status, Status.SUCCESS, point)
            else:
                self.assertEqual(status, Status.SUCCESS, point)
                self.assertEqual([outputs[r] for r in rows], want, point)

    @parametrize("point", BOUNDS)
    def test_never_a_wrong_value_at_the_int64_bounds(self, point):
        # an intermediate the evaluator cannot hold is a miss, never a
        # wrapped value; a success is Python's value
        expressions = [FloorDiv(S, T), S * T, S + T, PythonMod(S, T), FloorDiv(S, -4)]
        compiled, rows = _lowered(expressions)
        full = (1, 1, *point)
        want = [_python(e, full) for e in expressions]
        status, outputs = compiled.evaluate_inputs(full)
        if any(v is None or not MIN_I64 <= v <= MAX_I64 for v in want):
            self.assertNotEqual(status, Status.SUCCESS)
        if status == Status.SUCCESS:
            self.assertEqual([outputs[r] for r in rows], want)

    def test_proven_signs_take_the_short_form_and_miss_where_they_fail(self):
        # p is declared positive: p // 4 is one division, no select; a call
        # with p < 0 fails the division's domain instead of computing -2
        program = IntegerProgram(list(HINT))
        lowering = Lowering(program, {P: ("boxed", 0)})
        lowering.lower(FloorDiv(P, 4))
        ops = [r[0] for r in program.instructions]
        self.assertEqual(ops, ["boxed", "constant", "floordiv"])
        compiled = compile_program(program)
        status, _ = compiled.evaluate_inputs((-8, 6, 7, -3))
        self.assertEqual(status, Status.DIVISION_DOMAIN)

    def test_declines(self):
        F = sympy.Symbol("f", real=True)
        A = sympy.Symbol("a", integer=True)  # an allocation address: no source
        cases = [
            F + 1,
            A + S,
            S / 2,
            IntTrueDiv(S, T),
            PowByNatural(3, S),
            sympy.Piecewise((S, sympy.Gt(S, 0)), (T, True)),
            sympy.Integer(2**63),
            sympy.And(S, sympy.Gt(S, 0)),
        ]
        for e in cases:
            with self.assertRaises(Declined, msg=str(e)):
                _lowered([e])
        # a value the hints cannot hold
        with self.assertRaisesRegex(Declined, "fails at the hints"):
            _lowered([P * Q], hint=(2**40, 2**40, 0, 0))
        with self.assertRaisesRegex(Declined, "fails at the hints"):
            _lowered([FloorDiv(S, T)], hint=(1, 1, 1, 0))


def _holds(compiled, row, inputs):
    result = compiled.evaluate_inputs(inputs)
    if result is None:  # an input of another kind than the trace's
        return False
    status, outputs = result
    return status == Status.SUCCESS and outputs[row] == 1


class TestPredicate(TestCase):
    def test_guards_from_a_trace(self):
        # T1's guards for a host that compares sizes, pins c, reads a*b and
        # tests a modulo by a stride
        env = _TraceShapeEnv()
        a, b, c = (env.symbol(v, n, positive=True) for n, v in zip("abc", (8, 8, 1)))
        st = env.symbol(8, "st")
        taken = [a == b, c == 1, (c == 1) | (st == 8), a * b == 64, a * b % st == 0]
        self.assertTrue(all(map(bool, taken)))
        symbols = [x.node.expr for x in (a, b, c, st)]
        program = IntegerProgram([8, 8, 1, 8])
        lowering = Lowering(program, {x: ("boxed", i) for i, x in enumerate(symbols)})
        predicate = lowering.predicate([g.expr for g in env.guards])
        compiled = compile_program(program)
        cases = [
            ((8, 8, 1, 8), True),
            ((8, 8, 1, -8), True),  # Python's 64 % -8 is 0
            ((8, 8, 1, 3), False),
            ((8, 8, 1, 0), False),  # Ne(st, 0), and the modulo's domain
            ((8, 8, 2, 8), False),
            ((4, 16, 1, 8), False),
            ((8, 8, 1.0, 8), False),  # not an int
        ]
        for point, want in cases:
            self.assertEqual(_holds(compiled, predicate, point), want, point)

    def test_declared_signs_are_checked(self):
        # sympy decides a > 0 from a's declaration, so no guard says it; the
        # predicate does
        env = _TraceShapeEnv()
        a = env.symbol(4, "a", positive=True)
        self.assertTrue(bool(a > 0))
        self.assertTrue(bool(a < 5))
        program = IntegerProgram([4])
        predicate = Lowering(program, {a.node.expr: ("boxed", 0)}).predicate(
            [g.expr for g in env.guards]
        )
        compiled = compile_program(program)
        holds = [_holds(compiled, predicate, [v]) for v in (4, 1, 5, 0, -3)]
        self.assertEqual(holds, [True, True, False, False, False])

    def test_next_power_of_2(self):
        # triton.next_power_of_2's body on a SymInt: the host's shifts are
        # floor divisions, its ors BitwiseFn_bitwise_or
        env = _TraceShapeEnv()
        n = env.symbol(1000, "n", positive=True)
        m = n - 1
        for shift in (1, 2, 4, 8, 16, 32):
            m |= m >> shift
        self.assertTrue(bool(m + 1 == 1024))
        program = IntegerProgram([1000])
        predicate = Lowering(program, {n.node.expr: ("boxed", 0)}).predicate(
            [g.expr for g in env.guards]
        )
        compiled = compile_program(program)
        holds = [_holds(compiled, predicate, [v]) for v in (1000, 513, 1024, 512, 1025)]
        self.assertEqual(holds, [True, True, True, False, False])

    def test_tensor_leaves(self):
        x = torch.empty(4, 6)[:, 1:]
        n, m, st = sympy.symbols("n m st", integer=True, positive=True)
        sources = {n: ("size", 0, 0), m: ("size", 0, 1), st: ("stride", 0, 0)}
        program = IntegerProgram([x])
        guards = [sympy.Eq(FloorDiv(st, m), 1), sympy.Lt(n * m, 64)]
        predicate = Lowering(program, sources).predicate(guards)
        compiled = compile_program(program)
        self.assertTrue(_holds(compiled, predicate, [x]))
        self.assertTrue(_holds(compiled, predicate, [torch.empty(2, 5)]))
        self.assertFalse(_holds(compiled, predicate, [torch.empty(2, 5).t()]))
        self.assertFalse(_holds(compiled, predicate, [torch.empty(16, 5)]))
        self.assertFalse(_holds(compiled, predicate, [torch.empty(0, 5)]))  # n > 0
        self.assertFalse(_holds(compiled, predicate, [3]))

    def test_a_guard_false_at_the_hints_is_an_internal_error(self):
        program = IntegerProgram([3])
        with self.assertRaisesRegex(AssertionError, "false at the hints"):
            Lowering(program, {S: ("boxed", 0)}).predicate([sympy.Gt(S, 5)])


instantiate_parametrized_tests(TestLowering)


def setUpModule():
    import torch.cuda._host_trace as host_trace

    host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
