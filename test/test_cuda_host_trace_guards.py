# Owner(s): ["module: cuda graphs"]

import sympy

import torch
from torch.cuda._host_trace import _TraceShapeEnv
from torch.fx.experimental.symbolic_shapes import ShapeEnv, statically_known_false, statically_known_true
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.functions import FloorDiv, Mod, PythonMod


def _syms(env, pos, **values):
    # symbol() also serves a plain ShapeEnv, the reference for what the host
    # decides
    return [_TraceShapeEnv.symbol(env, v, n, positive=pos) for n, v in values.items()]


class TestTraceShapeEnv(TestCase):
    def test_guards_are_the_expressions_the_host_evaluated(self):
        # every branch is decided by its hint and recorded as evaluated: an
        # int read is `Eq(expr, value)`, never a replacement; the same relation
        # is recorded once; a constant is no guard
        def program(env):
            a, b, c = _syms(env, True, a=8, b=8, c=1)
            (st,) = _syms(env, False, st=8)
            taken = [
                bool(a == b),
                bool(4 * a == 4 * b),  # the same fact in another form
                bool(a == b),  # a repeat
                bool(c == 1),  # a specialization
                bool((c == 1) | (st == 8)),
                int(a * b),  # an int read
                bool(st != 0),  # the host's own test of the divisor ...
                bool(a * b % st == 0),  # ... is the modulo's domain guard
                bool(a == a),  # a constant
            ]
            return taken, [x.node.expr for x in (a, b, c, st)]

        plain = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        env = _TraceShapeEnv()
        taken, (A, B, C, ST) = program(env)
        self.assertEqual(taken, program(plain)[0])
        self.assertTrue(plain.replacements)
        self.assertFalse(env.replacements)
        want = [
            sympy.Eq(A, B),
            sympy.Eq(4 * A, 4 * B),
            sympy.Eq(C, 1),
            sympy.Or(sympy.Eq(C, 1), sympy.Eq(ST, 8)),
            sympy.Eq(A * B, 64),
            sympy.Ne(ST, 0),
            sympy.Eq(PythonMod(A * B, ST), 0),
        ]
        self.assertEqual([g.expr for g in env.guards], want)

    def test_a_hash_is_a_specialization(self):
        # a key built from sizes (quack's jit_cache) guards on their values;
        # any other ShapeEnv's SymInt stays unhashable (einops relies on it)
        env = _TraceShapeEnv()
        (n,) = _syms(env, True, n=1024)
        self.assertEqual({(n, 2): "compiled"}[(1024, 2)], "compiled")
        self.assertEqual(hash(n), hash(1024))
        self.assertEqual([g.expr for g in env.guards], [sympy.Eq(n.node.expr, 1024)])
        (p,) = _syms(ShapeEnv(), True, p=1024)
        with self.assertRaisesRegex(TypeError, "unhashable type: non-nested SymInt"):
            hash(p)

    def test_a_static_read_is_a_guard(self):
        # statically_known_true/false decide by the hint where eager reads a
        # concrete value; any other ShapeEnv's stays undecided without a guard
        env = _TraceShapeEnv()
        a, b = _syms(env, True, a=8, b=8)
        self.assertTrue(statically_known_true(a == b))
        self.assertTrue(statically_known_false(a == 4))
        A, B = a.node.expr, b.node.expr
        self.assertEqual([g.expr for g in env.guards], [sympy.Eq(A, B), sympy.Ne(A, 4)])
        plain = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        p, q = _syms(plain, True, p=8, q=8)
        self.assertFalse(statically_known_true(p == q))
        self.assertFalse(plain.guards)

    def test_ranges_are_never_refined(self):
        # a guard never lets a later one be decided statically: each is
        # checked on its own at replay
        env = _TraceShapeEnv()
        (a,) = _syms(env, True, a=4)
        self.assertTrue(bool(a < 5))
        self.assertTrue(bool(a < 10))
        A = a.node.expr
        self.assertEqual([g.expr for g in env.guards], [A < 5, A < 10])

    def test_partial_operations_record_their_domain_when_created(self):
        # `Ne(divisor, 0)` is recorded when the division is created, so the
        # ordered guards check it before any guard built on the division,
        # whether or not the host tests the divisor itself; a domain the
        # declared ranges decide (a size, a literal) is no guard
        def program(env, order):
            a, b = _syms(env, True, a=8, b=6)
            st, zf = _syms(env, False, st=2, zf=0.5)
            taken = []
            if order == "div-test":
                taken.append(bool(a // st > 0))
            elif order == "test-div":
                taken.append(bool(st != 0))
                taken.append(bool(a % st == 0))
            elif order == "pin-div":
                taken.append(bool(st == 2))
                torch.sym_min(a // st, 0)
            elif order == "div-pin":
                a // st
                taken.append(bool(st == 2))
            else:
                for v in (a % b, (a - 1) % b, a // 4, (a // 2) % b, a >> st):
                    taken.append(v.node.hint)
                taken.append((a / st).node.hint)  # IntTrueDiv by st: a guard
                taken.append((1.0 / zf).node.hint)  # a float division too
            return taken, [x.node.expr for x in (a, st, zf)]

        for order in ("div-test", "test-div", "pin-div", "div-pin", "none"):
            env = _TraceShapeEnv()
            plain = ShapeEnv(duck_shape=False, specialize_zero_one=False)
            taken, (A, ST, ZF) = program(env, order)
            self.assertEqual(taken, program(plain, order)[0], order)
            want = {
                "div-test": [sympy.Ne(ST, 0), sympy.Gt(FloorDiv(A, ST), 0)],
                "test-div": [sympy.Ne(ST, 0), sympy.Eq(PythonMod(A, ST), 0)],
                "pin-div": [sympy.Eq(ST, 2), sympy.Ne(ST, 0)],
                "div-pin": [sympy.Ne(ST, 0), sympy.Eq(ST, 2)],
                "none": [sympy.Ne(ST, 0), sympy.Ne(ZF, 0)],
            }[order]
            self.assertEqual([g.expr for g in env.guards], want, order)

    def test_torch_mod_records_nonnegative_operands(self):
        # torch's Mod is defined for nonnegative operands only; sym_node.py
        # builds it when the declared ranges show both are (no guard),
        # otherwise PythonMod
        env = _TraceShapeEnv()
        a, b = _syms(env, True, a=8, b=6)
        (st,) = _syms(env, False, st=2)
        A, ST = a.node.expr, st.node.expr
        self.assertIsInstance(((a // 2) % b).node.expr, Mod)
        self.assertIsInstance((a % st).node.expr, PythonMod)
        self.assertEqual([g.expr for g in env.guards], [sympy.Ne(ST, 0)])
        env.domain(ST, A, Mod(ST, A))
        env.domain(A, FloorDiv(A, ST), FloorDiv(A, FloorDiv(A, ST)))
        want = [sympy.Ge(ST, 0), sympy.Ne(FloorDiv(A, ST), 0)]
        self.assertEqual([g.expr for g in env.guards][1:], want)


def setUpModule():
    import torch.cuda._host_trace as host_trace

    host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
