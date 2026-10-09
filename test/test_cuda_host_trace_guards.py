# Owner(s): ["module: cuda graphs"]

import subprocess
import sys
from unittest import mock

import sympy

import torch
import torch.fx.experimental.sym_node as sym_node
from torch.cuda import _host_trace_ir as _ir
from torch.cuda._host_trace import _TraceShapeEnv, Declined
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


class TestIRSymNodeDunders(TestCase):
    # SymInt's binary dunders call an IRSymNode's method directly for ints
    # and non-constant IR ints; the generic path must build the same nodes
    # and record the same guards
    @staticmethod
    def _state(env, out):
        nodes = [(_ir.render(n), n.hint) for n in env.ctx.nodes]
        records = [(_ir.render(g), w and (w[0], *map(_ir.render, w[1:]))) for g, w in env.records]
        return nodes, records, [(type(v), v.node.hint if isinstance(v, (torch.SymInt, torch.SymBool, torch.SymFloat)) else v) for v in out]

    @staticmethod
    def _program(env):
        x = env.symbol(12, "x", positive=True)
        y = env.symbol(5, "y", positive=True)
        z = env.symbol(-3, "z")
        out = [
            x + 3, 3 + x, x - y, 5 - x, x * y, 2 * z, x // 4, 7 // y, x % y, z % 4,
            3 < x, x >= y, z == -3, 2 != y, x <= 12, y > z,
            torch.sym_max(3, x), torch.sym_min(x, y), torch.sym_max(z, 0),
            x**2, x & 7, x << 1, x + True, x * 1.5, x - torch.SymInt(x.node.wrap_int(4)),
        ]
        out += [bool(x * y > 50), bool(z < 0), int(x // y), bool(x % y == 2)]
        return out

    def test_direct_dunders_match_the_generic_path(self):
        generic = mock.patch.object(sym_node, "_DIRECT_INT_NODES", set())
        to_node = mock.patch.object(sym_node, "to_node", wraps=sym_node.to_node)
        env = _ir.Env()
        with to_node as calls:
            fast = self._state(env, self._program(env))
        # only the bool, the float and the constant node take the generic path
        self.assertEqual(calls.call_count, 3)
        env = _ir.Env()
        with generic:
            slow = self._state(env, self._program(env))
        self.assertEqual(fast, slow)
        self.assertGreater(len(fast[1]), 0)

    def test_symint_dunders_change_only_with_an_env(self):
        # a process that never builds an IR Env keeps SymInt's generic dunders
        code = "import torch.cuda._host_trace; from torch import SymInt; from torch.cuda import _host_trace_ir as ir; a = SymInt.__add__; ir.Env(); print(a.__name__, SymInt.__add__.__name__)"
        out = subprocess.check_output([sys.executable, "-c", code], text=True)
        self.assertEqual(out.split(), ["binary_magic_impl", "direct_binary_magic_impl"])

    def test_binary_is_the_dunder(self):
        import operator

        ops = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "int_floordiv": operator.floordiv, "mod": operator.mod}
        ops |= {m: getattr(operator, m) for m in ("eq", "ne", "lt", "le", "gt", "ge")}
        ops |= {"sym_max": torch.sym_max, "sym_min": torch.sym_min}
        for method, f in ops.items():
            for pair in ((0, 1), (0, 4), (4, 0)):
                states = []
                for op in (f, lambda a, b: _ir.binary(method, a, b)):
                    env = _ir.Env()
                    v = [env.symbol(12, "x", positive=True), env.symbol(5, "y", positive=True)] + [2, 3, 7]
                    r = op(v[pair[0]], v[pair[1]])
                    bool(r)
                    states.append(self._state(env, [r]))
                self.assertEqual(states[0], states[1], (method, pair))
        env = _ir.Env()
        self.assertIs(_ir.binary("add", 2, 3), NotImplemented)
        self.assertIs(_ir.binary("add", env.symbol(1, "x"), 1.5), NotImplemented)


class TestIRContexts(TestCase):
    # two traces' contexts number their nodes alike: an id of one names an
    # unrelated node of the other, so a node is only ever its own context's
    def test_transfer_never_resolves_a_node_of_another_context_by_id(self):
        variant, fresh, other = _ir.Env(), _ir.Env(), _ir.Env()
        v = variant.symbol(7, "v")
        mine, foreign = (fresh.symbol(1000, "y") + 15) // 16, (other.symbol(1000, "x") + 15) // 16
        self.assertEqual(mine.node.node.id, foreign.node.node.id)
        memo = {}
        symbols = {fresh.ctx.nodes[0].args[0]: v.node.node}
        variant.ctx.transfer(mine.node.node, fresh.ctx, symbols, memo)
        with self.assertRaises(KeyError):
            variant.ctx.transfer(foreign.node.node, fresh.ctx, symbols, memo)

    def test_an_operation_on_symbols_of_two_traces_is_refused(self):
        x, y = _ir.Env().symbol(8, "x"), _ir.Env().symbol(8, "y")
        with self.assertRaisesRegex(Declined, "symbols of two traces"):
            x + y

    def test_a_guard_on_another_traces_symbol_is_refused(self):
        variant, fresh = _ir.Env(), _ir.Env()
        n = variant.symbol(1000, "n")

        class Running:
            shape_env, declined = fresh, None

            def decline(self, msg):
                self.declined = Declined(msg)
                return self.declined

        running = Running()
        with mock.patch.object(_ir.ACTIVE, "trace", running, create=True):
            with self.assertRaisesRegex(Declined, "a guard on a symbol of another trace"):
                bool(n % 16 == 8)
        self.assertIsNotNone(running.declined)
        self.assertEqual(variant.records, [])
        # outside any trace (a fold, a lowering) the variant records its own
        self.assertTrue(bool(n % 16 == 8))
        self.assertEqual(len(variant.records), 1)


def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace as host_trace

    _host_trace_hint_audit.enable_for_tests()
    host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
