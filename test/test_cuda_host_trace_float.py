# Owner(s): ["module: cuda graphs"]
# Float arithmetic in traced host code: float32 and float64 rows evaluated in the host's operation order (the program's
# reference and Program.cpp), comparisons as guards, argmin's k - 1 guards, and the sympy export (Identity per operation).
import itertools
import math
import random
import struct
import unittest
from fractions import Fraction

import numpy as np
import sympy

import torch
import torch.cuda._host_trace as ht
from torch.cuda import _host_trace_cover as cover, _host_trace_ir as _ir
from torch.cuda._host_trace_lower import IRLowering
from torch.cuda._host_trace_program import (
    _step,
    compile_program,
    f32_bits,
    f64,
    f64_bits,
    IntegerProgram,
    Status,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase
from torch.utils._sympy.functions import Identity
from torch.utils._triton import has_triton


f32 = np.float32
FLOAT_ROWS = ("fadd", "fsub", "fmul", "f32r", "f32add", "f32sub", "f32mul", "f32quot", "f32fma")
MAIN = {64: 1.68, 32: 1.48, 16: 1.2, 8: 1.0}
RED = {64: 1.2, 32: 1.08, 16: 1.03, 8: 1.0}


def exact_fma32(a, b, c):
    # an independent reference: the nearest float32 to the exact a * b + c, by search
    v = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    if abs(v) >= Fraction(float(np.finfo(f32).max)) + Fraction(2) ** 103:  # past FLT_MAX by half an ulp
        return math.copysign(math.inf, v)
    with np.errstate(over="ignore"):
        x = f32(float(v))
        near = [y for y in (np.nextafter(x, f32(-np.inf)), x, np.nextafter(x, f32(np.inf))) if np.isfinite(y)]
    return float(min(near, key=lambda y: (abs(Fraction(float(y)) - v), int(np.array(y, np.float32).view(np.uint32)) & 1)))


def evaluate(program, inputs):
    """The program's rows at the boxed inputs by the Python reference (_step)."""
    values = []
    for op, *x in program.instructions:
        if op == "constant":
            v = x[0]
        elif op == "boxed":
            v = inputs[x[0]]
        else:
            v = _step(op, [values[i] for i in x])
            if isinstance(v, Status):
                return v, None
        values.append(v)
    return Status.SUCCESS, values


def float_rows(cases):
    """A program of constant rows and one row per (op, operands), compiled; the rows' indices."""
    program = IntegerProgram([])
    rows = []
    for op, operands in cases:
        args = [program.emit("constant", a if op == "f32i" else f64_bits(a)) for a in operands]
        rows.append(program.emit(op, *args))
    return program, rows


class TestFloatRows(TestCase):
    def cases(self):
        rng = random.Random(0)
        pool = [0.0, -0.0, 1.0, 1.5, 3.0, 2.0**-126, 2.0**-149, 3.4028234663852886e38, 1e-30, 16777216.0, 16777217.0]
        values = [float(f32(x)) for x in pool] + [float(f32(rng.uniform(-1e6, 1e6))) for _ in range(300)]
        values += [float(f32(rng.choice((1, -1)) * 2.0 ** rng.uniform(-40, 40))) for _ in range(300)]
        out = []
        for _ in range(3000):
            a, b, c = rng.choice(values), rng.choice(values), rng.choice(values)
            out += [("f32add", (a, b)), ("f32sub", (a, b)), ("f32mul", (a, b)), ("f32fma", (a, b, c))]
            if b != 0.0:
                out.append(("f32quot", (a, b)))
            x, y = rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3)
            out += [("fadd", (x, y)), ("fsub", (x, y)), ("fmul", (x, y)), ("f32r", (x * 1e-3,))]
        ints = [0, 1, -1, 2**24 + 1, 2**24 + 3, -(2**24 + 1), 2**53 + 2**29 + 1, 2**62 + 2**38 + 1, 2**63 - 1, -(2**63)]
        out += [("f32i", (n,)) for n in ints + [rng.randrange(-(2**40), 2**40) for _ in range(200)]]
        return out

    def expected(self, op, args):
        if op == "f32fma":
            return exact_fma32(*args)
        if op == "f32i":
            return float(Fraction(args[0]).limit_denominator(1)) if abs(args[0]) < 2**24 else _nearest_int32(args[0])
        if op == "f32r":
            return float(f32(args[0]))
        if op.startswith("f32"):
            x, y = f32(args[0]), f32(args[1])
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                return float({"f32add": x + y, "f32sub": x - y, "f32mul": x * y, "f32quot": x / y}[op])
        x, y = args
        return {"fadd": x + y, "fsub": x - y, "fmul": x * y}[op]

    def test_reference_rows_are_ieee(self):
        # each row is one IEEE operation: float32 ones as numpy's float32 ufuncs (one operation each), the fused
        # multiply-add rounded once, an integer converted once (not through a double above 2**53)
        for op, args in self.cases():
            got = f64(_step(op, [args[0]] if op == "f32i" else [f64_bits(a) for a in args]))
            want = self.expected(op, args)
            self.assertEqual(struct.pack("<d", got), struct.pack("<d", want), (op, args, got, want))

    def test_compiled_rows_match_the_reference(self):
        cases = self.cases()
        program, rows = float_rows(cases)
        status, values = compile_program(program).evaluate_inputs(())
        self.assertEqual(status, Status.SUCCESS)
        for (op, args), row in zip(cases, rows):
            self.assertEqual(values[row], program.values[row], (op, args))

    def test_f32_div_converts_its_integers_once(self):
        # static_cast<float>(int64) rounds once: 2**53 + 2**29 + 1 is above the float32 midpoint 2**53 + 2**29, so it
        # rounds up; through a double it lands on the midpoint and ties to even (down)
        n = 2**53 + 2**29 + 1
        want = struct.unpack("<i", struct.pack("<f", 2.0**53 + 2.0**30))[0]
        self.assertEqual(f32_bits(n, 1), want)
        program = IntegerProgram([])
        row = program.emit("f32div", program.emit("constant", n), program.emit("constant", 1))
        self.assertEqual(compile_program(program).evaluate_inputs(())[1][row], want)


def _nearest_int32(n):
    sh = abs(n).bit_length() - 24
    q, r = divmod(abs(n), 1 << sh)
    half = 1 << (sh - 1)
    q += r > half or (r == half and q & 1)
    return math.copysign(float(q << sh), n)


def cost_terms(mk, batch, sm, tiles):
    """The trtllm-gen tile cost model's integer inputs per candidate (selectTileSizeQForGqaGeneration's shape, a
    simplified computeCtaAndClusterConfig): (tile, seq per CTA, CTAs per KV sequence, waves), mk and batch ints or
    SymInts; the min is an expression (sym_min), so only the comparisons guard."""
    out = []
    for tile, (step, ctas) in tiles.items():
        kv = torch.sym_min(-(-mk // (2 * step)), torch.sym_max(1, sm // (ctas * batch)))
        seq = -(-(-(-mk // step)) // kv) * step
        out.append((tile, seq, kv, -(-(ctas * kv * batch) // sm)))
    return out


def binary_costs(terms):
    # fmha_gen.so's order on aarch64: (factor*M)*seq, then fma(R*128, kv, that), times waves
    costs = []
    for tile, seq, kv, waves in terms:
        main = ht.f32_mul(ht.f32_mul(1.0, ht.f32(MAIN[tile])), ht.f32(seq))
        t = ht.f32_fma(ht.f32_mul(ht.f32(RED[tile]), 128.0), ht.f32(kv), main)
        costs.append(ht.f32_mul(ht.f32(waves), t))
    return costs


def reference_costs(terms):
    # the same order in numpy float32 and the search-based fused multiply-add
    out = []
    for tile, seq, kv, waves in terms:
        main = f32(f32(1.0) * f32(MAIN[tile])) * f32(seq)
        t = exact_fma32(f32(f32(RED[tile]) * f32(128.0)), f32(kv), main)
        out.append(f32(waves) * f32(t))
    return out


def first_min(costs, below):
    best, k = below, -1
    for i, c in enumerate(costs):
        if c < best:
            best, k = c, i
    return k


class TestFloatGuards(TestCase):
    TILES = {64: (128, 2), 32: (128, 4), 16: (128, 8), 8: (64, 16)}

    def trace(self, mk, batch, sm=148):
        env = _ir.Env()
        m, b = env.symbol(mk, "mk", positive=True), env.symbol(batch, "b", positive=True)
        tiles = list(self.TILES)
        k = ht.argmin(binary_costs(cost_terms(m, b, sm, self.TILES)), below=ht.FLT_MAX)
        program = IntegerProgram([mk, batch])
        lowering = IRLowering(program, {m.node.node: ("boxed", 0), b.node.node: ("boxed", 1)}, env.ctx)
        predicate = lowering.predicate(env.records)
        return tiles[k], env, program, predicate

    def holds(self, program, predicate, point):
        status, values = evaluate(program, point)
        return status == Status.SUCCESS and values[predicate] == 1

    def test_argmin_guards_the_winner_only(self):
        # k - 1 comparisons and the bound, each against the winner; a tie goes to the first
        env = _ir.Env()
        x = env.symbol(10, "x", positive=True)
        costs = [ht.f32_mul(ht.f32(x), c) for c in (3.0, 2.0, 2.0, 5.0)]
        self.assertEqual(ht.argmin(costs, below=ht.FLT_MAX), 1)
        self.assertEqual(len(env.records), 4)
        self.assertEqual(ht.argmin([3.0, 2.0, 2.0, 5.0]), 1)
        self.assertEqual(ht.argmin([3.0, 2.0], below=1.0), -1)
        y = env.symbol(10, "y", positive=True)
        self.assertEqual(ht.argmin([ht.f32(y)], below=4.0), -1)

    def test_guards_hold_exactly_where_the_winner_holds(self):
        # the guard program at every (max_kv, batch) of a box around the hint: it holds where the binary's order picks
        # the traced tile (numpy float32 reference), else not; the integer parts are expressions
        for mk, batch in ((300, 1), (1500, 2), (2100, 1), (5000, 3)):
            tile, env, program, predicate = self.trace(mk, batch)
            tiles = list(self.TILES)
            seen = set()
            for m, b in itertools.product(range(max(1, mk - 700), mk + 700, 3), range(1, 6)):
                want = tiles[first_min(reference_costs(cost_terms(m, b, 148, self.TILES)), f32(ht.FLT_MAX))]
                self.assertEqual(self.holds(program, predicate, (m, b)), want == tile, (mk, batch, m, b, tile, want))
                seen.add(want)
            self.assertGreater(len(seen), 1, (mk, batch))  # the box crosses a decision boundary

    def test_the_fused_order_decides_an_exact_tie(self):
        # Q/KV heads 32/8, max_q 16, batch 1, 148 SMs, max_kv 2049: tiles 32 and 16 cost 1448.96 in exact arithmetic;
        # the binary's fused reduction term picks 16, an unfused order 32
        terms = [(64, 512, 5, 1), (32, 512, 5, 1), (16, 768, 4, 1), (8, 1280, 2, 1)]
        self.assertEqual(ht.argmin(binary_costs(terms), below=ht.FLT_MAX), 2)
        unfused = []
        for tile, seq, kv, waves in terms:
            main = ht.f32_mul(ht.f32(MAIN[tile]), ht.f32(seq))
            unfused.append(ht.f32_mul(ht.f32(waves), ht.f32_add(main, ht.f32_mul(ht.f32_mul(ht.f32(RED[tile]), 128.0), ht.f32(kv)))))
        self.assertEqual(ht.argmin(unfused, below=ht.FLT_MAX), 1)

    def test_large_integers_round_once(self):
        # f32(n) of a symbol is static_cast<float>: n = 2**24 + 1 converts to 2**24, so 2**24 + 1 and 2**24 compare equal
        env = _ir.Env()
        n = env.symbol(2**24 + 1, "n", positive=True)
        self.assertTrue(bool(ht.f32(n) == ht.f32(2**24)))
        program = IntegerProgram([2**24 + 1])
        predicate = IRLowering(program, {n.node.node: ("boxed", 0)}, env.ctx).predicate(env.records)
        self.assertTrue(self.holds(program, predicate, (2**24,)))
        self.assertFalse(self.holds(program, predicate, (2**24 + 2,)))
        self.assertTrue(self.holds(program, predicate, (2**24 + 1,)))

    def test_float32_operands_are_checked(self):
        env = _ir.Env()
        x = ht.f32(env.symbol(3, "x", positive=True))
        with self.assertRaisesRegex(TypeError, "float32 value"):
            ht.f32_mul(x, 2.2)
        with self.assertRaisesRegex(TypeError, "float32 value"):
            ht.f32_mul(x, env.symbol(3, "y", positive=True) * 1.5)
        with self.assertRaisesRegex(TypeError, "float32 value"):
            ht.f32_add(2.2, 1.0)
        self.assertEqual(ht.f32_mul(x, ht.f32(2.2)).node.node.hint, float(f32(3.0) * f32(2.2)))

    def test_cover_evaluates_float_guards_by_the_rows(self):
        # the cover solver evaluates an IR float node by the rows' rule over the box, not its sympy export
        env = _ir.Env()
        s, k = env.symbol(512, "s", positive=True), env.symbol(5, "k", positive=True)
        t = ht.f32_fma(ht.f32_mul(ht.f32(1.08), 128.0), ht.f32(k), ht.f32_mul(ht.f32(1.48), ht.f32(s)))
        leaves = {s.node.node.args[0]: np.arange(1, 4000, dtype=np.int64), k.node.node.args[0]: 5}
        got = cover._ir_vector(t.node.node, leaves, {}, True, {})
        want = [exact_fma32(f32(f32(1.08) * f32(128.0)), f32(5), f32(f32(1.48) * f32(v))) for v in range(1, 4000)]
        self.assertEqual(got.tolist(), want)


class TestFloatExport(TestCase):
    def export(self):
        env = _ir.Env()
        s, k = env.symbol(512, "s", positive=True), env.symbol(5, "k", positive=True)
        fused = ht.f32_fma(ht.f32_mul(ht.f32(1.08), 128.0), ht.f32(k), ht.f32_mul(ht.f32(1.48), ht.f32(s)))
        unfused = ht.f32_add(ht.f32_mul(ht.f32_mul(ht.f32(1.08), 128.0), ht.f32(k)), ht.f32_mul(ht.f32(1.48), ht.f32(s)))
        return env, s, k, fused, unfused

    def test_identity_blocks_simplification(self):
        # one Identity per float operation: sympy's simplify, expand, factor, cancel and substitution leave the export
        # as it is (no reassociation or folding across an operation)
        env, s, k, fused, unfused = self.export()
        r = sympy.Symbol("r", integer=True, positive=True)
        for v in (fused, unfused):
            e = env.export.expr(v.node.node)
            self.assertTrue(isinstance(e, Identity))
            for f in (sympy.simplify, sympy.expand, sympy.factor, sympy.cancel, sympy.together, lambda x: x.doit()):
                self.assertEqual(f(e), e)
            sym_s, sym_k = env.export.expr(s.node.node), env.export.expr(k.node.node)
            self.assertEqual(e.subs(sym_s, r).subs(r, sym_s), e)
            # numbers in: every operation keeps its Identity (no fold across it)
            at = e.subs({sym_s: 512, sym_k: 5})
            self.assertEqual(sum(isinstance(x, Identity) for x in sympy.preorder_traversal(at)), sum(isinstance(x, Identity) for x in sympy.preorder_traversal(e)))
        self.assertNotEqual(env.export.expr(fused.node.node), env.export.expr(unfused.node.node))

    def test_rows_evaluate_in_the_operation_order(self):
        # the IR's rows are the evaluation: at (512, 5) the fused and unfused orders differ by an ulp
        env, s, k, fused, unfused = self.export()
        program = IntegerProgram([512, 5])
        lowering = IRLowering(program, {s.node.node: ("boxed", 0), k.node.node: ("boxed", 1)}, env.ctx)
        rf, ru = lowering._float(fused.node.node), lowering._float(unfused.node.node)
        self.assertEqual(f64(program.values[rf]), 1448.9600830078125)
        self.assertEqual(f64(program.values[ru]), 1448.9599609375)
        self.assertEqual([program.instructions[rf][0], program.instructions[ru][0]], ["f32fma", "f32add"])


HAS_TRITON = has_triton()
if HAS_TRITON:
    import triton
    import triton.language as tl
    from torch.cuda._host_trace_replay import HostTraceReplay as _Replay

    @triton.jit
    def _add(x_ptr, y_ptr, n, s, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        m = i < n
        tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) + s, mask=m)

    class HostTraceReplay(_Replay):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._called = True


@unittest.skipIf(not TEST_CUDA or not HAS_TRITON, "CUDA and Triton")
class TestFloatCostModelOp(TestCase):
    def test_a_float_cost_model_dispatches_again_where_its_choice_changes(self):
        # a launcher's block choice by a float cost model over n, inside a dispatch_unit: the comparisons are the
        # unit's guards (argmin's, against the winner), so a new n replays while the choice holds and redispatches
        # the unit where it changes; no call traces again
        blocks = (256, 512, 1024, 2048)

        def costs(n):
            out = []
            for b in blocks:
                k = -(-n // b)
                out.append(ht.f32_fma(ht.f32(0.05), ht.f32(k), ht.f32_mul(ht.f32(-(-k // 132)), ht.f32(1.0 + 0.37 * b / 256.0))))
            return out

        @ht.dispatch_unit
        def pick(x):
            n = x.numel()
            b = blocks[ht.argmin(costs(n))]
            y = torch.empty_like(x)
            _add[(triton.cdiv(n, b),)](x, y, n, 1, B=b)
            return y

        def f(x):
            return pick(x).sin()

        r = HostTraceReplay(f)
        # all 1 mod 16: Triton's own specialization of n (% 16, == 1) is fixed
        sizes = (1009, 1009, 2001, 3825, 3841, 5009, 1505, 20001, 70001, 14833, 4001, 200001, 90001)
        for n in sizes:
            x = torch.randn(n, device="cuda")
            self.assertEqual(r(x), f(x), atol=0, rtol=0)
        choice = [ht.argmin(costs(n)) for n in sizes]
        self.assertGreater(len(set(choice)), 2)
        # the trace is the first call's; each choice first seen after it is a redispatch (a choice seen before replays)
        seen, expected = {choice[0]}, 0
        for c in choice[1:]:
            expected += c not in seen
            seen.add(c)
        self.assertEqual((r.traces, r.eager, r.redispatches, len(r.variants)), (1, 0, expected, 1))


if __name__ == "__main__":
    run_tests()
