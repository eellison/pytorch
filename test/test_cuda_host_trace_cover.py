# Owner: module: cuda graphs
# Cover (torch.cuda._host_trace_cover): the solver's acceptance against the replay's own evaluate at sampled points,
# and a covered box replays at every region's sample points with no trace, bitwise equal to eager.
import collections
import random
from unittest import mock

import numpy as np
import sympy

import torch
import torch.cuda._host_trace as ht
import torch.utils._sympy.functions as F
from torch.cuda._host_trace import BitLength, F32Div
from torch.cuda import _host_trace_cover as cv
from torch.cuda._host_trace_replay import HostTraceReplay
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TestCase


def f1(x, w):
    return torch.softmax(x, 1) + (x * w).sum(0), x.cumsum(0)


def f2(x, y, w):
    # x: (nd + T, 64), y: (nd + 1, 64)
    return (x * w).sum(0) + y.amax(0), torch.relu(y) * 2


def decoder(x, *ws):
    # naive attention: the softmax and the reductions over T pick kernels by T, the same guards in every layer
    for w in ws:
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        x = x + torch.softmax(h @ h.t(), -1) @ h @ w
    return x, x.sum(0)


def decode(q, page_table, kv):
    # q: (bs, 64); page_table: (bs, ceil(S / 64)) page ids into kv: (pages, 64, 64)
    k = kv[page_table].flatten(1, 2)
    p = torch.softmax((q[:, None] * k).sum(-1), -1)
    return (p[..., None] * k).sum(1)


class _NoMemo(dict):
    def __setitem__(self, k, v):
        pass


class TestCover(TestCase):
    def check(self, r, f, at, solver, points, make):
        # the solver's acceptance is the native evaluate's (graph guards), and with selectors every op selects an
        # entry (no dispatch again, no fold)
        live = [v for v in r.variants if not v.learns]
        cov = [(solver.accepts(v), solver.selected(v)) for v in live]
        for p in points:
            idx = tuple(x - lo for x, lo in zip(p if isinstance(p, tuple) else (p,), solver.lo))
            ev = [v.native.evaluate(at(p)) for v in live]
            want = [(e is not None, e is not None and not e[3]) for e in ev]
            self.assertEqual([(bool(a[idx]), bool(b[idx])) for a, b in cov], want, p)
        # a covered point is a replay: no trace, bitwise equal to eager
        traced = []
        for p in points:
            args = make(p)
            before = (r.traces, r.folds, r.redispatches, r.eager, r.replays + 1)
            self.assertEqual(r(*args), f(*args), atol=0, rtol=0)
            if (r.traces, r.folds, r.redispatches, r.eager, r.replays) != before:
                traced.append(p)
        self.assertEqual(traced, [])

    @parametrize("mode", ["graph", "op"])
    def test_cover_1d(self, mode):
        w = torch.randn(64, device="cuda")
        r = HostTraceReplay(f1)
        got = {}
        for t in (100, 300, 700):
            got[t] = (torch.randn(t, 64, device="cuda"), w)
            r(*got[t])
        at = cv.ArgsAt(got)
        lo, hi = 1, 1 << 14
        c = cv.cover(r, at, lambda t: r(torch.randn(t, 64, device="cuda"), w), lo, hi, mode=mode)
        self.assertTrue(c["proven"], c)
        self.assertEqual([f for f in c["fallbacks"] if f[1] != "vector"], [])
        self.assertEqual(c["guards"].get("fallback_per_T", 0), 0)
        solver = cv.Cover(at, lo, hi)
        self.assertEqual(solver.uncovered(r, selectors=True), [])
        points = cv.points(solver) + random.Random(0).sample(range(lo, hi + 1), 200)
        self.check(r, f1, at, solver, points, lambda t: (torch.randn(t, 64, device="cuda"), w))

    @parametrize("mode", ["graph", "op"])
    def test_cover_2d(self, mode):
        w = torch.randn(64, device="cuda")
        r = HostTraceReplay(f2)

        def make(p):
            nd, t = p
            return torch.randn(nd + t, 64, device="cuda"), torch.randn(nd + 1, 64, device="cuda"), w

        got = {}
        for p in ((3, 100), (3, 300), (9, 100), (9, 700)):
            got[p] = make(p)
            r(*got[p])
        at = cv.ArgsAt(got)
        self.assertEqual(at.dims, 2)
        lo, hi = (0, 1), (64, 2048)
        c = cv.cover(r, at, lambda p: r(*make(p)), lo, hi, mode=mode)
        self.assertTrue(c["proven"], c)
        self.assertEqual([f for f in c["fallbacks"] if f[1] != "vector"], [])
        solver = cv.Cover(at, lo, hi)
        self.assertEqual(solver.uncovered(r, selectors=True), [])
        rng = random.Random(0)
        points = cv.points(solver) + [(rng.randint(0, 64), rng.randint(1, 2048)) for _ in range(200)]
        self.check(r, f2, at, solver, points, make)

    def test_cover_decode(self):
        # a decode step over (bs, S): the page table's width is ceil(S / 64), not affine in S
        kv = torch.randn(64, 64, 64, device="cuda")

        def make(p):
            b, s = p
            return torch.randn(b, 64, device="cuda"), torch.randint(0, 64, (b, -(-s // 64)), device="cuda"), kv

        r = HostTraceReplay(decode)
        got = {p: make(p) for p in ((2, 1000), (2, 3000), (6, 1000), (6, 5000))}
        for args in got.values():
            r(*args)
        with self.assertRaisesRegex(ValueError, "affine"):
            cv.ArgsAt(got, axes=("bs", "S"))
        with self.assertRaisesRegex(ValueError, "argument 1 dim 1"):
            cv.ArgsAt(got, axes=("bs", "S"), sizes={(1, 1): F.CeilDiv(cv.axis("S"), 32)})
        at = cv.ArgsAt(got, axes=("bs", "S"), sizes={(1, 1): F.CeilDiv(cv.axis("S"), 64)})
        lo, hi = (1, 1), (16, 8192)
        c = cv.cover(r, at, lambda p: r(*make(p)), lo, hi, mode="graph", count_regions=False)
        self.assertTrue(c["proven"], c)
        self.assertIsNone(c["regions"])
        self.assertEqual([f for f in c["fallbacks"] if f[1] != "vector"], [])
        solver = cv.Cover(at, lo, hi)
        self.assertEqual(solver.uncovered(r, selectors=True), [])
        print(f"decode: traces {c['delta']['traces']} splits {collections.Counter(solver.split_guards.values())}")
        rng = random.Random(0)
        points = cv.points(solver) + [(rng.randint(1, 16), rng.randint(1, 8192)) for _ in range(100)]
        self.check(r, decode, at, solver, points, make)

    def test_shared_rows_are_one_object(self):
        # a guard shared across variants is reduced to one row object, and the solver's memos change no result
        kv = torch.randn(64, 64, 64, device="cuda")

        def make(p):
            return torch.randn(p[0], 64, device="cuda"), torch.randint(0, 64, (p[0], -(-p[1] // 64)), device="cuda"), kv

        r = HostTraceReplay(decode)
        got = {p: make(p) for p in ((2, 1000), (2, 3000), (6, 1000), (6, 5000))}
        for args in got.values():
            r(*args)
        at, lo, hi = cv.ArgsAt(got, axes=("bs", "S"), sizes={(1, 1): F.CeilDiv(cv.axis("S"), 64)}), (1, 1), (16, 8192)
        self.assertTrue(cv.cover(r, at, lambda p: r(*make(p)), lo, hi, count_regions=False)["proven"])
        live = [v for v in r.variants if not v.learns]
        new, old = cv.Cover(at, lo, hi), cv.Cover(at, lo, hi)
        old._reduced, old._anded, old._ored = _NoMemo(), _NoMemo(), _NoMemo()
        for v in live:
            self.assertTrue(np.array_equal(new.accepts(v), old.accepts(v)) and np.array_equal(new.selected(v), old.selected(v)))
        rows = [x for x in new._rows.values() if x is not None]
        self.assertLess(len({id(x) for x in rows}), len(rows))

    @parametrize("case", ["1d", "decode"])
    def test_ir_rows_match_substituted(self, case):
        # a guard's row read off its IR record equals the substituted sympy guard's (ir_rows=False), the int64
        # evaluation equals the exact one, and the cover exports no tape's guards to sympy
        if case == "1d":
            w = torch.randn(64, device="cuda")
            r = HostTraceReplay(f1)
            got = {t: (torch.randn(t, 64, device="cuda"), w) for t in (100, 300, 700)}
        else:
            kv = torch.randn(64, 64, 64, device="cuda")
            r = HostTraceReplay(decode)
            got = {(b, s): (torch.randn(b, 64, device="cuda"), torch.randint(0, 64, (b, -(-s // 64)), device="cuda"), kv) for b, s in ((2, 1000), (2, 3000), (6, 1000), (6, 5000))}
        for args in got.values():
            r(*args)
        if case == "1d":
            at, lo, hi = cv.ArgsAt(got), 1, 1 << 14
        else:
            at, lo, hi = cv.ArgsAt(got, axes=("bs", "S"), sizes={(1, 1): F.CeilDiv(cv.axis("S"), 64)}), (1, 1), (16, 8192)
        live = [v for v in r.variants if not v.learns]
        exported = ["guards" in v.tape.__dict__ for v in live]
        new = cv.Cover(at, lo, hi)
        cov = [(new.accepts(v), new.selected(v)) for v in live]
        self.assertEqual(["guards" in v.tape.__dict__ for v in live], exported)
        self.assertGreater(new.counts["guards_ir"], 0)
        with mock.patch.object(cv, "ir_rows", False):
            old = cv.Cover(at, lo, hi)
            for v, rows in zip(live, cov):
                self.assertTrue(np.array_equal(old.accepts(v), rows[0]) and np.array_equal(old.selected(v), rows[1]))
        self.assertEqual(new._rows.keys(), old._rows.keys())
        for k, row in old._rows.items():
            a, b = (np.broadcast_to(True if x is None else x, new.shape) for x in (new._rows[k], row))
            self.assertTrue((a == b).all(), k)
        for v in live:
            leaves = new._variant_leaves(v).values["box"]
            for node, written in v.tape.shape_env.records:
                for x in (node,) if written is None else written[1:]:
                    try:
                        a = cv._ir_vector(x, leaves[np.int64], {}, True, {})
                    except cv._EVAL_ERRORS:
                        continue
                    b = cv._ir_vector(x, leaves[object], {}, False, {})
                    self.assertTrue((np.broadcast_to(a, new.shape) == np.broadcast_to(b, new.shape)).all(), x)

    @parametrize("shared", [False, True])
    def test_op_cover_of_identical_layers(self, shared):
        # with shared_op_guards the layers' common guards are their ops', so a region of them is one redispatch of
        # those ops (one group); without, they are graph-level and each region is a trace
        ws = [torch.randn(64, 64, device="cuda") / 8 for _ in range(4)]

        def make(t):
            return (torch.randn(t, 64, device="cuda"), *ws)

        with mock.patch.object(ht, "shared_op_guards", shared):
            r = HostTraceReplay(decoder)
            got = {t: make(t) for t in (100, 300, 700)}
            for args in got.values():
                r(*args)
            at = cv.ArgsAt(got)
            lo, hi = 1, 2048
            c = cv.cover(r, at, lambda t: r(*make(t)), lo, hi, mode="op")
            self.assertTrue(c["proven"], c)
            solver = cv.Cover(at, lo, hi)
            self.check(r, decoder, at, solver, cv.points(solver), make)
        print(f"shared {shared}: traces {c['delta']['traces']} redispatches {c['delta']['redispatches']} groups {c['groups']} regions {c['regions']}")
        selectors = sum(len(v.captured.lowered.selectors) for v in r.variants if not v.learns)
        if shared:
            # identical ops across the layers solve and dispatch once
            self.assertLess(len(c["groups"]), selectors)


T, ND = cv.T, cv.ND
log2 = F.OpaqueUnaryFn_log2


def row(g, lo, hi, first=0):
    counts, stats, fallbacks = collections.Counter(), collections.Counter(), []
    with mock.patch.object(cv, "vector_first", first):
        return cv._guard_row(g, lo, hi, counts, stats, fallbacks, ("g",)), counts, stats, fallbacks


class _At:
    def __init__(self, axes):
        self.dims, self.axes = len(axes), axes


class TestCoverSolver(TestCase):
    # CPU only: guards built by hand
    # each needs the fallback: a modulus over a quarter of the range, an opaque function, a bitwise function,
    # Min/Max of floors, the host trace's own functions (BitLength is test_cover_1d's: a reduction's config), a select
    FALLBACKS = [
        sympy.Eq(F.FloorDiv(T, 1024), 1),
        sympy.Lt(F.FloorToInt(log2(T)), 7),
        sympy.Eq(F.BitwiseFn_bitwise_and(T, T - 1), 0),
        sympy.Gt(F.Max(F.ModularIndexing(T, 3, 512), F.CeilDiv(T, 700)), 100),
        sympy.Lt(BitLength(T * 64 - 1), 13),
        sympy.Ge(F32Div(T, 7), 1120403456),
        sympy.Gt(F.Where(sympy.Gt(F.CeilDiv(T, 256), 8), F.FloorDiv(T, 7), 2 * T), 400),
    ]

    def test_vector_fallback_matches_per_t(self):
        lo, hi = 1, 3000
        for g in self.FALLBACKS:
            got, counts, stats, fallbacks = row(g, lo, hi)
            self.assertEqual(counts["guards_fallback"], 1, (g, fallbacks))
            self.assertEqual([f[1] for f in fallbacks], ["vector"], (g, fallbacks))
            self.assertLess(stats["fallback_s"], 1.0, g)
            want = np.array([cv._truth(g, x) for x in range(lo, hi + 1)])
            self.assertTrue((got == want).all(), (g, np.flatnonzero(got != want)[:5]))

    def test_undefined_points_are_false(self):
        # a divisor that is 0 on part of the range (a tile count below one tile): there evaluate misses
        lo, hi = 1, 3000
        got, counts, _, fallbacks = row(sympy.Eq(F.Mod(64, F.FloorDiv(T, 128)), 0), lo, hi)
        self.assertEqual(counts["guards_fallback"], 1)
        self.assertEqual([f[1] for f in fallbacks], ["per_T"])
        want = [t >= 128 and 64 % (t // 128) == 0 for t in range(lo, hi + 1)]
        self.assertEqual(got.tolist(), want)

    def test_vector_matches_solver(self):
        # guards the solver reduces: the vector form agrees with the solved row
        lo, hi = 1, 5000
        for g in [sympy.Ge(T, 4097), sympy.Eq(F.Mod(T - 1, 16), 0), sympy.Le(F.Min(T, 2048) * 3, F.FloorDiv(T, 2) + 900)]:
            solved, counts, _, _ = row(g, lo, hi)
            self.assertEqual(counts["guards_solved"], 1, g)
            ts = np.arange(lo, hi + 1).astype(object)
            vec = np.broadcast_to(np.asarray(cv._vector(g, ts), dtype=bool), ts.shape)
            self.assertTrue((solved == vec).all(), g)

    def test_vector_first(self):
        # a short axis is evaluated vectorized before the solver, with the solver's result
        lo, hi = 1, 3000
        for g in [sympy.Eq(F.Mod(T - 1, 16), 0), self.FALLBACKS[3]]:
            got, counts, _, fallbacks = row(g, lo, hi, first=16384)
            self.assertEqual((counts["guards_vector"], fallbacks), (1, []), g)
            self.assertEqual(got.tolist(), row(g, lo, hi)[0].tolist(), g)
        # undefined somewhere: the solver's route
        _, counts, _, _ = row(sympy.Eq(F.Mod(64, F.FloorDiv(T, 128)), 0), lo, hi, first=16384)
        self.assertEqual((counts["guards_vector"], counts["guards_fallback"]), (0, 1))

    def test_2d_axes_at_zero_and_one(self):
        lo, hi = (0, 1), (64, 2048)
        nd = np.arange(lo[0], hi[0] + 1)[:, None]
        t = np.arange(lo[1], hi[1] + 1)[None, :]
        cases = {
            sympy.Eq(ND + 1, 1): np.broadcast_to(nd == 0, (65, 2048)),
            sympy.Ne(T + ND, 1): (nd + t) != 1,
            sympy.Gt(T, 5 * ND): t > 5 * nd,
            sympy.Eq(T, 1): np.broadcast_to(t == 1, (65, 2048)),
            sympy.Ge(T * ND, 7): t * nd >= 7,
            sympy.Lt(F.FloorDiv(T + ND, 128), 3): (nd + t) // 128 < 3,
        }
        c = cv.Cover(_At((ND, T)), lo, hi)
        for g, want in cases.items():
            got = np.broadcast_to(c._solve(g, (str(g),)), c.shape)
            self.assertTrue((got == want).all(), (g, np.argwhere(got != want)[:5].tolist()))
        self.assertEqual(c.fallbacks, [])
        self.assertEqual(c.stats["sums"], 2)

    def test_named_axes(self):
        bs, S = cv.axis("bs"), cv.axis("S")
        lo, hi = (1, 1), (32, 4096)
        b = np.arange(lo[0], hi[0] + 1)[:, None]
        s = np.arange(lo[1], hi[1] + 1)[None, :]
        cases = {
            sympy.Eq(bs, 1): np.broadcast_to(b == 1, (32, 4096)),
            sympy.Ge(F.CeilDiv(S, 16) * bs, 64): -(-s // 16) * b >= 64,
            sympy.Lt(F.FloorDiv(S + 15, 16), 4): np.broadcast_to((s + 15) // 16 < 4, (32, 4096)),
        }
        c = cv.Cover(_At((bs, S)), lo, hi)
        for g, want in cases.items():
            got = np.broadcast_to(c._solve(g, (str(g),)), c.shape)
            self.assertTrue((got == want).all(), (g, np.argwhere(got != want)[:5].tolist()))
        self.assertEqual((c.stats["sums"], c.stats["boxes"], c.stats["slices"]), (0, 1, 0))

    # The guards a Qwen3-8B prefill step traced with on SGLang (quack's GEMM under the CuTe route, symbolic CuTe
    # integer ops and bit lengths), in the printed forms the runs logged: a ceil is -((-x)//d), the tile counts are
    # against 148 SMs, and a CuTe if/else picks the raster group.
    _tiles = -F.FloorDiv(-T, 256)
    _group = F.Where(sympy.Eq(F.Where(sympy.Lt(8 * F.FloorDiv(7 + _tiles, 8), 24), 0, 1), 0), _tiles, 24)

    @staticmethod
    def _py_group(t):
        tiles = -(-t // 256)
        return tiles if 8 * ((7 + tiles) // 8) < 24 else 24

    PREFILL = [
        (sympy.Ne(T, 1), lambda t: t != 1),
        (sympy.Gt(T, 8), lambda t: t > 8),
        (sympy.Le(T, 64), lambda t: t <= 64),
        (sympy.Eq(F.Mod(T, 16), 0), lambda t: t % 16 == 0),
        (sympy.Gt(4096 * T, 4096), lambda t: 4096 * t > 4096),
        (sympy.Ne(-F.FloorDiv(-T, 256), 24), lambda t: -(-t // 256) != 24),
        (sympy.Gt(-64 * F.FloorDiv(T, -16), 148), lambda t: -64 * (t // -16) > 148),
        (sympy.Le(-64 * F.FloorDiv(T, -32), 148), lambda t: -64 * (t // -32) <= 148),
        (sympy.Ge(F.FloorDiv(_group, F.Min(8, _group)), 2), lambda t: TestCoverSolver._py_group(t) // min(8, TestCoverSolver._py_group(t)) >= 2),
        (sympy.Lt(F.PowByNatural(2, F.Max(0, BitLength(F.Min(8, _group) - 1))), 8), lambda t: 2 ** max(0, (min(8, TestCoverSolver._py_group(t)) - 1).bit_length()) < 8),
    ]

    def test_prefill_guards(self):
        lo, hi = 1, 4096
        for g, f in self.PREFILL:
            want = [bool(f(t)) for t in range(lo, hi + 1)]
            for first in (0, 16384):
                got, _, _, fallbacks = row(g, lo, hi, first)
                self.assertEqual(got.tolist(), want, (g, first))
                self.assertEqual([x for x in fallbacks if x[1] != "vector"], [], (g, first))
        # a tile count that is 0 past T = 2048 (the TGV GEMM's split): False there, by the solver's per-T route
        got, _, _, fallbacks = row(sympy.Eq(F.Mod(4096, F.FloorDiv(2048, T)), 0), lo, hi, 16384)
        self.assertEqual(got.tolist(), [t <= 2048 and 4096 % (2048 // t) == 0 for t in range(lo, hi + 1)])

    def test_mixed_guards(self):
        # a mixed step: nd decode tokens and T prefill tokens; the GEMM guards are on nd + T, trtllm's
        # BS_BLOCK = next_power_of_2(nd + 1) on nd's bit length
        lo, hi = (1, 1), (64, 2048)
        nd = np.arange(lo[0], hi[0] + 1)[:, None]
        t = np.arange(lo[1], hi[1] + 1)[None, :]
        bits = np.vectorize(lambda n: int(n).bit_length())(nd)
        cases = {
            sympy.Gt(T + ND, 8): nd + t > 8,
            sympy.Le(T + ND, 64): nd + t <= 64,
            sympy.Eq(F.Mod(T + ND, 16), 0): (nd + t) % 16 == 0,
            sympy.Ne(-F.FloorDiv(-T - ND, 256), 24): -(-(nd + t) // 256) != 24,
            sympy.Gt(-64 * F.FloorDiv(T + ND, -16), 148): -64 * ((nd + t) // -16) > 148,
            sympy.Le(-96 * F.FloorDiv(T + ND, -32), 148): -96 * ((nd + t) // -32) <= 148,
            sympy.Eq(BitLength(ND), 3): np.broadcast_to(bits == 3, (64, 2048)),
            sympy.Le(F.PowByNatural(2, BitLength(ND)), 16): np.broadcast_to(2**bits <= 16, (64, 2048)),
        }
        c = cv.Cover(_At((ND, T)), lo, hi)
        for g, want in cases.items():
            got = np.broadcast_to(c._solve(g, (str(g),)), c.shape)
            self.assertTrue((got == want).all(), (g, np.argwhere(got != want)[:5].tolist()))
        self.assertEqual([x for x in c.fallbacks if x[1] != "vector"], [])
        self.assertEqual((c.stats["sums"], c.stats["slices"]), (6, 0))


    def test_decode_guards(self):
        # the guards test_cover_decode's step traced with: the page table's width ceil(S / 64) (substituted from
        # ArgsAt's sizes) and the reduction config inductor picks from bs and 64 * ceil(S / 64)
        bs, S = cv.axis("bs"), cv.axis("S")
        r = 64 * F.CeilDiv(S, 64)

        def p2(x, cap):
            return F.Where(sympy.Lt(x, cap), F.PowByNatural(2, sympy.Max(0, BitLength(x) - 1)), cap)

        def py_p2(x, cap):
            return 2 ** max(0, x.bit_length() - 1) if x < cap else cap

        rblock = sympy.Min(F.FloorDiv(128, sympy.Min(32, p2(16 * bs, 128))), p2(r, 128))
        d = F.Where(sympy.Ge(r, sympy.Min(256, 16 * rblock)), rblock, 1)

        def py_d(b, s):
            rs = 64 * -(-s // 64)
            rb = min(128 // min(32, py_p2(16 * b, 128)), py_p2(rs, 128))
            return rs, (rb if rs >= min(256, 16 * rb) else 1)

        cases = [
            (sympy.Eq(F.CeilDiv(S, 64), 1), lambda b, s: -(-s // 64) == 1),
            (sympy.Le(r, 2048), lambda b, s: 64 * -(-s // 64) <= 2048),
            (sympy.Eq(F.FloorDiv(r + 1023, 1024), 3), lambda b, s: (64 * -(-s // 64) + 1023) // 1024 == 3),
            (sympy.Ne(bs, 1), lambda b, s: b != 1),
            (sympy.Ne(d, 1), lambda b, s: py_d(b, s)[1] != 1),
            (sympy.Ge(F.FloorDiv(r + d - 1, d), 256), lambda b, s: (lambda rs, dd: (rs + dd - 1) // dd >= 256)(*py_d(b, s))),
        ]
        lo, hi = (1, 1), (32, 4096)
        c = cv.Cover(_At((bs, S)), lo, hi)
        for g, fn in cases:
            want = np.array([[fn(b, s) for s in range(lo[1], hi[1] + 1)] for b in range(lo[0], hi[0] + 1)])
            got = np.broadcast_to(c._solve(g, (str(g),)), c.shape)
            self.assertTrue((got == want).all(), (g, np.argwhere(got != want)[:5].tolist()))
        self.assertEqual([f for f in c.fallbacks if f[-2] != "vector"], [])
        self.assertEqual((c.stats["boxes"], c.stats["slices"]), (2, 0))

    def test_vector_int64(self):
        # over int64 where every value provably fits, else (an overflow, a float, a function with no int64 form)
        # over Python ints; both raise on a division by zero
        ts = np.arange(1, 1 << 14)
        for g in [*(self.FALLBACKS[k] for k in (0, 2, 3, 4, 6)), sympy.Gt(F.PowByNatural(2, BitLength(T)) * T, 5000), sympy.Eq(F.Mod(T * T, 7), 3)]:
            got = cv._vector(g, ts.astype(np.int64))
            self.assertEqual(got, cv._vector(g, ts.astype(object)), g)
        for g, err in [(sympy.Lt(F.FloorToInt(log2(T)), 7), cv._Fallback), (sympy.Ge(F32Div(T, 7), 1120403456), cv._Fallback),
                       (sympy.Eq(F.Mod(T ** 5 * 2 ** 40, 7), 3), OverflowError), (sympy.Lt(T / 3, 7), cv._Fallback)]:
            with self.assertRaises(err):
                cv._vector(g, ts.astype(np.int64))
            self.assertEqual(cv._exact(g, ts.astype), cv._vector(g, ts.astype(object)), g)
        with self.assertRaises(ZeroDivisionError):
            cv._exact(sympy.Gt(F.FloorDiv(1000, T - 5), 3), ts.astype)


instantiate_parametrized_tests(TestCover)

def setUpModule():
    from torch.cuda import _host_trace_hint_audit

    _host_trace_hint_audit.enable_for_tests()


if __name__ == "__main__":
    run_tests()
