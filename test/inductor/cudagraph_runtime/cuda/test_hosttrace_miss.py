# Owner(s): ["module: inductor"]
"""The miss policy of direct_hosttrace.HostTraceReplay: a call that misses every
variant's guards is traced at its own inputs and served by the new variant, natively;
the ordinary host runs only when that trace declines (reported once per reason, the
declined class remembered); a cap on the variants of one entry raises instead of
falling back. The host executes exactly once per call, a miss included: the runtime
team's mutation, RNG and cold-GEMM cases (COORDINATION.md 2026-09-19 15:26 / 16:23 UTC)."""

import contextlib
import gc
import os
import sys
import warnings
import weakref
from unittest import mock

import sympy

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_hosttrace_decode import D, decode_step, DH, DTYPE, H, LMAX, V


K = N = 4096


def prefill(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def scaled(x, alpha):
    return x * alpha


def add(x, y):
    return x + y


def linear(x, w, b):
    return F.linear(x, w, b)


def sometimes_declines(x):
    y = x * 2
    if x.shape[0] > 4:
        # a host read of a traced tensor: the trace of this class declines by name
        y = y + y.sum().item()
    return y


def update_in_place(x):
    x.copy_(x + 1)
    if x.shape[0] >= 8:
        return x * 3
    return x * 2


def fresh_rows(rows, width=16):
    return torch.full((rows, width), 10.0, device="cuda")[:, ::2]


def draw(x, alpha):
    p = 0.1 if x.shape[0] < 8 else 0.2
    first = torch.native_dropout(x, p, True)[0]
    second = torch.native_dropout(first[:, :1024].contiguous(), alpha, True)[0]
    return first, second


def draw_then_decline(x):
    value = torch.native_dropout(x, 0.2, True)[0]
    if x.shape[0] >= 8:
        return value + value.sum().item()
    return value * 2


def mutate_then_decline(x):
    x.add_(1.0)
    return x + x.sum().item()


def two_views_out(x):
    t1 = x * 2.0
    return (t1 + 1.0)[:, ::2], t1.sum(0)


def update_then_addmm(x, weight, bias):
    x.copy_(x + 1)
    if x.shape[0] >= 8:
        return torch.addmm(bias, x + 0, weight)
    return x * 2


class TestHostTraceMiss(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.skipTest("CUDA with flash attention required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        torch.manual_seed(0)
        self._flash = torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        )
        self._flash.__enter__()
        self.addCleanup(self._flash.__exit__, None, None, None)

    def _replay(self, fn, args, **kw):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args, **kw)
        self.addCleanup(replay.close)
        return replay

    def _call(self, replay, fn, args):
        """Bitwise against eager; returns (missed the variants, served natively)."""
        want = fn(*args)
        misses, ordinary = replay.misses, replay.ordinary
        got = replay(*args)
        want = (want,) if isinstance(want, torch.Tensor) else tuple(want)
        got = (got,) if isinstance(got, torch.Tensor) else tuple(got)
        for g, w in zip(got, want):
            self.assertEqual(g, w, atol=0, rtol=0)
        return replay.misses != misses, replay.ordinary == ordinary

    @staticmethod
    def _qkv(L, B=4, heads=12, dh=64):
        return tuple(
            torch.randn(B, heads, L, dh, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )

    def test_prefill_other_residue_class_builds_a_second_variant(self):
        replay = self._replay(prefill, self._qkv(128))
        # the traced class: flash rounds seqlen_q to a multiple of 128 and the tape
        # recorded the branch it took, so 256 serves at once
        self.assertEqual(self._call(replay, prefill, self._qkv(256)), (False, True))
        # 100 misses that branch: traced, built and served natively on this call
        self.assertEqual(self._call(replay, prefill, self._qkv(100)), (True, True))
        self.assertEqual(len(replay.variants), 2)
        _, why, outcome = replay.miss_log[-1]
        self.assertIn("Mod(", why)
        self.assertIn("128", why)
        self.assertEqual(outcome, "variant 2")
        # both classes hit from now on, and the residue class is the only boundary
        for L in (128, 100, 128, 100, 90, 200, 384, 129):
            self.assertEqual(
                self._call(replay, prefill, self._qkv(L)), (False, True), L
            )
        self.assertEqual((replay.misses, replay.ordinary), (1, 0))

    def _decode_setup(self):
        table = torch.randn(V, D, device="cuda", dtype=DTYPE)
        ln_w = 1 + 0.1 * torch.randn(D, device="cuda", dtype=DTYPE)
        ln_b = 0.1 * torch.randn(D, device="cuda", dtype=DTYPE)
        wq, wk, wv, w_out = (
            torch.randn(D, device="cuda", dtype=DTYPE) for _ in range(4)
        )

        def caches(B):
            return (
                torch.randn(B, H, LMAX, DH, device="cuda", dtype=DTYPE),
                torch.randn(B, H, LMAX, DH, device="cuda", dtype=DTYPE),
            )

        def args(B, L, k, v):
            ids = torch.randint(0, V, (B,), dtype=torch.int64).pin_memory()
            return (ids, table, ln_w, ln_b, wq, wk, wv, w_out, k[:, :, :L], v[:, :, :L])

        return caches, args

    def test_decode_chain_batches_build_their_variants(self):
        caches, args = self._decode_setup()
        replay = self._replay(decode_step, args(4, 16, *caches(4)))
        missed = {}
        for round_ in range(2):
            for B in (4, 2, 8, 1):
                e = caches(B)
                r = tuple(c.clone() for c in e)
                ids = torch.randint(0, V, (B,), dtype=torch.int64).pin_memory()
                want = decode_step(ids, *args(B, 20, *e)[1:])
                misses = replay.misses
                got = replay(ids, *args(B, 20, *r)[1:])
                for g, w in zip(got, want):
                    self.assertEqual(g, w, atol=0, rtol=0)
                for ec, rc in zip(e, r):
                    self.assertEqual(ec[:, :, :20], rc[:, :, :20], atol=0, rtol=0)
                if round_ == 0:
                    missed[B] = replay.misses != misses
                else:
                    self.assertEqual(replay.misses, misses, B)
        # every call was served natively; batch 1 is its own class (the sibling's
        # coalescing), the classes of 2 and 8 are whatever the tape recorded
        self.assertEqual(replay.ordinary, 0)
        self.assertTrue(missed[1])
        self.assertEqual(replay.misses, len(replay.variants) - 1)
        self.assertEqual(replay.misses, sum(missed.values()))

    def test_a_declined_class_runs_eager_and_is_reported_once(self):
        replay = self._replay(sometimes_declines, (torch.randn(4, 8, device="cuda"),))
        x = torch.randn(8, 8, device="cuda")
        with self.assertWarnsRegex(RuntimeWarning, "declined"):
            self.assertEqual(
                self._call(replay, sometimes_declines, (x,)), (True, False)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(
                self._call(replay, sometimes_declines, (x,)), (True, False)
            )
        self.assertEqual(len(replay.declines), 1)
        self.assertEqual(len(replay.variants), 1)
        self.assertEqual(replay.miss_log[-1][2], "declined class")
        # the traced class still serves
        y = torch.randn(4, 8, device="cuda")
        self.assertEqual(self._call(replay, sometimes_declines, (y,)), (False, True))
        self.assertEqual((replay.misses, replay.ordinary), (2, 2))

    def test_a_declined_class_is_not_traced_again(self):
        from torch.cuda import _host_trace

        replay = self._replay(sometimes_declines, (torch.randn(4, 8, device="cuda"),))
        x = torch.randn(8, 8, device="cuda")
        with (
            mock.patch.object(_host_trace, "trace", wraps=_host_trace.trace) as traced,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            for _ in range(3):
                self.assertEqual(
                    self._call(replay, sometimes_declines, (x,)), (True, False)
                )
        self.assertEqual(traced.call_count, 1)
        self.assertEqual(sum("declined" in str(w.message) for w in caught), 1)
        self.assertEqual(
            [entry[2] for entry in replay.miss_log],
            ["declined", "declined class", "declined class"],
        )
        # another exact class of the declining branch is traced once more, then
        # remembered: the guards so far of this function's partial trace read the
        # allocation of `x * 2`, which the inputs alone cannot decide, so the entry
        # remembers the decline by its exact inputs (commit 1's rule)
        y = torch.randn(9, 8, device="cuda")
        with mock.patch.object(_host_trace, "trace", wraps=_host_trace.trace) as traced:
            for _ in range(2):
                self.assertEqual(
                    self._call(replay, sometimes_declines, (y,)), (True, False)
                )
        self.assertEqual(traced.call_count, 1)
        self.assertEqual(len(replay.declined_classes), 2)
        self.assertEqual(len(replay.declines), 1)
        # a size outside the declining branch: the traced variant serves it
        z = torch.randn(3, 8, device="cuda")
        self.assertEqual(self._call(replay, sometimes_declines, (z,)), (False, True))
        self.assertEqual(len(replay.variants), 1)

    def test_each_call_mutates_its_input_once(self):
        # the runtime team's exactly-once regression: a miss traced with a warm-up ran the
        # host once more than the call; the constructor's warm-up is the entry's one
        # execution at construction
        for warm_up in (True, False):
            example = fresh_rows(4)
            replay = self._replay(update_in_place, (example,), warm_up=warm_up)
            torch.cuda.synchronize()
            self.assertEqual(example[0, 0].item(), 11.0 if warm_up else 10.0)
            # an initial hit, a new-shape miss, a new-shape hit, an old-shape hit
            for rows in (4, 12, 12, 4):
                actual, expected = fresh_rows(rows), fresh_rows(rows)
                want = update_in_place(expected)
                got = replay(actual)
                self.assertEqual(actual, expected, atol=0, rtol=0)
                self.assertEqual(got, want, atol=0, rtol=0)
            self.assertEqual(
                (replay.misses, replay.ordinary, len(replay.variants)), (1, 0, 2)
            )

    def test_guard_and_constant_misses_draw_once(self):
        # the outputs and the complete generator state match one eager call per call
        # across a shape miss, a changed-constant family and the hits of every class
        for warm_up in (True, False):
            replay = self._replay(
                draw, (torch.ones(4, 4096, device="cuda"), 0.2), warm_up=warm_up
            )
            phases = [(4, 0.2), (12, 0.2), (12, 0.2), (12, 0.3), (12, 0.3), (4, 0.2)]
            inputs = [
                (torch.ones(rows, 4096, device="cuda"), alpha) for rows, alpha in phases
            ]
            torch.cuda.manual_seed(1234)
            expected = []
            for x, alpha in inputs:
                out = draw(x, alpha)
                torch.cuda.synchronize()
                expected.append((out, torch.cuda.get_rng_state()))
            torch.cuda.manual_seed(1234)
            missed = []
            for (x, alpha), (want, want_state) in zip(inputs, expected):
                misses = replay.misses
                got = replay(x, alpha)
                torch.cuda.synchronize()
                for g, w in zip(got, want):
                    self.assertEqual(g, w, atol=0, rtol=0)
                self.assertEqual(torch.cuda.get_rng_state(), want_state)
                missed.append(replay.misses - misses)
            self.assertEqual(missed, [0, 1, 0, 1, 0, 0], warm_up)
            self.assertEqual(replay.ordinary, 0)

    def test_a_declined_miss_draws_once(self):
        replay = self._replay(draw_then_decline, (torch.ones(4, 1024, device="cuda"),))
        x = torch.ones(12, 1024, device="cuda")
        torch.cuda.manual_seed(5678)
        want = draw_then_decline(x)
        want_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(5678)
        with self.assertWarnsRegex(RuntimeWarning, "declined"):
            got = replay(x)
        torch.cuda.synchronize()
        self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(torch.cuda.get_rng_state(), want_state)
        self.assertEqual(
            (replay.misses, replay.ordinary, len(replay.variants)), (1, 1, 1)
        )

    def test_a_new_addmm_branch_is_served_before_any_eager_reference(self):
        # the miss harvests and prepares the new branch's region on the call itself: the
        # replay runs before eager ever ran that branch (the runtime team's cold-GEMM case)
        weight = torch.ones(64, 32, device="cuda")
        bias = torch.full((32,), 2.0, device="cuda")
        replay = self._replay(update_then_addmm, (fresh_rows(4, 128), weight, bias))
        self.assertEqual(len(replay.tape.regions), 0)
        for rows, missed, variants in ((12, 1, 2), (12, 0, 2), (4, 0, 2)):
            actual, expected = fresh_rows(rows, 128), fresh_rows(rows, 128)
            misses, ordinary = replay.misses, replay.ordinary
            got = replay(actual, weight, bias)
            torch.cuda.synchronize()
            want = update_then_addmm(expected, weight, bias)
            self.assertEqual(actual, expected, atol=0, rtol=0)
            self.assertEqual(got, want, atol=0, rtol=0)
            self.assertEqual(replay.misses - misses, missed, rows)
            self.assertEqual(replay.ordinary, ordinary)
            self.assertEqual(len(replay.variants), variants)
        self.assertEqual([len(v.tape.regions) for v in replay.variants], [0, 1])

    def test_a_miss_on_another_stream_is_refused_by_name_before_a_trace(self):
        # the family's dispatch is bound to the stream of its preparation: a hit on
        # another stream is refused by the runtime, and a miss there is refused the
        # same way before a trace, a lowering or a build (round 8's F2: it used to
        # build a variant the dispatch then rejected, on every call)
        from torch.cuda import _host_trace

        replay = self._replay(add, (torch.randn(64, 256, device="cuda"),) * 2)
        big = (torch.randn(4, 64, 256, device="cuda"),) * 2
        side = torch.cuda.Stream()
        with (
            mock.patch.object(_host_trace, "trace", wraps=_host_trace.trace) as traced,
            torch.cuda.stream(side),
        ):
            for _ in range(3):
                with self.assertRaisesRegex(RuntimeError, "bound to stream"):
                    replay(*big)
            # a hit class there: the runtime's refusal
            with self.assertRaisesRegex(RuntimeError, "bound device and stream"):
                replay(*(torch.randn(64, 256, device="cuda"),) * 2)
        self.assertEqual(traced.call_count, 0)
        self.assertEqual((replay.traces, len(replay.variants)), (1, 1))
        self.assertEqual(replay.misses, 0)
        # the same class on the bound stream is built once and served
        self.assertEqual(self._call(replay, add, big), (True, True))
        self.assertEqual((replay.traces, len(replay.variants)), (2, 2))

    def test_grad_mode_is_the_callers_contract_a_replay_serves_without_history(self):
        # the input contract is the caller's (E32, O48): the entry reads no grad mode
        # and scans no box for requires_grad. Under grad mode a call with an input
        # requiring grad is served natively like any other and its outputs carry no
        # autograd history (eager's would); an in-place op on a leaf requiring grad
        # is served where eager raises. The runners call the entry under no_grad
        # (round 8's F3, kept as the record of the documented behaviour)
        example = (torch.randn(64, 256, device="cuda"), torch.randn(64, 256, device="cuda"))
        replay = self._replay(add, example)
        x = torch.randn(64, 256, device="cuda", requires_grad=True)
        y = torch.randn(64, 256, device="cuda")
        self.assertTrue(torch.is_grad_enabled())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = replay(x, y)
        self.assertIsNone(out.grad_fn)
        self.assertFalse(out.requires_grad)
        self.assertEqual(out, (x + y).detach(), atol=0, rtol=0)
        self.assertEqual((replay.calls, replay.ordinary, replay.misses), (1, 0, 0))
        with torch.no_grad():
            self.assertEqual(replay(x, y), x + y, atol=0, rtol=0)
        self.assertEqual((replay.calls, replay.ordinary, replay.misses), (2, 0, 0))
        inplace = self._replay(lambda a, b: a.add_(b), example)
        leaf = torch.randn(64, 256, device="cuda", requires_grad=True)
        expected = leaf.detach() + y
        with self.assertRaisesRegex(RuntimeError, "leaf Variable that requires grad"):
            leaf.add_(y)
        inplace(leaf, y)
        self.assertEqual(leaf.detach(), expected, atol=0, rtol=0)
        self.assertEqual((inplace.calls, inplace.ordinary, inplace.misses), (1, 0, 0))

    def test_a_trace_declining_after_the_constructor_warm_up_keeps_its_result(self):
        # the constructor's warm-up is the entry's first call (E24): when the trace
        # declines after it ran, the entry is usable in the declined state and the
        # warm-up's return value is on `warm_up_outputs` (round 8's N3: the caller
        # used to get an exception after a mutation); with warm_up=False nothing ran
        # and the decline raises as before
        from torch.cuda import _host_trace

        x = torch.zeros(8, 8, device="cuda")
        want = mutate_then_decline(torch.zeros(8, 8, device="cuda"))
        with self.assertWarnsRegex(RuntimeWarning, "declined"):
            replay = self._replay(mutate_then_decline, (x,))
        self.assertEqual(x, torch.ones(8, 8, device="cuda"), atol=0, rtol=0)
        self.assertEqual(replay.warm_up_outputs, want, atol=0, rtol=0)
        self.assertEqual((len(replay.variants), len(replay.declines)), (0, 1))
        self.assertEqual(replay.traces, 1)
        self.assertTrue(replay.single)
        y = torch.zeros(8, 8, device="cuda")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = replay(y)
        self.assertEqual(out, want, atol=0, rtol=0)
        self.assertEqual(y, torch.ones(8, 8, device="cuda"), atol=0, rtol=0)
        self.assertEqual((replay.ordinary, replay.traces), (1, 1))
        why, outcome = replay.miss_log[-1][1:]
        self.assertEqual(why, "the entry has no variant yet")
        self.assertEqual(outcome, "declined class")
        replay.close()
        self.assertIsNone(replay.warm_up_outputs)
        z = torch.zeros(8, 8, device="cuda")
        with self.assertRaises(_host_trace.Declined):
            self.module.HostTraceReplay(mutate_then_decline, (z,), warm_up=False)
        self.assertEqual(z, torch.zeros(8, 8, device="cuda"), atol=0, rtol=0)

    def test_the_miss_log_names_a_guard_over_an_opaque_rebind(self):
        # the reduction's block rebinds (last_pow2, the divider shift) feed guards the
        # log used to leave symbolic and then mislabel as the arena's (round 8's N1):
        # the rebinds are re-run at the call's inputs and the guard is named
        replay = self._replay(two_views_out, (torch.randn(64, 512, device="cuda"),))
        big = (torch.randn(1024, 512, device="cuda"),)
        self.assertEqual(self._call(replay, two_views_out, big), (True, True))
        why = replay.miss_log[-1][1]
        self.assertTrue(why.startswith("guard failed: "), why)
        self.assertIn("Min(", why)
        self.assertNotIn("arena", why)

    def test_the_variant_cap_raises_instead_of_falling_back(self):
        replay = self._replay(
            scaled, (torch.randn(4, 8, device="cuda"), 2.0), max_variants=2
        )
        # another rank is another class: the second variant
        x3 = torch.randn(2, 4, 8, device="cuda")
        self.assertEqual(self._call(replay, scaled, (x3, 2.0)), (True, True))
        self.assertEqual(len(replay.variants), 2)
        with self.assertRaisesRegex(RuntimeError, "max_variants=2"):
            replay(torch.randn(4, 8, device="cuda", dtype=torch.float16), 2.0)
        # the existing classes still serve
        self.assertEqual(
            self._call(replay, scaled, (torch.randn(4, 8, device="cuda"), 2.0)),
            (False, True),
        )
        self.assertEqual(self._call(replay, scaled, (x3, 2.0)), (False, True))

    def test_another_non_tensor_argument_is_its_own_family(self):
        replay = self._replay(scaled, (torch.randn(4, 8, device="cuda"), 2.0))
        # the argument contract (a constant of the tape) is part of the miss
        self.assertEqual(
            self._call(replay, scaled, (torch.randn(4, 8, device="cuda"), 3.0)),
            (True, True),
        )
        self.assertEqual(len(replay.variants), 2)
        for alpha in (2.0, 3.0, 2.0, 3.0):
            x = torch.randn(4, 8, device="cuda")
            self.assertEqual(
                self._call(replay, scaled, (x, alpha)), (False, True), alpha
            )
        self.assertEqual((replay.misses, replay.ordinary), (1, 0))

    def test_a_new_region_key_is_harvested_and_served_on_the_same_call(self):
        w = torch.randn(N, K, device="cuda", dtype=DTYPE) / K**0.5
        b = torch.randn(N, device="cuda", dtype=DTYPE)

        def x(m):
            return torch.randn(m, K, device="cuda", dtype=DTYPE)

        replay = self._replay(linear, (x(4), w, b))
        # M = 8 is a new cuBLAS key of the same tape: harvested at the miss and served by
        # the one variant on the same call
        self.assertEqual(self._call(replay, linear, (x(8), w, b)), (True, True))
        self.assertEqual(len(replay.variants), 1)
        self.assertEqual(replay.miss_log[-1][2], "variant 1 after the harvest")
        self.assertEqual(self._call(replay, linear, (x(8), w, b)), (False, True))
        self.assertEqual(replay.ordinary, 0)

    # The runtime team's standalone-decline properties (their
    # cpu/test_standalone_host_trace_declines.py drives C's internals with mocks: the
    # dispatcher slot, `_miss`, the lock); stated here against the entry's surface.

    def _mutating(self):
        def fn(x):
            x.add_(1)
            if x.shape[0] >= 8:
                return x * 3
            return x * 2

        return fn

    def test_a_recognized_decline_runs_the_ordinary_host_once_per_call(self):
        from torch._inductor.runtime.cudagraph_launch_association import (
            UnsupportedCapture,
        )

        for stage, exception in (
            ("lower_tape", UnsupportedCapture("unsupported lowering")),
            (
                "lower_tape",
                self.module.HostTraceLoweringDeclined("unsupported lowering"),
            ),
            ("prepare_hosttrace", UnsupportedCapture("unsupported preparation")),
        ):
            with self.subTest(stage=stage, exception=type(exception).__name__):
                fn = self._mutating()
                replay = self._replay(fn, (torch.zeros(4, 8, device="cuda"),))
                x = torch.zeros(8, 8, device="cuda")
                with (
                    mock.patch.object(
                        self.module, stage, side_effect=exception
                    ) as patched,
                    warnings.catch_warnings(record=True) as caught,
                ):
                    warnings.simplefilter("always")
                    for call in (1, 2):
                        out = replay(x)
                        # the host ran exactly once for this call: the miss trace, or the
                        # remembered class's ordinary call
                        self.assertEqual(
                            x, torch.full((8, 8), float(call), device="cuda")
                        )
                        self.assertEqual(out, x * 3)
                        self.assertEqual(
                            (replay.calls, replay.misses, replay.ordinary),
                            (call, call, call),
                        )
                        self.assertFalse(replay.lock.locked())
                self.assertEqual(patched.call_count, 1)
                self.assertEqual(sum("declined" in str(w.message) for w in caught), 1)
                self.assertEqual(len(replay.variants), 1)
                self.assertEqual(
                    [e[2] for e in replay.miss_log], ["declined", "declined class"]
                )
                # the traced class still serves natively (the host mutates: no _call)
                y = torch.zeros(4, 8, device="cuda")
                misses, ordinary = replay.misses, replay.ordinary
                out = replay(y)
                self.assertEqual(y, torch.ones(4, 8, device="cuda"))
                self.assertEqual(out, torch.full((4, 8), 2.0, device="cuda"))
                self.assertEqual((replay.misses, replay.ordinary), (misses, ordinary))

    def test_an_unexpected_error_propagates_without_retry(self):
        for stage in ("lower_tape", "prepare_hosttrace"):
            with self.subTest(stage=stage):
                fn = self._mutating()
                replay = self._replay(fn, (torch.zeros(4, 8, device="cuda"),))
                x = torch.zeros(8, 8, device="cuda")
                with mock.patch.object(
                    self.module,
                    stage,
                    side_effect=RuntimeError("injected preparation error"),
                ) as patched:
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        replay(x)
                    self.assertEqual(patched.call_count, 1)
                # the miss trace captures without executing the host, and the error
                # came before any variant: nothing ran, nothing was retried or admitted
                self.assertEqual(x, torch.zeros(8, 8, device="cuda"))
                self.assertEqual(len(replay.variants), 1)
                self.assertFalse(replay.lock.locked())
                y = torch.zeros(4, 8, device="cuda")
                misses = replay.misses
                self.assertEqual(replay(y), torch.full((4, 8), 2.0, device="cuda"))
                self.assertEqual(replay.misses, misses)

    def test_a_publication_error_closes_only_the_new_entry(self):
        replay = self._replay(scaled, (torch.randn(4, 8, device="cuda"), 2.0))
        real_prepare = self.module.prepare_hosttrace
        prepared = []

        def preparing(lowered, args, **kw):
            entry = real_prepare(lowered, args, **kw)
            prepared.append(mock.Mock(wraps=entry))
            return prepared[-1]

        # a new family (another argument contract) publishes a new dispatch: its failure
        # closes the prepared entry it was made for and nothing else
        with (
            mock.patch.object(self.module, "prepare_hosttrace", side_effect=preparing),
            mock.patch.object(
                torch._C,
                "_cuda_make_boxed_dispatch",
                side_effect=ValueError("invalid publication"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "invalid publication"):
                replay(torch.randn(4, 8, device="cuda"), 3.0)
        self.assertEqual(len(prepared), 1)
        prepared[0].close.assert_called_once_with()
        self.assertEqual(len(replay.variants), 1)
        self.assertFalse(replay.lock.locked())
        self.assertEqual(
            self._call(replay, scaled, (torch.randn(4, 8, device="cuda"), 2.0)),
            (False, True),
        )

    def test_a_decline_allows_a_later_successful_preparation(self):
        from torch._inductor.runtime.cudagraph_launch_association import (
            UnsupportedCapture,
        )

        replay = self._replay(scaled, (torch.randn(4, 8, device="cuda"), 2.0))
        real = self.module.lower_tape
        x3 = torch.randn(2, 4, 8, device="cuda")
        # the first class to miss declines at lowering: remembered, served ordinary
        with mock.patch.object(
            self.module,
            "lower_tape",
            side_effect=[UnsupportedCapture("first trace declines")],
        ):
            with self.assertWarnsRegex(RuntimeWarning, "declined"):
                self.assertEqual(self._call(replay, scaled, (x3, 2.0)), (True, False))
        self.assertEqual(self._call(replay, scaled, (x3, 2.0)), (True, False))
        self.assertEqual(len(replay.variants), 1)
        # a later miss of another class prepares and serves
        with mock.patch.object(self.module, "lower_tape", side_effect=real):
            half = torch.randn(4, 8, device="cuda", dtype=torch.float16)
            self.assertEqual(self._call(replay, scaled, (half, 2.0)), (True, True))
        self.assertEqual(len(replay.variants), 2)
        self.assertFalse(replay.lock.locked())

    def test_the_boxed_entry_declines_a_pageable_input_after_the_ordinary_result(self):
        def ordinary(source):
            source.add_(1)
            return source

        replay = self.module.HostTraceReplay(ordinary)
        self.addCleanup(replay.close)
        source = torch.zeros(3)
        box = [source]
        with self.assertWarnsRegex(RuntimeWarning, "declined"):
            (output,) = replay(box)
        self.assertIs(output, source)
        self.assertEqual(source, torch.ones(3))
        self.assertEqual(box, [])
        self.assertEqual(replay.variants, [])
        self.assertEqual((replay.calls, replay.misses, replay.ordinary), (1, 1, 1))
        self.assertFalse(replay.lock.locked())
        replay.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            replay([source])

    # the runtime team's R22 lifetime cases on this line (cleanup_r22/lifetime): a miss
    # that failed retains nothing of the call, and a prepared variant retains the tape's
    # records, not the call it was traced at

    def test_a_failed_miss_releases_the_call_it_traced(self):
        for stage in ("trace", "prepare_hosttrace"):
            with self.subTest(stage=stage):
                mutating = self._mutating()

                def host(x):
                    out = mutating(x)
                    if stage == "trace" and x.shape[0] >= 8:
                        raise RuntimeError("injected host error")
                    return out

                replay = self._replay(host, (torch.zeros(4, 8, device="cuda"),))
                x = torch.zeros(8, 8, device="cuda")
                ref = weakref.ref(x)

                def failing_preparation(*args, **kwargs):
                    raise RuntimeError("injected preparation error")

                # a plain replacement: a Mock would record the call's arguments
                patch = (
                    mock.patch.object(
                        self.module, "prepare_hosttrace", new=failing_preparation
                    )
                    if stage == "prepare_hosttrace"
                    else contextlib.nullcontext()
                )
                with patch:
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        replay(x)
                # the miss trace executed nothing and the error came before any variant
                self.assertEqual(x, torch.zeros(8, 8, device="cuda"))
                del x
                gc.collect()
                self.assertIsNone(ref())
                self.assertFalse(replay.lock.locked())
                self.assertFalse(replay.closed)
                self.assertEqual(len(replay.variants), 1)
                y = torch.zeros(4, 8, device="cuda")
                self.assertEqual(replay(y), torch.full((4, 8), 2.0, device="cuda"))
                self.assertEqual(y, torch.ones(4, 8, device="cuda"))

    def test_a_prepared_variant_retains_the_records_not_the_call(self):
        x = torch.randn(4, 8, device="cuda")
        ref = weakref.ref(x)
        replay = self._replay(scaled, (x, 2.0))
        self.assertIsNone(replay.tape.args)
        del x
        gc.collect()
        self.assertIsNone(ref())
        y = torch.randn(4, 8, device="cuda")
        self.assertEqual(self._call(replay, scaled, (y, 2.0)), (False, True))
        # a second variant, from a miss, releases its call the same way
        z = torch.randn(4, 8, device="cuda", dtype=torch.float16)
        ref = weakref.ref(z)
        self.assertEqual(self._call(replay, scaled, (z, 2.0)), (True, True))
        self.assertEqual(len(replay.variants), 2)
        self.assertIsNone(replay.variants[1].tape.args)
        del z
        gc.collect()
        self.assertIsNone(ref())

    # guards with connectives and ordered domains (the round-7 overlap fix and the
    # domain guards, both cascade 12): the predicate renders Not / And over two inputs'
    # pointers and evaluates a domain guard before the division it guards

    def _traced_with_guards(self, fn, args, guards_of):
        """The entry of `fn` at `args` whose first tape carries `guards_of(tape)`
        ahead of its recorded guards."""
        from torch.cuda import _host_trace

        real = _host_trace.trace

        def tracing(fn, args, **kw):
            tape = real(fn, args, **kw)
            tape.guards = [*guards_of(tape), *tape.guards]
            return tape

        with mock.patch.object(_host_trace, "trace", side_effect=tracing):
            return self._replay(fn, args)

    def test_an_overlap_guard_over_two_inputs_misses_on_overlapping_views(self):
        E = self.module._expr

        def span(rec):
            begin = E(rec.root.sym) + rec.root.itemsize * E(rec.offset)
            return begin, begin + rec.root.itemsize * E(rec.sizes[0])

        def overlap_guard(tape):
            (a0, a1), (b0, b1) = (span(rec) for rec in tape.inputs)
            # eager's aliasing question as the tape records it: not (a and b)
            return [sympy.Not(sympy.And(sympy.Lt(a0, b1), sympy.Lt(b0, a1)))]

        x, y = torch.arange(16.0, device="cuda"), torch.arange(16.0, device="cuda") + 1
        replay = self._traced_with_guards(add, (x, y), overlap_guard)
        self.assertEqual(self._call(replay, add, (x, y)), (False, True))
        # one storage, disjoint 32-byte aligned windows (the tape's alignment guards
        # hold): served; overlapping windows: the connective misses
        base = torch.arange(32.0, device="cuda")
        self.assertEqual(self._call(replay, add, (base[16:], base[:16])), (False, True))
        self.assertTrue(self._call(replay, add, (base[:16], base[8:24]))[0])
        self.assertIn("guard failed: ", replay.miss_log[-1][1])
        self.assertIn("&", replay.miss_log[-1][1])

    def test_a_payload_that_overflows_int64_misses_before_selection(self):
        # the plan's payload arithmetic (a kernel's numel field, an arena block's
        # rounded byte count) is int64: the lowering emits a signed-range obligation
        # per sum, product and power it lowers, evaluated checked in the predicate, so
        # a call whose payload would overflow misses before a variant is selected
        # (the native evaluation would raise after it). A real tape's predicate, fed
        # the facts of a box no tensor can carry (sizes of 2**32).
        import ctypes

        from torch._inductor.runtime._cudagraph import direct_hosttrace as dh

        x = torch.randn(8, 8, device="cuda")
        replay = self._replay(add, (x, x.clone()))
        lowered = replay.lowered
        bounds = [
            g
            for g in lowered.extra_guards
            if isinstance(g, (sympy.Le, sympy.Ge)) and abs(int(g.rhs)) >= 2**63 - 1
        ]
        self.assertTrue(any(g.has(sympy.Mul) for g in bounds), lowered.extra_guards)
        y = torch.randn(8, 8, device="cuda")
        self.assertEqual(replay(y, y), y + y, atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)
        box = lowered.box(
            (y, y), replay._hot.arena.tensor if replay._hot.arena else None
        )
        self.assertTrue(dh.check_predicate(lowered, box))
        values = [box[i].data_ptr() for i in lowered.pointer_indices]
        values.extend(box[i].storage_offset() for i in lowered.offset_indices)
        facts = list(lowered.facts)
        values.extend(dh._fact_value(f, box[f.index]) for f in facts)
        predicate = ctypes.CFUNCTYPE(
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        )(lowered.predicate_address)
        offset = len(lowered.pointer_indices) + len(lowered.offset_indices)
        for k, f in enumerate(facts):
            if f.kind == "size":
                values[offset + k] = 2**32  # a numel of 2**64: the product overflows
            elif f.kind == "stride" and f.dim == 0:
                values[offset + k] = 2**32
        bits = (ctypes.c_uint64 * len(values))(*(v % (2**64) for v in values))
        self.assertEqual(
            predicate(ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64)), None), 0
        )
        # the switch is what emits them (the A/B measurement's)
        with mock.patch.object(dh, "_PAYLOAD_BOUNDS", False):
            plain = dh.lower_tape(replay.tape, (y, y), arena=replay.arena_enabled)
        self.assertEqual(plain.facts, lowered.facts)
        self.assertEqual([g for g in plain.extra_guards if g in bounds], [])
        self.assertEqual(
            set(plain.extra_guards) | set(bounds), set(lowered.extra_guards)
        )

    def test_an_ordered_domain_guard_misses_before_the_division_it_guards(self):
        from torch.utils._sympy.functions import FloorDiv

        E = self.module._expr

        def domain_guards(tape):
            (rec,) = tape.inputs
            size, stride = E(rec.sizes[0]), E(rec.strides[0])
            # the producer's domain guard ahead of the relation that divides
            return [sympy.Ne(stride, 0), sympy.Eq(FloorDiv(size, stride), 8)]

        replay = self._traced_with_guards(
            scaled, (torch.arange(8.0, device="cuda"), 2.0), domain_guards
        )
        self.assertEqual(
            self._call(replay, scaled, (torch.arange(8.0, device="cuda"), 2.0)),
            (False, True),
        )
        expanded = torch.ones(1, device="cuda").expand(8)
        self.assertTrue(self._call(replay, scaled, (expanded, 2.0))[0])
        self.assertTrue(replay.miss_log[-1][1].startswith("guard failed: Ne("))


if __name__ == "__main__":
    run_tests()
