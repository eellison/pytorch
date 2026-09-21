# Owner(s): ["module: inductor"]
"""The planned arena of the native adapter (hosttrace_arena): the tape's temporaries in
one boxed arena input, the outputs as the runtime's buffers."""

import gc
import sys

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


D = 512


def _scale_then_norm(x, w, ln_w, ln_b):
    # every temporary scales with the rows of x: h, h * w, silu(...)
    h = F.layer_norm(x, (D,), ln_w, ln_b)
    return F.silu(h * w) + h


def _three_temporaries(x, y, z):
    # t1 lives across the call; t2 (y * 2) and t3 (z * 3) never overlap (each dies in
    # the in-place add that consumes it), so the plan may put t3 under t1 where t2
    # sat, a decision the sizes of y and z make
    t1 = x * 2.0
    y.add_(y * 2.0)
    z.add_(z * 3.0)
    return t1 + 1.0


class TestHostTraceArena(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        torch.manual_seed(0)

    def _replay(self, fn, args, **kw):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args, **kw)
        self.addCleanup(replay.close)
        return replay

    def _allocations_during(self, fn):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = fn()
        return torch.cuda.memory_stats()["allocation.all.allocated"] - before, out

    # ---- the decode chain (commit 6's composition) as the tape with reuse

    def _chain(self):
        if not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.skipTest("flash attention required")
        import test_hosttrace_decode as dt

        flash = torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        )
        flash.__enter__()
        self.addCleanup(flash.__exit__, None, None, None)
        dtype = dt.DTYPE
        weights = (
            torch.randn(dt.V, dt.D, device="cuda", dtype=dtype),
            1 + 0.1 * torch.randn(dt.D, device="cuda", dtype=dtype),
            0.1 * torch.randn(dt.D, device="cuda", dtype=dtype),
            *(torch.randn(dt.D, device="cuda", dtype=dtype) for _ in range(4)),
        )

        def caches(B):
            return tuple(
                torch.randn(B, dt.H, dt.LMAX, dt.DH, device="cuda", dtype=dtype)
                for _ in range(2)
            )

        def args(B, L, c):
            ids = torch.randint(0, dt.V, (B,), dtype=torch.int64).pin_memory()
            return (ids, *weights, c[0][:, :, :L], c[1][:, :, :L])

        return dt.decode_step, caches, args

    def test_temporaries_share_the_arena_and_outputs_stay_buffers(self):
        step, caches, args = self._chain()
        replay = self._replay(step, args(4, 16, caches(4)))
        lowered = replay.lowered
        plan = lowered.arena
        self.assertIsNotNone(plan)
        # the two outputs are the runtime's; every other allocation a launch, a memset
        # or a copy touches (the H2D destination of the ids gather among them) is a
        # block of the arena, boxed as the last input
        self.assertEqual(len(lowered.allocations), 2)
        self.assertGreaterEqual(len(plan.blocks), 10)
        self.assertEqual(lowered.input_names[-1], "arena")
        self.assertEqual(len(lowered.input_names), len(replay.tape.inputs) + 1)
        stats = replay.arena_stats()
        self.assertEqual(stats["families"][0]["capacity"] % 512, 0)
        self.assertGreaterEqual(
            stats["families"][0]["capacity"], stats["variants"][0]["bytes_at_hints"]
        )
        e_caches = caches(4)
        r_caches = tuple(c.clone() for c in e_caches)
        for L in range(16, 25):
            a = args(4, L, e_caches)
            want = step(*a)
            a = (a[0], *a[1:8], r_caches[0][:, :, :L], r_caches[1][:, :, :L])
            n, got = self._allocations_during(lambda: replay(*a))
            self.assertEqual(n, 2)  # the buffers; no allocator call for a temporary
            self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(e_caches, r_caches, atol=0, rtol=0)
        self.assertEqual((replay.misses, replay.arena_grows), (0, 0))

    def test_nonoverlap_assertion_checks_every_served_call(self):
        step, caches, args = self._chain()
        replay = self._replay(step, args(4, 16, caches(4)), arena_check=True)
        c = caches(4)
        for L in (16, 20, 24):
            a = args(4, L, c)
            self.assertEqual(replay(*a), step(*a), atol=0, rtol=0)
        # the check is real: a plan that puts two live blocks at one offset fails it
        plan = replay.lowered.arena
        blocks = list(plan.blocks.values())
        a, b = next(
            (a, b)
            for i, a in enumerate(blocks)
            for b in blocks[i + 1 :]
            if a.first <= b.last and b.first <= a.last
        )
        offset = b.offset
        b.offset = a.offset
        try:
            with self.assertRaisesRegex(AssertionError, "overlap while both live"):
                replay._check_arena(replay._hot, args(4, 24, c))
        finally:
            b.offset = offset
        replay._check_arena(replay._hot, args(4, 24, c))

    def test_outputs_held_across_calls_stay_valid(self):
        step, caches, args = self._chain()
        replay = self._replay(step, args(4, 16, caches(4)))
        e_caches, r_caches = caches(4), None
        r_caches = tuple(c.clone() for c in e_caches)
        held = []
        for L in range(16, 28):
            a = args(4, L, e_caches)
            want = tuple(t.clone() for t in step(*a))
            got = replay(a[0], *a[1:8], r_caches[0][:, :, :L], r_caches[1][:, :, :L])
            held.append((got, want))
        # twelve more calls reuse the arena and hand out new output buffers
        for L in range(28, 40):
            a = args(4, L, e_caches)
            replay(a[0], *a[1:8], r_caches[0][:, :, :L], r_caches[1][:, :, :L])
        torch.cuda.synchronize()
        for got, want in held:
            self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)

    # ---- growth and plan classes

    def test_the_arena_grows_with_the_call_and_replays_bitwise(self):
        w = torch.randn(D, device="cuda")
        ln_w = 1 + 0.1 * torch.randn(D, device="cuda")
        ln_b = 0.1 * torch.randn(D, device="cuda")

        def a(rows):
            return (torch.randn(rows, D, device="cuda"), w, ln_w, ln_b)

        replay = self._replay(_scale_then_norm, a(64))
        capacity = replay.arena_stats()["families"][0]["capacity"]
        self.assertEqual(len(replay.lowered.allocations), 1)
        grown = []
        for rows in (64, 128, 1024, 4096, 256, 8192):
            args = a(rows)
            self.assertEqual(replay(*args), _scale_then_norm(*args), atol=0, rtol=0)
            now = replay.arena_stats()["families"][0]["capacity"]
            grown.append(now > capacity)
            capacity = now
        # three temporaries of rows x D floats: 384 KiB at 64 rows, grown at 128 (768
        # KiB), 1024 (6 MiB) and 4096 rows, not at 256, again at 8192; a growth is a
        # rebind of the arena root, not a miss and not a new variant
        self.assertEqual(grown, [False, True, True, True, False, True])
        self.assertEqual(replay.arena_grows, 4)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.traces), (0, 1, 1)
        )
        self.assertTrue(
            replay.arena_stats()["families"][0]["capacity"] >= 3 * 4 * 8192 * D
        )

    def _entry_call(self, replay, args):
        """`entry(box)` over the tape's tensors alone (the runtime team's boxed surface),
        the box consumed; returns the outputs and the Python frames the call ran."""
        box, frames = list(args), []

        def profile(frame, event, arg):
            if event == "call":
                frames.append(frame.f_code)

        sys.setprofile(profile)
        try:
            outputs = replay.entry(box)
        finally:
            sys.setprofile(None)
        self.assertEqual(box, [])
        return outputs, frames

    def test_entry_takes_the_callers_box_and_binds_the_arena_itself(self):
        # the arena is the dispatch's own trailing input: a caller of `entry(box)` passes
        # the tape's tensors alone, a hit runs without a Python frame, a growth rebinds
        # the arena inside the dispatch (the next hit is native again), and a call the
        # family rejects runs the entry's policy from the dispatch's cold callback
        w = torch.randn(D, device="cuda")
        ln_w = 1 + 0.1 * torch.randn(D, device="cuda")
        ln_b = 0.1 * torch.randn(D, device="cuda")

        def a(rows):
            return (torch.randn(rows, D, device="cuda"), w, ln_w, ln_b)

        replay = self._replay(_scale_then_norm, a(64))
        self.assertIs(replay.entry, replay.dispatch)
        for rows, native in ((64, True), (1024, False), (256, True)):
            args = a(rows)
            outputs, frames = self._entry_call(replay, args)
            self.assertEqual(outputs[0], _scale_then_norm(*args), atol=0, rtol=0)
            self.assertEqual(frames == [], native)
        self.assertEqual((replay.arena_grows, replay.misses), (1, 0))
        self.assertEqual(len(replay.variants), 1)

        def b(nx, ny, nz):
            return tuple(torch.randn(n, device="cuda") for n in (nx, ny, nz))

        replay = self._replay(_three_temporaries, b(1024, 4096, 512))
        for args, native, misses in (
            (b(1024, 4096, 512), True, 0),
            # outside the plan class: the callback's policy builds the second variant
            (b(1024, 4096, 8192), False, 1),
            (b(1024, 4096, 8192), True, 0),  # the second variant serves natively
            (b(1024, 4096, 512), True, 0),
        ):
            e = tuple(t.clone() for t in args)
            want = _three_temporaries(*e)
            before = replay.misses
            outputs, frames = self._entry_call(replay, args)
            self.assertEqual(outputs[0], want, atol=0, rtol=0)
            self.assertEqual(args, e, atol=0, rtol=0)
            self.assertEqual(frames == [], native)
            self.assertEqual(replay.misses - before, misses)
        self.assertEqual((len(replay.variants), replay.traces), (2, 1))

    def test_a_plan_class_miss_builds_a_second_variant_from_the_same_tape(self):
        def a(nx, ny, nz):
            return tuple(torch.randn(n, device="cuda") for n in (nx, ny, nz))

        replay = self._replay(_three_temporaries, a(1024, 4096, 512))
        plan = replay.lowered.arena
        self.assertEqual(len(plan.blocks), 3)
        self.assertGreaterEqual(len(plan.class_guards), 1)

        def check(args, misses):
            e = tuple(t.clone() for t in args)
            want = _three_temporaries(*e)
            before = replay.misses
            got = replay(*args)
            self.assertEqual(got, want, atol=0, rtol=0)
            self.assertEqual(args, e, atol=0, rtol=0)
            self.assertEqual(replay.misses - before, misses)

        check(a(1024, 4096, 512), 0)
        check(a(2048, 8192, 1024), 0)  # the same size order: the class holds
        # z larger than y: the plan made at the hints is not valid here; the same
        # tape is built at these inputs (no trace) and serves the call
        check(a(1024, 4096, 8192), 1)
        self.assertEqual(
            (len(replay.variants), replay.traces, replay.ordinary), (2, 1, 0)
        )
        self.assertIn("arena plan class", " ".join(str(row) for row in replay.miss_log))
        check(a(1024, 4096, 8192), 0)
        check(a(1024, 4096, 512), 0)  # the first variant still serves its class

    def test_the_consumer_entry_keeps_per_call_buffers(self):
        # the runtime team's entry (prepare_host_trace) lowers without an arena: the
        # box is the tape's tensors and every allocation a buffer of the runtime
        from torch.cuda import _host_trace

        w = torch.randn(D, device="cuda")
        args = (torch.randn(64, D, device="cuda"), w, 1 + 0.1 * w, 0.1 * w)
        tape = _host_trace.trace(_scale_then_norm, args)
        prepared = self.module.prepare_host_trace(tape, args)
        self.addCleanup(prepared.close)
        self.assertIsNone(prepared.lowered.arena)
        self.assertEqual(len(prepared.lowered.input_names), len(tape.inputs))
        self.assertGreaterEqual(len(prepared.lowered.allocations), 3)


if __name__ == "__main__":
    run_tests()
