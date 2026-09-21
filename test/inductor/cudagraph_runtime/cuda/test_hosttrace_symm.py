# Owner(s): ["module: inductor"]
"""The symmetric-memory one-shot all-reduce (commit 11) served natively: the handle's
peer-pointer table, signal pads and rank are per-call lookups from the buffer's base
address (E19), lowered as pointer fields whose values the numeric plan's call op computes
per call (retirement stage B, cell 6). Two ranks; every rank traces the same program at
the same shapes, so misses, declines and variants are identical on all ranks. Each test
prepares the interim replay beside the native entry from one tape (host_trace_oracle)."""

import statistics
import time
import unittest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    PLATFORM_SUPPORTS_SYMM_MEM,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    skipIfRocm,
)


if torch.cuda.is_available():
    from torch.cuda import _host_trace as ht

D = 4096
B_MAX = 32
symm_ops = torch.ops.symm_mem


def all_reduce_in_buffer(buf, x, group_name):
    inp = buf.narrow(0, 0, x.numel()).view(x.shape)
    return symm_ops.one_shot_all_reduce(inp, "sum", group_name)


def all_reduce_copy(buf, x, group_name):
    inp = buf.narrow(0, 0, x.numel()).view(x.shape)
    return symm_ops.one_shot_all_reduce_copy(inp, x + x, "sum", group_name)


def all_reduce_copy_of_input(buf, x, group_name):
    inp = buf.narrow(0, 0, x.numel()).view(x.shape)
    return symm_ops.one_shot_all_reduce_copy(inp, x, "sum", group_name)


def tp_layer(buf, x, w, gamma, group_name):
    partial = x * w
    inp = buf.narrow(0, 0, partial.numel()).view(partial.shape)
    full = symm_ops.one_shot_all_reduce_copy(inp, partial, "sum", group_name)
    return F.layer_norm(full + x, (x.shape[-1],), gamma, None)


@unittest.skipIf(
    not PLATFORM_SUPPORTS_SYMM_MEM, "symmetric memory is not supported on this platform"
)
@requires_cuda_p2p_access()
class TestHostTraceSymm(MultiProcContinuousTest):
    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _init(self, dtype=torch.bfloat16):
        torch.cuda.set_device(self.device)
        self.group_name = dist.group.WORLD.group_name
        buf = symm_mem.empty(B_MAX * D, dtype=dtype, device=self.device)
        symm_mem.rendezvous(buf, group=self.group_name)
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        return buf

    def _x(self, B, seed, dtype=torch.bfloat16):
        g = torch.Generator(device=self.device)
        g.manual_seed(1000 * seed + self.rank)
        return torch.randn(B, D, generator=g, device=self.device).to(dtype)

    def _quiescent(self):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    def _pads(self, buf):
        h = symm_mem.rendezvous(buf, group=self.group_name)
        n = symm_mem.get_signal_pad_size() // 4
        return h.get_signal_pad(self.rank, [n], torch.uint32)

    def _pads_zero_at_rest(self, buf):
        # both ranks read their pads between two barriers: a rank that has passed
        # its read must not issue the next collective (which signals the peer's
        # pads) before the peer has read its own
        self._quiescent()
        zero = (self._pads(buf) == 0).all().item()
        self._quiescent()
        return zero

    def _fill(self, buf, x):
        buf[: x.numel()].copy_(x.reshape(-1))
        return buf

    def _check(self, got, want):
        self.assertEqual(got.dtype, want.dtype)
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(torch.equal(got, want))

    def _oracle(self, fn, args):
        from torch.testing._internal.host_trace_oracle import Oracle

        oracle = Oracle(fn, args)
        self.addCleanup(oracle.close)
        self.assertIsNone(oracle.refused, oracle.refused)
        return oracle

    def _replay(self, fn, args):
        replay = self.module.HostTraceReplay(fn, args)
        self.addCleanup(replay.close)
        return replay

    # -- the collective alone -------------------------------------------------

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_lowering_shape(self):
        # the handle's lookups: two pointer fields and the rank are plan call values
        # over the buffer's base address (a pointer value less the offset's bytes),
        # the world size an opaque guard of the predicate
        buf = self._init()
        replay = self._replay(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        lookups = sorted({o["fn"] for o in replay.tape.opaque})
        self.assertEqual(
            lookups,
            [
                "symm_buffer_ptrs",
                "symm_rank",
                "symm_signal_pad_ptrs",
                "symm_world_size",
            ],
        )
        calls = [
            f
            for call in replay.lowered.calls
            for f in call.fields
            if f.kind != "pointer" and f.source.expression.op == "pcall"
        ]
        self.assertGreaterEqual(len(calls), 3)
        self.assertTrue(any(f.kind == "i64" for f in calls))
        # the world size is the one opaque guard: the predicate calls its host
        # function by address on the buffer's base (the pointer slot less the
        # offset's bytes) and compares with the traced value
        guards = [o for o in replay.tape.opaque if o["kind"] == "guard"]
        self.assertEqual([o["fn"] for o in guards], ["symm_world_size"])
        self.assertEqual(int(guards[0]["expected"]), self.world_size)
        self.assertIn(
            f"uintptr_t({int(guards[0]['impl'])})", replay.lowered.guard_source
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_trace_and_replay_across_sizes(self):
        buf = self._init()
        for fn in (all_reduce_in_buffer, all_reduce_copy):
            x = self._x(4, 1)
            self._fill(buf, x)
            oracle = self._oracle(fn, (buf, x, self.group_name))
            names = [launch["kernel"] for launch in oracle.tape.launches]
            self.assertTrue(
                any("one_shot_all_reduce_kernel" in n for n in names), names
            )
            served, missed = [], []
            for B, seed in (
                (4, 2),
                (2, 3),
                (3, 4),
                (5, 5),
                (8, 6),
                (16, 7),
                (1, 8),
                (B_MAX + 1, 9),
            ):
                xb = self._x(B, seed)
                if B > B_MAX:
                    oracle.expect_miss((buf, xb, self.group_name))
                    missed.append(B)
                    continue
                self._fill(buf, xb)
                got = oracle.variant.try_replay((buf, xb, self.group_name))
                if got is None:
                    missed.append(B)
                    continue
                served.append(B)
                self._check(got[0], fn(buf, xb, self.group_name))
            self.assertEqual(served, [4, 2, 3, 5, 8, 16], (served, missed))
            # B=1 takes the single-block arm (a different launch config), a
            # size above the buffer fails the narrow's bound
            self.assertEqual(missed, [1, B_MAX + 1])
            self.assertEqual(oracle.native.misses, 2)
            x1 = self._x(1, 10)
            self._fill(buf, x1)
            one = self._oracle(fn, (buf, x1, self.group_name))
            x1b = self._x(1, 11)
            self._fill(buf, x1b)
            one.check((buf, x1b, self.group_name))

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_back_to_back_replays_leave_the_pads_reset(self):
        # the signal pads are a compare-and-swap toggle: every slot is back at 0
        # when a collective returns, so replays need no per-call state
        buf = self._init()
        replay = self._replay(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        # a pad reads 0 only while no collective is in flight on either rank: the
        # native preparation is seconds of rank-local work (the predicate compiles),
        # so the ranks meet before each read, or the faster rank's next kernel has
        # already signaled the slower one's pad
        self.assertTrue(self._pads_zero_at_rest(buf))
        outs = []
        xs = [self._x((4, 8, 2, 16, 3)[i % 5], 20 + i) for i in range(100)]
        for xb in xs:
            outs.append(replay(buf, xb, self.group_name))
        self.assertTrue(self._pads_zero_at_rest(buf))
        for xb, got in zip(xs[-5:], outs[-5:]):
            self._check(got, all_reduce_copy(buf, xb, self.group_name))
        self.assertEqual(replay.misses, 0)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_interleaved_with_eager_collectives(self):
        buf = self._init()
        oracle = self._oracle(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        for seed in range(5):
            xb = self._x(4, 30 + seed)
            (got,) = oracle.check((buf, xb, self.group_name))
            nccl = xb + xb
            dist.all_reduce(nccl)
            self.assertTrue(
                torch.allclose(got.float(), nccl.float(), atol=2e-2, rtol=2e-2)
            )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_two_buffers_serve_through_their_own_tables(self):
        # the handle is looked up from the buffer's address at every call, as eager
        # does: a second rendezvoused buffer of the same shape is served through its
        # own peer table and signal pads, by the same variant
        buf = self._init()
        oracle = self._oracle(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        buf2 = symm_mem.empty(B_MAX * D, dtype=torch.bfloat16, device=self.device)
        symm_mem.rendezvous(buf2, group=self.group_name)
        self.assertNotEqual(buf.data_ptr(), buf2.data_ptr())
        for i in range(6):
            b = (buf, buf2)[i % 2]
            oracle.check((b, self._x(4, 40 + i), self.group_name))
        self.assertEqual((oracle.native.misses, len(oracle.native.variants)), (0, 1))
        args = (buf, self._x(4, 1), self.group_name)
        self._fill(buf, args[1])
        in_buffer = self._oracle(all_reduce_in_buffer, args)
        for i in range(4):
            b = (buf2, buf)[i % 2]
            xb = self._x(4, 50 + i)
            self._fill(b, xb)
            in_buffer.check((b, xb, self.group_name))

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_buffer_without_rendezvous_raises_eagers_error(self):
        # the lookup runs before any launch on every rank (in the predicate, then in
        # the plan); a plain CUDA tensor in the buffer's slot fails the way eager
        # fails, with no hang, and the entry is intact
        buf = self._init()
        replay = self._replay(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        plain = torch.empty(B_MAX * D, dtype=torch.bfloat16, device=self.device)
        xb = self._x(4, 2)
        t0 = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "empty_strided_p2p"):
            all_reduce_copy(plain, xb, self.group_name)
        with self.assertRaises((RuntimeError, ValueError)):
            replay(plain, xb, self.group_name)
        self.assertLess(time.monotonic() - t0, 60.0)
        self._check(
            replay(buf, xb, self.group_name), all_reduce_copy(buf, xb, self.group_name)
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_never_rendezvoused_buffer_is_rendezvoused_on_demand(self):
        # eager's op rendezvous's a symmetric buffer on first use (a host-side
        # collective inside the allocator's lookup); the native lookup is the same
        # call, so it does the same, identically on every rank
        buf = self._init()
        oracle = self._oracle(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        fresh = symm_mem.empty(B_MAX * D, dtype=torch.bfloat16, device=self.device)
        t0 = time.monotonic()
        oracle.check((fresh, self._x(4, 2), self.group_name))
        self.assertLess(time.monotonic() - t0, 60.0)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_different_world_size_misses(self):
        # the kernel is instantiated for the world size, so it is a guard: a buffer
        # rendezvoused on a single-rank group misses, and the entry's variant built
        # at that miss serves it (E24), the two-rank variant its own group
        buf = self._init()
        replay = self._replay(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        groups = [dist.new_group([r]) for r in range(self.world_size)]
        solo = groups[self.rank]
        try:
            solo_buf = symm_mem.empty(
                B_MAX * D, dtype=torch.bfloat16, device=self.device
            )
            symm_mem.rendezvous(solo_buf, group=solo.group_name)
            xb = self._x(4, 2)
            all_reduce_copy(solo_buf, xb, solo.group_name)
        except Exception as e:
            self.skipTest(
                f"a single-rank symmetric group is not constructible here: {e}"
            )
        # the group name is a constant of the contract: the solo call is its own
        # family (its tape carries the solo world size in its guard)
        self._check(
            replay(solo_buf, xb, solo.group_name),
            all_reduce_copy(solo_buf, xb, solo.group_name),
        )
        self.assertEqual((replay.misses, len(replay.variants)), (1, 2))
        solo_guard = next(
            o for o in replay.variants[1].tape.opaque if o["fn"] == "symm_world_size"
        )
        self.assertEqual(int(solo_guard["expected"]), 1)
        xc = self._x(4, 3)
        self._check(
            replay(solo_buf, xc, solo.group_name),
            all_reduce_copy(solo_buf, xc, solo.group_name),
        )
        self._check(
            replay(buf, xb, self.group_name), all_reduce_copy(buf, xb, self.group_name)
        )
        self.assertEqual(replay.misses, 1)

    # -- declines, identical on every rank, before any collective launch ----

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_rank_dependent_local_input_declines(self):
        # serve or miss must be a function of rank-shared state only: a rank-local
        # input tensor as the collective's local input declines at the trace, on
        # every rank, before any launch (the entry raises at construction)
        buf = self._init()
        full = self._x(8, 1)
        view = full[self.rank : self.rank + 4]
        with self.assertRaisesRegex(ht.Declined, "rank-local fact"):
            self.module.HostTraceReplay(
                all_reduce_copy_of_input, (buf, view, self.group_name), warm_up=False
            )
        oracle = self._oracle(all_reduce_copy, (buf, self._x(4, 3), self.group_name))
        oracle.check((buf, self._x(4, 4), self.group_name))

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_declines_are_rank_symmetric_and_leave_the_trace_clean(self):
        buf = self._init()
        buf16 = symm_mem.empty(B_MAX * D, dtype=torch.float16, device=self.device)
        symm_mem.rendezvous(buf16, group=self.group_name)
        with self.assertRaises(ht.Declined):
            self.module.HostTraceReplay(
                all_reduce_copy,
                (buf16, self._x(4, 1, torch.float16), self.group_name),
                warm_up=False,
            )
        closure_buf = buf

        def closure(x, group_name):
            inp = closure_buf[: 4 * D].view(4, D)
            return symm_ops.one_shot_all_reduce_copy(inp, x, "sum", group_name)

        with self.assertRaisesRegex(ht.Declined, "not an input of the traced function"):
            self.module.HostTraceReplay(
                closure, (self._x(4, 1), self.group_name), warm_up=False
            )
        oracle = self._oracle(all_reduce_copy, (buf, self._x(4, 1), self.group_name))
        oracle.check((buf, self._x(4, 2), self.group_name))

    # -- a tensor-parallel layer stand-in ------------------------------------

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_tp_layer_chain(self):
        buf = self._init()
        torch.manual_seed(7)
        w = torch.randn(D, device=self.device).to(torch.bfloat16)
        gamma = torch.randn(D, device=self.device).to(torch.bfloat16)
        args = (buf, self._x(4, 1), w, gamma, self.group_name)
        oracle = self._oracle(tp_layer, args)
        for B, seed in ((4, 2), (2, 3), (3, 4), (8, 5), (16, 6)):
            oracle.check((buf, self._x(B, seed), w, gamma, self.group_name))
        self.assertEqual(oracle.native.misses, 0)
        native, interim = oracle.native, oracle.interim
        xb = self._x(4, 7)
        call = (buf, xb, w, gamma, self.group_name)
        for _ in range(5):
            native(*call)
            interim.replay(call)
            tp_layer(*call)
        torch.cuda.synchronize()
        dist.barrier()

        def timed(fn):
            t = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) / 20 * 1e6)
            return statistics.median(t)

        eager_us = timed(lambda: tp_layer(*call))
        native_us = timed(lambda: native(*call))
        interim_us = timed(lambda: interim.replay(call))
        print(
            f"[rank {self.rank}] tp_layer B=4 D={D}: eager {eager_us:.0f} us, native {native_us:.0f} us, interim {interim_us:.0f} us per step",
            flush=True,
        )


if __name__ == "__main__":
    run_tests()
