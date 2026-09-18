# Owner(s): ["module: cuda"]

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
    # the data already sits in the symmetric buffer (the caller wrote it
    # before the call): the collective reduces a view of the buffer shaped
    # like x; x only carries the shape here
    inp = buf.narrow(0, 0, x.numel()).view(x.shape)
    return symm_ops.one_shot_all_reduce(inp, "sum", group_name)


def all_reduce_copy(buf, x, group_name):
    # the kernel copies the local input into the buffer itself; the local
    # input is a value computed in the traced function (x + x), as a model's
    # partial product is: a rank-local input tensor in that slot would put
    # its own alignment into the collective's guards (see
    # test_a_rank_dependent_local_input_declines)
    inp = buf.narrow(0, 0, x.numel()).view(x.shape)
    return symm_ops.one_shot_all_reduce_copy(inp, x + x, "sum", group_name)


def all_reduce_copy_of_input(buf, x, group_name):
    # the local input is the rank-local input tensor itself
    inp = buf.narrow(0, 0, x.numel()).view(x.shape)
    return symm_ops.one_shot_all_reduce_copy(inp, x, "sum", group_name)


def tp_layer(buf, x, w, gamma, group_name):
    # a tensor-parallel layer stand-in: a column-parallel projection (an
    # elementwise stand-in for the GEMM), the all-reduce of the partial, the
    # residual and a layer norm
    partial = x * w
    inp = buf.narrow(0, 0, partial.numel()).view(partial.shape)
    full = symm_ops.one_shot_all_reduce_copy(inp, partial, "sum", group_name)
    # (a trailing reduction would add the reduce config's own named misses
    # across batch sizes; the layer norm output is the chain's result here)
    return F.layer_norm(full + x, (x.shape[-1],), gamma, None)


def buffer_inside(x, group_name):
    buf = symm_mem.empty(x.numel(), dtype=x.dtype, device=x.device)
    symm_mem.rendezvous(buf, group=group_name)
    return all_reduce_copy(buf, x, group_name)


@unittest.skipIf(
    not PLATFORM_SUPPORTS_SYMM_MEM, "symmetric memory is not supported on this platform"
)
@requires_cuda_p2p_access()
class HostTraceSymmTest(MultiProcContinuousTest):
    """Every rank traces the same program with the same shapes, so guards,
    misses and declines are identical on all ranks and every collective below
    runs in lockstep (the replays, the eager references and the misses are
    the same sequence on every rank).

    The one-shot kernel's barrier spins until the peer's kernel arrives, with
    no timeout of its own, so on a GPU shared with other processes a test is
    bounded only by the harness's per-test timeout; a rank that times out
    leaves its peer spinning, and the base class would then join that peer
    without a timeout (a hang in place of a failure): tearDownClass gives
    the workers the class timeout to exit and terminates the rest by rank."""

    @classmethod
    def tearDownClass(cls):
        if cls.__dict__.get("_processes_spawned", False):
            for task_queue in cls.task_queues:
                task_queue.put(None)
            stuck = []
            for rank, p in enumerate(cls.processes):
                p.join(timeout=cls.timeout.total_seconds())
                if p.is_alive():
                    stuck.append(rank)
                    p.terminate()
                    p.join(timeout=30)
                    if p.is_alive():
                        p.kill()
            if stuck:
                # the base class joins the terminated workers below
                super().tearDownClass()
                raise AssertionError(
                    f"rank(s) {stuck} did not exit within {cls.timeout.total_seconds():.0f} s "
                    "after the class finished and were terminated: a collective whose "
                    "peer never arrived (see the failing test above, or the box load)"
                )
        super().tearDownClass()

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _init(self, dtype=torch.bfloat16):
        torch.cuda.set_device(self.device)
        self.group_name = dist.group.WORLD.group_name
        buf = symm_mem.empty(B_MAX * D, dtype=dtype, device=self.device)
        symm_mem.rendezvous(buf, group=self.group_name)
        return buf

    def _x(self, B, seed, dtype=torch.bfloat16):
        # different data per rank, the same on every replay of a seed
        g = torch.Generator(device=self.device)
        g.manual_seed(1000 * seed + self.rank)
        return torch.randn(B, D, generator=g, device=self.device).to(dtype)

    def _handle(self, buf):
        return symm_mem.rendezvous(buf, group=self.group_name)

    def _pads(self, buf):
        h = self._handle(buf)
        n = symm_mem.get_signal_pad_size() // 4
        return h.get_signal_pad(self.rank, [n], torch.uint32)

    def _fill(self, buf, x):
        # the caller's write into the symmetric buffer, outside the trace
        buf[: x.numel()].copy_(x.reshape(-1))
        return buf

    def _check(self, got, want):
        self.assertEqual(got.dtype, want.dtype)
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(torch.equal(got, want))

    # -- the collective alone -------------------------------------------------

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_trace_and_replay_across_sizes(self):
        buf = self._init()
        for fn in (all_reduce_in_buffer, all_reduce_copy):
            x = self._x(4, 1)
            self._fill(buf, x)
            args = (buf, x, self.group_name)
            tape = ht.trace(fn, args)
            names = [launch["kernel"] for launch in tape.launches]
            self.assertTrue(
                any("one_shot_all_reduce_kernel" in n for n in names), names
            )
            variant = ht.build(tape, fn, args)
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
                    self.assertIsNone(variant.try_replay((buf, xb, self.group_name)))
                    missed.append(B)
                    continue
                self._fill(buf, xb)
                got = variant.try_replay((buf, xb, self.group_name))
                want = fn(buf, xb, self.group_name)
                if got is None:
                    missed.append(B)
                    continue
                served.append(B)
                self._check(got[0], want)
            self.assertEqual(served, [4, 2, 3, 5, 8, 16], (served, missed))
            # B=1 takes the single-block arm (a different launch config), a
            # size above the buffer fails the narrow's bound
            self.assertEqual(missed, [1, B_MAX + 1])
            # a batch-1 tape serves batch 1
            x1 = self._x(1, 10)
            self._fill(buf, x1)
            tape1 = ht.trace(fn, (buf, x1, self.group_name))
            v1 = ht.build(tape1, fn, (buf, x1, self.group_name))
            x1b = self._x(1, 11)
            self._fill(buf, x1b)
            self._check(
                v1.replay((buf, x1b, self.group_name))[0],
                fn(buf, x1b, self.group_name),
            )

    def _assert_pads_reset(self, buf):
        # this rank's pads after its own collectives completed. The peer may
        # be ahead: its next collective's first barrier signals this rank's
        # pad as soon as the peer launches it, so both ranks read their pads
        # before either goes on (the barrier); without it the check failed on
        # the lagging rank under box load and left the peer spinning
        torch.cuda.synchronize()
        pads = self._pads(buf)
        self.assertEqual((pads != 0).nonzero().flatten().tolist(), [])
        dist.barrier()

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_back_to_back_replays_leave_the_pads_reset(self):
        # the signal pads are a compare-and-swap toggle: every slot is back
        # at 0 when a collective returns, so replays need no per-call state
        buf = self._init()
        args = (buf, self._x(4, 1), self.group_name)
        variant = ht.build(ht.trace(all_reduce_copy, args), all_reduce_copy, args)
        self._assert_pads_reset(buf)
        outs = []
        xs = [self._x((4, 8, 2, 16, 3)[i % 5], 20 + i) for i in range(100)]
        for xb in xs:
            outs.append(variant.replay((buf, xb, self.group_name))[0])
        self._assert_pads_reset(buf)
        for xb, got in zip(xs[-5:], outs[-5:]):
            self._check(got, all_reduce_copy(buf, xb, self.group_name))

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_interleaved_with_eager_collectives(self):
        buf = self._init()
        args = (buf, self._x(4, 1), self.group_name)
        variant = ht.build(ht.trace(all_reduce_copy, args), all_reduce_copy, args)
        for seed in range(5):
            xb = self._x(4, 30 + seed)
            got = variant.replay((buf, xb, self.group_name))[0]
            ref = all_reduce_copy(buf, xb, self.group_name)
            nccl = xb + xb  # the copy form reduces x + x
            dist.all_reduce(nccl)
            self._check(got, ref)
            self.assertTrue(
                torch.allclose(got.float(), nccl.float(), atol=2e-2, rtol=2e-2)
            )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_two_buffers_serve_through_their_own_tables(self):
        # the handle is looked up from the buffer's address at every replay,
        # as eager does: a second rendezvoused buffer of the same shape is
        # served through its own peer table and signal pads
        buf = self._init()
        args = (buf, self._x(4, 1), self.group_name)
        tape = ht.trace(all_reduce_copy, args)
        lookups = [o["fn"] for o in tape.opaque]
        self.assertEqual(
            sorted(set(lookups)),
            [
                "symm_buffer_ptrs",
                "symm_rank",
                "symm_signal_pad_ptrs",
                "symm_world_size",
            ],
        )
        variant = ht.build(tape, all_reduce_copy, args)
        buf2 = symm_mem.empty(B_MAX * D, dtype=torch.bfloat16, device=self.device)
        symm_mem.rendezvous(buf2, group=self.group_name)
        self.assertNotEqual(buf.data_ptr(), buf2.data_ptr())
        for i in range(6):
            b = (buf, buf2)[i % 2]
            xb = self._x(4, 40 + i)
            got = variant.replay((b, xb, self.group_name))[0]
            self._check(got, all_reduce_copy(b, xb, self.group_name))
        # the in-buffer form too: the data sits in each buffer
        args = (buf, self._x(4, 1), self.group_name)
        v2 = ht.build(ht.trace(all_reduce_in_buffer, args), all_reduce_in_buffer, args)
        for i in range(4):
            b = (buf2, buf)[i % 2]
            xb = self._x(4, 50 + i)
            self._fill(b, xb)
            got = v2.replay((b, xb, self.group_name))[0]
            self._fill(b, xb)
            self._check(got, all_reduce_in_buffer(b, xb, self.group_name))

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_buffer_without_rendezvous_raises_eagers_error(self):
        # the lookup runs before any launch on every rank; a plain CUDA
        # tensor in the buffer's slot fails the way eager fails, with no hang
        buf = self._init()
        args = (buf, self._x(4, 1), self.group_name)
        variant = ht.build(ht.trace(all_reduce_copy, args), all_reduce_copy, args)
        plain = torch.empty(B_MAX * D, dtype=torch.bfloat16, device=self.device)
        xb = self._x(4, 2)
        t0 = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "empty_strided_p2p"):
            variant.replay((plain, xb, self.group_name))
        with self.assertRaisesRegex(RuntimeError, "empty_strided_p2p"):
            all_reduce_copy(plain, xb, self.group_name)
        self.assertLess(time.monotonic() - t0, 60.0)
        # the variant is intact: the next replay on the real buffer serves
        self._check(
            variant.replay((buf, xb, self.group_name))[0],
            all_reduce_copy(buf, xb, self.group_name),
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_never_rendezvoused_buffer_is_rendezvoused_on_demand(self):
        # eager's op rendezvous's a symmetric buffer on first use (a host-side
        # collective inside the allocator's lookup); the replay's lookup is
        # the same call, so it does the same, identically on every rank
        buf = self._init()
        args = (buf, self._x(4, 1), self.group_name)
        variant = ht.build(ht.trace(all_reduce_copy, args), all_reduce_copy, args)
        fresh = symm_mem.empty(B_MAX * D, dtype=torch.bfloat16, device=self.device)
        xb = self._x(4, 2)
        t0 = time.monotonic()
        got = variant.replay((fresh, xb, self.group_name))[0]
        self.assertLess(time.monotonic() - t0, 60.0)
        self._check(got, all_reduce_copy(fresh, xb, self.group_name))

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_different_world_size_misses(self):
        # the kernel is instantiated for the world size, so it is a guard: a
        # buffer rendezvoused on a single-rank group misses by name, and a
        # fresh trace on that group serves
        buf = self._init()
        args = (buf, self._x(4, 1), self.group_name)
        variant = ht.build(ht.trace(all_reduce_copy, args), all_reduce_copy, args)
        # new_group is collective: every rank creates every single-rank group
        # and uses its own
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
        self.assertIsNone(variant.try_replay((solo_buf, xb, solo.group_name)))
        solo_args = (solo_buf, xb, solo.group_name)
        v1 = ht.build(ht.trace(all_reduce_copy, solo_args), all_reduce_copy, solo_args)
        xc = self._x(4, 3)
        self._check(
            v1.replay((solo_buf, xc, solo.group_name))[0],
            all_reduce_copy(solo_buf, xc, solo.group_name),
        )
        # the two-rank tape still serves its own group
        self._check(
            variant.replay((buf, xb, self.group_name))[0],
            all_reduce_copy(buf, xb, self.group_name),
        )

    # -- declines, identical on every rank, before any collective launch ----

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_a_rank_dependent_local_input_declines(self):
        # serve or miss must be a function of rank-shared state only (the
        # collective's guards may not reference a rank-local buffer's offset
        # or base): a rank-local input tensor as the collective's local input
        # declines by name on every rank, before any launch, with no exchange
        buf = self._init()
        full = self._x(8, 1)
        view = full[self.rank : self.rank + 4]  # a rank-dependent offset
        with self.assertRaisesRegex(ht.Declined, "rank-local fact"):
            ht.trace(
                all_reduce_copy_of_input, (buf, view, self.group_name), warm_up=False
            )
        # the same for an input whose offset happens to agree on every rank:
        # the fact is rank-local either way
        with self.assertRaisesRegex(ht.Declined, "rank-local fact"):
            ht.trace(
                all_reduce_copy_of_input,
                (buf, self._x(4, 2), self.group_name),
                warm_up=False,
            )
        # a value computed in the traced function is fine (all_reduce_copy)
        args = (buf, self._x(4, 3), self.group_name)
        variant = ht.build(ht.trace(all_reduce_copy, args), all_reduce_copy, args)
        xb = self._x(4, 4)
        self._check(
            variant.replay((buf, xb, self.group_name))[0],
            all_reduce_copy(buf, xb, self.group_name),
        )

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_declines_are_rank_symmetric_and_leave_the_trace_clean(self):
        buf = self._init()
        # the host supports float32 and bfloat16 only: a float16 buffer
        # declines in the host's checks before its launch, on every rank
        buf16 = symm_mem.empty(B_MAX * D, dtype=torch.float16, device=self.device)
        symm_mem.rendezvous(buf16, group=self.group_name)
        with self.assertRaises(ht.Declined):
            ht.trace(
                all_reduce_copy,
                (buf16, self._x(4, 1, torch.float16), self.group_name),
                warm_up=False,
            )
        # a buffer that is not an input of the traced function (captured by a
        # closure) has no traced root: declined by name
        closure_buf = buf

        def closure(x, group_name):
            # a concrete view of the captured buffer reaches the collective
            inp = closure_buf[: 4 * D].view(4, D)
            return symm_ops.one_shot_all_reduce_copy(inp, x, "sum", group_name)

        def closure_symbolic(x, group_name):
            # a symbolic view of the captured buffer cannot be taken at all
            return all_reduce_copy(closure_buf, x, group_name)

        with self.assertRaisesRegex(ht.Declined, "not an input of the traced function"):
            ht.trace(closure, (self._x(4, 1), self.group_name), warm_up=False)
        with self.assertRaisesRegex(ht.Declined, "does not own"):
            ht.trace(closure_symbolic, (self._x(4, 1), self.group_name), warm_up=False)
        # a buffer allocated inside the traced function: symm_mem.empty uses
        # torch.cuda.use_mem_pool, which under a stream capture hands the
        # MemPool object to CUDAGraph._retain_pool where a pool id is expected
        # (an upstream TypeError, torch/cuda/memory.py, not a host-tracing
        # decline); either way nothing launches
        with self.assertRaises((ht.Declined, TypeError)):
            ht.trace(buffer_inside, (self._x(4, 1), self.group_name), warm_up=False)
        # the process is clean afterwards: a fresh trace serves
        args = (buf, self._x(4, 1), self.group_name)
        variant = ht.build(ht.trace(all_reduce_copy, args), all_reduce_copy, args)
        xb = self._x(4, 2)
        self._check(
            variant.replay((buf, xb, self.group_name))[0],
            all_reduce_copy(buf, xb, self.group_name),
        )

    # -- a tensor-parallel layer stand-in ------------------------------------

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_tp_layer_chain(self):
        buf = self._init()
        torch.manual_seed(7)  # the same weights on every rank
        w = torch.randn(D, device=self.device).to(torch.bfloat16)
        gamma = torch.randn(D, device=self.device).to(torch.bfloat16)
        args = (buf, self._x(4, 1), w, gamma, self.group_name)
        tape = ht.trace(tp_layer, args)
        variant = ht.build(tape, tp_layer, args)
        for B, seed in ((4, 2), (2, 3), (3, 4), (8, 5), (16, 6)):
            xb = self._x(B, seed)
            got = variant.replay((buf, xb, w, gamma, self.group_name))[0]
            self._check(got, tp_layer(buf, xb, w, gamma, self.group_name))
        # per-step cost, eager vs the interim replay (the native lowering is
        # the other team's evaluator); barrier so both ranks start together
        xb = self._x(4, 7)
        for _ in range(5):
            variant.replay((buf, xb, w, gamma, self.group_name))
            tp_layer(buf, xb, w, gamma, self.group_name)
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

        eager_us = timed(lambda: tp_layer(buf, xb, w, gamma, self.group_name))
        replay_us = timed(lambda: variant.replay((buf, xb, w, gamma, self.group_name)))
        print(
            f"[rank {self.rank}] tp_layer B=4 D={D}: eager {eager_us:.0f} us, replay {replay_us:.0f} us per step",
            flush=True,
        )


if __name__ == "__main__":
    run_tests()
