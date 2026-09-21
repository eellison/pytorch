# Owner(s): ["module: inductor"]
"""The output arena of the native adapter (hosttrace_arena.plan_outputs / OutputRing):
the escaping outputs of a call are typed views of one block; the family keeps a ring
of two, a block is reused only when the caller holds no view of it, and a call with
both held takes a block of its own that the caller's views own."""

import collections
import gc

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


D = 512


def _scale_then_norm(x, w, ln_w, ln_b):
    h = F.layer_norm(x, (D,), ln_w, ln_b)
    return F.silu(h * w) + h


def _four_outputs(x, w):
    # a whole allocation, a dtype-changing view of another, a sliced view of the
    # first and a third dtype: four views of one block
    h = x * w
    return (
        h + 1.0,
        torch.view_as_complex((h * 2.0).view(-1, D // 2, 2)),
        h[1:],
        (h * 3.0).to(torch.bfloat16),
    )


def _inplace_step(x, y, w):
    # the live state is returned as itself (7a): not an arena output
    x.add_(y * w)
    return x, (x * 2.0)


class TestHostTraceOutputArena(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        torch.manual_seed(0)

    def _replay(self, fn, args, **kw):
        gc.collect()
        kw.setdefault("output_arena", True)
        replay = self.module.HostTraceReplay(fn, args, **kw)
        self.addCleanup(replay.close)
        return replay

    def _allocations_during(self, fn):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = fn()
        return torch.cuda.memory_stats()["allocation.all.allocated"] - before, out

    def _stn_args(self):
        w = torch.randn(D, device="cuda")
        ln_w = 1 + 0.1 * torch.randn(D, device="cuda")
        ln_b = 0.1 * torch.randn(D, device="cuda")
        return lambda rows: (torch.randn(rows, D, device="cuda"), w, ln_w, ln_b)

    def _ring(self, replay):
        return replay._hot.outputs

    # ---- what an output is

    def test_outputs_are_views_of_the_block_and_no_allocator_call_per_call(self):
        a = self._stn_args()
        replay = self._replay(_scale_then_norm, a(64))
        lowered = replay.lowered
        self.assertEqual(len(lowered.allocations), 0)
        self.assertEqual(lowered.input_names[-2:], ("arena", "outputs"))
        (output,) = lowered.outputs
        self.assertEqual(type(output).__name__, "TensorViewOutput")
        self.assertEqual(output.source.index, len(replay.tape.inputs) + 1)
        self.assertEqual(output.dtype, torch.float32)
        for rows in (64, 64, 32, 64):
            args = a(rows)
            n, got = self._allocations_during(lambda: replay(*args))
            self.assertEqual(n, 0)
            self.assertEqual(got, _scale_then_norm(*args), atol=0, rtol=0)
            # the output is a view of the boxed block: same storage, offset zero, the
            # block's dtype is uint8 and the view's its own
            block = self._ring(replay).blocks[0][0]
            self.assertEqual(got.untyped_storage().data_ptr(), block.data_ptr())
            self.assertEqual(got.storage_offset(), 0)
            self.assertEqual(got.data_ptr() % 512, 0)
            self.assertEqual(got.dtype, torch.float32)
            del got
        self.assertEqual((replay.misses, replay.output_grows), (0, 0))
        stats = replay.arena_stats()["families"][0]
        self.assertEqual(stats["output_blocks"], 1)
        self.assertEqual(stats["output_held"], 0)

    def test_four_views_of_one_block_bitwise(self):
        w = torch.randn(D, device="cuda")
        a = lambda rows: (torch.randn(rows, D, device="cuda"), w)  # noqa: E731
        replay = self._replay(_four_outputs, a(64))
        plan = replay.lowered.output_arena
        # h, h + 1, h * 2 (viewed complex) and the bf16 result: four blocks of the
        # output plan in a chain, no class guard; h * 3 is a temporary of the arena
        self.assertEqual(len(plan.blocks), 4)
        self.assertEqual(plan.class_guards, ())
        self.assertEqual(len(replay.lowered.allocations), 0)
        for rows in (64, 128, 7, 1024):
            args = a(rows)
            want = _four_outputs(*args)
            n, got = self._allocations_during(lambda: replay(*args))
            for g, wt in zip(got, want):
                self.assertEqual(g.dtype, wt.dtype)
                self.assertEqual(g.stride(), wt.stride())
                self.assertEqual(g, wt, atol=0, rtol=0)
            storages = {g.untyped_storage().data_ptr() for g in got}
            self.assertEqual(len(storages), 1)
            self.assertEqual({g.data_ptr() % 8 for g in got}, {0})
            del got
        self.assertEqual(replay.misses, 0)

    def test_the_live_state_keeps_its_identity_and_is_not_an_arena_output(self):
        w = torch.randn(D, device="cuda")
        x, y = torch.randn(8, D, device="cuda"), torch.randn(8, D, device="cuda")
        replay = self._replay(_inplace_step, (x, y, w))
        kinds = [type(o).__name__ for o in replay.lowered.outputs]
        self.assertEqual(kinds, ["BorrowedInputOutput", "TensorViewOutput"])
        self.assertEqual(len(replay.lowered.output_arena.blocks), 1)
        e_x, r_x = torch.randn(8, D, device="cuda"), None
        r_x = e_x.clone()
        for _ in range(3):
            yy = torch.randn(8, D, device="cuda")
            want = _inplace_step(e_x, yy, w)
            got = replay(r_x, yy, w)
            self.assertIs(got[0], r_x)
            self.assertEqual(got[1], want[1], atol=0, rtol=0)
            self.assertEqual(r_x, e_x, atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)

    # ---- holding outputs

    def test_outputs_held_eight_deep_stay_bitwise_and_blocks_are_reused_after_release(
        self,
    ):
        a = self._stn_args()
        replay = self._replay(_scale_then_norm, a(64))
        ring = self._ring(replay)
        held = collections.deque()
        allocations = []
        for k in range(40):
            args = a(64)
            want = _scale_then_norm(*args)
            n, got = self._allocations_during(lambda: replay(*args))
            allocations.append(n)
            ptr = got.data_ptr()
            held.append((got, want, ptr))
            if len(held) > 8:
                old, want_old, ptr_old = held.popleft()
                # the held value never changed under the caller, nor its address
                self.assertEqual(old, want_old, atol=0, rtol=0)
                self.assertEqual(old.data_ptr(), ptr_old)
                del old
            del got
        torch.cuda.synchronize()
        for got, want, ptr in held:
            self.assertEqual(got, want, atol=0, rtol=0)
        del got
        # the ring keeps two blocks (the second taken at the first held call); with
        # eight generations held a call allocates one block of its own (one
        # allocation, not one per output) unless a ring block was released just
        # before it (the FIFO hands the ring's two back now and then)
        stats = replay.arena_stats()["families"][0]
        self.assertEqual(stats["output_blocks"], 2)
        self.assertEqual(stats["output_held"], 2)
        self.assertEqual(allocations[0], 0)
        self.assertLessEqual(max(allocations), 1)
        self.assertEqual(sum(allocations), ring.overflow + 1)
        self.assertGreaterEqual(ring.overflow, 30)
        self.assertEqual(replay.output_grows, 0)  # the floor never rose
        # released, the caller-owned blocks go back to the allocator at once
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        held.clear()
        self.assertLess(torch.cuda.memory_allocated(), allocated - 6 * 128 * 1024)
        self.assertEqual(ring.held(), 0)
        # after the release every call reuses the same block: the caller that drops
        # the previous outputs gets the same address back
        args = a(64)
        ptrs = set()
        for _ in range(5):
            n, out = self._allocations_during(lambda: replay(*args))
            self.assertEqual(n, 0)
            ptrs.add(out.data_ptr())
            del out
        self.assertEqual(len(ptrs), 1)
        self.assertEqual(replay.misses, 0)

    def test_holding_every_output_never_waits_and_keeps_the_ring_at_two(self):
        a = self._stn_args()
        replay = self._replay(_scale_then_norm, a(64))
        ring = self._ring(replay)
        kept = []
        for k in range(24):
            args = a(64)
            got = replay(*args)
            kept.append((got, _scale_then_norm(*args)))
            # the call was served at once: no miss, no wait for a release
            self.assertLessEqual(len(ring.blocks), 2)
        self.assertEqual(ring.overflow, 22)
        self.assertEqual(replay.misses, 0)
        torch.cuda.synchronize()
        for got, want in kept:
            self.assertEqual(got, want, atol=0, rtol=0)
        addresses = {got.data_ptr() for got, _ in kept}
        self.assertEqual(len(addresses), 24)

    def test_one_block_keeps_a_caller_that_feeds_outputs_back_at_zero_allocations(self):
        # decode's pattern: the previous call's outputs are the next call's inputs (a
        # hold of one): the two blocks alternate and nothing is allocated
        w = torch.randn(D, device="cuda")
        ln_w = 1 + 0.1 * torch.randn(D, device="cuda")
        ln_b = 0.1 * torch.randn(D, device="cuda")
        x = torch.randn(64, D, device="cuda")
        replay = self._replay(_scale_then_norm, (x, w, ln_w, ln_b))
        want = x
        allocations = []
        for _ in range(12):
            want = _scale_then_norm(want, w, ln_w, ln_b)
            n, x = self._allocations_during(lambda: replay(x, w, ln_w, ln_b))
            allocations.append(n)
            self.assertEqual(x, want, atol=0, rtol=0)
        # the second call finds the first block held by its input and takes the
        # ring's second block; from then on the two alternate
        self.assertEqual(allocations, [0, 1] + [0] * 10)
        self.assertEqual(self._ring(replay).overflow, 0)
        self.assertEqual(len(self._ring(replay).blocks), 2)

    def test_a_held_output_survives_growth_a_second_variant_and_close(self):
        a = self._stn_args()
        replay = self._replay(_scale_then_norm, a(64))
        args = a(64)
        want = _scale_then_norm(*args)
        held = replay(*args)
        # a larger call raises the floor and takes a new block (the held one stays)
        big = a(4096)
        self.assertEqual(replay(*big), _scale_then_norm(*big), atol=0, rtol=0)
        self.assertGreaterEqual(replay.output_grows, 1)
        self.assertEqual(held, want, atol=0, rtol=0)
        # a second entry of the same function beside it
        other = self._replay(_scale_then_norm, a(16))
        self.assertEqual(other(*args), want, atol=0, rtol=0)
        self.assertEqual(held, want, atol=0, rtol=0)
        replay.close()
        torch.cuda.synchronize()
        self.assertEqual(held, want, atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)

    def test_the_caller_may_write_an_output_in_place(self):
        a = self._stn_args()
        replay = self._replay(_scale_then_norm, a(64))
        out = replay(*a(64))
        out.fill_(7.0)
        for _ in range(3):
            other = replay(*a(64))
            del other
        torch.cuda.synchronize()
        self.assertTrue(bool((out == 7.0).all()))
        # a view the caller makes of an output keeps the block held too
        piece = out[3:5]
        del out
        self.assertEqual(self._ring(replay).held(), 1)
        del piece
        self.assertEqual(self._ring(replay).held(), 0)

    # ---- the block's capacity

    def test_the_block_grows_with_the_call_and_replays_bitwise(self):
        a = self._stn_args()
        replay = self._replay(_scale_then_norm, a(64))
        ring = self._ring(replay)
        capacities = [ring.capacity]
        for rows in (64, 128, 1024, 4096, 256, 8192):
            args = a(rows)
            self.assertEqual(replay(*args), _scale_then_norm(*args), atol=0, rtol=0)
            capacities.append(ring.capacity)
        # rows x D floats: 128 KiB at 64 rows (the 64 KiB rounding); the floor rises
        # at 128 (doubled: 256 KiB), 1024 (2 MiB), 4096 (8 MiB) and 8192 (16 MiB),
        # not at 256; a rise is a rebind of the block root at the retried dispatch,
        # not a miss and not a new variant
        self.assertEqual(replay.output_grows, 4)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.traces), (0, 1, 1)
        )
        self.assertTrue(capacities[-1] >= 4 * 8192 * D)
        # the free blocks under the floor were dropped for the new ones
        self.assertEqual(len(ring.blocks), 1)
        self.assertEqual(ring.replaced, 4)

    def test_arena_check_covers_the_output_plan(self):
        a = self._stn_args()
        replay = self._replay(_scale_then_norm, a(64), arena_check=True)
        for rows in (64, 32, 128):
            args = a(rows)
            self.assertEqual(replay(*args), _scale_then_norm(*args), atol=0, rtol=0)
        plan = replay.lowered.output_arena
        (block,) = plan.blocks.values()
        rounded = block.rounded
        block.rounded = rounded * 10**6  # a block the boxed block cannot hold
        try:
            with self.assertRaisesRegex(AssertionError, "exceeds the arena"):
                replay._check_arena(replay._hot, a(64))
        finally:
            block.rounded = rounded
        replay._check_arena(replay._hot, a(64))

    # ---- switches and the other roots

    def test_the_default_and_the_off_switch_keep_per_call_buffers(self):
        a = self._stn_args()
        self.assertFalse(self.module._OUTPUT_ARENA_DEFAULT)
        default = self.module.HostTraceReplay(_scale_then_norm, a(64))
        self.addCleanup(default.close)
        self.assertIsNone(default.lowered.output_arena)
        # the raw boxed surface takes the tape's tensors alone (the runtime team's
        # entry(box) tests): the block would be a third boxed input
        replay = self._replay(_scale_then_norm, a(64), output_arena=False)
        self.assertIsNone(replay.lowered.output_arena)
        self.assertEqual(len(replay.lowered.allocations), 1)
        self.assertEqual(replay.lowered.input_names[-1], "arena")
        args = a(64)
        n, got = self._allocations_during(lambda: replay(*args))
        self.assertEqual(n, 1)
        self.assertEqual(got, _scale_then_norm(*args), atol=0, rtol=0)
        self.assertIsNone(replay._hot.outputs)

    def test_the_consumer_entry_has_no_output_arena(self):
        from torch.cuda import _host_trace

        a = self._stn_args()
        args = a(64)
        tape = _host_trace.trace(_scale_then_norm, args)
        prepared = self.module.prepare_host_trace(tape, args)
        self.addCleanup(prepared.close)
        self.assertIsNone(prepared.lowered.output_arena)
        self.assertEqual(len(prepared.lowered.input_names), len(tape.inputs))

    def test_a_memcpy_destination_is_planned_like_any_temporary(self):
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
        caches = tuple(
            torch.randn(4, dt.H, dt.LMAX, dt.DH, device="cuda", dtype=dtype)
            for _ in range(2)
        )

        def args(L):
            ids = torch.randint(0, dt.V, (4,), dtype=torch.int64).pin_memory()
            return (ids, *weights, caches[0][:, :, :L], caches[1][:, :, :L])

        replay = self._replay(dt.decode_step, args(16))
        lowered = replay.lowered
        # the H2D copy's destination is a temporary of the arena like any other (the
        # arena holds memcpy endpoints: the two-entry crash was the pinned source's
        # lifetime, integration-06 370da8228f5); the two outputs are blocks of the
        # output arena; no runtime buffer is left
        self.assertEqual(len(lowered.allocations), 0)
        self.assertEqual(len(lowered.output_arena.blocks), 2)
        e_caches = tuple(c.clone() for c in caches)
        for L in (16, 20, 24):
            a = args(L)
            want = dt.decode_step(a[0], *a[1:8], e_caches[0][:, :, :L], e_caches[1][:, :, :L])
            n, got = self._allocations_during(lambda: replay(*a))
            self.assertEqual(n, 0)
            self.assertEqual(got, want, atol=0, rtol=0)
            del got  # held across the next call, it would cost that call a block
        self.assertEqual(caches, e_caches, atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)


if __name__ == "__main__":
    run_tests()
