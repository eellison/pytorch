# Owner(s): ["module: inductor"]
"""The allocator-order replay of the native adapter (hosttrace_allocseq): the tape's
temporaries as roots the family binds by replaying the tape's allocation and
deallocation sequence through the caching allocator in a private pool before the
launch; the outputs stay the runtime's buffers."""

import gc

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import run_tests, TestCase


D = 512
MiB = 1 << 20


def _scale_then_norm(x, w, ln_w, ln_b):
    h = F.layer_norm(x, (D,), ln_w, ln_b)
    return F.silu(h * w) + h


def _three_temporaries(x, y, z):
    # t1 lives across the call; t2 (y * 2) and t3 (z * 3) never overlap (each dies in
    # the in-place add that consumes it), so the sequence hands t3 the block t2 freed
    t1 = x * 2.0
    y.add_(y * 2.0)
    z.add_(z * 3.0)
    return t1 + 1.0


def _rank_dependent(x):
    if x.dim() == 2:
        return x * 2.0 + 1.0
    return (x * 2.0 + 1.0).sum(0) + x.sum(0)


class TestHostTraceAllocSeq(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        torch.manual_seed(0)

    def _replay(self, fn, args, **kw):
        gc.collect()
        kw.setdefault("allocseq", True)
        replay = self.module.HostTraceReplay(fn, args, **kw)
        self.addCleanup(replay.close)
        return replay

    def _allocations_during(self, fn):
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = fn()
        return torch.cuda.memory_stats()["allocation.all.allocated"] - before, out

    def _three(self, rows=1024):
        return tuple(torch.randn(rows, D, device="cuda") for _ in range(3))

    def _sequence(self, replay):
        return replay._hot.sequence

    # ---- the sequence: roots, addresses, modes

    def test_temporaries_are_roots_and_outputs_stay_buffers(self):
        replay = self._replay(_three_temporaries, self._three())
        lowered = replay.lowered
        plan = lowered.sequence
        self.assertIsNotNone(plan)
        self.assertIsNone(lowered.arena)
        # the output is the runtime's one buffer; the three temporaries are roots boxed
        # after the tape's tensors, allocated before their first use and freed after
        # their last, in tape order
        self.assertEqual(len(lowered.allocations), 1)
        self.assertEqual(len(plan.names), 3)
        self.assertEqual(lowered.input_names[-3:], ("seq0", "seq1", "seq2"))
        self.assertEqual(plan.program, (0, 1, ~1, 2, ~2, ~0))
        sequence = self._sequence(replay)
        # t3 took the block t2 freed: two roots at one address, the peak two blocks
        self.assertEqual(sequence.addresses[1], sequence.addresses[2])
        self.assertNotEqual(sequence.addresses[0], sequence.addresses[1])
        self.assertEqual(plan.peak_at_hints, 2 * 1024 * D * 4)
        self.assertEqual(plan.sum_at_hints, 3 * 1024 * D * 4)
        self.assertEqual(sequence.peak, plan.peak_at_hints)
        for _ in range(3):
            x, y, z = self._three()
            want = _three_temporaries(x, y.clone(), z.clone())
            n, got = self._allocations_during(lambda: replay(x, y, z))
            self.assertEqual(
                n, 1
            )  # the output buffer; no allocator call for a temporary
            self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)
        self.assertEqual(sequence.binds, 1)

    def test_hold_keeps_the_blocks_while_the_sizes_fit(self):
        replay = self._replay(_three_temporaries, self._three(1024))
        sequence = self._sequence(replay)
        addresses = list(sequence.addresses)
        for rows in (1024, 512, 1024, 256):
            x, y, z = self._three(rows)
            want = _three_temporaries(x, y.clone(), z.clone())
            self.assertEqual(replay(x, y, z), want, atol=0, rtol=0)
        # smaller calls fit the bound blocks: no sequence, the same addresses
        self.assertEqual(sequence.binds, 1)
        self.assertEqual(sequence.addresses, addresses)
        self.assertEqual(replay.sequence_rebinds, 0)
        x, y, z = self._three(4096)
        want = _three_temporaries(x, y.clone(), z.clone())
        self.assertEqual(replay(x, y, z), want, atol=0, rtol=0)
        # a larger call outgrew them: the predicate's size terms failed, the miss path
        # ran the sequence at the call's sizes and rebound the roots, no miss
        self.assertEqual((sequence.binds, replay.sequence_rebinds), (2, 1))
        self.assertEqual(replay.misses, 0)
        self.assertEqual(len(replay.variants), 1)
        self.assertEqual(sequence.nbytes, [4096 * D * 4] * 3)
        self.assertEqual(sequence.peak, 2 * 4096 * D * 4)
        x, y, z = self._three(1024)
        want = _three_temporaries(x, y.clone(), z.clone())
        self.assertEqual(replay(x, y, z), want, atol=0, rtol=0)
        self.assertEqual(sequence.binds, 2)

    def test_replay_mode_runs_the_sequence_every_call(self):
        replay = self._replay(_three_temporaries, self._three(), allocseq_mode="replay")
        sequence = self._sequence(replay)
        for k in range(1, 4):
            x, y, z = self._three()
            want = _three_temporaries(x, y.clone(), z.clone())
            n, got = self._allocations_during(lambda: replay(x, y, z))
            self.assertEqual(got, want, atol=0, rtol=0)
            # the output buffer plus one allocation per temporary
            self.assertEqual(n, 1 + len(sequence.plan.names))
            self.assertEqual(sequence.binds, 1 + k)
        # the same sizes in the same pool give the same addresses: nothing rebound
        self.assertEqual(sequence.rebinds, 3)
        self.assertEqual(replay.misses, 0)

    def test_the_pool_holds_the_peak_not_the_sum(self):
        rows = (
            8192 * 4
        )  # three 64 MiB temporaries: exact-size segments, no 20 MiB rounding
        replay = self._replay(_three_temporaries, self._three(rows))
        sequence = self._sequence(replay)
        plan = replay.lowered.sequence
        self.assertEqual(plan.peak_at_hints, 2 * rows * D * 4)
        reserved = sequence.reserved()
        self.assertGreaterEqual(reserved, plan.peak_at_hints)
        self.assertLess(reserved, plan.sum_at_hints)
        # every root reads its size: the pool's segments are the entry's hold
        self.assertEqual(sum(sequence.nbytes), plan.sum_at_hints)

    def test_pressure_moves_the_next_sequence_to_a_fresh_pool(self):
        from torch._inductor.runtime._cudagraph import hosttrace_allocseq as hs

        replay = self._replay(_three_temporaries, self._three(1024))
        sequence = self._sequence(replay)
        first = sequence.pool
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved()
        hs._ooms[0] += 1  # what the out-of-memory observer records
        x, y, z = self._three(1024)
        want = _three_temporaries(x, y.clone(), z.clone())
        self.assertEqual(replay(x, y, z), want, atol=0, rtol=0)
        # hold mode: nothing binds while the sizes fit, so the pool stays
        self.assertIs(sequence.pool, first)
        x, y, z = self._three(2048)
        want = _three_temporaries(x, y.clone(), z.clone())
        self.assertEqual(replay(x, y, z), want, atol=0, rtol=0)
        # the rebind saw the pressure: a fresh pool, the first dropped (its segments
        # released to the driver at once)
        self.assertIsNot(sequence.pool, first)
        self.assertEqual((sequence.pools, sequence.pressure_releases), (2, 1))
        torch.cuda.synchronize()
        self.assertLessEqual(
            torch.cuda.memory_reserved() - reserved, sequence.reserved()
        )
        self.assertEqual(replay.misses, 0)

    def test_nonoverlap_assertion_checks_every_served_call(self):
        replay = self._replay(_three_temporaries, self._three(), arena_check=True)
        for rows in (1024, 2048, 512):
            x, y, z = self._three(rows)
            want = _three_temporaries(x, y.clone(), z.clone())
            self.assertEqual(replay(x, y, z), want, atol=0, rtol=0)
        plan = replay.lowered.sequence
        sequence = self._sequence(replay)
        # the check is real: roots live at once (t1 and t2) at one address fail it
        addresses = list(sequence.addresses)
        addresses[1] = addresses[0]
        with self.assertRaisesRegex(AssertionError, "overlap while both live"):
            plan.check(sequence.nbytes, addresses)
        with self.assertRaisesRegex(AssertionError, "not 512-byte aligned"):
            plan.check(sequence.nbytes, [a + 256 for a in sequence.addresses])
        plan.check(sequence.nbytes, sequence.addresses)

    # ---- families and the other placements

    def test_a_variant_with_another_sequence_starts_a_new_family(self):
        x2 = torch.randn(64, D, device="cuda")
        replay = self._replay(_rank_dependent, (x2,))
        x3 = torch.randn(4, 64, D, device="cuda")
        self.assertEqual(replay(x3), _rank_dependent(x3), atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (1, 2))
        # the second tape has other temporaries: its own family, its own sequence
        self.assertEqual(len(replay._families), 2)
        first, second = (f.sequence for f in replay._families)
        self.assertIsNot(first, second)
        self.assertNotEqual(first.plan.signature, second.plan.signature)
        for x in (x2, x3, x2):
            self.assertEqual(replay(x), _rank_dependent(x), atol=0, rtol=0)
        self.assertEqual(replay.misses, 1)

    def test_the_sequence_replaces_the_arena(self):
        args = self._three()
        with self.assertRaisesRegex(ValueError, "replaces the planned arena"):
            self.module.HostTraceReplay(
                _three_temporaries, args, allocseq=True, arena=True
            )
        replay = self._replay(_three_temporaries, args)
        self.assertIsNone(replay._hot.arena)
        self.assertIsNone(replay._hot.outputs)
        self.assertEqual(replay.arena_stats()["variants"], [None])
        # off by default: the arena serves the temporaries, no sequence
        if self.module._ALLOCSEQ_DEFAULT:
            self.skipTest("TORCH_HOST_TRACE_ALLOCSEQ=1 in the environment")
        plain = self._replay(_three_temporaries, args, allocseq=None)
        self.assertIsNone(plain.lowered.sequence)
        self.assertIsNone(plain._hot.sequence)
        self.assertEqual(plain.sequence_stats()["families"], [None])

    def test_the_output_arena_composes_with_the_sequence(self):
        args = self._three()
        replay = self._replay(
            _three_temporaries, args, output_arena=True, arena_check=True
        )
        lowered = replay.lowered
        self.assertIsNotNone(lowered.sequence)
        self.assertIsNotNone(lowered.output_arena)
        # the box: the tape's tensors, the output block, then the roots; nothing is
        # left to the runtime's per-call buffers
        self.assertEqual(lowered.input_names[-4:], ("outputs", "seq0", "seq1", "seq2"))
        self.assertEqual(lowered.sequence.input_index, len(lowered.tape.inputs) + 1)
        self.assertEqual(len(lowered.allocations), 0)
        family = replay._hot
        self.assertIsNotNone(family.outputs)
        self.assertIsNotNone(family.sequence)
        for _ in range(3):
            x, y, z = self._three()
            want = _three_temporaries(x, y.clone(), z.clone())
            n, got = self._allocations_during(lambda: replay(x, y, z))
            # the ring's block serves the dropped output again and the roots hold
            self.assertEqual(n, 0)
            self.assertEqual(got, want, atol=0, rtol=0)
            del got
        self.assertEqual(replay.misses, 0)
        self.assertEqual(len(family.outputs.blocks), 1)
        self.assertEqual(self._sequence(replay).binds, 1)

    def test_programmatic_consumers_move_the_free_point(self):
        import sympy

        from torch._inductor.runtime._cudagraph.hosttrace_allocseq import plan_sequence

        # five nodes in launch order at the recorder's seqs (allocations take seqs too);
        # a dies at node 2, b at 3, c at 5, d at 7
        rows = [(n, sympy.Integer(1024)) for n in "abcd"]
        uses = {"a": [0, 2], "b": [3], "c": [5], "d": [7]}
        order = {"a": 0, "b": 1, "c": 2, "d": 3}
        nodes = (0, 2, 3, 5, 7)
        plain = plan_sequence(rows, uses, {}, 4, order)
        self.assertEqual(plain.program, (0, ~0, 1, ~1, 2, ~2, 3, ~3))
        self.assertEqual(plain.peak_at_hints, 1024)
        self.assertEqual(
            plan_sequence(rows, uses, {}, 4, order, (), nodes).program, plain.program
        )
        # nodes 3 and 5 behind programmatic edges, 7 waited: a (last use 2) and b are
        # reusable from node 7 on, so b and c cannot take a's block and the peak is three
        moved = plan_sequence(
            rows, uses, {}, 4, order, programmatic=(3, 5), nodes=nodes
        )
        self.assertEqual(moved.program, (0, 1, 2, ~0, ~1, ~2, 3, ~3))
        self.assertEqual(moved.intervals, ((0, 3), (1, 4), (2, 5), (6, 7)))
        self.assertEqual(moved.peak_at_hints, 3 * 1024)
        # the run stops at the first node that waited: with node 3 full, a is freed at
        # its last use again and only b's free point moves past node 5
        moved = plan_sequence(rows, uses, {}, 4, order, programmatic=(5,), nodes=nodes)
        self.assertEqual(moved.program, (0, ~0, 1, 2, ~1, ~2, 3, ~3))
        self.assertEqual(moved.peak_at_hints, 2 * 1024)
        with self.assertRaisesRegex(ValueError, "launch order"):
            plan_sequence(rows, uses, {}, 4, order, programmatic=(5,))

    def test_a_region_behind_a_programmatic_edge_moves_the_free_point(self):
        # the rule wired from the lowered tape (A326 / A337): t dies at the launch
        # right before the GEMM region, and cuBLAS launches the region's first kernel
        # with programmatic stream serialization, so the region's output o may not
        # take t's block: t is freed after the region (the program allocates o before
        # it frees t and y), where the unwired plan freed t at its last use
        def block(x, w):
            t = x * 2.0
            y = t + 1.0
            o = y @ w
            return o + 1.0

        w = torch.randn(D, D, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(64, D, device="cuda", dtype=torch.bfloat16)
        replay = self._replay(block, (x, w), arena_check=True)
        lowered = replay.lowered
        plan = lowered.sequence
        (region,) = lowered.regions
        (site,) = replay.region_stats()["sites"]
        if not site["programmatic"][0]:
            self.skipTest("cuBLAS launched this GEMM without programmatic serialization")
        self.assertEqual(plan.programmatic, (region.seq,))
        self.assertEqual(len(plan.names), 3)
        self.assertEqual(plan.program, (0, 1, 2, ~0, ~1, ~2))
        self.assertEqual(plan.peak_at_hints, plan.sum_at_hints)
        # the driver's edge into the region's first node is the flag the plan used
        self.assertEqual(lowered.node_seqs[lowered.programmatic_nodes[0]], region.seq)
        self.assertEqual(
            replay.sequence_stats()["variants"][0]["programmatic"], 1
        )
        # a larger call outgrows the held roots: the miss path rebinds them (no
        # re-trace); its new region key is a harvest, the one kind of miss here
        for rows in (64, 128, 64):
            x = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(replay(x, w), block(x, w), atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 1)
        self.assertEqual(replay.ordinary, 0)
        self.assertGreaterEqual(replay.sequence_rebinds, 1)
        self.assertTrue(all("harvest" in row[2] for row in replay.miss_log))

    def test_the_consumer_entry_keeps_per_call_buffers(self):
        # the runtime team's entry (prepare_host_trace) lowers without a sequence:
        # every allocation stays a per-call buffer of its box
        from torch.cuda import _host_trace

        args = self._three()
        tape = _host_trace.trace(_three_temporaries, args)
        prepared = self.module.prepare_host_trace(tape, args)
        self.addCleanup(prepared.close)
        self.assertIsNone(prepared.lowered.sequence)
        self.assertEqual(len(prepared.lowered.allocations), 4)

    # ---- the decode chain (commit 6's composition): a memcpy endpoint, a growing cache

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

    def test_the_chain_grows_its_cache_and_replays_bitwise(self):
        step, caches, args = self._chain()
        replay = self._replay(step, args(4, 16, caches(4)))
        lowered = replay.lowered
        # the two outputs are the runtime's buffers; every other allocation a launch
        # or the H2D copy touches is a root (the arena rule since integration-06's
        # 370da8228f5: a memcpy endpoint is planned like any temporary)
        self.assertEqual(len(lowered.allocations), 2)
        self.assertGreaterEqual(len(lowered.sequence.names), 10)
        sequence = self._sequence(replay)
        e_caches = caches(4)
        r_caches = tuple(c.clone() for c in e_caches)
        for L in range(16, 25):
            a = args(4, L, e_caches)
            want = step(*a)
            a = (a[0], *a[1:8], r_caches[0][:, :, :L], r_caches[1][:, :, :L])
            n, got = self._allocations_during(lambda: replay(*a))
            self.assertEqual(n, 2)  # the two output buffers; no allocator call for a temporary
            self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual(e_caches, r_caches, atol=0, rtol=0)
        # the cache grows but no temporary of this chain depends on its length (flash
        # materializes no scores): the blocks of the preparation serve every step
        self.assertEqual(sequence.binds, 1)
        self.assertEqual((replay.misses, replay.sequence_rebinds), (0, 0))
        self.assertEqual(len(replay.variants), 1)

    def test_two_entries_own_their_pools(self):
        step, caches, args = self._chain()
        first = self._replay(step, args(4, 16, caches(4)))
        second = self._replay(_three_temporaries, self._three())
        p1, p2 = self._sequence(first).pool, self._sequence(second).pool
        self.assertNotEqual(p1.id, p2.id)
        c = caches(4)
        for L in (17, 18, 19):
            a = args(4, L, c)
            self.assertEqual(first(*a), step(*a), atol=0, rtol=0)
            x, y, z = self._three()
            want = _three_temporaries(x, y.clone(), z.clone())
            self.assertEqual(second(x, y, z), want, atol=0, rtol=0)
        sequence = self._sequence(second)
        second.close()
        self.assertIsNone(sequence.pool)
        a = args(4, 20, c)
        self.assertEqual(first(*a), step(*a), atol=0, rtol=0)
        self.assertEqual((first.misses, second.misses), (0, 0))


if __name__ == "__main__":
    run_tests()
