# Owner(s): ["module: inductor"]
"""Local misses served by partitioning (hosttrace_partition, plan item 38 stage 2 in
its partition form, E41): a call that fails only kernel-only guards of one op of the
tape is served by the program split at that op (the segments before and after it as
programs of their own, the op through eager), bitwise against eager, with no re-trace;
a metadata miss still re-traces; a misclassified guard is caught at the swap check; the
allocator's peak of a partitioned call stays at eager's."""

import gc

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


N = 4096
MiB = 1 << 20


def _misaligned(w, elements=2):
    # the same values two elements (4 bytes for bf16) into a larger storage: the
    # layer norm's alignment class flips, nothing else
    flat = torch.empty(w.numel() + elements, device=w.device, dtype=w.dtype)
    out = flat[elements : elements + w.numel()]
    out.copy_(w)
    return out


def _chain(x, y, w, b):
    h = x + y
    h = F.layer_norm(h, (h.shape[-1],), w, b, 1e-5)
    return F.gelu(h)


def _crossing(x, w, b):
    # t is read on both sides of the layer norm: an escaping output of the prefix
    # and a prebound root of the suffix
    t = F.gelu(x)
    h = F.layer_norm(t, (t.shape[-1],), w, b, 1e-5)
    u = h * t
    return u.sum(dim=1)


def _two_lookups(ids, table, positions):
    return F.embedding(ids, table) + F.embedding(ids, positions)


def _add(x, y):
    # one op: a later op's own metadata guards over the broadcast dims (the mul's
    # Ne(s, 1) rows, metadata-tagged) would classify the miss as not local before
    # the swap check is reached
    return x + y


class TestHostTracePartition(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace
        torch.manual_seed(0)

    def _replay(self, fn, args, **kw):
        gc.collect()
        kw.setdefault("partition", True)
        with torch.no_grad():
            replay = self.module.HostTraceReplay(fn, args, **kw)
        self.addCleanup(replay.close)
        return replay

    def _inputs(self, M=64, dtype=torch.bfloat16):
        x = torch.randn(M, N, device="cuda", dtype=dtype)
        y = torch.randn(M, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype)
        b = torch.randn(N, device="cuda", dtype=dtype)
        return x, y, w, b

    def test_a_kernel_only_miss_is_served_by_the_partition_bitwise(self):
        x, y, w, b = self._inputs()
        replay = self._replay(_chain, (x, y, w, b))
        with torch.no_grad():
            self.assertEqual(replay(x, y, w, b), _chain(x, y, w, b), atol=0, rtol=0)
            w2 = _misaligned(w)
            got = replay(x, y, w2, b)
            want = _chain(x, y, w2, b)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(replay.traces, 1)
        self.assertEqual(replay.partition_builds, 1)
        self.assertEqual(replay.partition_serves, 1)
        self.assertEqual(len(replay.variants), 1)
        (partition,) = replay.partitions
        stats = partition.stats()
        self.assertEqual(
            [op[1] for op in stats["ops"]], ["aten.native_layer_norm.default"]
        )
        # the prefix (the add) escapes its output to the eager layer norm; the suffix
        # (the gelu) reads the layer norm's output as a prebound root
        self.assertEqual(stats["segments"][0]["escapes"], ["alloc0"])
        self.assertEqual(stats["segments"][0]["launches"], 1)
        self.assertEqual(stats["segments"][1]["launches"], 1)
        self.assertEqual(len(stats["segments"][1]["prebound"]), 1)
        self.assertEqual(stats["segments"][1]["t_outputs"], [0])
        self.assertIn("local miss: kernel-only guards of op", replay.miss_log[-1][1])
        # the same class again serves from the built partition, bitwise
        with torch.no_grad():
            for _ in range(3):
                self.assertTrue(torch.equal(replay(x, y, w2, b), want))
            self.assertTrue(torch.equal(replay(x, y, w, b), _chain(x, y, w, b)))
        self.assertEqual(replay.partition_serves, 4)
        self.assertEqual(replay.partition_builds, 1)
        self.assertEqual(replay.traces, 1)

    def test_a_temporary_read_across_the_cut_is_a_boundary(self):
        x, _, w, b = self._inputs()
        replay = self._replay(_crossing, (x, w, b))
        with torch.no_grad():
            w2 = _misaligned(w)
            got = replay(x, w2, b)
            want = _crossing(x, w2, b)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(replay.traces, 1)
        self.assertEqual(replay.partition_builds, 1)
        (partition,) = replay.partitions
        stats = partition.stats()
        prefix, suffix = stats["segments"]
        # the gelu's output crosses the cut: an escape of the prefix and, with the
        # layer norm's output, a prebound root of the suffix
        self.assertEqual(len(prefix["escapes"]), 1)
        self.assertEqual(len(suffix["prebound"]), 2)
        self.assertGreaterEqual(suffix["sequence_roots"], 2)

    def test_a_metadata_miss_still_re_traces(self):
        x, y, w, b = self._inputs()
        replay = self._replay(_chain, (x, y, w, b))
        x2, y2, w2, b2 = (t[..., : N // 2].contiguous() for t in (x, y, w, b))
        with torch.no_grad():
            got = replay(x2, y2, w2, b2)
            want = _chain(x2, y2, w2, b2)
        self.assertTrue(torch.equal(got, want))
        # a hidden-size change fails metadata and validity guards: a trace, no partition
        self.assertEqual(replay.traces, 2)
        self.assertEqual(replay.partition_builds, 0)
        self.assertEqual(replay.partition_serves, 0)
        self.assertEqual(len(replay.variants), 2)

    def test_two_kernel_only_misses_are_two_cuts(self):
        table = torch.randn(1000, 256, device="cuda", dtype=torch.bfloat16)
        positions = torch.randn(1000, 256, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 1000, (4, 1), device="cuda")
        replay = self._replay(_two_lookups, (ids, table, positions))
        ids64 = torch.randint(0, 1000, (64, 1), device="cuda")
        with torch.no_grad():
            got = replay(ids64, table, positions)
            want = _two_lookups(ids64, table, positions)
        self.assertTrue(torch.equal(got, want))
        # index_select's numel <= 16 path: one kernel-only guard per lookup
        self.assertEqual(replay.traces, 1)
        self.assertEqual(replay.partition_builds, 1)
        (partition,) = replay.partitions
        stats = partition.stats()
        # the cut is at the top-level op (the embedding composite around each
        # index_select); the second lookup raises the same relation over the same ids,
        # which the record keeps once: its `also` row names the second op
        self.assertEqual(
            [op[1] for op in stats["ops"]],
            ["aten.embedding.default", "aten.embedding.default"],
        )
        # nothing runs between the two lookups: an empty middle segment; the add is
        # the suffix over two prebound roots (the eager lookups' outputs)
        self.assertEqual(stats["segments"][1]["launches"], 0)
        self.assertEqual(stats["segments"][2]["launches"], 1)
        self.assertEqual(len(stats["segments"][2]["prebound"]), 2)
        with torch.no_grad():
            self.assertTrue(torch.equal(replay(ids64, table, positions), want))
            self.assertTrue(
                torch.equal(
                    replay(ids, table, positions), _two_lookups(ids, table, positions)
                )
            )
        self.assertEqual(replay.partition_serves, 2)

    def test_a_misclassified_guard_is_caught_at_the_swap_check(self):
        x = torch.randn(8, 4, 64, device="cuda")
        y = torch.randn(8, 4, 64, device="cuda")
        replay = self._replay(_add, (x, y))
        tape = replay.tape
        add = next(op for op in tape.ops if op.depth == 0)
        self.assertEqual(str(add.func), "aten.add.Tensor")
        # the add's output names dim 1 by one operand's symbol (the raw expression,
        # before the broadcast equality unified the two); the call where that operand
        # is 1 there broadcasts in eager and differs from the tape's expression
        out_dim = add.outputs.shape[1].node._expr
        which = next(
            i for i, t in enumerate(add.args) if t.shape[1].node._expr == out_dim
        )
        # deliberately misclassify: every metadata-tagged guard of the add (its
        # broadcast equalities) becomes a kernel-choice guard
        rows = 0
        for raw in range(*add.guard_range):
            op_index, depth, origin = tape.guard_rows[raw]
            if depth == 0:
                tape.guard_rows[raw] = (op_index, 1, origin)
                rows += 1
        self.assertGreater(rows, 0)
        small = [x, y]
        small[which] = small[which][:, :1].contiguous()
        with torch.no_grad():
            got = replay(*small)
            want = _add(*small)
        self.assertTrue(torch.equal(got, want))
        # the swap check refused the partition (the eager output's sizes differ from
        # the tape's expressions at the call), the call re-traced, the guards are
        # reclassified as metadata
        self.assertEqual(replay.swap_refusals, 1)
        self.assertEqual(replay.partition_builds, 0)
        self.assertEqual(replay.traces, 2)
        self.assertIn("swap check refused", replay.miss_log[-2][1])
        self.assertTrue(
            all(tape.guard_rows[raw][1] == 0 for raw in range(*add.guard_range))
        )

    def test_the_allocator_peak_of_a_partitioned_call_stays_at_eagers(self):
        x, _, w, b = self._inputs(M=1024)
        replay = self._replay(_crossing, (x, w, b), allocseq=True)
        w2 = _misaligned(w)

        def peak(fn):
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            out = fn()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base, out

        with torch.no_grad():
            eager_peak, want = peak(lambda: _crossing(x, w2, b))
            del want
            first_peak, got = peak(lambda: replay(x, w2, b))
            del got
            steady_peak, got = peak(lambda: replay(x, w2, b))
            native_peak, _ = peak(lambda: replay(x, w, b))
        self.assertEqual(replay.partition_builds, 1)
        self.assertTrue(torch.equal(got, _crossing(x, w2, b)))
        # the eager layer norm's outputs and the prefix's escaping output live from
        # their allocation to the suffix's launch, as they do in eager
        self.assertLessEqual(steady_peak, eager_peak)
        self.assertLessEqual(native_peak, eager_peak)


class TestHostTraceCachedPartition(TestCase):
    @parametrize("change", ("dtype", "broadcast"))
    def test_cached_empty_segments_reject_before_mutation(self, device, change):
        from torch._inductor.runtime._cudagraph import (
            direct_hosttrace,
            hosttrace_partition,
        )

        def add_(x, y):
            return x.add_(y)

        examples = (
            torch.ones(256, device=device),
            torch.full((256,), 2.0, device=device),
        )
        with torch.no_grad():
            replay = direct_hosttrace.HostTraceReplay(add_, examples, partition=True)
            self.addCleanup(replay.close)
            x = torch.ones_like(examples[0])
            y = _misaligned(examples[1], elements=1)
            self.assertIs(replay(x, y), x)
            self.assertEqual(x, torch.full_like(x, 3.0), atol=0, rtol=0)
        self.assertEqual(replay.traces, 1)
        self.assertEqual(replay.partition_builds, 1)
        self.assertEqual(replay.partition_serves, 1)
        (partition,) = replay.partitions
        self.assertEqual(
            [op[1] for op in partition.stats()["ops"]], ["aten.add_.Tensor"]
        )
        self.assertEqual(len(partition.segments), 2)
        self.assertTrue(all(seg.part is None for seg in partition.segments))
        self.assertTrue(all(seg.lowered is None for seg in partition.segments))

        dtype = torch.float64 if change == "dtype" else torch.float32
        x = torch.ones(256, device=device, dtype=dtype)
        y = torch.full(
            (1 if change == "broadcast" else 256,), 2.0, device=device, dtype=dtype
        )
        before = (x.clone(), y.clone())
        args = (x, y)
        family = partition.variant.family
        with torch.no_grad():
            self.assertIsNone(
                hosttrace_partition.serve_existing(
                    replay, family, args, family.box(args)
                )
            )
            self.assertEqual(args, before, atol=0, rtol=0)
            self.assertEqual(replay.partition_serves, 1)
            self.assertEqual(replay.traces, 1)

            self.assertIs(replay(*args), x)
            self.assertEqual(x, before[0] + before[1], atol=0, rtol=0)
            self.assertEqual(y, before[1], atol=0, rtol=0)
            self.assertEqual(replay.traces, 2)
            self.assertEqual(len(replay.variants), 2)
            self.assertEqual(replay.partition_builds, 1)
            self.assertEqual(replay.partition_serves, 1)
            self.assertIs(replay(*args), x)
            self.assertEqual(x, before[0] + 2 * before[1], atol=0, rtol=0)
        self.assertEqual(replay.traces, 2)
        self.assertEqual(replay.ordinary, 0)
        self.assertEqual(replay.declines, [])


instantiate_device_type_tests(TestHostTraceCachedPartition, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
