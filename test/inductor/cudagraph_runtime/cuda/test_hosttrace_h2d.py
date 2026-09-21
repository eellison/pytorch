# Owner(s): ["module: inductor"]
"""Host-to-device data (HostTable, copy_h2d, pinned inputs) lowered into the shared native replay."""

import gc
import json
import sys
from pathlib import Path
from unittest import mock

import torch

from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from host_trace_h2d_probe import probe


C = torch._C
V, D = 512, 64


def grouped(*tensors):
    return probe().grouped_mul(*tensors)


def gather(table, ids):
    return probe().gather(table, ids)


def _groups(sizes):
    out = []
    for n in sizes:
        out.append(torch.randn(n, device="cuda"))
        out.append(torch.randn(n, device="cuda"))
    return tuple(out)


def _ref_grouped(tensors):
    return torch.cat([a * b for a, b in zip(tensors[0::2], tensors[1::2])])


def _ids(n, pinned=True):
    ids = torch.randint(0, V, (n,), dtype=torch.int64)
    return ids.pin_memory() if pinned else ids


class TestHostTraceH2D(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace

    def _replay(self, fn, args, **kw):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args, **kw)
        self.addCleanup(replay.close)
        return replay

    def test_table_host_serves_new_sizes_and_addresses(self):
        replay = self._replay(grouped, _groups((64, 128, 32)))
        # the pointer table and the size table, each copied once per call
        self.assertEqual(len(replay.lowered.host_tables), replay.tape.num_host_buffers)
        self.assertEqual(len(replay.lowered.memcpys), replay.tape.num_memcpys)
        self.assertEqual(len(replay.lowered.memcpys), 2)
        for sizes in ((64, 128, 32), (16, 16, 16), (200, 8, 1024)):
            args = _groups(sizes)
            before = replay.misses
            self.assertEqual(replay(*args), _ref_grouped(args), atol=0, rtol=0)
            self.assertEqual(replay.misses, before)

    def test_back_to_back_replays_without_a_host_sync(self):
        for depth in (1, 2):
            replay = self._replay(grouped, _groups((64, 128, 32)), staging_depth=depth)
            gc.collect()
            xs = [_groups((64 + 8 * i, 128, 32 + i)) for i in range(6)]
            expected = [_ref_grouped(x) for x in xs]
            outs = [replay(*x) for x in xs]
            del xs
            gc.collect()
            torch.cuda.synchronize()
            replay.close()
            gc.collect()
            for out, reference in zip(outs, expected):
                self.assertEqual(out, reference, atol=0, rtol=0)

    def test_fixed_pinned_source_rewritten_in_place_has_no_source_rebinds(self):
        table = torch.randn(V, D, device="cuda")
        ids = _ids(64)
        replay = self._replay(gather, (table, ids))
        for _ in range(4):
            replay.wait_for_h2d()
            new = torch.randint(0, V, (64,), dtype=torch.int64)
            ids.copy_(new)
            self.assertEqual(replay(table, ids), table[new.cuda()], atol=0, rtol=0)
        self.assertEqual(replay.memcpy_stats[1], 0)

    def test_fresh_pinned_source_per_call_rebinds_once(self):
        table = torch.randn(V, D, device="cuda")
        replay = self._replay(gather, (table, _ids(64)))
        previous = None
        for step in range(4):
            ids = _ids(64)
            want = table[ids.cuda()]
            out = replay(table, ids)
            if previous is not None:
                replay.wait_for_h2d()
                previous.fill_(V - 1)
            self.assertEqual(out, want, atol=0, rtol=0)
            self.assertEqual(replay.memcpy_stats[1], step + 1)
            previous = ids
        for n in (1, 8, 200):
            ids = _ids(n)
            self.assertEqual(replay(table, ids), table[ids.cuda()], atol=0, rtol=0)
        self.assertEqual(replay.misses, 0)

    def test_pageable_and_cuda_ids_miss_to_the_ordinary_host(self):
        # the pinned fact misses; the ordinary host then refuses the source as eager does
        table = torch.randn(V, D, device="cuda")
        replay = self._replay(gather, (table, _ids(64)))
        for bad in (_ids(64, pinned=False), _ids(64).cuda()):
            with self.assertRaises(RuntimeError):
                replay(table, bad)
        self.assertEqual(replay.misses, 2)
        ids = _ids(64)
        self.assertEqual(replay(table, ids), table[ids.cuda()], atol=0, rtol=0)
        self.assertEqual(replay.misses, 2)

    def test_agrees_with_the_interim_replay(self):
        from torch.cuda import _host_trace

        base = _groups((64, 128, 32))
        replay = self._replay(grouped, base)
        interim = _host_trace.build(replay.tape, grouped, base)
        for sizes in ((16, 16, 16), (200, 8, 1024)):
            args = _groups(sizes)
            ours = interim.replay(args)[0]
            self.assertEqual(replay(*args), ours, atol=0, rtol=0)


class TestHostTracePinnedVariants(TestCase):
    def test_capture_pinned_source_survives_host_cache_flush(self, device):
        from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay

        replay = HostTraceReplay(gather)
        self.addCleanup(replay.close)
        table = torch.randn(V, D, device=device)
        ids = _ids(64)
        source = StorageWeakRef(ids.untyped_storage())
        expected = table[ids.to(device)]
        (output,) = replay([table, ids])
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 1)
        misses = replay.misses
        del ids
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertFalse(source.expired())
        C._host_emptyCache()
        ids = _ids(64)
        rebound = StorageWeakRef(ids.untyped_storage())
        expected = table[ids.to(device)]
        with mock.patch.object(
            replay,
            "fn",
            side_effect=AssertionError("Expected native hit"),
        ):
            (output,) = replay([table, ids])
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(replay.misses, misses)
        del ids
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertTrue(source.expired())
        self.assertFalse(rebound.expired())
        replay.close()
        gc.collect()
        self.assertTrue(rebound.expired())

    @parametrize("boxed", (False, True))
    def test_older_pinned_variant_retains_current_source(self, device, boxed):
        from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay

        def gather_by_device(table, ids):
            if ids.device.type == "cpu":
                return probe().gather(table, ids)
            return C._host_trace_ti_index_select(table, 0, ids)

        table = torch.randn(V, D, device=device)
        initial_ids = _ids(64)
        replay = (
            HostTraceReplay(gather_by_device)
            if boxed
            else HostTraceReplay(gather_by_device, (table, initial_ids))
        )
        self.addCleanup(replay.close)

        def invoke(ids):
            if boxed:
                box = [table, ids]
                (output,) = replay(box)
                self.assertEqual(box, [])
                return output
            return replay(table, ids)

        self.assertEqual(invoke(initial_ids), table[initial_ids.to(device)])
        self.assertEqual(len(replay.variants), 1)
        gpu_ids = initial_ids.to(device)
        self.assertEqual(invoke(gpu_ids), table[gpu_ids])
        self.assertEqual(len(replay.variants), 2)
        misses = replay.misses
        current_ids = _ids(64)
        source = StorageWeakRef(current_ids.untyped_storage())
        reference = table[current_ids.to(device)]
        with mock.patch.object(
            C,
            "_host_trace_record_host_event",
            side_effect=AssertionError(
                "Native H2D replay called the Python event hook"
            ),
        ):
            output = invoke(current_ids)
            self.assertEqual(replay.misses, misses)
            gpu_output = invoke(gpu_ids)
        del initial_ids, current_ids
        replay.wait_for_h2d()
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertFalse(source.expired())
        C._host_emptyCache()
        current_ids = _ids(64)
        rebound = StorageWeakRef(current_ids.untyped_storage())
        rebound_reference = table[current_ids.to(device)]
        with mock.patch.object(
            replay,
            "fn",
            side_effect=AssertionError("Expected an older native variant hit"),
        ):
            flushed_output = invoke(current_ids)
            gpu_output = invoke(gpu_ids)
        self.assertEqual(output, reference, atol=0, rtol=0)
        self.assertEqual(flushed_output, rebound_reference, atol=0, rtol=0)
        self.assertEqual(gpu_output, table[gpu_ids], atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 2)
        self.assertEqual(replay.misses, misses)
        del current_ids
        replay.wait_for_h2d()
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertTrue(source.expired())
        self.assertFalse(rebound.expired())
        replay.close()
        gc.collect()
        self.assertTrue(rebound.expired())

    @parametrize("phase", ("captured", "served"))
    def test_dormant_pinned_variant_retains_storage_after_set(self, device, phase):
        from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay

        def gather_by_device(table, ids):
            if ids.device.type == "cpu":
                return probe().gather(table, ids)
            return C._host_trace_ti_index_select(table, 0, ids)

        replay = HostTraceReplay(gather_by_device)
        self.addCleanup(replay.close)
        table = torch.randn(V, D, device=device)
        ids = _ids(64)
        (output,) = replay([table, ids])
        self.assertEqual(output, table[ids.to(device)], atol=0, rtol=0)
        if phase == "served":
            ids = _ids(64)
            (output,) = replay([table, ids])
            self.assertEqual(output, table[ids.to(device)], atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 1)
        gpu_ids = ids.to(device)
        (gpu_output,) = replay([table, gpu_ids])
        self.assertEqual(gpu_output, table[gpu_ids], atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 2)
        misses = replay.misses
        source = StorageWeakRef(ids.untyped_storage())
        torch.cuda.synchronize(device)

        ids.set_(_ids(64))
        gc.collect()
        self.assertFalse(source.expired(), "set_ released the bound copy storage")
        C._host_emptyCache()
        self.assertFalse(source.expired())
        rebound = StorageWeakRef(ids.untyped_storage())
        expected = table[ids.to(device)]
        with mock.patch.object(
            replay,
            "fn",
            side_effect=AssertionError("Expected an older native variant hit"),
        ):
            (gpu_output,) = replay([table, gpu_ids])
            self.assertFalse(source.expired())
            (output,) = replay([table, ids])
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(gpu_output, table[gpu_ids], atol=0, rtol=0)
        self.assertEqual(len(replay.variants), 2)
        self.assertEqual(replay.misses, misses)
        del ids
        torch.cuda.synchronize(device)
        gc.collect()
        self.assertTrue(source.expired())
        self.assertFalse(rebound.expired())
        replay.close()
        gc.collect()
        self.assertTrue(rebound.expired())


class TestHostTraceTableCopyHistories(TestCase):
    @parametrize("rewrite", (False, True))
    def test_each_copy_reads_its_own_table_image(self, device, rewrite):
        from torch._inductor.runtime._cudagraph.direct_hosttrace import HostTraceReplay

        fn = probe().copy_rewrite_copy if rewrite else probe().copy_twice
        replay = HostTraceReplay(fn)
        self.addCleanup(replay.close)
        inputs = [torch.randn(size, 8, device=device) for size in (3, 5, 7, 11)]
        self.assertEqual(len({source.data_ptr() for source in inputs}), len(inputs))
        expected = [
            torch.tensor(
                [size * 100, size * 200, size * 300]
                if rewrite
                else [size * 100 + i for i in range(4)] * 2,
                device=device,
            )
            for size in (3, 5, 7, 11)
        ]
        box = [inputs[0]]
        outputs = [replay(box)[0]]
        self.assertEqual(box, [])
        self.assertEqual(len(replay.variants), 1)
        misses = replay.misses
        with mock.patch.object(
            replay,
            "fn",
            side_effect=AssertionError("Native table replay invoked the ordinary host"),
        ):
            for source in inputs[1:]:
                box = [source]
                outputs.append(replay(box)[0])
                self.assertEqual(box, [])
        self.assertEqual(replay.misses, misses)
        self.assertEqual(len(replay.variants), 1)

        lowered = replay.variants[0].lowered
        records = json.loads(lowered.tape.to_json())
        images, copies = records["host_buffers"], records["memcpys"]
        self.assertEqual(len(images), 2)
        self.assertEqual(len(copies), 2)
        self.assertEqual(images[0]["name"], images[1]["name"])
        self.assertNotEqual(images[0]["root"], images[1]["root"])
        self.assertEqual(
            [copy["src"] for copy in copies], [image["root"] for image in images]
        )
        self.assertEqual(
            [len(image["elements"]) for image in images], [1, 2] if rewrite else [4, 4]
        )
        self.assertEqual([source for _, source, _, _ in lowered.memcpys], [0, 1])
        first, second = lowered.host_tables
        self.assertLess(first.seq, lowered.memcpys[0][0])
        self.assertLess(lowered.memcpys[0][0], second.seq)
        self.assertLess(second.seq, lowered.memcpys[1][0])
        if rewrite:
            self.assertNotEqual(
                images[0]["elements"][0]["expr"], images[1]["elements"][0]["expr"]
            )

        inputs.clear()
        gc.collect()
        torch.cuda.synchronize(device)
        replay.close()
        for output, reference in zip(outputs, expected, strict=True):
            self.assertEqual(output, reference, atol=0, rtol=0)


class TestHostTracePinnedCopyStreams(TestCase):
    @parametrize("forked", (False, True))
    def test_pinned_copy_requires_the_origin_capture_stream(self, device, forked):
        from torch.cuda import _host_trace

        side = torch.cuda.Stream(device=device)
        table = torch.randn(16, 8, device=device)
        ids = torch.tensor([3, 1, 7, 4], dtype=torch.int64).pin_memory()
        expected = table[ids.to(device)]
        torch.cuda.synchronize(device)

        def on_side(table, ids):
            origin = torch.cuda.current_stream(device)
            if forked:
                side.wait_stream(origin)
            try:
                with torch.cuda.stream(side):
                    return gather(table, ids)
            finally:
                if forked:
                    origin.wait_stream(side)

        output = on_side(table, ids)
        side.synchronize()
        self.assertEqual(output, expected, atol=0, rtol=0)
        reason = (
            "host_trace: a copy on a side stream joined to the trace"
            if forked
            else "host_trace: a copy on a stream that is not the trace's capturing stream"
        )
        with self.assertRaisesRegex(_host_trace.Declined, reason):
            _host_trace.trace(on_side, (table, ids), warm_up=False)
        self.assertFalse(C._host_trace_tracing())
        self.assertFalse(torch.cuda.is_current_stream_capturing())
        healthy = _host_trace.trace(gather, (table, ids), warm_up=False)
        self.assertEqual(len(healthy.memcpys), 1)
        self.assertEqual(len(healthy.launches), 1)


class TestHostTraceH2DUnion(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from torch._inductor.runtime._cudagraph import direct_hosttrace

        self.module = direct_hosttrace

    def _replay(self, fn, args, **kw):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args, **kw)
        self.addCleanup(replay.close)
        return replay

    def test_pinned_source_survives_a_host_cache_flush_between_calls(self, device):
        # the exec's copy node names the caller's pinned block; a host-cache flush
        # (every torch.cuda.graph capture runs one) must not free it under the node, or
        # the next fresh block at the same address launches an unrefreshed node
        table = torch.randn(V, D, device=device)
        replay = self._replay(gather, (table, _ids(64)))
        torch._C._host_emptyCache()
        for step in range(100):
            ids = _ids(64)
            self.assertEqual(replay(table, ids), table[ids.to(device)], atol=0, rtol=0)
            del ids
            torch._C._host_emptyCache()
        self.assertEqual(replay.misses, 0)

    def test_a_second_entry_between_calls_of_the_first(self, device):
        # two entries of different functions: the second's preparation capture flushes
        # the host cache between calls of the first, whose next call updates kernel nodes
        table = torch.randn(V, D, device=device)
        replay = self._replay(gather, (table, _ids(64)))
        self.assertEqual(replay(table, _ids(64)).shape, (64, D))
        self._replay(lambda x: x * 2, (torch.randn(4096, device=device),))
        for step in range(100):
            ids = _ids(64 + step % 3)
            self.assertEqual(replay(table, ids), table[ids.to(device)], atol=0, rtol=0)
            del ids
            if step % 25 == 24:
                self._replay(lambda x: x + 1.0, (torch.randn(1024, device=device),))
        self.assertEqual(replay.misses, 0)

    def test_a_dormant_variants_pinned_input_survives_set_and_a_flush(self, device):
        # the runtime team's shape (COORDINATION 2026-09-20 16:04 UTC): a variant that
        # is not the one serving retains the pinned block its copy node names; the
        # caller's set_ moves that tensor to another storage, the active variant serves
        # a fresh pinned input and the host cache is flushed. The dormant exec's binding
        # stays valid because the owner holds the storage, not the tensor (which has
        # lost the allocation): the storage's use count keeps the hold, the flush
        # cannot free it and no fresh block takes its address; the dormant variant's
        # next call rebinds and releases it
        def use_count(storage):
            return torch._C._storage_Use_Count(storage._cdata)

        table = torch.randn(V, D, device=device)
        ids = _ids(64)
        replay = self._replay(gather, (table, ids))
        self.assertEqual(replay(table, ids), table[ids.to(device)], atol=0, rtol=0)
        half = table.half()
        other = _ids(64)
        self.assertEqual(replay(half, other), half[other.to(device)], atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (1, 2))
        storage, former = ids.untyped_storage(), ids.data_ptr()
        ids.set_(_ids(64))
        self.assertGreaterEqual(use_count(storage), 2)  # this reference and the hold
        replay.wait_for_h2d()
        torch.cuda.synchronize(device)
        torch._C._host_emptyCache()
        fresh = _ids(64)
        self.assertEqual(replay(half, fresh), half[fresh.to(device)], atol=0, rtol=0)
        torch._C._host_emptyCache()
        self.assertGreaterEqual(use_count(storage), 2)
        pins = [_ids(64) for _ in range(64)]
        self.assertNotIn(former, [t.data_ptr() for t in pins])
        del pins
        self.assertEqual(replay(table, ids), table[ids.to(device)], atol=0, rtol=0)
        self.assertEqual((replay.misses, replay.declines), (1, []))
        replay.wait_for_h2d()
        torch.cuda.synchronize(device)
        self.assertEqual(use_count(storage), 1)

    def test_set_and_resize_on_a_pinned_input_after_preparation(self, device):
        # set_ moves the input to another pinned storage: the next call is served
        # through a rebind (the facts hold: no miss, no decline) and that rebind
        # releases the former storage; a storage-growing resize_ on a pinned tensor is
        # refused by eager itself (the caching host allocator's storage is not
        # resizable), so no allocation moves under a held storage, and a shrinking
        # resize_ keeps the allocation and is served
        table = torch.randn(V, D, device=device)
        ids = _ids(64)
        replay = self._replay(gather, (table, ids))
        self.assertEqual(replay(table, ids), table[ids.to(device)], atol=0, rtol=0)
        storage = ids.untyped_storage()
        rebinds = replay.memcpy_stats[1]
        ids.set_(_ids(64))
        replay.wait_for_h2d()
        torch.cuda.synchronize(device)
        torch._C._host_emptyCache()
        self.assertEqual(replay(table, ids), table[ids.to(device)], atol=0, rtol=0)
        self.assertEqual(replay.memcpy_stats[1], rebinds + 1)
        self.assertEqual((replay.misses, replay.declines), (0, []))
        replay.wait_for_h2d()
        torch.cuda.synchronize(device)
        self.assertEqual(torch._C._storage_Use_Count(storage._cdata), 1)
        with self.assertRaisesRegex(RuntimeError, "not resizable"):
            ids.resize_(4096)
        ids.resize_(32)
        self.assertEqual(replay(table, ids), table[ids.to(device)], atol=0, rtol=0)
        self.assertEqual((replay.misses, replay.declines), (0, []))

    @parametrize(
        "form",
        (
            "clone_contiguous",
            "clone_transposed",
            "contiguous",
            "copy_row",
            "copy_column",
        ),
    )
    def test_d2d_copies_lower_to_memcpy_nodes(self, device, form):
        # cascade 10b's declared "d2d" records (a contiguous clone, a clone of a dense
        # transposed view, copy_ into a contiguous row) lower natively to memcpy nodes
        # of the record's kind; the strided forms are the copy kernel of the sibling
        forms = {
            "clone_contiguous": (lambda t, o: t.clone(), 1),
            "clone_transposed": (lambda t, o: t.t().clone(), 1),
            "contiguous": (lambda t, o: t.t().contiguous(), 0),
            "copy_row": (
                lambda t, o: o[1].copy_(t[0]),
                1,
            ),
            "copy_column": (
                lambda t, o: o[:, 1].copy_(t[:, 0]),
                0,
            ),
        }

        def args(rows, cols):
            return (
                torch.randn(rows, cols, device=device, dtype=torch.bfloat16),
                torch.randn(rows, cols, device=device, dtype=torch.bfloat16),
            )

        fn, memcpys = forms[form]
        base = args(48, 96)
        replay = self._replay(fn, base)
        self.assertEqual([m["kind"] for m in replay.tape.memcpys], ["d2d"] * memcpys)
        self.assertEqual(replay.lowered.memcpy_kinds, ("d2d",) * memcpys)
        self.assertEqual(replay.lowered.pinned_positions, ())
        for new in (args(48, 96), args(7, 130), args(200, 16)):
            t, o = new
            want = fn(t.clone(), o.clone())
            got = replay(t, o)
            self.assertEqual(got, want, atol=0, rtol=0)
            self.assertEqual(got.stride(), want.stride(), form)
        self.assertEqual(replay.misses, 0)
        if memcpys:
            self.assertGreater(replay.memcpy_stats[0], 0)

    def test_wait_for_h2d_covers_every_entry_reading_a_shared_buffer(self, device):
        # retirement stage B: two entries read the same pinned buffer; one entry's wait
        # covers the other's pending copy from that buffer, and the module-level wait
        # takes the buffer itself (the interim's wait_for_h2d(pinned))
        from torch.testing._internal.host_trace_oracle import Oracle

        table = torch.randn(V, D, device=device)
        ids = _ids(64)
        r1 = self._replay(gather, (table, ids))
        r2 = self._replay(gather, (table, ids))
        want = table[ids.to(device)]
        o1, o2 = r1(table, ids), r2(table, ids)
        r1.wait_for_h2d()
        self.assertTrue(r2._last_event.query())
        self.assertEqual((o1, o2), (want, want), atol=0, rtol=0)
        o1, o2 = r1(table, ids), r2(table, ids)
        self.module.wait_for_h2d(ids)
        self.assertTrue(r1._last_event.query() and r2._last_event.query())
        ids.copy_(torch.randint(0, V, (64,), dtype=torch.int64))
        torch.cuda.synchronize(device)
        self.assertEqual((o1, o2), (want, want), atol=0, rtol=0)
        # the same tape on both backends
        oracle = Oracle(gather, (table, ids))
        self.addCleanup(oracle.close)
        self.assertIsNone(oracle.refused)
        for n in (64, 8, 200):
            oracle.check((table, _ids(n)))


instantiate_device_type_tests(TestHostTraceH2DUnion, globals(), only_for="cuda")
instantiate_device_type_tests(TestHostTracePinnedVariants, globals(), only_for="cuda")
instantiate_device_type_tests(
    TestHostTraceTableCopyHistories, globals(), only_for="cuda"
)
instantiate_device_type_tests(
    TestHostTracePinnedCopyStreams, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
