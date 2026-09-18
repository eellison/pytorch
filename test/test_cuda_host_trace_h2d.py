# Owner(s): ["module: cuda"]

import json
import unittest

from host_trace_h2d_probe import probe
from host_trace_testing import HostTraceTestCase

import torch
from torch.testing._internal.common_utils import run_tests, skipIfRocm


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

C = torch._C


def grouped(*tensors):
    # (a0, b0, a1, b1, ...): group g multiplies a_g and b_g into one output
    return probe().grouped_mul(*tensors)


def gather(table, ids):
    return probe().gather(table, ids)


def _groups(sizes, device="cuda"):
    out = []
    for n in sizes:
        out.append(torch.randn(n, device=device))
        out.append(torch.randn(n, device=device))
    return tuple(out)


def _ref_grouped(tensors):
    return torch.cat([a * b for a, b in zip(tensors[0::2], tensors[1::2])])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceH2D(HostTraceTestCase):
    V, D = 512, 64

    def _table(self):
        return torch.randn(self.V, self.D, device="cuda")

    def _ids(self, n, pinned=True):
        ids = torch.randint(0, self.V, (n,), dtype=torch.int64)
        return ids.pin_memory() if pinned else ids

    def _roundtrip(self, fn, base, news, **kw):
        tape, variant, cases = self._replay_cases(fn, base, news, **kw)
        served = [a for a, c in zip(news, cases) if c.out is not None]
        return tape, variant, served, [a for a, c in zip(news, cases) if c.out is None]

    def test_grouped_tape_records_tables_and_copies(self):
        base = _groups([256, 1000, 64])
        tape = ht.trace(grouped, base)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(tape.num_allocations, 2)  # the output and the device table
        self.assertEqual(tape.num_host_buffers, 2)  # pointers, sizes
        self.assertEqual(tape.num_memcpys, 2)
        parsed = json.loads(tape.to_json())
        ptrs, sizes = parsed["host_buffers"]
        self.assertEqual(ptrs["name"], "grouped_ptrs")
        self.assertEqual([q["kind"] for q in ptrs["elements"]], ["ptr"] * 9)
        self.assertTrue(all(not q["const"] for q in ptrs["elements"]))
        self.assertEqual([q["kind"] for q in sizes["elements"]], ["i32"] * 3)
        # the size elements are the input lengths, not numbers
        self.assertTrue(all(not q["const"] for q in sizes["elements"]))
        # each copy reads its table's root
        self.assertEqual(
            [m["src"] for m in parsed["memcpys"]], [ptrs["root"], sizes["root"]]
        )
        # both tables land at the copy's place in host order, after the writes
        self.assertLess(parsed["allocations"][-1]["seq"], ptrs["seq"])
        self.assertLess(ptrs["seq"], parsed["launches"][0]["seq"])

    def test_grouped_replays_at_new_addresses_and_sizes(self):
        base = _groups([256, 1000, 64])
        news = [
            _groups([256, 1000, 64]),  # the same sizes, other buffers
            _groups([512, 7, 4096]),
            _groups([1, 1, 1]),
            _groups([3000, 3000, 3000]),
        ]
        tape, variant, served, missed = self._roundtrip(grouped, base, news)
        self.assertEqual(len(served), 4, missed)
        # every replay moved the tables (ring slot) and the output: the copies
        # were re-pointed, the launch's image changed
        self.assertGreater(variant.exec.dirty_memcpy_nodes, 0)
        self.assertGreater(variant.exec.dirty_nodes, 0)

    def test_back_to_back_replays_without_a_host_sync(self):
        # two calls in a row, no synchronize in between: the second must not
        # overwrite a staging slot the first copy is still reading
        base = _groups([4096, 4096, 4096])
        for depth in (1, 2):
            tape = ht.trace(grouped, base)
            variant = ht.build(tape, grouped, base, staging_depth=depth)
            xs = [_groups([4096, 4096, 4096]) for _ in range(6)]
            outs = [variant.replay(x)[0] for x in xs]
            torch.cuda.synchronize()
            for x, out in zip(xs, outs):
                self.assertTrue(torch.equal(out, _ref_grouped(x)), f"depth {depth}")

    def test_gather_from_a_pinned_input(self):
        table = self._table()
        base = (table, self._ids(32))
        tape = ht.trace(gather, base)
        self.assertEqual(tape.num_host_buffers, 0)
        self.assertEqual(tape.num_memcpys, 1)
        parsed = json.loads(tape.to_json())
        self.assertEqual(parsed["inputs"][1]["device"], "cpu")
        self.assertTrue(parsed["inputs"][1]["pinned"])
        self.assertEqual(parsed["inputs"][0]["device"], "cuda")
        variant = ht.build(tape, gather, base)
        for n in (32, 1, 8, 200):
            ids = self._ids(n)
            self.assertTrue(
                torch.equal(variant.replay((table, ids))[0], table[ids.cuda()])
            )

    def test_a_copy_on_write_table_stays_lazy(self):
        # the kernel-read table is an input through the const accessor: a lazy
        # clone stays copy-on-write through an ordinary gather, the trace's
        # warm-up and the build; the gather's output goes through the mutable form.
        table = self._table()
        lazy = torch._lazy_clone(table)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        ids = self._ids(32)
        want = table[ids.cuda()]
        self.assertTrue(torch.equal(gather(lazy, ids), want))
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        base = (lazy, ids)
        tape = ht.trace(gather, base)
        variant = ht.build(tape, gather, base)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        self.assertTrue(torch.equal(variant.replay(base)[0], want))
        self.assertTrue(torch._C._is_cow_tensor(lazy))

    def test_fixed_staging_input_rewritten_in_place_has_no_source_rebinds(self):
        # pattern (a): the caller keeps one pinned buffer and rewrites it
        # between calls, after wait_for_h2d(); the copy's source never moves
        table = self._table()
        ids = self._ids(64)
        tape = ht.trace(gather, (table, ids))
        variant = ht.build(tape, gather, (table, ids))
        for _ in range(5):
            variant.wait_for_h2d()
            new = torch.randint(0, self.V, (64,), dtype=torch.int64)
            ids.copy_(new)
            out = variant.replay((table, ids))[0]
            self.assertTrue(torch.equal(out, table[new.cuda()]))
        self.assertEqual(variant.source_rebinds, 0)

    def test_fresh_pinned_input_per_call_rebinds_the_source_once(self):
        # pattern (b): a new pinned buffer every call; one source rebind per
        # call, and the previous buffer is not read again
        table = self._table()
        base = (table, self._ids(64))
        tape = ht.trace(gather, base)
        variant = ht.build(tape, gather, base)
        previous = None
        for step in range(5):
            ids = self._ids(64)
            want = table[ids.cuda()]
            out = variant.replay((table, ids))[0]
            if previous is not None:
                variant.wait_for_h2d()
                previous.fill_(self.V - 1)  # the old buffer: must not matter
            self.assertTrue(torch.equal(out, want))
            self.assertEqual(variant.source_rebinds, step + 1)
            previous = ids

    def test_pageable_source_declines_at_the_trace_and_misses_at_replay(self):
        table = self._table()
        with self.assertRaisesRegex(ht.Declined, "pageable CPU memory"):
            ht.trace(gather, (table, self._ids(16, pinned=False)))
        base = (table, self._ids(16))
        tape = ht.trace(gather, base)
        variant = ht.build(tape, gather, base)
        before = (
            variant.exec.dirty_memcpy_nodes,
            variant.exec.dirty_nodes,
            variant.calls,
        )
        with self.assertRaisesRegex(ht.Miss, "pageable"):
            variant.replay((table, self._ids(16, pinned=False)))
        # the miss came before any node was touched
        self.assertEqual(
            (variant.exec.dirty_memcpy_nodes, variant.exec.dirty_nodes, variant.calls),
            before,
        )
        with self.assertRaisesRegex(ht.Miss, "pinned CPU tensor"):
            variant.replay((table, self._ids(16).cuda()))

    def test_raw_address_in_a_table_declines(self):
        x = torch.randn(8, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "raw_ptrs.*raw address"):
            ht.trace(probe().raw_table_address, (x,))
        # the recorder is clean afterwards
        self.assertFalse(C._host_trace_tracing())

    def test_raw_memcpy_in_a_host_declines_by_node_type(self):
        x = torch.randn(8, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "produced a memcpy node"):
            ht.trace(probe().raw_memcpy, (x,))
        self.assertFalse(C._host_trace_tracing())
        # and a trace right after is fine
        base = _groups([16, 16])
        self.assertEqual(ht.trace(grouped, base).num_memcpys, 2)

    def test_two_variants_interleaved_keep_their_own_staging(self):
        base1, base2 = _groups([128, 256]), _groups([1024, 8])
        v1 = ht.build(ht.trace(grouped, base1), grouped, base1)
        v2 = ht.build(ht.trace(grouped, base2), grouped, base2)
        xs1 = [_groups([128, 256]) for _ in range(4)]
        xs2 = [_groups([1024, 8]) for _ in range(4)]
        outs = []
        for a, b in zip(xs1, xs2):
            outs.append((v1.replay(a)[0], v2.replay(b)[0]))
        torch.cuda.synchronize()
        for (a, b), (oa, ob) in zip(zip(xs1, xs2), outs):
            self.assertTrue(torch.equal(oa, _ref_grouped(a)))
            self.assertTrue(torch.equal(ob, _ref_grouped(b)))

    def test_ordinary_path(self):
        ts = _groups([100, 37, 2048])
        self.assertTrue(torch.equal(grouped(*ts), _ref_grouped(ts)))
        table, ids = self._table(), self._ids(40)
        self.assertTrue(torch.equal(gather(table, ids), table[ids.cuda()]))

    def test_copy_byte_count_is_guarded_by_the_table_capacity(self):
        # the host copies size(0) * 8 bytes from a 32-byte table: traced at
        # 2 the byte count is a guard, so 3 and 4 serve and 5..8 miss instead
        # of reaching the runtime with a range past the staging slot
        copy_count = probe().copy_count
        base = (torch.randn(2, 8, device="cuda"),)
        tape = ht.trace(copy_count, base)
        variant = ht.build(tape, copy_count, base)
        for n in (3, 4, 2):
            got = variant.replay((torch.randn(n, 8, device="cuda"),))[0]
            self.assertEqual(got.item(), n * 100 + n - 1)
        for n in (5, 6, 8):
            self.assertIsNone(variant.try_replay((torch.randn(n, 8, device="cuda"),)))
        with self.assertRaisesRegex(ht.Miss, "guard failed"):
            variant.replay((torch.randn(5, 8, device="cuda"),))
        # the same call is a typed error on the ordinary path
        with self.assertRaisesRegex(RuntimeError, "40 bytes from a 32-byte host table"):
            copy_count(torch.randn(5, 8, device="cuda"))

    def test_zero_byte_copy_declines_at_the_trace(self):
        x = torch.randn(8, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "copy of no bytes"):
            ht.trace(probe().copy_zero, (x,))
        self.assertFalse(C._host_trace_tracing())

    def test_wait_for_h2d_covers_every_variant_reading_a_shared_buffer(self):
        # two variants read the same pinned buffer; waiting on one variant
        # waits for the other's pending copy too, and the module-level wait
        # takes the buffer itself
        table = self._table()
        ids = self._ids(64)
        v1 = ht.build(ht.trace(gather, (table, ids)), gather, (table, ids))
        v2 = ht.build(ht.trace(gather, (table, ids)), gather, (table, ids))
        o1 = v1.replay((table, ids))[0]
        o2 = v2.replay((table, ids))[0]
        v1.wait_for_h2d()
        self.assertTrue(v2._last_event.query())
        want = table[ids.cuda()]
        self.assertTrue(torch.equal(o1, want) and torch.equal(o2, want))
        o1 = v1.replay((table, ids))[0]
        o2 = v2.replay((table, ids))[0]
        ht.wait_for_h2d(ids)
        self.assertTrue(v1._last_event.query() and v2._last_event.query())
        ids.copy_(torch.randint(0, self.V, (64,), dtype=torch.int64))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(o1, want) and torch.equal(o2, want))

    def test_ordinary_copy_h2d_refuses_what_the_trace_refuses(self):
        copy_into = probe().copy_into
        dst = torch.zeros(16, dtype=torch.int64, device="cuda")
        src = torch.arange(16, dtype=torch.int64).pin_memory()
        copy_into(dst, src)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(dst.cpu(), src))
        with self.assertRaisesRegex(RuntimeError, "pinned CPU tensor; got a cuda"):
            copy_into(dst, src.cuda())
        with self.assertRaisesRegex(RuntimeError, "pageable CPU memory"):
            copy_into(dst, torch.arange(16, dtype=torch.int64))
        with self.assertRaisesRegex(RuntimeError, "destination must be device memory"):
            copy_into(torch.zeros(16, dtype=torch.int64).pin_memory(), src)
        # under a trace: the warm-up call raises the ordinary error first, and
        # without warm-up the trace declines a host destination at assembly
        x = torch.randn(4, device="cuda")
        host_dst = torch.zeros(16, dtype=torch.int64).pin_memory()
        beside = probe().copy_into_beside
        with self.assertRaisesRegex(RuntimeError, "destination must be device memory"):
            ht.trace(beside, (x, host_dst, src))
        with self.assertRaisesRegex(ht.Declined, "destination must be device memory"):
            ht.trace(beside, (x, host_dst, src), warm_up=False)
        self.assertFalse(C._host_trace_tracing())

    def test_repeated_copies_of_one_table_are_one_image_each(self):
        # one table copied unchanged into two buffers: the tape holds the
        # table once per copy, each image right before its own copy in host
        # order and read by that copy alone
        copy_twice = probe().copy_twice
        base = (torch.randn(3, 8, device="cuda"),)
        tape = ht.trace(copy_twice, base)
        self.assertEqual((tape.num_host_buffers, tape.num_memcpys), (2, 2))
        parsed = json.loads(tape.to_json())
        images, copies = parsed["host_buffers"], parsed["memcpys"]
        self.assertEqual([hb["name"] for hb in images], ["twice_table"] * 2)
        self.assertNotEqual(images[0]["root"], images[1]["root"])
        self.assertEqual([m["src"] for m in copies], [hb["root"] for hb in images])
        self.assertEqual([len(hb["elements"]) for hb in images], [4, 4])
        self.assertLess(images[0]["seq"], copies[0]["seq"])
        self.assertLess(copies[0]["seq"], images[1]["seq"])
        self.assertLess(images[1]["seq"], copies[1]["seq"])
        variant = ht.build(tape, copy_twice, base)
        self.assertEqual(len(variant._rings), 2)
        for n in (3, 1, 7):
            x = torch.randn(n, 8, device="cuda")
            want = torch.tensor([n * 100 + i for i in range(4)] * 2, device="cuda")
            self.assertTrue(torch.equal(variant.replay((x,))[0], want))
            self.assertTrue(torch.equal(copy_twice(x), want))

    def test_a_rewrite_between_two_copies_keeps_the_earlier_image(self):
        # write slot 0, copy it, rewrite slot 0 and write slot 1, copy both:
        # each copy's image holds the table as that copy read it, and the
        # replay renders the two images into slots of their own
        fn = probe().copy_rewrite_copy
        base = (torch.randn(2, 8, device="cuda"),)
        tape = ht.trace(fn, base)
        first, second = json.loads(tape.to_json())["host_buffers"]
        self.assertEqual((len(first["elements"]), len(second["elements"])), (1, 2))
        self.assertNotEqual(first["elements"][0]["expr"], second["elements"][0]["expr"])
        variant = ht.build(tape, fn, base)
        for n in (2, 5, 1):
            x = torch.randn(n, 8, device="cuda")
            want = torch.tensor([n * 100, n * 200, n * 300], device="cuda")
            self.assertTrue(torch.equal(variant.replay((x,))[0], want))
            self.assertTrue(torch.equal(fn(x), want))

    def test_a_copy_on_a_forked_side_stream_declines(self):
        # the copy record has no stream field: a copy on a stream forked from
        # the capturing stream and joined back declines at the copy, and the
        # recorder is clean afterwards
        copy_on = probe().copy_on_stream
        x = torch.randn(4, 8, device="cuda")
        want = torch.tensor([400, 401], device="cuda")
        self.assertTrue(torch.equal(copy_on(x, True), want))
        with self.assertRaisesRegex(ht.Declined, "side stream joined to the trace"):
            ht.trace(lambda y: copy_on(y, True), (x,))
        self.assertFalse(C._host_trace_tracing())
        base = _groups([16, 16])
        self.assertEqual(ht.trace(grouped, base).num_memcpys, 2)

    def test_a_copy_on_a_non_capturing_stream_declines(self):
        copy_on = probe().copy_on_stream
        x = torch.randn(4, 8, device="cuda")
        self.assertTrue(
            torch.equal(copy_on(x, False), torch.tensor([400, 401], device="cuda"))
        )
        with self.assertRaisesRegex(ht.Declined, "not the trace's capturing stream"):
            ht.trace(lambda y: copy_on(y, False), (x,))
        self.assertFalse(C._host_trace_tracing())

    def test_staging_slots_are_recorded_on_the_replay_stream(self):
        base = _groups([64, 64])
        variant = ht.build(ht.trace(grouped, base), grouped, base)
        variant.replay(_groups([64, 64]))
        stream = torch.cuda.current_stream().cuda_stream
        for k, ring in enumerate(variant._rings):
            slot = ring[variant._ring_used[k]]
            # a caching-host-allocator block: the allocator defers its reuse
            self.assertTrue(C._host_trace_record_host_event(slot, stream))
        torch.cuda.synchronize()

    # ---- device-to-device copies (copy_d2d): clone and the contiguous copy_

    def test_clone_is_a_memcpy_record(self):
        x = torch.randn(64, 256, device="cuda")

        def clone(t):
            return t.clone()

        tape = ht.trace(clone, (x,))
        self.assertEqual(
            (tape.num_launches, tape.num_memcpys, tape.num_allocations), (0, 1, 1)
        )
        parsed = json.loads(tape.to_json())
        m = parsed["memcpys"][0]
        # source over the input's root, destination over the allocation's,
        # the byte count an expression of the sizes
        self.assertIn(parsed["inputs"][0]["root"], m["src"])
        self.assertIn(parsed["allocations"][0]["root"], m["dst"])
        self.assertIsInstance(m["bytes"], str)
        variant = ht.build(tape, clone, (x,))
        self.assertEqual(variant.exec.num_memcpy_nodes, 1)
        self.assertEqual(variant.exec.num_nodes, 0)
        for t in (
            x,
            torch.randn(64, 256, device="cuda"),
            torch.randn(3, 5, device="cuda"),
            torch.randn(1000, 7, device="cuda"),
        ):
            (got,) = variant.replay((t,))
            self.assertTrue(torch.equal(got, t))
            self.assertNotEqual(got.data_ptr(), t.data_ptr())
        self.assertGreater(variant.exec.dirty_memcpy_nodes, 0)
        # a device-to-device copy holds no host source: no event bookkeeping
        self.assertFalse(variant._host_copies)
        self.assertEqual(len(variant._held), 0)
        # the ordinary path of the sibling is the same memcpy
        y = torch.empty_like(x)
        C._host_trace_ti_copy_(y, x)
        self.assertTrue(torch.equal(y, x))

    def test_d2d_copy_forms(self):
        # which copies are a memcpy node and which a kernel, as in Copy.cu's
        # copy_device_to_device: a contiguous pair of one dtype (a contiguous
        # clone, a clone of a dense transposed view under preserve_format,
        # copy_ into a contiguous row) is the memcpy; a strided pair is the
        # copy kernel of the sibling
        forms = {
            "clone contiguous": (lambda t, o: t.clone(), (0, 1)),
            "clone transposed (preserve_format)": (lambda t, o: t.t().clone(), (0, 1)),
            "contiguous() of a transposed view": (
                lambda t, o: t.t().contiguous(),
                (1, 0),
            ),
            "clone of a slice-step view": (lambda t, o: t[:, ::2].clone(), (1, 0)),
            "copy_ into a contiguous row of an input": (
                lambda t, o: o[1].copy_(t[0]),
                (0, 1),
            ),
            "copy_ into a strided column of an input": (
                lambda t, o: o[:, 1].copy_(t[:, 0]),
                (1, 0),
            ),
        }

        def args(rows, cols):
            return (
                torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16),
                torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16),
            )

        for name, (fn, (launches, memcpys)) in forms.items():
            with self.subTest(form=name):
                base = args(48, 96)
                tape = ht.trace(fn, base)
                self.assertEqual(
                    (tape.num_launches, tape.num_memcpys), (launches, memcpys)
                )
                variant = ht.build(tape, fn, base)
                for new in (args(48, 96), args(7, 130), args(200, 16)):
                    t, o = new
                    want = fn(t.clone(), o.clone())
                    (got,) = variant.replay(new)
                    self.assertTrue(torch.equal(got, want), name)
                    self.assertEqual(got.stride(), want.stride(), name)

    def test_d2d_copy_declines_by_name(self):
        # a copy between a traced tensor and one the trace does not own: the
        # sibling's device check, before any record
        held = torch.randn(8, 8, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "did not create"):
            ht.trace(lambda t: held.copy_(t), (torch.randn(8, 8, device="cuda"),))

    def test_memcpy_records_declare_their_kind(self):
        # the record says which copy it is (declared, not derived from its
        # addresses): the recording site sets kind, the JSON carries it, and
        # the build pairs the record with a memcpy node of that kind
        table = self._table()
        tape = ht.trace(gather, (table, self._ids(32)))
        self.assertEqual([m["kind"] for m in tape.memcpys], ["h2d"])
        (m,) = json.loads(tape.to_json())["memcpys"]
        self.assertEqual(m["kind"], "h2d")

        def clone(t):
            return t.clone()

        tape = ht.trace(clone, (table,))
        (m,) = json.loads(tape.to_json())["memcpys"]
        self.assertEqual(m["kind"], "d2d")
        variant = ht.build(tape, clone, (table,))
        self.assertEqual(variant.exec.memcpy_kind(0), "d2d")

    def test_d2d_memcpy_between_overlapping_views_declines_and_is_guarded(self):
        # copy_device_to_device's memcpy branch sits behind the iterator's
        # overlap check: a contiguous pair over one storage whose ranges
        # intersect is what eager refuses, and a memcpy over such ranges is
        # undefined. The sibling declines by name at the trace (with or
        # without the warm-up); a tape built on separate inputs keeps the two
        # apart with an address guard, so a replay on overlapping views
        # misses instead of issuing the memcpy
        message = "some elements of the input tensor and the written-to tensor"

        def copy_(a, b):
            a.copy_(b)
            return a

        def flat():
            f = torch.randn(8 * 64 + 1, device="cuda")
            return f[:-1], f[1:]

        def rows():
            x = torch.randn(9, 64, device="cuda")
            return x[:-1], x[1:]

        for overlapping in (flat, rows):
            with self.assertRaisesRegex(RuntimeError, message):
                copy_(*overlapping())
            with self.assertRaisesRegex(ht.Declined, message):
                ht.trace(copy_, overlapping(), warm_up=False)
            with self.assertRaisesRegex(RuntimeError, message):
                ht.trace(copy_, overlapping())
            self.assertFalse(C._host_trace_tracing())
        for overlapping, separate in (
            (
                flat,
                lambda: (
                    torch.randn(512, device="cuda"),
                    torch.randn(512, device="cuda"),
                ),
            ),
            (
                rows,
                lambda: (
                    torch.randn(8, 64, device="cuda"),
                    torch.randn(8, 64, device="cuda"),
                ),
            ),
        ):
            base = separate()
            tape = ht.trace(copy_, base)
            self.assertEqual((tape.num_launches, tape.num_memcpys), (0, 1))
            variant = ht.build(tape, copy_, base)
            for new in (separate(), separate()):
                a, b = new
                (got,) = variant.replay(new)
                self.assertTrue(torch.equal(got, b))
                self.assertTrue(torch.equal(a, b))
            with self.assertRaises(ht.Miss):
                variant.replay(overlapping())
            self.assertIsNone(variant.try_replay(overlapping()))

        # the same refusal for two views of one input (the recorder alone)
        def shifted(x):
            x.view(-1)[:-1].copy_(x.view(-1)[1:])
            return x

        with self.assertRaisesRegex(ht.Declined, message):
            ht.trace(shifted, (torch.randn(8, 64, device="cuda"),), warm_up=False)
        self.assertFalse(C._host_trace_tracing())

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


if __name__ == "__main__":
    run_tests()
