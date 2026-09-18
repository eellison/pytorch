# Owner(s): ["module: cuda"]

import json
import re
import unittest

from host_trace_h2d_probe import probe
from host_trace_testing import assert_eager_function_handles, bits, HostTraceTestCase

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfRocm,
    TEST_CUDA_PYTHON_BINDINGS,
)


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

C = torch._C

# the kernel template up to its type arguments: the sibling launches the same
# instantiation, so the whole mangled name must agree, except that a lambda or
# a proxy never appears in these kernels' signatures
_FAMILY = re.compile(
    r"^(_ZN2at6native\d+(?:indexSelectSmallIndex|vectorized_gather_kernel|CatArrayBatchedCopy\w*)I[^E]*)"
)


def _family(name):
    m = _FAMILY.match(name)
    return m.group(1) if m else name


def _one(r):
    # a replay returns the outputs as a list; the functions here return one tensor
    return r[0] if isinstance(r, (list, tuple)) else r


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceEmbedCat(HostTraceTestCase):
    def _same_launch(self, real, ours, ranges=()):
        # the same kernel instantiation with the same launch configuration; the
        # argument image compared on the byte ranges the host writes (the rest
        # of a by-value block is uninitialized in the real op)
        self.assertEqual(len(ours), len(real))
        for (name_r, grid_r, block_r, smem_r, image_r), (
            name_o,
            grid_o,
            block_o,
            smem_o,
            image_o,
        ) in zip(real, ours):
            self.assertEqual(_family(name_o), _family(name_r))
            self.assertEqual((grid_o, block_o, smem_o), (grid_r, block_r, smem_r))
            self.assertEqual(len(image_o), len(image_r))
            for lo, hi in ranges:
                self.assertEqual(image_o[lo:hi], image_r[lo:hi], f"bytes {lo}:{hi}")

    # ---- index_select

    # TensorInfo<T, unsigned>: data 8, sizes 25 x 4, strides 25 x 4, dims 4, padding 4 (216)
    _TI = 216

    def _index_select_ranges(self, dims_out, dims_self, dims_idx, n_idx):
        # the three TensorInfo blocks at 0 / 216 / 432, then dstSelectDim (int),
        # srcSelectDim (int), innerSize (unsigned), srcSelectDimSize (int64)
        ranges = []
        for base, dims in (
            (0, dims_out),
            (self._TI, dims_self),
            (2 * self._TI, dims_idx),
        ):
            # sizes[0:dims], strides[0:dims], dims; the data pointer of the
            # output differs per capture, the inputs' agree
            ranges.append((base + 8, base + 8 + 4 * dims))
            ranges.append((base + 108, base + 108 + 4 * dims))
            ranges.append((base + 208, base + 212))
        ranges.append((self._TI + 0, self._TI + 8))  # self's pointer
        ranges.append((2 * self._TI + 0, 2 * self._TI + 8))  # the ids' pointer
        tail = 3 * self._TI
        ranges.append((tail, tail + 8))  # dstSelectDim, srcSelectDim
        ranges.append((tail + 8, tail + 12))  # innerSize
        ranges.append((tail + 16, tail + 24))  # srcSelectDimSize
        return ranges

    def test_index_select_parity_with_the_real_op(self):
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for idx_dtype in (torch.int64, torch.int32):
                for V, Dd in ((5, 8), (100, 64), (1000, 100), (300, 128), (64, 4096)):
                    for B in (1, 2, 3, 8, 15, 16):
                        w = torch.randn(V, Dd, device="cuda", dtype=dtype)
                        ids = torch.randint(0, V, (B,), device="cuda", dtype=idx_dtype)
                        want = torch.index_select(w, 0, ids)
                        got = C._host_trace_ti_index_select(w, 0, ids)
                        torch.cuda.synchronize()
                        self.assertEqual(got.shape, want.shape)
                        self.assertEqual(got.stride(), want.stride())
                        self.assertTrue(torch.equal(bits(got), bits(want)))
                        real, _ = self._capture(lambda: torch.index_select(w, 0, ids))
                        ours, _ = self._capture(
                            lambda: C._host_trace_ti_index_select(w, 0, ids)
                        )
                        # a 2-D contiguous weight collapses to dims 2 (the selected
                        # dim preserved), a 1-D contiguous ids to 1
                        self._same_launch(
                            real, ours, self._index_select_ranges(2, 2, 1, B)
                        )

    def test_index_select_extra_dtypes_parity(self):
        # the real op's whole dtype list: int8/uint8/int16, complex, float8 on both routes
        torch.manual_seed(0)
        for dtype in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.complex64,
            torch.complex128,
            torch.float8_e4m3fn,
        ):
            if dtype.is_complex:
                table = torch.randn(200, 16, device="cuda", dtype=dtype)
            elif dtype.is_floating_point:
                table = torch.randn(200, 16, device="cuda").to(dtype)
            else:
                table = torch.randint(-100, 100, (200, 16), device="cuda").to(dtype)
            for n_idx in (8, 32):
                ids = torch.randint(0, 200, (n_idx,), device="cuda")
                want = torch.index_select(table, 0, ids)
                got = C._host_trace_ti_index_select(table, 0, ids)
                torch.cuda.synchronize()
                self.assertEqual(got.dtype, dtype)
                self.assertTrue(torch.equal(bits(got), bits(want)), f"{dtype} {n_idx}")
                real, _ = self._capture(lambda: torch.index_select(table, 0, ids))
                ours, _ = self._capture(
                    lambda: C._host_trace_ti_index_select(table, 0, ids)
                )
                self._same_launch(real, ours)

    def test_index_select_dim_1_and_3d_parity(self):
        torch.manual_seed(0)
        w = torch.randn(32, 40, 24, device="cuda", dtype=torch.float32)
        for dim, B in ((1, 4), (2, 3), (0, 8)):
            ids = torch.randint(0, w.shape[dim], (B,), device="cuda")
            want = torch.index_select(w, dim, ids)
            got = C._host_trace_ti_index_select(w, dim, ids)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(got, want))
            real, _ = self._capture(lambda: torch.index_select(w, dim, ids))
            ours, _ = self._capture(lambda: C._host_trace_ti_index_select(w, dim, ids))
            self._same_launch(real, ours)

    def test_index_select_large_route_parity(self):
        # more than 16 indices: at::gather_out's vectorized fast path (dim 0,
        # 16-byte slices); the same kernel, grid and block
        torch.manual_seed(0)
        for dtype, Dd in (
            (torch.bfloat16, 64),
            (torch.float32, 128),
            (torch.float16, 4096),
        ):
            w = torch.randn(1000, Dd, device="cuda", dtype=dtype)
            for B in (17, 32, 64, 257):
                ids = torch.randint(0, 1000, (B,), device="cuda")
                want = torch.index_select(w, 0, ids)
                got = C._host_trace_ti_index_select(w, 0, ids)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(bits(got), bits(want)))
                real, _ = self._capture(lambda: torch.index_select(w, 0, ids))
                ours, _ = self._capture(
                    lambda: C._host_trace_ti_index_select(w, 0, ids)
                )
                # out, inp, idx, num_ind, slice_size, ind_dim_size, inp_stride, out_stride, allow_neg
                self._same_launch(real, ours, [(8, 24), (24, 28), (32, 64), (64, 65)])

    def test_index_select_large_route_declines_where_the_generic_kernel_runs(self):
        w = torch.randn(
            1000, 100, device="cuda", dtype=torch.bfloat16
        )  # 200-byte rows: not 16-aligned
        ids = torch.randint(0, 1000, (32,), device="cuda")
        with self.assertRaisesRegex(ht.Declined, "generic kernel"):
            C._host_trace_ti_index_select(w, 0, ids)
        w2 = torch.randn(1000, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ht.Declined, "generic kernel"):
            C._host_trace_ti_index_select(
                w2, 1, torch.randint(0, 64, (32,), device="cuda")
            )

    def _roundtrip(self, fn, base_args, new_args_list):
        tape, variant, cases = self._replay_cases(
            fn,
            base_args,
            new_args_list,
            msg=lambda args: f"replay differs at {args[1].shape}",
        )
        return tape, variant, [True if c.out is not None else None for c in cases]

    def test_index_select_replays_at_new_ids_shapes(self):
        torch.manual_seed(0)
        w = torch.randn(1000, 64, device="cuda", dtype=torch.bfloat16)

        def fn(w, ids):
            return torch.index_select(w, 0, ids)

        base = (w, torch.randint(0, 1000, (8,), device="cuda"))
        news = [
            (w, torch.randint(0, 1000, (b,), device="cuda")) for b in (8, 3, 16, 1, 5)
        ]
        tape, variant, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served[:3], [True, True, True])
        # ids of 1: collapse_dims folds the size-1 dim, a different launch: a named miss
        self.assertIsNone(served[3])
        self.assertTrue(served[4])
        # a weight with another row count: the row count is data to the kernel
        # (srcSelectDimSize) and its size is a value, served
        w2 = torch.randn(500, 64, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(
            variant.try_replay((w2, torch.randint(0, 500, (8,), device="cuda")))
            is not None
        )
        # more than 16 ids: the route selector misses by name
        with self.assertRaisesRegex(ht.Miss, "16"):
            variant.replay((w, torch.randint(0, 1000, (32,), device="cuda")))

    def test_embedding_replays_for_1d_and_2d_ids(self):
        torch.manual_seed(0)
        table = torch.randn(1000, 128, device="cuda", dtype=torch.bfloat16)

        def fn(table, ids):
            return F.embedding(ids, table)

        base = (table, torch.randint(0, 1000, (4,), device="cuda"))
        news = [
            (table, torch.randint(0, 1000, (b,), device="cuda")) for b in (4, 8, 16, 2)
        ]
        tape, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served, [True] * 4)
        base2 = (table, torch.randint(0, 1000, (2, 4), device="cuda"))
        news2 = [
            (table, torch.randint(0, 1000, s, device="cuda"))
            for s in ((2, 4), (4, 4), (1, 8), (3, 5))
        ]
        tape2, _, served2 = self._roundtrip(fn, base2, news2)
        self.assertEqual(tape2.num_launches, 1)
        self.assertEqual(served2, [True] * 4)

    def test_embedding_from_pinned_ids_through_h2d(self):
        # the decode use: ids arrive in pinned memory, cross through the H2D
        # path into a device ids tensor, then index the table
        torch.manual_seed(0)
        table = torch.randn(1000, 64, device="cuda", dtype=torch.bfloat16)

        def fn(table, ids):
            dev = torch.empty(ids.shape[0], dtype=torch.int64, device="cuda")
            probe().copy_into(dev, ids)
            return F.embedding(dev, table)

        def pinned(b):
            return torch.randint(0, 1000, (b,), dtype=torch.int64).pin_memory()

        base = (table, pinned(4))
        tape = ht.trace(fn, base)
        variant = ht.build(tape, fn, base)
        self.assertEqual(tape.num_launches, 1)
        for b in (4, 8, 3, 16):
            ids = pinned(b)
            got = variant.try_replay((table, ids))
            self.assertIsNotNone(got)
            got = _one(got)
            want = table[ids.cuda()]
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_embedding_declines_by_name(self):
        table = torch.randn(100, 32, device="cuda")
        ids = torch.randint(0, 100, (4,), device="cuda")
        # max_norm renorms the table in place first: an unconverted op
        with self.assertRaises(ht.Declined):
            ht.trace(lambda t, i: F.embedding(i, t, max_norm=1.0), (table, ids))
        with self.assertRaises(RuntimeError):
            ht.trace(lambda t, i: F.embedding(i.float(), t), (table, ids))

    # ---- cat

    # CatArrInputTensorMetadata<T, unsigned, 128, 1>: input 128 x 8, offset / dimSize /
    # nElements 128 x 4, isContiguous 128, tensorStride 32 (2720); after the output
    # pointer (8) come the metadata (2720), os (32), concatDim (4), the last unsigned (4)
    def _cat_ranges(self, n, ndims_os):
        m = 8
        ranges = [
            (m, m + 8 * n),
            (m + 1024, m + 1024 + 4 * n),
            (m + 1536, m + 1536 + 4 * n),
            (m + 2048, m + 2048 + 4 * n),
            (m + 2560, m + 2560 + n),
        ]
        os_ = m + 2720
        ranges.append((os_, os_ + 4 * ndims_os))
        ranges.append((os_ + 16, os_ + 16 + 4 * ndims_os))
        ranges.append((os_ + 32, os_ + 40))
        return ranges

    def test_cat_parity_with_the_real_op(self):
        torch.manual_seed(0)
        cases = []
        for dtype in (
            torch.bfloat16,
            torch.float32,
            torch.int64,
            torch.float16,
            torch.bool,
            torch.int8,
        ):
            for n in (2, 3, 4, 8):
                for widths in ((32, 32), (3, 4, 8, 1), (16, 17, 5), (64, 64, 64, 64)):
                    cases.append((dtype, n, widths))
        for dtype, n, widths in cases:
            for dim, shape_of in (
                (1, lambda w: (6, w)),
                (0, lambda w: (w, 24)),
                (-1, lambda w: (2, 3, w)),
                (2, lambda w: (2, 3, w, 4)),
            ):
                tensors = [
                    (
                        torch.randn(shape_of(widths[i % len(widths)]), device="cuda")
                        * 4
                    ).to(dtype)
                    for i in range(n)
                ]
                want = torch.cat(tensors, dim)
                got = C._host_trace_ti_cat(tensors, dim)
                torch.cuda.synchronize()
                self.assertEqual(got.shape, want.shape)
                self.assertTrue(
                    torch.equal(bits(got), bits(want)),
                    f"{dtype} {n} {widths} dim {dim}",
                )
                real, _ = self._capture(lambda: torch.cat(tensors, dim))
                ours, _ = self._capture(lambda: C._host_trace_ti_cat(tensors, dim))
                nd = (
                    want.dim()
                    if "vectorized" not in real[0][0]
                    else (dim % want.dim()) + 1
                )
                self._same_launch(real, ours, self._cat_ranges(n, nd))

    def test_cat_more_inputs_than_one_batch(self):
        torch.manual_seed(0)
        tensors = [
            torch.randn(4, 8, device="cuda", dtype=torch.bfloat16) for _ in range(130)
        ]
        want = torch.cat(tensors, 1)
        got = C._host_trace_ti_cat(tensors, 1)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(bits(got), bits(want)))
        real, _ = self._capture(lambda: torch.cat(tensors, 1))
        ours, _ = self._capture(lambda: C._host_trace_ti_cat(tensors, 1))
        self.assertEqual(len(real), 2)
        self._same_launch(real, ours, self._cat_ranges(128, 2))

    def test_cat_fallback_paths_match(self):
        # a single input is a plain copy in the real op (a D2D memcpy node) and in
        # the sibling (the traced copy); the legacy 1-D empty input is skipped
        torch.manual_seed(0)
        a = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        for tensors, dim in (
            ([a], 0),
            ([a, torch.empty(0, device="cuda", dtype=torch.bfloat16)], 0),
        ):
            want = torch.cat(tensors, dim)
            got = C._host_trace_ti_cat(tensors, dim)
            torch.cuda.synchronize()
            self.assertEqual(got.shape, want.shape)
            self.assertTrue(torch.equal(bits(got), bits(want)))

    # CatArrInputTensorMetadata<T, unsigned, 64, 64>, the non-contiguous batched kernel's
    # block: input 64 x 8, offset / dimSize / nElements 64 x 4, isContiguous 64, then 64
    # TensorSizeStride entries of 32 (3392); after the output pointer (8) come the
    # metadata (3392), os (32), concatDim (4), the last unsigned (4)
    def _cat_ranges_strided(self, n, ndims, skip=()):
        m = 8
        ranges = [
            (m, m + 8 * n),
            (m + 512, m + 512 + 4 * n),
            (m + 768, m + 768 + 4 * n),
            (m + 1024, m + 1024 + 4 * n),
            (m + 1280, m + 1280 + n),
        ]
        for k in range(n):
            if k in skip:
                # a skipped legacy empty: the real host reads past its arrays there
                continue
            ss = m + 1344 + 32 * k
            ranges.append((ss, ss + 4 * ndims))
            ranges.append((ss + 16, ss + 16 + 4 * ndims))
        os_ = m + 3392
        ranges.append((os_, os_ + 4 * ndims))
        ranges.append((os_ + 16, os_ + 16 + 4 * ndims))
        ranges.append((os_ + 32, os_ + 40))
        return ranges

    def test_cat_non_contiguous_inputs_take_the_batched_kernel(self):
        # cat_out_cuda's second branch: a non-contiguous input (or an expanded one)
        # sends the whole list through the batched kernel that carries per-input
        # strides, one launch per 64 inputs; the sibling launches the same
        # instantiation with the same block
        torch.manual_seed(0)
        a = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16).t()
        for tensors, dim in (
            ([a, b], 1),
            ([a[:, ::2], a], 1),
            ([a[:1].expand(4, 16), a], 0),
            ([b, a, b], 1),
        ):
            want = torch.cat(tensors, dim)
            got = C._host_trace_ti_cat(tensors, dim)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(bits(got), bits(want)))
            real, _ = self._capture(lambda: torch.cat(tensors, dim))
            ours, _ = self._capture(lambda: C._host_trace_ti_cat(tensors, dim))
            self.assertEqual(len(real), 1)
            self.assertRegex(real[0][0], r"\d+CatArrayBatchedCopyI")
            self._same_launch(
                real, ours, self._cat_ranges_strided(len(tensors), want.dim())
            )

    def test_cat_legacy_empty_takes_the_non_contiguous_batched_kernel(self):
        # TensorShape.cpp clears all_contiguous for a skipped legacy 1-D empty
        # tensor, so eager launches the non-contiguous batched kernel with the
        # empty in the batch (dimSize 0); 65 and 129 inputs plus the empty split
        # into two and three launches
        torch.manual_seed(0)
        e = torch.empty(0, device="cuda", dtype=torch.bfloat16)
        for n, launches in ((2, 1), (65, 2), (129, 3)):
            tensors = [
                torch.randn(4, 8, device="cuda", dtype=torch.bfloat16) for _ in range(n)
            ]
            tensors.insert(1, e)
            want = torch.cat(tensors, 1)
            got = C._host_trace_ti_cat(tensors, 1)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(bits(got), bits(want)))
            real, _ = self._capture(lambda: torch.cat(tensors, 1))
            ours, _ = self._capture(lambda: C._host_trace_ti_cat(tensors, 1))
            self.assertEqual(len(real), launches)
            self.assertRegex(real[0][0], r"\d+CatArrayBatchedCopyI")
            self._same_launch(
                real, ours, self._cat_ranges_strided(min(n + 1, 64), 2, skip=(1,))
            )

    def test_cat_traced_non_contiguous_and_legacy_empty_replay(self):
        torch.manual_seed(0)

        def fn(x, y):
            return torch.cat([x, y.t()], 1)

        def fn_empty(x, y):
            return torch.cat([x, torch.empty(0, device=x.device, dtype=x.dtype), y], 0)

        def pair(m, a, b):
            return (
                torch.randn(m, a, device="cuda", dtype=torch.bfloat16),
                torch.randn(b, m, device="cuda", dtype=torch.bfloat16),
            )

        _, _, served = self._roundtrip(
            fn, pair(4, 16, 8), [pair(4, 16, 8), pair(6, 32, 8), pair(4, 8, 24)]
        )
        self.assertEqual(served, [True, True, True])
        a, b = (
            torch.randn(4, 8, device="cuda", dtype=torch.bfloat16),
            torch.randn(3, 8, device="cuda", dtype=torch.bfloat16),
        )
        _, _, served = self._roundtrip(
            fn_empty,
            (a, b),
            [
                (a, b),
                (a[:2], b),
                (torch.randn(7, 8, device="cuda", dtype=torch.bfloat16), b),
            ],
        )
        self.assertEqual(served, [True, True, True])

    def test_cat_channels_last_ambiguous_declines_by_name(self):
        # (N, C, 1, 1) in channels_last is also default-contiguous; the meta still
        # allocates the output channels-last (suggest_memory_format), a layout the
        # sibling's batched path does not produce: it declines instead of returning
        # contiguous strides
        amb = torch.randn(2, 3, 1, 1, device="cuda").to(
            memory_format=torch.channels_last
        )
        amb2 = torch.randn(2, 5, 1, 1, device="cuda").to(
            memory_format=torch.channels_last
        )
        want = torch.cat([amb, amb2], 1)
        self.assertEqual(want.stride(), (8, 1, 8, 8))
        with self.assertRaisesRegex(ht.Declined, "channels-last"):
            C._host_trace_ti_cat([amb, amb2], 1)
        with self.assertRaisesRegex(ht.Declined, "channels-last"):
            ht.trace(lambda x, y: torch.cat([x, y], 1), (amb, amb2))
        cl3 = torch.randn(2, 3, 1, 1, 1, device="cuda").to(
            memory_format=torch.channels_last_3d
        )
        with self.assertRaisesRegex(ht.Declined, "channels-last"):
            C._host_trace_ti_cat([cl3, cl3.clone()], 1)
        # one channels-last and one contiguous input: the meta picks contiguous, served
        cl = torch.randn(2, 8, 4, 4, device="cuda").to(
            memory_format=torch.channels_last
        )
        con = torch.randn(2, 8, 4, 4, device="cuda")
        want = torch.cat([cl, con], 1)
        got = C._host_trace_ti_cat([cl, con], 1)
        torch.cuda.synchronize()
        self.assertEqual(got.stride(), want.stride())
        self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_cat_replays_with_symbolic_per_input_sizes(self):
        torch.manual_seed(0)

        def fn(x, y):
            return torch.cat([x, y], -1)

        def pair(m, a, b):
            return (
                torch.randn(m, a, device="cuda", dtype=torch.bfloat16),
                torch.randn(m, b, device="cuda", dtype=torch.bfloat16),
            )

        base = pair(8, 32, 32)
        news = [
            pair(8, 32, 32),
            pair(16, 32, 32),
            pair(4, 64, 64),
            pair(8, 48, 16),
            pair(3, 8, 8),
            pair(3, 6, 6),
        ]
        tape, variant, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served[:5], [True] * 5)
        # 12-byte slices are not 16-byte multiples: the host selects another kernel, a named miss
        self.assertIsNone(served[5])
        # a third input changes the launch (grid.y, the slots written): a contract miss
        with self.assertRaises(ht.Miss):
            variant.replay(pair(8, 32, 32) + (base[0],))

    def test_rotate_half_traces_and_replays(self):
        # the rotary pattern: cat([-x2, x1], -1) with x1 / x2 halves of the
        # head dim; x1 is a non-contiguous view, so the sibling takes the
        # copy fallback for it
        torch.manual_seed(0)
        neg_one = torch.tensor(-1.0, device="cuda", dtype=torch.bfloat16)

        def rotate_half(x, neg_one):
            h = x.shape[-1] // 2
            return torch.cat([x[..., h:] * neg_one, x[..., :h]], -1)

        def make(b, dh):
            return (
                torch.randn(b, 8, 1, dh, device="cuda", dtype=torch.bfloat16),
                neg_one,
            )

        base = make(4, 64)
        news = [make(4, 64), make(8, 64), make(1, 64), make(2, 128)]
        tape, variant, served = self._roundtrip(rotate_half, base, news)
        self.assertEqual(served[:2], [True, True])
        self.assertIn(served[2], (None, True))
        self.assertIn(served[3], (None, True))

    def test_cat_of_a_split_head_view_builds(self):
        # the HF GPT-2 idiom: a piece of the qkv row viewed (B, 1, H, DH) and
        # permuted, then concatenated onto the cache. The view's size-1 dim
        # takes computeStride's stride (the piece's row length), as eager's
        # does, so cat's image, which carries the stride table, matches at
        # the build's byte check
        B, H, L, DH = 4, 12, 15, 64
        past = torch.randn(B, H, L, DH, device="cuda", dtype=torch.bfloat16)
        row = torch.randn(B, 1, 3 * H * DH, device="cuda", dtype=torch.bfloat16)

        def key_of(r):
            return r.narrow(2, H * DH, H * DH).view(B, 1, H, DH).permute(0, 2, 1, 3)

        def cat_inside(p, r):
            return torch.cat((p, key_of(r)), -2)

        want = cat_inside(past, row)
        tape = ht.trace(cat_inside, (past, row))
        self.assertEqual(tape.num_launches, 1)
        variant = ht.build(tape, cat_inside, (past, row))
        got = _one(variant.replay((past, row)))
        self.assertEqual(got.stride(), want.stride())
        self.assertTrue(torch.equal(bits(got), bits(want)))
        longer = torch.randn(B, H, 31, DH, device="cuda", dtype=torch.bfloat16)
        row2 = torch.randn_like(row)
        got = _one(variant.replay((longer, row2)))
        self.assertTrue(torch.equal(bits(got), bits(cat_inside(longer, row2))))

    def test_cat_declines_by_name(self):
        a = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        with self.assertRaisesRegex(ht.Declined, "mixed dtypes"):
            ht.trace(lambda x, y: torch.cat([x, y], 1), (a, b))
        cl = torch.randn(2, 8, 4, 4, device="cuda").to(
            memory_format=torch.channels_last
        )
        with self.assertRaisesRegex(ht.Declined, "channels-last"):
            ht.trace(lambda x, y: torch.cat([x, y], 1), (cl, cl.clone()))

    def test_copy_on_write_inputs_stay_lazy(self):
        # the embedding weight, the ids and cat's inputs are read through the
        # const accessor: lazy clones stay copy-on-write through the ordinary
        # ops, the trace's warm-up and the build; outputs go through the
        # mutable form.
        table = torch.randn(1000, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.randint(0, 1000, (4,), device="cuda")
        lazy_table = torch._lazy_clone(table)
        self.assertTrue(torch._C._is_cow_tensor(lazy_table))
        want = F.embedding(ids, table)
        self.assertTrue(torch.equal(F.embedding(ids, lazy_table), want))
        self.assertTrue(torch._C._is_cow_tensor(lazy_table))

        def emb(t, i):
            return F.embedding(i, t)

        args = (lazy_table, ids)
        variant = ht.build(ht.trace(emb, args), emb, args)
        self.assertTrue(torch._C._is_cow_tensor(lazy_table))
        self.assertTrue(torch.equal(variant.replay(args)[0], want))
        self.assertTrue(torch._C._is_cow_tensor(lazy_table))

        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        lazy_x = torch._lazy_clone(x)
        want_cat = torch.cat([x, x], -1)
        self.assertTrue(torch.equal(torch.cat([lazy_x, lazy_x], -1), want_cat))
        self.assertTrue(torch._C._is_cow_tensor(lazy_x))

        def cat2(a):
            return torch.cat([a, a], -1)

        variant = ht.build(ht.trace(cat2, (lazy_x,)), cat2, (lazy_x,))
        self.assertTrue(torch.equal(variant.replay((lazy_x,))[0], want_cat))
        self.assertTrue(torch._C._is_cow_tensor(lazy_x))

    def test_ordinary_ops_are_untouched(self):
        w = torch.randn(100, 64, device="cuda")
        ids = torch.randint(0, 100, (4,), device="cuda")
        nodes, _ = self._capture(lambda: torch.index_select(w, 0, ids))
        self.assertEqual(len(nodes), 1)
        self.assertIn("indexSelectSmallIndex", nodes[0][0])
        nodes, _ = self._capture(lambda: torch.cat([w, w], 1))
        self.assertEqual(len(nodes), 1)
        self.assertIn("CatArrayBatchedCopy", nodes[0][0])

    # ---- index_copy_ / index_put_ (the StaticCache writes)

    def _cache_args(self, b, pos, n=1, heads=4, length=64, dim=32):
        return (
            torch.randn(b, heads, length, dim, device="cuda", dtype=torch.bfloat16),
            torch.arange(pos, pos + n, device="cuda"),
            torch.randn(b, heads, n, dim, device="cuda", dtype=torch.bfloat16),
        )

    def _check_write(self, variant, fn, args, what):
        cache, pos, k = args
        want = fn(cache.clone(), pos, k)
        (got,) = variant.replay(args)
        self.assertTrue(torch.equal(got, want), what)
        self.assertTrue(torch.equal(cache, want), what)  # written in place

    def test_index_copy_into_a_cache_slice_replays(self):
        # StaticCache.update: k_out.index_copy_(2, cache_position, key_states),
        # the position a value. The count of positions is a symbol too, but a
        # count of one is the iterator's broadcast case (a size-1 dim, stride
        # 0: a guard), so a decode trace serves decode and a prefill trace
        # serves prefill
        def write(cache, pos, k):
            return cache.index_copy_(2, pos, k)

        base = self._cache_args(2, 5)
        tape = ht.trace(write, base)
        self.assertEqual((tape.num_launches, tape.num_memcpys), (1, 0))
        variant = ht.build(tape, write, base)
        # (batch 1 is the same broadcast question on the batch dim: a trace at
        # batch 2 does not serve it, as for every sibling-iterator op)
        for args in (
            self._cache_args(2, 6),
            self._cache_args(3, 17),
            self._cache_args(4, 63),
        ):
            self._check_write(variant, write, args, f"pos {args[1].tolist()}")
        with self.assertRaisesRegex(ht.Miss, "== 1"):
            variant.replay(self._cache_args(2, 0, n=4))
        prefill = self._cache_args(2, 0, n=4)
        variant = ht.build(ht.trace(write, prefill), write, prefill)
        for args in (
            self._cache_args(2, 8, n=4),
            self._cache_args(3, 10, n=3),
            self._cache_args(4, 1, n=7),
        ):
            self._check_write(variant, write, args, f"pos {args[1].tolist()}")

        # the functional form: a memcpy of self into the result, then the kernel
        def functional(cache, pos, k):
            return torch.index_copy(cache, 2, pos, k)

        tape = ht.trace(functional, base)
        self.assertEqual((tape.num_launches, tape.num_memcpys), (1, 1))
        variant = ht.build(tape, functional, base)
        for args in (self._cache_args(2, 9), self._cache_args(5, 1)):
            cache, pos, k = args
            (got,) = variant.replay(args)
            self.assertTrue(torch.equal(got, torch.index_copy(cache, 2, pos, k)))
        # the meta's checks with its texts
        cache, pos, k = base
        with self.assertRaisesRegex(RuntimeError, "Expected a long tensor for index"):
            ht.trace(write, (cache, pos.int(), k))
        with self.assertRaisesRegex(IndexError, "Number of indices"):
            ht.trace(write, (cache, torch.arange(2, device="cuda"), k))
        with self.assertRaisesRegex(RuntimeError, "same slice shapes"):
            ht.trace(write, (cache, pos, k[:, :2]))

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_index_writes_replay_eager_function_handles(self):
        # the siblings are compiled into IndexKernel.cu and launch that file's
        # index_elementwise_kernel over eager's IndexCopyFunctor and
        # IndexFunctor<IndexPutFunctor>: the entry's node (index_copy_), the
        # tape's launch and the replay's node hold the function handle eager's
        # capture holds (E36), for the decode write (one position) and the
        # prefill write, in bf16 and f32
        def copy_real(cache, pos, k):
            return cache.index_copy_(2, pos, k)

        def copy_entry(cache, pos, k):
            return C._host_trace_ti_index_copy_(cache, 2, pos, k)

        def put_real(cache, pos, k):
            cache[:, :, pos] = k
            return cache

        for n in (1, 4):
            for dtype in (torch.bfloat16, torch.float32):
                cache, pos, k = self._cache_args(2, 5, n=n)
                args = (cache.to(dtype), pos, k.to(dtype))
                with self.subTest(op="index_copy_", n=n, dtype=dtype):
                    eager = assert_eager_function_handles(
                        self, copy_real, args, copy_entry, launches=1
                    )
                    self.assertEqual(len(eager), 1)
                with self.subTest(op="index_put_", n=n, dtype=dtype):
                    eager = assert_eager_function_handles(
                        self, put_real, args, launches=1
                    )
                    self.assertEqual(len(eager), 1)

    def test_index_put_replays_for_the_bracket_write(self):
        # k_out[:, :, cache_position] = key_states (index_put_ with two
        # empty indices then the positions; accumulate False)
        def write(cache, pos, k):
            cache[:, :, pos] = k
            return cache

        base = self._cache_args(2, 5)
        tape = ht.trace(write, base)
        self.assertEqual((tape.num_launches, tape.num_memcpys), (1, 0))
        variant = ht.build(tape, write, base)
        for args in (
            self._cache_args(2, 6),
            self._cache_args(3, 17),
            self._cache_args(4, 63),
        ):
            self._check_write(variant, write, args, f"pos {args[1].tolist()}")
        prefill = self._cache_args(2, 0, n=4)
        variant = ht.build(ht.trace(write, prefill), write, prefill)
        for args in (self._cache_args(2, 8, n=4), self._cache_args(3, 10, n=3)):
            self._check_write(variant, write, args, f"pos {args[1].tolist()}")

        # two index tensors broadcast together, a value broadcast over the
        # indexed rows, an index not at the front (transposeToFront)
        def two(x, i, j, v):
            x[i, :, j] = v
            return x

        def two_args(n, cols):
            return (
                torch.randn(16, 3, cols, device="cuda"),
                torch.randint(0, 16, (n,), device="cuda"),
                torch.randint(0, cols, (n,), device="cuda"),
                torch.randn(n, 3, device="cuda"),
            )

        base2 = two_args(5, 8)
        tape = ht.trace(two, base2)
        self.assertEqual(tape.num_launches, 1)
        variant = ht.build(tape, two, base2)
        for args in (two_args(5, 8), two_args(2, 11), two_args(9, 3)):
            x, i, j, v = args
            want = two(x.clone(), i, j, v)
            (got,) = variant.replay(args)
            self.assertTrue(torch.equal(got, want))

        # the functional form clones first (a memcpy), then writes
        def functional(cache, pos, k):
            return torch.ops.aten.index_put.default(cache, [None, None, pos], k)

        tape = ht.trace(functional, base)
        self.assertEqual((tape.num_launches, tape.num_memcpys), (1, 1))
        variant = ht.build(tape, functional, base)
        cache, pos, k = self._cache_args(2, 11)
        (got,) = variant.replay((cache, pos, k))
        self.assertTrue(torch.equal(got, functional(cache, pos, k)))
        # declines by name: accumulate, a boolean mask, a CPU index
        cache, pos, k = base
        with self.assertRaisesRegex(ht.Declined, "accumulate"):
            ht.trace(
                lambda c, p, v: torch.ops.aten.index_put_.default(
                    c, [None, None, p], v, True
                ),
                base,
            )
        mask = torch.zeros(64, dtype=torch.bool, device="cuda")
        mask[5] = True
        with self.assertRaisesRegex(ht.Declined, "mask"):
            ht.trace(
                lambda c, m, v: torch.ops.aten.index_put_.default(
                    c, [None, None, m], v
                ),
                (cache, mask, k),
            )
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            ht.trace(write, (cache, pos, k[:, :, :, :8]))

    # ---- triu / tril

    def _triu_tril_entry(self, upper):
        def entry(t, diagonal=0):
            out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
            return C._host_trace_ti_triu_tril(t, diagonal, upper, out)

        return entry

    def test_triu_tril_parity_with_the_real_op(self):
        torch.manual_seed(0)
        for upper in (True, False):
            real = torch.triu if upper else torch.tril
            entry = self._triu_tril_entry(upper)
            for dtype in (
                torch.float32,
                torch.bfloat16,
                torch.int64,
                torch.bool,
                torch.float64,
            ):
                for shape in (
                    (5, 7),
                    (7, 5),
                    (3, 6, 4),
                    (1, 9),
                    (64, 65),
                    (2, 3, 33, 17),
                ):
                    for diagonal in (0, 1, -1, 3, -4):
                        x = torch.randn(shape, device="cuda").to(dtype)
                        if dtype is torch.bool:
                            x = torch.rand(shape, device="cuda") > 0.5
                        want = real(x, diagonal)
                        got = entry(x, diagonal)
                        torch.cuda.synchronize()
                        with self.subTest(
                            upper=upper, dtype=dtype, shape=shape, diagonal=diagonal
                        ):
                            self.assertTrue(torch.equal(got, want))
                            self.assertEqual(got.stride(), want.stride())
                            # the same kernel instantiation and launch configuration
                            real_nodes, _ = self._capture(lambda: real(x, diagonal))
                            ours, _ = self._capture(lambda: entry(x, diagonal))
                            self.assertEqual(
                                [n[:4] for n in ours], [n[:4] for n in real_nodes]
                            )
            # in place: the inplace instantiation
            x = torch.randn(6, 9, device="cuda")
            want = real(x, 2)
            y = x.clone()
            self.assertTrue(
                torch.equal(C._host_trace_ti_triu_tril(y, 2, upper, y), want)
            )
            real_nodes, _ = self._capture(
                lambda: (x.clone().triu_(2) if upper else x.clone().tril_(2))
            )
            ours, _ = self._capture(lambda: C._host_trace_ti_triu_tril(y, 2, upper, y))
            self.assertEqual([n[0] for n in ours], [n[0] for n in real_nodes])

    def test_triu_tril_replay_with_constant_and_symbolic_diagonals(self):
        for upper in (True, False):
            real = torch.triu if upper else torch.tril
            for diagonal in (0, 1, -1):

                def fn(x, d=diagonal):
                    return real(x, d)

                base = (torch.randn(5, 7, device="cuda", dtype=torch.bfloat16),)
                news = [
                    (torch.randn(7, 5, device="cuda", dtype=torch.bfloat16),),
                    (torch.randn(64, 130, device="cuda", dtype=torch.bfloat16),),
                    (torch.randn(1, 3, device="cuda", dtype=torch.bfloat16),),
                ]
                tape = ht.trace(fn, base)
                self.assertEqual(tape.num_launches, 1)
                variant = ht.build(tape, fn, base)
                for new in news:
                    (got,) = variant.replay(new)
                    self.assertTrue(
                        torch.equal(got, fn(*new)), f"{upper} {diagonal} {new[0].shape}"
                    )

            # a diagonal computed from a size is a value of the launch, not a
            # pin: the replay follows the new width
            def sym(x):
                return real(x, x.shape[1] - 3)

            base = (torch.randn(6, 8, device="cuda"),)
            variant = ht.build(ht.trace(sym, base), sym, base)
            for new in (
                (torch.randn(6, 5, device="cuda"),),
                (torch.randn(4, 10, device="cuda"),),
            ):
                (got,) = variant.replay(new)
                self.assertTrue(torch.equal(got, sym(*new)))

            # the in-place op writes the input
            def inplace(x):
                return x.triu_(1) if upper else x.tril_(1)

            base = (torch.randn(9, 9, device="cuda"),)
            variant = ht.build(ht.trace(inplace, base), inplace, base)
            x = torch.randn(3, 12, device="cuda")
            want = inplace(x.clone())
            (got,) = variant.replay((x,))
            self.assertTrue(torch.equal(got, want))
            self.assertTrue(torch.equal(x, want))

        # the mask builder's piece at its real shape: triu of a full mask
        def mask(m):
            return torch.triu(m, diagonal=1)

        base = (torch.full((16, 16), -1e9, device="cuda", dtype=torch.bfloat16),)
        variant = ht.build(ht.trace(mask, base), mask, base)
        for n in (17, 48, 1):
            m = torch.full((n, n + 3), -1e9, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(torch.equal(variant.replay((m,))[0], mask(m)))
        with self.assertRaisesRegex(RuntimeError, "at least 2 dimensions"):
            ht.trace(mask, (torch.randn(5, device="cuda"),))

    def test_cat_with_an_empty_view_piece_passes_eagers_null(self):
        # eager's data_ptr() of a tensor with no elements is null whatever its
        # storage, and parallel_cat puts that null in its metadata for an
        # empty piece; the recorder's address of an empty traced view is the
        # same constant 0, so the build's byte check passes and the cat
        # replays at other sizes of the rest
        x = torch.randn(8, 64, device="cuda")
        cases = {
            "front": lambda t: torch.cat([t[:0], t]),
            "back": lambda t: torch.cat([t, t[:0]]),
            "middle": lambda t: torch.cat([t, t[:0], t]),
            "dim1": lambda t: torch.cat([t[:, :0], t], 1),
            "split_piece": lambda t: torch.cat(t.split([64, 0], 1)[::-1], 1),
            "legacy_1d": lambda t: torch.cat([t.view(-1), t.view(-1)[:0]]),
            "alloc_view": lambda t: torch.cat([(t * 2)[:0], t]),
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                tape = two_hint.trace_twice(fn, (x,))
                pointers = [
                    q["value"]
                    for q in tape.launches[-1]["params"]
                    if q["kind"] == "ptr"
                ]
                self.assertIn(0, pointers)
                variant = ht.build(tape, fn, (x,))
                for t in (x, torch.randn(13, 64, device="cuda")):
                    self.assertTrue(torch.equal(_one(variant.replay((t,))), fn(t)))

    def test_functional_index_ops_of_an_allocation_replay(self):
        # index_put / index_copy (the functional forms) clone their self first:
        # a copy between two allocations, decided by root identity
        def put(x, i, v):
            return (x * 2).index_put((i,), v)

        def copy(x, i, v):
            return (x * 2).index_copy(0, i, v)

        def args(m):
            return (
                torch.randn(m, 64, device="cuda"),
                torch.tensor([1, m - 1, 3], device="cuda"),
                torch.randn(3, 64, device="cuda"),
            )

        base = args(8)
        for fn in (put, copy):
            with self.subTest(fn=fn.__name__):
                tape = two_hint.trace_twice(fn, base)
                self.assertEqual((tape.num_launches, tape.num_memcpys), (2, 1))
                pairs = [r for r in tape.root_facts if r[0] != "domain"]
                self.assertEqual(pairs, [("a0", "a1")])
                variant = ht.build(tape, fn, base)
                for a in (base, args(13)):
                    self.assertTrue(torch.equal(_one(variant.replay(a)), fn(*a)))

    def test_index_ops_refuse_and_guard_overlap_like_the_real_ops(self):
        # index_copy_'s meta asserts no internal overlap of self and no overlap
        # (partial or full) of self with the index and the source;
        # _index_put_impl_ asserts no overlap of self with the value and each
        # index. The siblings evaluate the same predicates on the trace's
        # values: what eager refuses declines by name at the trace (the
        # recorder alone included), and a tape built on separate inputs
        # misses on overlapping views where eager raises
        partial = "some elements of the input tensor and the written-to tensor"
        internal = "more than one element of the written-to tensor"

        def index_copy(t, i, s):
            t.index_copy_(0, i, s)
            return t

        def index_put(t, i, v):
            t[i] = v
            return t

        def idx(vals):
            return torch.tensor(vals, device="cuda")

        def rows(n=8, cols=64):
            return torch.randn(n, cols, device="cuda")

        cases = {
            "index_copy_ source a view of self": (
                index_copy,
                lambda t: (t, idx([0, 5, 7]), t[1:4]),
            ),
            "index_copy_ source is self": (
                index_copy,
                lambda t: (t, torch.arange(8, device="cuda"), t),
            ),
            "index_put_ value a view of self": (
                index_put,
                lambda t: (t, idx([0, 5, 7]), t[2:5]),
            ),
            "index_put_ value is self": (
                index_put,
                lambda t: (t, torch.arange(8, device="cuda"), t),
            ),
        }
        for name, (fn, alias) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(RuntimeError, partial):
                    fn(*alias(rows()))
                with self.assertRaisesRegex(ht.Declined, partial):
                    ht.trace(fn, alias(rows()), warm_up=False)
                with self.assertRaisesRegex(RuntimeError, partial):
                    ht.trace(fn, alias(rows()))
                self.assertFalse(C._host_trace_tracing())

        # an index that is a view of an int64 self
        def put_by_own_row(t):
            t.index_put_((t[0],), torch.ones(64, dtype=torch.int64, device="cuda"))
            return t

        ids = torch.zeros(8, 64, dtype=torch.int64, device="cuda")
        with self.assertRaisesRegex(RuntimeError, partial):
            put_by_own_row(ids.clone())
        with self.assertRaisesRegex(ht.Declined, partial):
            ht.trace(put_by_own_row, (ids.clone(),), warm_up=False)

        # index_copy_ on an expanded self: eager's internal-overlap refusal,
        # reproduced by the sibling without the warm-up
        def expanded(x, i, s):
            x.expand(-1, 64).index_copy_(0, i, s)
            return x

        args = (torch.randn(8, 1, device="cuda"), idx([1, 0, 2]), rows(3))
        with self.assertRaisesRegex(RuntimeError, internal):
            expanded(*args)
        with self.assertRaisesRegex(ht.Declined, internal):
            ht.trace(expanded, args, warm_up=False)
        self.assertFalse(C._host_trace_tracing())
        # tapes on separate inputs serve that class and miss the overlapping one
        for fn, i in ((index_copy, idx([0, 5, 7])), (index_put, idx([0, 5, 7]))):
            with self.subTest(op=fn.__name__):
                base = (rows(), i, rows(3))
                tape = ht.trace(fn, base)
                self.assertEqual((tape.num_launches, tape.num_memcpys), (1, 0))
                variant = ht.build(tape, fn, base)
                for new in (
                    (rows(), i, rows(3)),
                    (rows(12, 32), idx([11, 0, 3]), rows(3, 32)),
                ):
                    t, ii, s = new
                    want = fn(t.clone(), ii, s)
                    (got,) = variant.replay(new)
                    self.assertTrue(torch.equal(got, want))
                t = rows()
                with self.assertRaises(ht.Miss):
                    variant.replay((t, i, t[1:4]))
                self.assertIsNone(variant.try_replay((t, i, t[1:4])))
        # the functional forms write a fresh result: no overlap to refuse, as
        # in eager (the meta checks only a defined result)
        t, i = rows(), idx([0, 5, 7])
        for fn in (
            lambda t, i: torch.index_copy(t, 0, i, t[1:4]),
            lambda t, i: torch.index_put(t, (i,), t[2:5]),
        ):
            want = fn(t.clone(), i)
            tape = ht.trace(fn, (t, i))
            variant = ht.build(tape, fn, (t, i))
            (got,) = variant.replay((t, i))
            self.assertTrue(torch.equal(got, want))
        self.assertFalse(C._host_trace_tracing())

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


# ---- embedding backward: the sibling of Embedding.cu's embedding_dense_backward_cuda

embedding_backward = torch.ops.aten.embedding_dense_backward.default
_DEVICE_TYPE_ATTRS = {"test_exclusions": "a device-type framework attribute"}


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceEmbeddingBackward(HostTraceTestCase):
    """embedding_dense_backward through the sibling appended to Embedding.cu:
    the zeroed table (a memset record) and the feature kernel when the index
    count is at most 3072 without frequency scaling, the real host's route as
    a guard; the sort-based route (cub) declines by name. Eager's feature
    kernel is deterministic (one leader warp per target row per chunk
    serializes the accumulation in shared memory), so every served replay is
    compared bitwise with eager."""

    # the feature kernel's by-value arguments: indices 0, grad 8, grad_weight
    # 16 (a fresh table per capture, not compared), n 24, stride 32,
    # padding_idx 40
    _RANGES = ((0, 16), (24, 28), (32, 40), (40, 44))

    def _case(
        self,
        device,
        B,
        L,
        V=1000,
        D=64,
        dtype=torch.bfloat16,
        ids_dtype=torch.int64,
        padding_idx=-1,
        seed=0,
    ):
        # heavy duplicates (V small against B * L) and rows at padding_idx,
        # so the accumulation and the padding skip are exercised; B None is
        # a 1-D ids tensor
        torch.manual_seed(seed)
        shape = (L,) if B is None else (B, L)
        ids = torch.randint(0, V, shape, device=device, dtype=ids_dtype)
        if padding_idx >= 0:
            ids.view(-1)[::5] = padding_idx
        grad = torch.randn(*shape, D, device=device, dtype=dtype)
        return (grad, ids, V, padding_idx, False)

    def test_parity_with_the_real_op(self, device):
        # the sibling's table is bitwise the real op's, and the captured
        # nodes agree: one memset of the whole table (value 0) and the same
        # feature kernel instantiation, grid, block and shared memory, the
        # argument image equal on the bytes the host writes
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for ids_dtype in (torch.int64, torch.int32):
                for padding_idx in (-1, 3):
                    for B, L in ((4, 128), (1, 1), (3, 5), (24, 128), (None, 300)):
                        with self.subTest(
                            dtype=dtype,
                            ids=ids_dtype,
                            padding_idx=padding_idx,
                            shape=(B, L),
                        ):
                            args = self._case(
                                device,
                                B,
                                L,
                                dtype=dtype,
                                ids_dtype=ids_dtype,
                                padding_idx=padding_idx,
                            )
                            want = embedding_backward(*args)
                            got = C._host_trace_ti_embedding_dense_backward(*args)
                            torch.cuda.synchronize()
                            self._assert_bitwise(got, want, stride=True)
                            real, real_memsets, _ = self._capture_with_memsets(
                                lambda: embedding_backward(*args)
                            )
                            ours, our_memsets, _ = self._capture_with_memsets(
                                lambda: C._host_trace_ti_embedding_dense_backward(*args)
                            )
                            self.assertEqual((len(real), len(ours)), (1, 1))
                            ((name_r, grid_r, block_r, smem_r, image_r),) = real
                            ((name_o, grid_o, block_o, smem_o, image_o),) = ours
                            self.assertIn(
                                "embedding_backward_feature_kernel", C._demangle(name_r)
                            )
                            self.assertEqual(name_o, name_r)
                            self.assertEqual(
                                (grid_o, block_o, smem_o), (grid_r, block_r, smem_r)
                            )
                            self.assertEqual(len(image_o), len(image_r))
                            for lo, hi in self._RANGES:
                                self.assertEqual(
                                    image_o[lo:hi], image_r[lo:hi], f"bytes {lo}:{hi}"
                                )
                            self.assertEqual(
                                (len(real_memsets), len(our_memsets)), (1, 1)
                            )
                            self.assertEqual(real_memsets[0][1:], our_memsets[0][1:])
                            self.assertEqual(
                                our_memsets[0][1:],
                                (want.numel() * want.element_size(), 0),
                            )

    def test_replays_at_other_shapes_with_new_addresses(self, device):
        # traced at (4, 128): (2, 64), (8, 96), (24, 128) (3072 indices, the
        # last count of the route), a single index and the traced shape at
        # fresh addresses serve bitwise; padding_idx set and clear, int32 ids
        for padding_idx, ids_dtype in (
            (-1, torch.int64),
            (3, torch.int64),
            (3, torch.int32),
        ):
            with self.subTest(padding_idx=padding_idx, ids=ids_dtype):
                kw = dict(padding_idx=padding_idx, ids_dtype=ids_dtype)
                base = self._case(device, 4, 128, **kw)
                shapes = ((2, 64), (8, 96), (24, 128), (1, 1), (4, 128))
                news = [
                    self._case(device, B, L, seed=s, **kw)
                    for s, (B, L) in enumerate(shapes, 1)
                ]
                tape, variant, cases = self._replay_cases(
                    embedding_backward, base, news, stride=True
                )
                self.assertEqual([c.miss for c in cases], [None] * len(news))
                self.assertEqual(
                    (tape.num_launches, len(tape.memsets), tape.num_allocations),
                    (1, 1, 1),
                )
                # the memset of the table precedes the kernel, as eager's zeros do
                self.assertLess(tape.memsets[0]["seq"], tape.launches[0]["seq"])
                # the grid is the row width over the warp, an expression of the
                # grad's last dim
                launch = json.loads(tape.to_json())["launches"][0]
                self.assertNotIsInstance(launch["grid"][0], int)
                # the table's row count is a constant of the direct call (an int
                # argument): another count is outside the argument contract
                grad, ids, V, pad, s = base
                with self.assertRaisesRegex(ht.Miss, "non-tensor arguments"):
                    variant.replay((grad, ids, V + 1, pad, s))

    def test_the_route_is_a_guard_and_the_sort_route_declines_by_name(self, device):
        # the real host's route (at most 3072 indices without frequency
        # scaling) is a recorded guard: 3072 serves, 3096 is a named miss
        # rather than a kernel eager would not launch; traced there, or with
        # scale_grad_by_freq, the sibling declines by name (cub's radix sort
        # and the sorted kernel are not traced)
        base = self._case(device, 4, 128)
        _, variant, cases = self._replay_cases(
            embedding_backward, base, [self._case(device, 24, 128, seed=1)]
        )
        self.assertEqual(cases[0].miss, None)
        with self.assertRaisesRegex(ht.Miss, "guard failed.*3072"):
            variant.replay(self._case(device, 24, 129, seed=2))
        with self.assertRaisesRegex(ht.Declined, "sort-based route"):
            ht.trace(embedding_backward, self._case(device, 24, 129))
        grad, ids, V, pad, _ = base
        with self.assertRaisesRegex(ht.Declined, "sort-based route"):
            ht.trace(embedding_backward, (grad, ids, V, pad, True))
        self.assertFalse(C._host_trace_tracing())
        # eager takes the other route there: no feature kernel among its nodes
        real, _ = self._capture(
            lambda: embedding_backward(*self._case(device, 24, 129))
        )
        self.assertFalse(
            any("embedding_backward_feature_kernel" in C._demangle(n) for n, *_ in real)
        )

    def test_tape_holds_the_device_work_eager_issues(self, device):
        # the tape's records are the nodes eager's call produces: the memset
        # of the table, then the feature kernel by its full name; 2-D and
        # 1-D ids, fp32 and bf16
        for B, L, dtype in ((4, 128, torch.bfloat16), (None, 300, torch.float32)):
            with self.subTest(shape=(B, L), dtype=dtype):
                args = self._case(device, B, L, dtype=dtype)
                real, memsets, _ = self._capture_with_memsets(
                    lambda: embedding_backward(*args)
                )
                tape = ht.trace(embedding_backward, args)
                self.assertEqual(
                    [rec["kernel"] for rec in tape.launches], [n for n, *_ in real]
                )
                self.assertEqual((len(memsets), len(tape.memsets)), (1, 1))
                self.assertLess(tape.memsets[0]["seq"], tape.launches[0]["seq"])

    def test_forward_and_backward_of_an_nn_embedding(self, device):
        # nn.Embedding through functional_call and torch.autograd.grad: the
        # forward is the index_select sibling, the backward node runs on the
        # engine's worker thread (the recorder's thread scope) and reaches this
        # sibling with the table's row count symbolic (weight.size(0)), so a
        # replay with another table serves; out and grad_weight bitwise
        for padding_idx in (None, 3):
            with self.subTest(padding_idx=padding_idx):
                emb = torch.nn.Embedding(1000, 64, padding_idx=padding_idx).to(
                    device, torch.bfloat16
                )

                def fwd_bwd(w, ids, cot):
                    leaf = w.detach().requires_grad_(True)
                    out = torch.func.functional_call(emb, {"weight": leaf}, (ids,))
                    (gw,) = torch.autograd.grad(out, leaf, grad_outputs=cot)
                    return out, gw

                def case(B, L, V, seed):
                    torch.manual_seed(seed)
                    w = torch.randn(V, 64, device=device, dtype=torch.bfloat16)
                    ids = torch.randint(0, V, (B, L), device=device)
                    cot = torch.randn(B, L, 64, device=device, dtype=torch.bfloat16)
                    return (w, ids, cot)

                base = case(4, 128, 1000, 0)
                tape = ht.trace(fwd_bwd, base)
                self.assertEqual((tape.num_launches, len(tape.memsets)), (2, 1))
                variant = ht.build(tape, fwd_bwd, base)
                for B, L, V, seed in (
                    (2, 64, 1000, 1),
                    (8, 96, 1500, 2),
                    (24, 128, 700, 3),
                ):
                    args = case(B, L, V, seed)
                    out, gw = variant.replay(args)
                    want_out, want_gw = fwd_bwd(*args)
                    torch.cuda.synchronize()
                    self._assert_bitwise(out, want_out, f"out at {(B, L, V)}")
                    self._assert_bitwise(gw, want_gw, f"grad_weight at {(B, L, V)}")

    def test_copy_on_write_inputs_stay_lazy(self, device):
        # grad and the ids are read through the const accessor: lazy clones
        # stay copy-on-write through the ordinary sibling call, the trace's
        # warm-up and the build
        grad, ids, V, pad, s = self._case(device, 4, 128)
        lazy = [torch._lazy_clone(t) for t in (grad, ids)]
        for t in lazy:
            self.assertTrue(C._is_cow_tensor(t))
        args = (lazy[0], lazy[1], V, pad, s)
        want = embedding_backward(grad, ids, V, pad, s)
        got = C._host_trace_ti_embedding_dense_backward(*args)
        self._assert_bitwise(got, want)
        variant = ht.build(ht.trace(embedding_backward, args), embedding_backward, args)
        self._assert_bitwise(variant.replay(args)[0], want)
        for t in lazy:
            self.assertTrue(C._is_cow_tensor(t))

    def test_every_backward_case_traces_the_same_program_under_other_hints(
        self, device
    ):
        two_hint.assert_family(self, exclude=_DEVICE_TYPE_ATTRS)


instantiate_device_type_tests(
    TestCudaHostTraceEmbeddingBackward, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
