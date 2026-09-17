# Owner(s): ["module: cuda"]

import json
import re
import unittest

from host_trace_testing import bits, HostTraceTestCase

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, skipIfRocm


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

C = torch._C

# the real op and the sibling entry as functions of (input, dims, keepdim)
OPS = (
    {
        "sum": (
            lambda x, dims, keepdim: torch.sum(x, dims, keepdim=keepdim),
            lambda x, dims, keepdim: C._host_trace_ti_sum(x, dims, keepdim),
        ),
        "mean": (
            lambda x, dims, keepdim: torch.mean(x, dims, keepdim=keepdim),
            lambda x, dims, keepdim: C._host_trace_ti_mean(x, dims, keepdim),
        ),
        "amax": (
            lambda x, dims, keepdim: torch.amax(x, dims, keepdim=keepdim),
            lambda x, dims, keepdim: C._host_trace_ti_amax(x, dims, keepdim),
        ),
    }
    if torch.cuda.is_available()
    else {}
)

# the kernel template and its leading integer arguments (block size, output
# vector size); the ReduceOp's ops functor is a lambda in the real op and a
# named struct in the sibling, so the R type is not compared by name
_FAMILY = re.compile(r"^(_ZN2at6native\d+reduce_kernelI(?:Li\d+E)*)")


def _family(name):
    m = _FAMILY.match(name)
    return m.group(1) if m else name


def _layout(op, dtype, n):
    """Byte offsets inside the ReduceOp image of length n: [ops][ident][config]
    [input_calc][output_calc] src dst[2] acc_buf cta_buf semaphores base_idx
    accumulate final_output noutputs. The accumulator (ident) is float for the
    16- and 32-bit floats, the input type for amax, 8 bytes for fp64 / int64;
    MeanOps holds its factor at 0, the other functors are empty."""
    if op == "mean":
        f = 8 if dtype is torch.float64 else 4
        ident, acc = f, f
    elif op == "amax":
        ident = acc = dtype.itemsize
    else:
        ident = acc = 8 if dtype in (torch.float64, torch.int64) else 4
    cfg = max(4, ident + acc)
    in_calc = cfg + 64
    out_calc = in_calc + 404
    src = (out_calc + 504 + 7) // 8 * 8
    if src + 64 != n:
        raise AssertionError(
            f"ReduceOp image of {n} bytes does not fit the layout for {op} {dtype}"
        )
    return ident, cfg, in_calc, out_calc, src


def _same_reduce_op_image(test, op, dtype, ours, real):
    # The real op's image carries uninitialized bytes where the sibling's are
    # zero: ReduceConfig's padding after its bool, the unused entries of the
    # offset calculators (only `dims` of MAX_DIMS are set) and the padding
    # before noutputs. Everything the kernel reads must agree, except the
    # pointers to each capture's own allocations.
    test.assertEqual(len(ours), len(real))
    n = len(real)
    ident, cfg, in_calc, out_calc, src = _layout(op, dtype, n)
    start = 0 if op == "mean" else ident
    test.assertEqual(ours[start:cfg], real[start:cfg], "ops / ident")
    test.assertEqual(ours[cfg : cfg + 57], real[cfg : cfg + 57], "config")
    test.assertEqual(
        ours[cfg + 60 : cfg + 64], real[cfg + 60 : cfg + 64], "config.output_vec_size"
    )
    for base, nargs in ((in_calc, 1), (out_calc, 2)):
        dims = int.from_bytes(real[base : base + 4], "little")
        test.assertEqual(ours[base : base + 4], real[base : base + 4], "calc.dims")
        test.assertEqual(
            ours[base + 4 : base + 4 + 12 * dims],
            real[base + 4 : base + 4 + 12 * dims],
            "calc.sizes_",
        )
        s0 = base + 304
        test.assertEqual(
            ours[s0 : s0 + 4 * nargs * dims],
            real[s0 : s0 + 4 * nargs * dims],
            "calc.strides_",
        )
    test.assertEqual(ours[src : src + 8], real[src : src + 8], "src")
    test.assertEqual(ours[src + 16 : src + 32], bytes(16), "dst[1], acc_buf")
    test.assertEqual(real[src + 16 : src + 32], bytes(16), "dst[1], acc_buf (real)")
    for off in (src + 32, src + 40):  # cta_buf, semaphores: both null or both set
        test.assertEqual(
            ours[off : off + 8] == bytes(8), real[off : off + 8] == bytes(8)
        )
    test.assertEqual(
        ours[src + 48 : src + 58],
        real[src + 48 : src + 58],
        "base_idx, accumulate, final_output",
    )
    test.assertEqual(ours[src + 60 : src + 64], real[src + 60 : src + 64], "noutputs")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceReduce(HostTraceTestCase):
    def _assert_parity(self, op, real, entry, x, dims, keepdim):
        want = real(x, dims, keepdim)
        got = entry(x, dims, keepdim)
        torch.cuda.synchronize()
        self._assert_bitwise(got, want, stride=True)
        real_k, real_m, _ = self._capture_with_memsets(lambda: real(x, dims, keepdim))
        ours_k, ours_m, _ = self._capture_with_memsets(lambda: entry(x, dims, keepdim))
        self.assertEqual(len(ours_k), len(real_k))
        self.assertEqual(len(ours_m), len(real_m))
        for (dst_r, bytes_r, value_r), (dst_o, bytes_o, value_o) in zip(real_m, ours_m):
            self.assertEqual((bytes_o, value_o), (bytes_r, value_r))
        for (name_r, grid_r, block_r, smem_r, image_r), (
            name_o,
            grid_o,
            block_o,
            smem_o,
            image_o,
        ) in zip(real_k, ours_k):
            if "reduce_kernel" not in name_r:
                # zero_ / fill_ of an empty result: the same elementwise kernel
                self.assertEqual(name_o, name_r)
                continue
            self.assertEqual(_family(name_o), _family(name_r))
            self.assertEqual((grid_o, block_o, smem_o), (grid_r, block_r, smem_r))
            _same_reduce_op_image(self, op, x.dtype, image_o, image_r)

    def _matrix(self, dtype):
        d = "cuda"
        x = torch.randn(4, 48, 3000, device=d).to(dtype)
        cases = {
            "inner": (x[0], [-1], False),
            "outer": (x[0], [0], False),
            "all": (x[0], [], False),
            "keepdim": (x[0], [-1], True),
            "two dims": (x, [0, 2], False),
            "middle": (x, [1], False),
            "transposed": (x[0].t().contiguous().t(), [-1], False),
            "outer transposed": (x[0].t().contiguous().t(), [0], False),
            "batch 1": (x[0, :1], [-1], False),
            "row": (x[0, 0], [0], False),
            "N 64": (x[0, :, :64], [-1], False),
            "N 128": (x[0, :, :128], [-1], False),
            "N 4096": (torch.randn(64, 4096, device=d).to(dtype), [-1], False),
            "split": (torch.randn(8, 262144, device=d).to(dtype), [-1], False),
            "outer split": (torch.randn(65536, 8, device=d).to(dtype), [0], False),
            "misaligned": (
                torch.randn(144001, device=d).to(dtype)[1:].view(144, 1000),
                [-1],
                False,
            ),
            "empty": (x[0, :0], [-1], False),
            # a 0-d input accepts dim 0 and -1 (they wrap to 0 as the real
            # make_dim_mask does) and reduces to itself
            "scalar dim 0": (x[0, 0, 0], [0], False),
            "scalar dim -1 keepdim": (x[0, 0, 0], [-1], True),
            "scalar all": (x[0, 0, 0], [], False),
        }
        return cases

    def test_parity_with_the_real_op(self):
        dtypes = {
            "sum": [
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.int64,
            ],
            "mean": [torch.float16, torch.bfloat16, torch.float32, torch.float64],
            "amax": [
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
                torch.int32,
                torch.int64,
            ],
        }
        for op, (real, entry) in OPS.items():
            for dtype in dtypes[op]:
                for case, (x, dims, keepdim) in self._matrix(dtype).items():
                    if case == "empty" and op == "amax":
                        continue  # the real op raises on an empty input with dims
                    if dtype in (torch.int32, torch.int64):
                        x = (x * 100).to(dtype)
                    with self.subTest(op=op, dtype=dtype, case=case):
                        self._assert_parity(op, real, entry, x, dims, keepdim)

    def test_max_all_parity(self):
        x = torch.randn(48, 3000, device="cuda", dtype=torch.bfloat16)
        want = torch.max(x)
        got = C._host_trace_ti_amax(x, [], False)
        self.assertTrue(torch.equal(bits(got), bits(want)))
        real_k, _, _ = self._capture_with_memsets(lambda: torch.max(x))
        ours_k, _, _ = self._capture_with_memsets(
            lambda: C._host_trace_ti_amax(x, [], False)
        )
        self.assertEqual(_family(ours_k[0][0]), _family(real_k[0][0]))
        self.assertEqual(ours_k[0][1:4], real_k[0][1:4])

    # ---- traced

    def _roundtrip(self, fn, base_args, new_args_list):
        tape, variant, cases = self._replay_cases(
            fn,
            base_args,
            new_args_list,
            msg=lambda args: f"{tuple(args[0].shape)} differs",
            stride=True,
        )
        served = [
            tuple(a[0].shape) for a, c in zip(new_args_list, cases) if c.out is not None
        ]
        return tape, variant, served, [c.miss for c in cases if c.miss is not None]

    def _x(self, *shape, dtype=torch.bfloat16, offset=0):
        flat = torch.randn(int(torch.tensor(shape).prod()) + offset, device="cuda").to(
            dtype
        )
        return flat[offset:].view(*shape)

    def test_sum_replays_at_new_shapes(self):
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        news = [
            (self._x(M, N),)
            for M, N in [
                (48, 3000),
                (64, 4096),
                (32, 4096),
                (128, 4096),
                (64, 2048),
                (7, 1000),
                (1024, 512),
                (16, 4096),
                (9, 4096),
            ]
        ]
        tape, variant, served, missed = self._roundtrip(fn, (self._x(64, 4096),), news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(tape.num_allocations, 1)
        self.assertEqual(len(tape.memsets), 0)
        # the power-of-two neighbours of the traced shape serve: block dims
        # come from last_pow2 as a rebind, not from an interval guard
        for shape in [(64, 4096), (32, 4096), (128, 4096), (64, 2048), (16, 4096)]:
            self.assertIn(shape, served, missed)
        self.assertGreaterEqual(len(served), 7, missed)

    def test_zero_dim_input_with_an_explicit_dim(self):
        # sum(scalar, 0) works in eager; the sibling used to index an empty
        # dim mask for it and fault. Traced, it serves another scalar and
        # misses a 1-D input on the rank.
        fn = lambda t: torch.sum(t, 0)  # noqa: E731
        x0 = torch.randn((), device="cuda")
        tape = ht.trace(fn, (x0,))
        variant = ht.build(tape, fn, (x0,))
        self.assertEqual(tape.num_launches, 1)
        y0 = torch.randn((), device="cuda")
        out = variant.replay((y0,))[0]
        torch.cuda.synchronize()
        self.assertEqual(out.shape, torch.Size([]))
        self.assertTrue(torch.equal(bits(out), bits(fn(y0))))
        with self.assertRaises(ht.Miss):
            variant.replay((torch.randn(5, device="cuda"),))
        for op, (real, entry) in OPS.items():
            for dims, keepdim in (([0], False), ([-1], True), ([0], True)):
                with self.subTest(op=op, dims=dims, keepdim=keepdim):
                    got = entry(x0, dims, keepdim)
                    want = real(x0, dims, keepdim)
                    torch.cuda.synchronize()
                    self.assertEqual(got.shape, want.shape)
                    self.assertTrue(torch.equal(bits(got), bits(want)))
        with self.assertRaisesRegex(RuntimeError, "appears multiple times"):
            C._host_trace_ti_sum(x0, [0, -1], False)

    def test_sum_size_sweep_reports_served_and_missed(self):
        # powers of two and primes across both dims from one contiguous trace;
        # a miss names its guard (the vectorize-input threshold at N < 128, the
        # warp split at values_per_thread >= 256)
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        shapes = [
            (64, 4096),
            (32, 4096),
            (128, 4096),
            (64, 2048),
            (64, 8192),
            (48, 3000),
            (33, 2049),
            (7, 1000),
            (1024, 512),
            (129, 65),
            (2, 3),
            (500, 4097),
            (97, 1009),
            (1, 4096),
            (3, 131072),
            (64, 128),
            (64, 127),
        ]
        tape, variant, served, missed = self._roundtrip(
            fn, (self._x(64, 4096),), [(self._x(M, N),) for M, N in shapes]
        )
        for m in missed:
            self.assertIn("guard failed", m)
        # served: the power-of-two neighbours and the primes; missed, each on a
        # named guard: N < 128 (no input vectorization), N = 8192 (the warp
        # split), and the small batches, where the block grows past a warp
        # and shared memory switches on
        for shape in [
            (64, 4096),
            (32, 4096),
            (128, 4096),
            (64, 2048),
            (97, 1009),
            (33, 2049),
            (64, 128),
            (1024, 512),
        ]:
            self.assertIn(shape, served, missed)
        self.assertGreaterEqual(len(served), 10, missed)

    def test_global_reduce_with_semaphores(self):
        # the split across CTAs: cta_buf and semaphores are allocations, the
        # semaphore memset a memset node updated per call; two replays in a
        # row are both right (the node re-zeroes the semaphores each launch)
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        base = (self._x(8, 262144),)
        news = [
            (self._x(M, N),)
            for M, N in [
                (8, 262144),
                (8, 200000),
                (5, 262144),
                (12, 262144),
                (9, 131072),
                (16, 262144),
                (8, 4096),
            ]
        ]
        tape, variant, served, missed = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(tape.num_allocations, 3)
        self.assertEqual(len(tape.memsets), 1)
        self.assertIn((8, 262144), served)
        self.assertIn((8, 200000), served)
        self.assertIn((12, 262144), served)
        self.assertGreaterEqual(len(served), 4, missed)
        x = self._x(8, 200000)
        a = variant.replay((x,))[0].clone()
        b = variant.replay((x,))[0].clone()
        want = fn(x)
        self.assertTrue(torch.equal(bits(a), bits(want)))
        self.assertTrue(torch.equal(bits(b), bits(want)))
        self.assertGreater(variant.exec.dirty_memset_nodes, 0)

    def test_outer_and_keepdim_and_multi_dim_replay(self):
        # each list: two shapes that keep the traced configuration (served)
        # and one that changes it (a miss on a named guard: the CTA split, the
        # warp split, the output vector width)
        for fn, base, news in [
            (
                lambda t: torch.sum(t, 0),
                (self._x(4096, 64),),
                [(self._x(4096, 32),), (self._x(8192, 64),), (self._x(3000, 48),)],
            ),
            (
                lambda t: torch.mean(t, -1, keepdim=True),
                (self._x(64, 4096),),
                [(self._x(48, 3000),), (self._x(16, 2048),), (self._x(1, 4096),)],
            ),
            (
                lambda t: torch.amax(t, (0, 2)),
                (self._x(4, 48, 3000),),
                [
                    (self._x(8, 64, 4096),),
                    (self._x(4, 48, 4096),),
                    (self._x(3, 33, 2049),),
                ],
            ),
            (
                lambda t: torch.sum(t),
                (self._x(64, 4096),),
                [(self._x(48, 3000),), (self._x(32, 8192),), (self._x(2, 5),)],
            ),
            # the all-dims mean and max entries through the trace mode
            (
                lambda t: torch.mean(t),
                (self._x(64, 4096),),
                [(self._x(48, 3000),), (self._x(32, 8192),), (self._x(2, 5),)],
            ),
            (
                lambda t: torch.max(t),
                (self._x(64, 4096),),
                [(self._x(48, 3000),), (self._x(32, 8192),), (self._x(2, 5),)],
            ),
        ]:
            _, _, served, missed = self._roundtrip(fn, base, news)
            self.assertEqual(len(served), 2, missed)
            self.assertEqual(len(missed), 1)
            self.assertIn("guard failed", missed[0])

    def test_mean_factor_is_exact_at_every_shape(self):
        # the factor is the float the real host computes, re-evaluated per call
        fn = lambda t: torch.mean(t, -1)  # noqa: E731
        news = [
            (self._x(M, N, dtype=torch.float32),)
            for M, N in [(64, 4096), (48, 3000), (97, 1009), (7, 1000), (3, 999)]
        ]
        tape, _, served, missed = self._roundtrip(
            fn, (self._x(64, 4096, dtype=torch.float32),), news
        )
        self.assertTrue(
            any(o["fn"].startswith("mean_factor_bits") for o in tape.opaque)
        )
        # (3, 999) splits across warps and (7, 1000) widens the block past a
        # warp: both miss on a named guard; the rest serve bitwise
        for shape in [(64, 4096), (48, 3000), (97, 1009)]:
            self.assertIn(shape, served, missed)

    def test_last_pow2_declares_a_positive_result(self):
        # set_block_dimension divides by min(32, last_pow2(dim0)) and by the
        # block dimensions built from it, values_per_thread and blocks_per_sm
        # by their product: four divisions per site whose domain is defined
        # only for a nonzero divisor. The host declares the opaque result
        # positive (last_pow2_impl returns >= 1 for every input), so the
        # trace mints a positive symbol, the record and the JSON carry the
        # domain, and no guard, raw or on the tape, says a division by them is
        # defined; the same divisions over an undeclared result record four
        import sympy

        for op, (real, _) in OPS.items():
            for shape in ((64, 4096), (48, 3000)):
                with self.subTest(op=op, shape=shape):
                    fn = lambda t: real(t, [-1], False)  # noqa: E731
                    tape = ht.trace(fn, (self._x(*shape, dtype=torch.float32),))
                    pow2 = [o for o in tape.opaque if o["fn"] == "last_pow2"]
                    self.assertEqual(len(pow2), 2)
                    syms = set()
                    for o in pow2:
                        self.assertEqual(o["domain"], "positive")
                        self.assertTrue(o["sym"].node.expr.is_positive)
                        syms.add(o["sym"].node.expr)
                    for o in tape.opaque:
                        if o["fn"] != "last_pow2":
                            self.assertEqual(o["domain"], "int", o["fn"])
                    parsed = json.loads(tape.to_json())["opaque"]
                    domains = [o["domain"] for o in parsed if o["fn"] == "last_pow2"]
                    self.assertEqual(domains, ["positive", "positive"])
                    raw = [g.expr for g in tape.shape_env.guards]
                    for g in raw + list(tape.guards):
                        if isinstance(g, sympy.Ne):
                            self.assertFalse(g.free_symbols & syms, g)

        def block_arithmetic(tr, positive):
            s = tr.symbol(64, "opaque s", positive=positive)
            t = tr.symbol(4096, "opaque t", positive=positive)
            width = torch.sym_min(s, 32)
            height = torch.sym_min(t, 512 // width)
            width = torch.sym_min(s, 512 // height)
            per_thread = (4096 + width - 1) // width
            blocks_per_sm = 2048 // (width * height)
            self.assertEqual((per_thread.node.hint, blocks_per_sm.node.hint), (128, 4))
            return [g.expr for g in tr.shape_env.guards]

        self.assertEqual(len(block_arithmetic(ht._Trace(0), False)), 4)
        self.assertEqual(block_arithmetic(ht._Trace(0), True), [])

    def test_a_rebind_outside_its_declared_domain_misses(self):
        # the interim re-evaluates last_pow2 per call and binds the result; a
        # value outside the declared domain is a Miss (the folded domain
        # guards assumed it), never bound
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        x = self._x(64, 4096)
        tape = ht.trace(fn, (x,))
        variant = ht.build(tape, fn, (x,))
        rec = next(o for o in tape.opaque if o["fn"] == "last_pow2")
        call = rec["call"]
        rec["call"] = lambda args: 0
        msg = "last_pow2 is 0 at these inputs, below its declared domain"
        try:
            with self.assertRaisesRegex(ht.Miss, msg):
                variant.replay((self._x(32, 4096),))
        finally:
            rec["call"] = call
        self.assertIsNotNone(variant.try_replay((self._x(32, 4096),)))

    def test_dtype_and_dims_changes_miss_or_decline(self):
        x = self._x(64, 4096)
        fn = lambda t: torch.sum(t, -1)  # noqa: E731
        tape = ht.trace(fn, (x,))
        variant = ht.build(tape, fn, (x,))
        self.assertIsNone(variant.try_replay((self._x(64, 4096, dtype=torch.float16),)))
        # a transposed input after a contiguous trace: the stride-order guard
        self.assertIsNone(variant.try_replay((self._x(4096, 64).t(),)))
        # dims and keepdim are constants of the variant
        fn2 = lambda t, d, k: torch.sum(t, d, keepdim=k)  # noqa: E731
        tape = ht.trace(fn2, (x, [-1], False))
        variant = ht.build(tape, fn2, (x, [-1], False))
        self.assertIsNotNone(variant.try_replay((x, [-1], False)))
        self.assertIsNone(variant.try_replay((x, [0], False)))
        self.assertIsNone(variant.try_replay((x, [-1], True)))

    def test_output_vectorization_is_guarded(self):
        # the outer reduction vectorizes the output by the input's alignment
        # and the output extent: traced at width 4 (M % 4 == 0), M = 66 misses
        fn = lambda t: torch.sum(t, 0)  # noqa: E731
        base = (self._x(4096, 64),)
        _, variant, served, missed = self._roundtrip(
            fn, base, [(self._x(3000, 64),), (self._x(4096, 66),), (self._x(4096, 96),)]
        )
        self.assertIn((3000, 64), served)
        self.assertIn((4096, 96), served)
        self.assertNotIn((4096, 66), served)

    def test_reduction_composes_with_elementwise(self):
        def fn(x):
            return torch.sum(F.silu(x), -1)

        tape, _, served, missed = self._roundtrip(
            fn, (self._x(32, 4096),), [(self._x(48, 3000),), (self._x(16, 2048),)]
        )
        self.assertEqual(tape.num_launches, 2)
        self.assertEqual(tape.num_allocations, 2)
        self.assertEqual(len(served), 2, missed)

        def rms_like(x):
            # the broadcast multiply against a keepdim mean: the strided
            # elementwise path after the reduction
            return torch.mul(x, torch.mean(x, -1, keepdim=True))

        tape, _, served, missed = self._roundtrip(
            rms_like, (self._x(32, 4096),), [(self._x(48, 3000),), (self._x(16, 2048),)]
        )
        self.assertEqual(tape.num_launches, 2)
        self.assertEqual(len(served), 2, missed)

    def test_declines_by_name(self):
        x = self._x(8, 4096)
        with self.assertRaisesRegex(ht.Declined, "aten.prod"):
            ht.trace(lambda t: torch.prod(t, -1), (x,))
        with self.assertRaisesRegex(ht.Declined, "promotes"):
            ht.trace(lambda t: torch.sum(t, -1, dtype=torch.float32), (x,))
        with self.assertRaisesRegex(ht.Declined, "promotes"):
            ht.trace(lambda t: torch.sum(t, -1), ((x * 10).to(torch.int32),))
        with self.assertRaisesRegex(ht.Declined, "amax on Bool"):
            ht.trace(lambda t: torch.amax(t, -1), ((x > 0),))
        self.assertFalse(C._host_trace_tracing())
        tape = ht.trace(lambda t: torch.sum(t, -1), (x,))
        self.assertEqual(tape.num_launches, 1)

    def test_sum_out_writes_the_callers_tensor(self):
        # at::sum_out (sum.IntList_out): the result written into a tensor of
        # the result's shape, contiguous or a dense transpose, with the same
        # kernel eager's out= form launches; flash's backward sums the GQA
        # head groups of dk / dv this way
        x = self._x(2, 64, 4, 32)

        def real(t, dims, keepdim, out):
            return torch.sum(t, dims, keepdim=keepdim, out=out)

        def entry(t, dims, keepdim, out):
            return C._host_trace_ti_sum(t, dims, keepdim, out)

        for name, make_out in {
            "contiguous": lambda: torch.empty(2, 64, 32, device="cuda", dtype=x.dtype),
            "transposed": lambda: torch.empty(
                2, 32, 64, device="cuda", dtype=x.dtype
            ).transpose(1, 2),
        }.items():
            with self.subTest(name):
                want_out, got_out = make_out(), make_out()
                want = real(x, [2], False, want_out)
                got = entry(x, [2], False, got_out)
                torch.cuda.synchronize()
                self.assertIs(got, got_out)
                self.assertEqual(got.stride(), want.stride())
                self.assertTrue(torch.equal(bits(got), bits(want)))
                real_k, _ = self._capture(lambda: real(x, [2], False, want_out))
                ours_k, _ = self._capture(lambda: entry(x, [2], False, got_out))
                self.assertEqual(len(ours_k), len(real_k))
                for (name_r, grid_r, block_r, smem_r, _), (
                    name_o,
                    grid_o,
                    block_o,
                    smem_o,
                    _,
                ) in zip(real_k, ours_k):
                    self.assertEqual(_family(name_o), _family(name_r))
                    self.assertEqual(
                        (grid_o, block_o, smem_o), (grid_r, block_r, smem_r)
                    )

        # under a trace the same, replayed at other shapes into the caller's out
        def fn(t, out):
            torch.sum(t, [2], out=out)
            return out

        out = torch.empty(2, 64, 32, device="cuda", dtype=x.dtype)
        tape = ht.trace(fn, (x, out))
        self.assertEqual(tape.num_launches, 1)
        variant = ht.build(tape, fn, (x, out))
        for B, S in [(2, 64), (3, 128)]:
            y = self._x(B, S, 4, 32)
            o = torch.empty(B, S, 32, device="cuda", dtype=x.dtype)
            variant.replay((y, o))
            self.assertTrue(torch.equal(o, torch.sum(y, [2])))
        # a size-1 batch coalesces differently in the iterator: a guard
        with self.assertRaisesRegex(ht.Miss, "guard failed"):
            variant.replay(
                (
                    self._x(1, 64, 4, 32),
                    torch.empty(1, 64, 32, device="cuda", dtype=x.dtype),
                )
            )
        # an out= tensor the real op would resize or cast declines (without
        # the warm-up, whose ordinary call would resize it first)
        with self.assertRaisesRegex(ht.Declined, "out= tensor"):
            ht.trace(
                fn,
                (x, torch.empty(2, 64, 16, device="cuda", dtype=x.dtype)),
                warm_up=False,
            )
        with self.assertRaisesRegex(ht.Declined, "out= tensor"):
            ht.trace(
                fn,
                (x, torch.empty(2, 64, 32, device="cuda", dtype=torch.float32)),
                warm_up=False,
            )

    def test_ordinary_ops_are_untouched(self):
        # the sibling never runs on the ordinary path (nothing is being
        # traced); the real op's ReduceOp names the combine functor both hosts
        # launch since E36 (ReduceSumProdKernel.cu's SumCombine)
        x = self._x(64, 4096)
        kernels, _, _ = self._capture_with_memsets(lambda: torch.sum(x, -1))
        self.assertEqual(len(kernels), 1)
        self.assertIn("SumCombine", kernels[0][0])
        self.assertIn("reduce_kernel", kernels[0][0])
        self.assertFalse(C._host_trace_tracing())

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


if __name__ == "__main__":
    run_tests()
