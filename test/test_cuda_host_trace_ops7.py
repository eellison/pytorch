# Owner(s): ["module: cuda"]

import re
import unittest

from host_trace_testing import bits, capture_graph, graph_functions, HostTraceTestCase

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

_FAMILY = re.compile(
    r"^(_ZN2at6native\d+(?:vectorized_elementwise_kernel|unrolled_elementwise_kernel|elementwise_kernel)I(?:Li\d+E)*)"
)
_SOFTMAX_FAMILY = re.compile(
    r"^(_ZN2at6native\d+(?:softmax_warp_forward|cunn_SoftMaxForward\w*)I)"
)


def _family(name):
    m = _FAMILY.match(name) or _SOFTMAX_FAMILY.match(name)
    return m.group(1) if m else name


def _softmax(t):
    return torch.softmax(t, -1)


def _log_softmax(t):
    return torch.log_softmax(t, -1)


def _to_f32(t):
    return t.to(torch.float32)


def _to_i32(t):
    return t.to(torch.int32)


def _softmax_entry(x, dim, half_to_float=False, log=False):
    dtype = torch.float32 if half_to_float else x.dtype
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    return C._host_trace_softmax_out(x, dim, half_to_float, log, out)


UNARY = (
    ("sin", torch.sin, "_host_trace_ti_sin"),
    ("cos", torch.cos, "_host_trace_ti_cos"),
    ("exp", torch.exp, "_host_trace_ti_exp"),
    ("rsqrt", torch.rsqrt, "_host_trace_ti_rsqrt"),
    ("neg", torch.neg, "_host_trace_ti_neg"),
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceOps7(HostTraceTestCase):
    """Commit 7: the softmax host, cast copies in the sibling, sin / cos /
    exp / rsqrt / neg. Parity is bitwise against the real op with the same
    kernel family and launch configuration; traced replays are bitwise against
    eager or a named miss."""

    def _capture(self, fn):
        # the launch configuration only: (name, grid, block, smem)
        nodes, out = super()._capture(fn)
        return [n[:4] for n in nodes], out

    def _assert_same_launches(self, real, entry, args):
        want = real(*args)
        got = entry(*args)
        torch.cuda.synchronize()
        self._assert_bitwise(got, want, stride=True)
        real_nodes, _ = self._capture(lambda: real(*args))
        ours, _ = self._capture(lambda: entry(*args))
        self.assertEqual(len(ours), len(real_nodes))
        for r, o in zip(real_nodes, ours):
            self.assertEqual(_family(o[0]), _family(r[0]))
            self.assertEqual(o[1:], r[1:])

    def _roundtrip(self, fn, base_args, new_args_list):
        tape, variant, cases = self._replay_cases(fn, base_args, new_args_list)
        return tape, variant, [c.out is not None for c in cases]

    # ---- softmax: the converted host

    def _softmax_matrix(self):
        # dim sizes on both sides of the persistent / block switch (2048
        # elements or 8 KiB) and of the smem / register choices; outer sizes
        # small and large; every dtype the host dispatches
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for N in (
                1,
                2,
                31,
                32,
                33,
                512,
                1023,
                1024,
                1025,
                2047,
                2048,
                2049,
                4096,
                8192,
                12345,
            ):
                for M in (1, 2, 7, 8, 9, 64):
                    yield dtype, M, N

    def test_softmax_parity_with_the_real_op(self):
        for dtype, M, N in self._softmax_matrix():
            x = torch.randn(M, N, device="cuda").to(dtype)
            for log in (False, True):
                real = _log_softmax if log else _softmax

                def entry(t, log=log):
                    return _softmax_entry(t, -1, False, log)

                with self.subTest(dtype=dtype, M=M, N=N, log=log):
                    self._assert_same_launches(real, entry, (x,))
        # half_to_float, and a misaligned input row (the smem choice tests the address)
        x = torch.randn(8, 4096, device="cuda", dtype=torch.float16)
        self._assert_same_launches(
            lambda t: torch._softmax(t, -1, True),
            lambda t: _softmax_entry(t, -1, True, False),
            (x,),
        )
        flat = torch.randn(8 * 4096 + 1, device="cuda", dtype=torch.float32)
        x = flat[1:].view(8, 4096)
        self._assert_same_launches(
            lambda t: torch.softmax(t, -1), lambda t: _softmax_entry(t, -1), (x,)
        )

    def test_softmax_replays_across_sizes_and_misses_on_kernel_choice(self):
        fn = _softmax
        # the persistent path, traced in the 512-bucket: served inside it,
        # missed by name outside (the warp kernel is selected by log2 bucket)
        base = (torch.randn(64, 400, device="cuda", dtype=torch.bfloat16),)
        news = [
            (torch.randn(64, 300, device="cuda", dtype=torch.bfloat16),),
            (torch.randn(8, 512, device="cuda", dtype=torch.bfloat16),),
            (torch.randn(3, 257, device="cuda", dtype=torch.bfloat16),),
            (torch.randn(64, 256, device="cuda", dtype=torch.bfloat16),),
            (torch.randn(64, 600, device="cuda", dtype=torch.bfloat16),),
            (torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16),),
        ]
        tape, variant, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served, [True, True, True, False, False, False])
        # the block path, traced at 8192 (block 1024, register kernel with
        # 8 registers): another row count and a width with the same register
        # count serve; 6000 (6 registers) and 12000 (the smem kernel) are
        # other kernels, 1024 the persistent path: named misses
        base = (torch.randn(8, 8192, device="cuda", dtype=torch.float32),)
        news = [
            (torch.randn(3, 8192, device="cuda", dtype=torch.float32),),
            (torch.randn(8, 8000, device="cuda", dtype=torch.float32),),
            (torch.randn(8, 6000, device="cuda", dtype=torch.float32),),
            (torch.randn(8, 12000, device="cuda", dtype=torch.float32),),
            (torch.randn(8, 1024, device="cuda", dtype=torch.float32),),
        ]
        tape, variant, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served, [True, True, False, False, False])
        # traced on the smem kernel (12000): served at another smem width
        base = (torch.randn(8, 12000, device="cuda", dtype=torch.float32),)
        news = [(torch.randn(5, 11000, device="cuda", dtype=torch.float32),)]
        _, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(served, [True])
        # log_softmax and a 3-D input (dim = -1 over the last dim)
        fn = _log_softmax
        base = (torch.randn(2, 8, 1000, device="cuda", dtype=torch.float16),)
        news = [(torch.randn(3, 5, 900, device="cuda", dtype=torch.float16),)]
        _, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(served, [True])

    def test_softmax_over_an_inner_dim_declines_by_name(self):
        # the spatial path is not converted: it reads a raw data pointer
        # before its launch, which the recorder turns into a decline naming
        # the op (before any node, so the completeness rule is not reached)
        x = torch.randn(4, 64, 32, device="cuda", dtype=torch.float32)
        with self.assertRaisesRegex(ht.Declined, "_softmax.*not converted"):
            ht.trace(lambda t: torch.softmax(t, 1), (x,))
        self.assertFalse(C._host_trace_tracing())
        # dim=0 of a 2-D input is the spatial path too; the last dim serves
        tape = ht.trace(lambda t: torch.softmax(t, -1), (x,))
        self.assertEqual(tape.num_launches, 1)

    # ---- cast copies in the sibling

    def test_cast_copy_parity_with_the_real_op(self):
        M, N = 48, 3000
        pairs = [
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.float16),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float32),
            (torch.float64, torch.float32),
            (torch.float32, torch.float64),
            (torch.bfloat16, torch.float16),
            (torch.int64, torch.float32),
            (torch.float32, torch.int32),
            (torch.int32, torch.int64),
            (torch.bool, torch.float32),
            (torch.float32, torch.bool),
            (torch.uint8, torch.int16),
        ]
        for src_dtype, dst_dtype in pairs:
            src = (torch.randn(M, N, device="cuda") * 3).to(src_dtype)
            for layout in ("contiguous", "strided", "transposed_dst"):
                if layout == "contiguous":
                    s, want = src, torch.empty(M, N, device="cuda", dtype=dst_dtype)
                elif layout == "strided":
                    s = src[:, ::2]
                    want = torch.empty(M, N // 2, device="cuda", dtype=dst_dtype)
                else:
                    s, want = src, torch.empty(N, M, device="cuda", dtype=dst_dtype).t()
                got = torch.empty_like(want)
                with self.subTest(src=src_dtype, dst=dst_dtype, layout=layout):
                    want.copy_(s)
                    C._host_trace_ti_copy_(got, s)
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(bits(got), bits(want)),
                        "cast copy differs bitwise",
                    )
                    real_nodes, _ = self._capture(lambda: want.copy_(s))
                    ours, _ = self._capture(lambda: C._host_trace_ti_copy_(got, s))
                    self.assertEqual(len(ours), len(real_nodes))
                    self.assertEqual(_family(ours[0][0]), _family(real_nodes[0][0]))
                    self.assertEqual(ours[0][1:], real_nodes[0][1:])

    def test_cast_copies_replay_and_to_routes_through_them(self):
        fn = _to_f32
        base = (torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16),)
        news = [
            (torch.randn(48, 3000, device="cuda", dtype=torch.bfloat16),),
            (torch.randn(5, 7, device="cuda", dtype=torch.bfloat16),),
            (
                torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
                .t()
                .contiguous()
                .t(),
            ),
        ]
        tape, variant, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        # the transposed source has other strides than the traced one: a miss;
        # traced on a transposed source it serves a transposed source
        self.assertEqual(served, [True, True, False])
        tbase = (base[0].t().contiguous().t(),)
        _, _, served = self._roundtrip(fn, tbase, [news[2]])
        self.assertEqual(served, [True])
        # a source dtype change at replay misses (the contract), never a wrong cast
        self.assertIsNone(
            variant.try_replay(
                (torch.randn(64, 4096, device="cuda", dtype=torch.float16),)
            )
        )
        # the generic dynamic-cast path (no specialized kernel for the pair)
        fn = _to_i32
        base = (torch.randn(64, 4096, device="cuda", dtype=torch.float32),)
        news = [(torch.randn(48, 3000, device="cuda", dtype=torch.float32),)]
        _, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(served, [True])
        # complex destinations decline at the trace
        with self.assertRaisesRegex(ht.Declined, "copy_ from"):
            ht.trace(lambda t: t.to(torch.complex64), base)
        self.assertFalse(C._host_trace_tracing())

    def test_copy_launches_eager_work_for_a_broadcast_read(self):
        # the GQA repeat_kv expansion (a stride-0 read into a contiguous write),
        # a transposed read, a cast copy and a contiguous copy: the sibling
        # launches eager's kernel family with eager's configuration, and the
        # tape's offset calculator is the iterator eager coalesces to (three
        # dims for the broadcast: the merged inner dims, the broadcast dim, the
        # merged outer dims), with constant and with symbolic sizes
        B, kvh, n_rep, L, hd = 4, 4, 8, 24, 64

        shape5, out4 = (B, kvh, n_rep, L, hd), (B, kvh * n_rep, L, hd)

        def repeat_kv(k):
            b, h, s, d = k.shape
            return k[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, -1, s, d)

        k = torch.randn(B, kvh, L, hd, device="cuda", dtype=torch.bfloat16)
        src = k[:, :, None, :, :].expand(shape5)
        dst5 = torch.empty(shape5, device="cuda", dtype=torch.bfloat16)
        kt = torch.randn(1536, 128, device="cuda", dtype=torch.bfloat16).t()
        cases = [
            (src, dst5),
            (kt, torch.empty(128, 1536, device="cuda", dtype=torch.bfloat16)),
            (kt, torch.empty(128, 1536, device="cuda", dtype=torch.float32)),
        ]
        for s, dst in cases:
            with self.subTest(src=tuple(s.stride()), dst=dst.dtype):
                self._assert_same_launches(
                    lambda d, x: d.copy_(x),
                    lambda d, x: C._host_trace_ti_copy_(d, x),
                    (dst, s),
                )
        # a contiguous pair of one dtype is eager's cudaMemcpyAsync: a memcpy
        # record and no launch
        tape = ht.trace(lambda t: t.clone(), (k,))
        self.assertEqual((tape.num_launches, len(tape.memcpys)), (0, 1))

        def offset_calc(tape):
            (launch,) = tape.launches
            vals = {p["name"]: p["value"] for p in launch["params"]}
            dims = vals["offset_calc.dims"]
            sizes = [vals[f"offset_calc.sizes_[{i}].divisor"] for i in range(dims)]
            divisors = [int(v) for v in sizes]
            strides = [
                [int(vals[f"offset_calc.strides_[{i}][{j}]"]) for j in range(2)]
                for i in range(dims)
            ]
            return dims, divisors, strides

        want = (
            3,
            [L * hd, n_rep, B * kvh],
            [[2, 2], [2 * L * hd, 0], [2 * n_rep * L * hd, 2 * L * hd]],
        )
        self.assertEqual(offset_calc(ht.trace(repeat_kv, (k,))), want)
        # the sizes as constants of the call (the trace still symbolizes the input)
        constant = lambda t: t[:, :, None, :, :].expand(shape5).reshape(out4)  # noqa: E731
        self.assertEqual(offset_calc(ht.trace(constant, (k,))), want)
        self.assertFalse(C._host_trace_tracing())

    @unittest.skipIf(not TEST_CUDA_PYTHON_BINDINGS, "cuda.bindings reads the nodes")
    def test_copy_replays_eager_function_handles(self):
        # the sibling is compiled into Copy.cu and launches the instantiations
        # eager's host makes: the entry's node, the tape's launch and the
        # replay's node hold the function handle eager's capture holds, not
        # only its kernel name (the contiguous same-dtype pair is a memcpy node
        # on both sides); the non-contiguous rows through CUDALoops.cuh's
        # StridedOp, the named functor both hosts launch
        B, kvh, n_rep, L, hd = 4, 4, 8, 24, 64
        shape5 = (B, kvh, n_rep, L, hd)
        k = torch.randn(B, kvh, L, hd, device="cuda", dtype=torch.bfloat16)
        kb = torch.randn(1536, 128, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(128, 1536, device="cuda")
        bf16, f32, i32 = torch.bfloat16, torch.float32, torch.int32

        def expand(t):
            return t[:, :, None, :, :].expand(shape5)

        def same(t):
            return t

        def empty(shape, dtype):
            return torch.empty(shape, device="cuda", dtype=dtype)

        rows = {
            "repeat_kv (stride-0 read)": (expand, k, empty(shape5, bf16)),
            "strided (transposed read)": (torch.t, kb, empty((128, 1536), bf16)),
            "strided cast (bf16 -> f32)": (torch.t, kb, empty((128, 1536), f32)),
            "contiguous cast (f32 -> bf16)": (same, x, empty((128, 1536), bf16)),
            "contiguous dynamic cast (f32 -> i32)": (same, x, empty((128, 1536), i32)),
            "contiguous (memcpy)": (same, x, empty((128, 1536), f32)),
        }
        for row, (view, base, dst) in rows.items():

            def real(d, t):
                return d.copy_(view(t))

            def entry(d, t):
                return C._host_trace_ti_copy_(d, view(t))

            with self.subTest(row=row):
                eager = graph_functions(capture_graph(lambda: real(dst, base)))
                ours = graph_functions(capture_graph(lambda: entry(dst, base)))
                tape = ht.trace(real, (dst, base))
                variant = ht.build(tape, real, (dst, base))
                replay = graph_functions(variant.graph)
                self.assertEqual(len(eager), 1)
                self.assertEqual(ours, eager)
                self.assertEqual(replay, eager)
                self.assertEqual(tape.num_launches, int(eager != ["memcpy"]))
        self.assertFalse(C._host_trace_tracing())

    # ---- unary ops

    def test_unary_parity_with_the_real_op(self):
        for name, real, binding in UNARY:
            entry = getattr(C, binding)
            dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
            if name == "neg":
                dtypes += [torch.int32, torch.int64]
            for dtype in dtypes:
                x = (torch.rand(48, 3000, device="cuda") + 0.5).to(dtype)
                cases = {
                    "contiguous": x,
                    "transposed": x.t().contiguous().t(),
                    "slice": x[:, ::2],
                    "misaligned": x.flatten()[1:],
                    "batch1": x[:1],
                }
                for case, t in cases.items():
                    with self.subTest(op=name, dtype=dtype, case=case):
                        self._assert_same_launches(real, entry, (t,))

    def test_unary_ops_replay(self):
        for name, real, _ in UNARY:
            base = (torch.rand(64, 4096, device="cuda", dtype=torch.bfloat16) + 0.5,)
            news = [
                (torch.rand(48, 3000, device="cuda", dtype=torch.bfloat16) + 0.5,),
                (torch.rand(5, 7, device="cuda", dtype=torch.bfloat16) + 0.5,),
            ]
            with self.subTest(op=name):
                tape, _, served = self._roundtrip(real, base, news)
                self.assertEqual(tape.num_launches, 1)
                self.assertEqual(served, [True, True])
        # complex sin is a jiterator kernel in the real op: declined by name
        with self.assertRaisesRegex(ht.Declined, "sin on"):
            ht.trace(
                torch.sin, (torch.randn(8, 8, device="cuda", dtype=torch.complex64),)
            )

    def test_rotary_pieces_compose(self):
        # sin / cos of a position table times the input, in one trace
        def fn(x, freqs):
            return x * torch.cos(freqs) + torch.neg(x) * torch.sin(freqs)

        base = (
            torch.randn(4, 64, device="cuda", dtype=torch.float32),
            torch.randn(4, 64, device="cuda", dtype=torch.float32),
        )
        news = [
            (
                torch.randn(9, 64, device="cuda", dtype=torch.float32),
                torch.randn(9, 64, device="cuda", dtype=torch.float32),
            )
        ]
        tape, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 6)
        self.assertEqual(served, [True])

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(self)


_log_softmax_backward = torch.ops.aten._log_softmax_backward_data.default
_softmax_backward = torch.ops.aten._softmax_backward_data.default
_nll_forward = torch.ops.aten.nll_loss_forward.default
_nll_backward = torch.ops.aten.nll_loss_backward.default
_BACKWARD_FAMILY = re.compile(
    r"^(_ZN2at6native\d+(?:softmax_warp_backward|cunn_SoftMaxBackward\w*|nll_loss_\w+_cuda_kernel\w*)I)"
)


def _bwd_family(name):
    m = _FAMILY.match(name) or _BACKWARD_FAMILY.match(name)
    return m.group(1) if m else name


# a device-generic class carries this attribute of the device-type framework;
# it is not a test of the family
_DEVICE_TYPE_ATTRS = {"test_exclusions": "a device-type framework attribute"}


class _LossHostChecks:
    """Parity bitwise against the real op with the same kernel family and
    launch configuration; traced replays bitwise against eager or a named
    miss (the softmax backward and nll_loss classes below)."""

    def _capture(self, fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(g, stream=stream, capture_error_mode="relaxed"):
                out = fn()
        stream.synchronize()
        e = C._HostTraceExec(g, torch.cuda.current_device())
        nodes = [
            (e.kernel_name(j), tuple(e.grid(j)), tuple(e.block(j)), e.smem(j))
            for j in range(e.num_nodes)
        ]
        return nodes, out

    def _assert_bitwise(self, got, want):
        got = got if isinstance(got, (list, tuple)) else (got,)
        want = want if isinstance(want, (list, tuple)) else (want,)
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            self.assertEqual(g.dtype, w.dtype)
            self.assertTrue(torch.equal(bits(g), bits(w)), "outputs differ bitwise")

    def _assert_same_launches(self, real, entry, args):
        want = real(*args)
        got = entry(*args)
        torch.cuda.synchronize()
        self._assert_bitwise(got, want)
        real_nodes, _ = self._capture(lambda: real(*args))
        ours, _ = self._capture(lambda: entry(*args))
        self.assertEqual(len(ours), len(real_nodes), (ours, real_nodes))
        for r, o in zip(real_nodes, ours):
            self.assertEqual(_bwd_family(o[0]), _bwd_family(r[0]))
            self.assertEqual(o[1:], r[1:])

    def _roundtrip(self, fn, base_args, new_args_list):
        tape = ht.trace(fn, base_args)
        variant = ht.build(tape, fn, base_args)
        served = []
        for new_args in new_args_list:
            out = variant.try_replay(new_args)
            if out is None:
                served.append(False)
                continue
            want = fn(*new_args)
            torch.cuda.synchronize()
            self._assert_bitwise(out, want)
            served.append(True)
        return tape, variant, served


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceSoftmaxBackward(_LossHostChecks, HostTraceTestCase):
    """The softmax / log_softmax backward host (SoftMax.cu), converted beside
    the forward and reached through an entry that allocates grad_input."""

    def _softmax_pair(self, M, N, device, dtype, log=True, offset=0):
        flat = torch.randn(2 * M * N + offset, device=device).to(dtype)
        x = flat[offset : offset + M * N].view(M, N)
        grad = flat[M * N + offset : 2 * M * N + offset].view(M, N)
        out = torch.log_softmax(x, -1) if log else torch.softmax(x, -1)
        return grad, out

    def test_softmax_backward_parity_with_the_real_op(self, device):
        # the persistent / block switch (1024 elements or 4 KiB), the smem /
        # plain choice, every dtype, both epilogues, at the GPT-2 vocabulary
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for M, N in (
                (64, 400),
                (8, 1024),
                (8, 1025),
                (3, 8192),
                (8, 12000),
                (4, 50257),
                (2, 60000),
            ):
                for log in (False, True):
                    real = _log_softmax_backward if log else _softmax_backward
                    grad, out = self._softmax_pair(M, N, device, dtype, log)

                    def entry(g, o, log=log):
                        gi = torch.empty(g.shape, dtype=g.dtype, device=g.device)
                        return C._host_trace_softmax_backward_out(
                            g, o, -1, g.dtype, log, gi
                        )

                    with self.subTest(dtype=dtype, M=M, N=N, log=log):
                        self._assert_same_launches(
                            lambda g, o, real=real: real(g, o, -1, g.dtype),
                            entry,
                            (grad, out),
                        )
        # half_to_float: a float grad and output, a half input dtype
        grad, out = self._softmax_pair(8, 4096, device, torch.float32)
        self._assert_same_launches(
            lambda g, o: _log_softmax_backward(g, o, -1, torch.float16),
            lambda g, o: C._host_trace_softmax_backward_out(
                g,
                o,
                -1,
                torch.float16,
                True,
                torch.empty(g.shape, dtype=torch.float16, device=g.device),
            ),
            (grad, out),
        )

    def test_softmax_backward_replays_across_sizes_and_misses_on_kernel_choice(
        self, device
    ):
        def fn(g, o):
            return _log_softmax_backward(g, o, -1, g.dtype)

        # the persistent path, traced in the 512-bucket
        base = self._softmax_pair(64, 400, device, torch.bfloat16)
        news = [
            self._softmax_pair(48, 300, device, torch.bfloat16),
            self._softmax_pair(8, 512, device, torch.bfloat16),
            self._softmax_pair(64, 256, device, torch.bfloat16),
            self._softmax_pair(64, 2048, device, torch.bfloat16),
        ]
        tape, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served, [True, True, False, False])
        # the block path at the GPT-2 vocabulary (fp32: block 512; 50257
        # elements exceed the 48 KiB shared-memory kernel's 12272, so the
        # plain kernel): another row count and other wide rows serve
        base = self._softmax_pair(4, 50257, device, torch.float32)
        news = [
            self._softmax_pair(3, 50257, device, torch.float32),
            self._softmax_pair(5, 40000, device, torch.float32),
            self._softmax_pair(2, 60000, device, torch.float32),
        ]
        tape, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(served, [True, True, True])
        self.assertIn("cunn_SoftMaxBackward<", C._demangle(tape.launches[0]["kernel"]))
        # the shared-memory kernel (12000 fp32 elements): another smem width
        # serves, a width past the shared-memory bound is the plain kernel, a
        # named miss
        base = self._softmax_pair(8, 12000, device, torch.float32)
        news = [
            self._softmax_pair(5, 11000, device, torch.float32),
            self._softmax_pair(8, 12500, device, torch.float32),
        ]
        tape, _, served = self._roundtrip(fn, base, news)
        self.assertEqual(served, [True, False])
        self.assertIn(
            "cunn_SoftMaxBackwardSmem", C._demangle(tape.launches[0]["kernel"])
        )
        # softmax's backward multiplies grad by output first (the sibling)
        # then runs the host

        def fn2(g, o):
            return _softmax_backward(g, o, -1, g.dtype)

        base = self._softmax_pair(64, 400, device, torch.bfloat16, log=False)
        tape, _, served = self._roundtrip(
            fn2, base, [self._softmax_pair(48, 300, device, torch.bfloat16, log=False)]
        )
        self.assertEqual((tape.num_launches, served), (2, [True]))

    def test_softmax_backward_over_an_inner_dim_declines_by_name(self, device):
        grad, out = self._softmax_pair(4, 64 * 32, device, torch.float32)
        grad, out = grad.view(4, 64, 32), torch.log_softmax(out.view(4, 64, 32), 1)
        with self.assertRaisesRegex(
            ht.Declined, "_log_softmax_backward_data.*not converted"
        ):
            ht.trace(lambda g, o: _log_softmax_backward(g, o, 1, g.dtype), (grad, out))
        self.assertFalse(C._host_trace_tracing())

    def test_every_softmax_backward_case_traces_the_same_program_under_other_hints(
        self, device
    ):
        two_hint.assert_family(self, exclude=_DEVICE_TYPE_ATTRS)


instantiate_device_type_tests(
    TestCudaHostTraceSoftmaxBackward, globals(), only_for="cuda"
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceNllLoss(_LossHostChecks, HostTraceTestCase):
    """The nll_loss forward / backward hosts (Loss.cu), converted in place and
    reached through entries that allocate the structured metas' outputs; the
    composition F.cross_entropy and its hand-composed backward trace as one
    tape."""

    def _softmax_pair(self, M, N, device, dtype, log=True, offset=0):
        flat = torch.randn(2 * M * N + offset, device=device).to(dtype)
        x = flat[offset : offset + M * N].view(M, N)
        grad = flat[M * N + offset : 2 * M * N + offset].view(M, N)
        out = torch.log_softmax(x, -1) if log else torch.softmax(x, -1)
        return grad, out

    def _nll_inputs(
        self,
        B,
        Cn,
        device,
        dtype,
        ignore_some=False,
        weight=False,
        one_d=False,
        byte_target=False,
    ):
        x = torch.log_softmax(torch.randn(B, Cn, device=device).to(dtype), -1)
        t = torch.randint(0, Cn, (B,), device=device)
        if ignore_some:
            t[::3] = -100
        if byte_target:
            t = t.clamp_min(0).to(torch.uint8)
        w = torch.rand(Cn, device=device).to(dtype) + 0.5 if weight else None
        if one_d:
            x, t = x[0], t[:1].view(())
        return x, t, w

    def _nll_entry_forward(self, x, t, w, reduction, ignore_index):
        shape = (x.shape[0],) if reduction == 0 and x.dim() == 2 else ()
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        tw = torch.empty((), dtype=x.dtype, device=x.device)
        return C._host_trace_nll_loss_forward_out(
            x, t, w, reduction, ignore_index, out, tw
        )

    def _nll_entry_backward(self, g, x, t, w, reduction, ignore_index, tw):
        gi = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        return C._host_trace_nll_loss_backward_out(
            g, x, t, w, reduction, ignore_index, tw, gi
        )

    def test_nll_loss_parity_with_the_real_op(self, device):
        # both reduced forms and the unreduced one, with and without a class
        # weight, with ignored targets, byte targets, 1-D input, every dtype;
        # forward and backward, bitwise with the same launches (the thread
        # count of the reduce kernels is the frame-count ladder)
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
            for B, Cn in ((512, 1000), (4, 50257), (736, 64), (11600, 16), (1, 8)):
                for reduction in (1, 2, 0):
                    for weight in (False, True):
                        for ignore in (False, True):
                            with self.subTest(
                                dtype=dtype,
                                B=B,
                                C=Cn,
                                reduction=reduction,
                                weight=weight,
                                ignore=ignore,
                            ):
                                x, t, w = self._nll_inputs(
                                    B, Cn, device, dtype, ignore, weight
                                )
                                self._assert_same_launches(
                                    lambda x, t: _nll_forward(x, t, w, reduction, -100),
                                    lambda x, t: self._nll_entry_forward(
                                        x, t, w, reduction, -100
                                    ),
                                    (x, t),
                                )
                                out, tw = _nll_forward(x, t, w, reduction, -100)
                                g = torch.randn(out.shape, device=device).to(dtype)
                                self._assert_same_launches(
                                    lambda g, x, t, tw: _nll_backward(
                                        g, x, t, w, reduction, -100, tw
                                    ),
                                    lambda g, x, t, tw: self._nll_entry_backward(
                                        g, x, t, w, reduction, -100, tw
                                    ),
                                    (g, x, t, tw),
                                )
        # a 1-D input (no batch dim), byte targets, an ignore_index that is a class
        x, t, w = self._nll_inputs(1, 8, device, torch.float32, one_d=True)
        for reduction in (0, 1, 2):
            with self.subTest(case="1-D", reduction=reduction):
                self._assert_same_launches(
                    lambda x, t: _nll_forward(x, t, None, reduction, -100),
                    lambda x, t: self._nll_entry_forward(x, t, None, reduction, -100),
                    (x, t),
                )
        x, t, _ = self._nll_inputs(300, 200, device, torch.float32, byte_target=True)
        with self.subTest(case="byte target, ignore_index a class"):
            self._assert_same_launches(
                lambda x, t: _nll_forward(x, t, None, 1, 3),
                lambda x, t: self._nll_entry_forward(x, t, None, 1, 3),
                (x, t),
            )

    def test_nll_loss_replays_and_the_thread_ladder_is_a_guard(self, device):
        # traced at 512 frames (32 threads): frame counts in the same rung
        # serve with another class count and new addresses; 736 frames is the
        # 64-thread rung, a named miss; the unreduced form's grid is symbolic
        for reduction in (1, 2):

            def fwd(x, t, r=reduction):
                return _nll_forward(x, t, None, r, -100)

            base = self._nll_inputs(512, 1000, device, torch.bfloat16)[:2]
            news = [
                self._nll_inputs(300, 1000, device, torch.bfloat16)[:2],
                self._nll_inputs(735, 50257, device, torch.bfloat16, ignore_some=True)[
                    :2
                ],
                self._nll_inputs(736, 1000, device, torch.bfloat16)[:2],
            ]
            tape, _, served = self._roundtrip(fwd, base, news)
            self.assertEqual((tape.num_launches, len(tape.memsets)), (1, 0))
            self.assertEqual(served, [True, True, False])
            self.assertEqual(tape.launches[0]["block"][0], 32)

            def bwd(g, x, t, tw, r=reduction):
                return _nll_backward(g, x, t, None, r, -100, tw)

            def full(B, Cn):
                x, t, _ = self._nll_inputs(
                    B, Cn, device, torch.bfloat16, ignore_some=True
                )
                out, tw = _nll_forward(x, t, None, reduction, -100)
                return torch.randn((), device=device, dtype=torch.bfloat16), x, t, tw

            tape, _, served = self._roundtrip(
                bwd,
                full(512, 1000),
                [full(300, 1000), full(735, 50257), full(736, 1000)],
            )
            # grad_input is zeroed by a memset, then one kernel
            self.assertEqual((tape.num_launches, len(tape.memsets)), (1, 1))
            self.assertEqual(served, [True, True, False])

        # reduction none: one block per 1024 frames, total_weight zeroed
        def fwd0(x, t):
            return _nll_forward(x, t, None, 0, -100)

        base = self._nll_inputs(512, 1000, device, torch.float32)[:2]
        news = [
            self._nll_inputs(3000, 300, device, torch.float32)[:2],
            self._nll_inputs(1, 5, device, torch.float32)[:2],
        ]
        tape, _, served = self._roundtrip(fwd0, base, news)
        self.assertEqual(
            (tape.num_launches, len(tape.memsets), served), (1, 1, [True, True])
        )
        # a class weight is an input of the tape, a weighted mean divides by
        # the device-side total weight
        x, t, w = self._nll_inputs(512, 1000, device, torch.float32, weight=True)
        x2, t2, w2 = self._nll_inputs(
            100, 777, device, torch.float32, weight=True, ignore_some=True
        )

        def fwdw(x, t, w):
            return _nll_forward(x, t, w, 1, -100)

        _, _, served = self._roundtrip(fwdw, (x, t, w), [(x2, t2, w2)])
        self.assertEqual(served, [True])

    def test_cross_entropy_forward_and_hand_composed_backward_trace(self, device):
        # F.cross_entropy is log_softmax then nll_loss_forward (mean): one
        # tape of two launches; its backward, composed by hand on the main
        # thread as autograd composes it (nll_loss_backward then
        # _log_softmax_backward_data), traces after it and equals autograd's
        # gradient bitwise
        V = 50257

        def ce(logits, tg):
            return F.cross_entropy(logits, tg)

        def ce_and_grad(logits, tg):
            logp = torch.log_softmax(logits, -1)
            loss, tw = _nll_forward(logp, tg, None, 1, -100)
            seed = torch.ones_like(loss)
            g_logp = _nll_backward(seed, logp, tg, None, 1, -100, tw)
            g_logits = _log_softmax_backward(g_logp, logp, -1, logits.dtype)
            return loss, g_logits

        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                logits = torch.randn(512, V, device=device, dtype=dtype)
                tg = torch.randint(0, V, (512,), device=device)
                tg[::7] = -100
                tape, _, served = self._roundtrip(
                    ce,
                    (logits, tg),
                    [
                        (
                            torch.randn(300, V, device=device, dtype=dtype),
                            torch.randint(0, V, (300,), device=device),
                        )
                    ],
                )
                self.assertEqual((tape.num_launches, served), (2, [True]))
                tape, _, served = self._roundtrip(
                    ce_and_grad,
                    (logits, tg),
                    [
                        (
                            torch.randn(300, V, device=device, dtype=dtype),
                            torch.randint(0, V, (300,), device=device),
                        )
                    ],
                )
                # log_softmax, nll forward, the seed's fill, nll backward
                # (memset + kernel), log_softmax backward
                self.assertEqual(tape.num_launches, 5)
                self.assertEqual(len(tape.memsets), 1)
                self.assertEqual(served, [True])
                leaf = logits.detach().requires_grad_(True)
                (want,) = torch.autograd.grad(F.cross_entropy(leaf, tg), leaf)
                _, got = ce_and_grad(logits, tg)
                self.assertTrue(torch.equal(bits(got), bits(want)))

    def test_every_nll_case_traces_the_same_program_under_other_hints(self, device):
        two_hint.assert_family(self, exclude=_DEVICE_TYPE_ATTRS)


instantiate_device_type_tests(TestCudaHostTraceNllLoss, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
