# Owner(s): ["module: cuda graphs"]

import collections
import functools
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from typing import NamedTuple, Optional
from unittest import mock

import torch
from torch.cuda import _host_trace_replay
from torch.cuda._host_trace import Declined
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_tape import EagerCall, trace
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils.dlpack import ReadOnlyTensorWrapper


class HostTraceReplay(_host_trace_replay.HostTraceReplay):
    # traces at its first call: these are tests of the trace; an entry's first
    # call runs eagerly (test_the_first_call_runs_eagerly in
    # test_cuda_host_trace_replay)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._called = True


try:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import (
        from_dlpack,
        make_fake_compact_tensor,
        make_fake_stream,
        make_ptr,
    )

    HAS_CUTE = True
except ImportError:
    HAS_CUTE = False

if HAS_CUTE:
    from torch.cuda import _host_trace_cute as htc

    # torch.cuda._host_trace_replay was imported before cutlass: compiles
    # from here on are described
    htc.install()

    @cute.kernel
    def _add_kernel(
        gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, n: cutlass.Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        i = bidx * bdim + tidx
        if i < n:
            gC[i] = gA[i] + gB[i]

    @cute.jit
    def _add(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda_driver.CUstream,
    ):
        m = cute.size(mA, mode=[0])
        _add_kernel(mA, mB, mC, n).launch(
            grid=[(m + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.jit
    def _add_on_the_null_stream(
        mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, n: cutlass.Int32
    ):
        m = cute.size(mA, mode=[0])
        _add_kernel(mA, mB, mC, n).launch(
            grid=[(m + 127) // 128, 1, 1], block=[128, 1, 1]
        )

    @cute.jit
    def _add_shifted(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        n: cutlass.Int32,
        stream: cuda_driver.CUstream,
    ):
        _add_kernel(mA, mB, mC, n).launch(
            grid=[(n + 127) >> 7, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def _add_pdl32_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, n: cutlass.Int32):
        cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gC[i] = gA[i] + gB[i]

    @cute.jit
    def _add_pdl32(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, n: cutlass.Int32, stream: cuda_driver.CUstream):
        m = cute.size(mA, mode=[0])
        _add_pdl32_kernel(mA, mB, mC, n).launch(
            grid=[(m + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def _add_pdl_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, n: cutlass.Int64):
        cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = cutlass.Int64(bidx) * 128 + tidx
        if i < n:
            gC[i] = gA[i] + gB[i]

    # FlashInfer's norms' shape: an i64 size whose grid extent the launch
    # truncates to i32, launched with programmatic stream serialization
    @cute.jit
    def _add_pdl(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, n: cutlass.Int64, stream: cuda_driver.CUstream):
        _add_pdl_kernel(mA, mB, mC, n).launch(
            grid=[cute.ceil_div(n, 128), 1, 1], block=[128, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def _plus_one_kernel(gA: cute.Tensor, gC: cute.Tensor, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            gC[i] = gA[i].to(cutlass.Float32) + cutlass.Float32(1.0)

    # TF32 and FP8 in, f32 out
    @cute.jit
    def _plus_one(mA: cute.Tensor, mC: cute.Tensor, n: cutlass.Int32, stream: cuda_driver.CUstream):
        _plus_one_kernel(mA, mC, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

    # FlashInfer's mxfp8 GEMM's grid: a static leaf among dynamic ones, over a tile of fewer modes
    @cute.jit
    def _plus_one_tiles(mA: cute.Tensor, mC: cute.Tensor, n: cutlass.Int32, stream: cuda_driver.CUstream):
        t = cute.ceil_div((n, 1, n), (128,))
        _plus_one_kernel(mA, mC, n).launch(grid=[t[0], t[1], 1], block=[128, 1, 1], stream=stream)

    def _compile_add(shape=None, env_stream=True):
        t = make_fake_compact_tensor(
            cutlass.Float32, shape or (cute.sym_int32(),), assumed_align=16
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=env_stream)
        return cute.compile(
            _add, t, t, t, cutlass.Int32(0), stream, options="--enable-tvm-ffi"
        )

    _COMPILED = {}

    @torch.library.custom_op("host_trace_test::cute_add", mutates_args=(), device_types="cuda")
    def _cute_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # the compile is picked from the size after the fake kernel fixed the output
        c = torch.empty_like(a)
        add = _COMPILED["add_1024"] if a.shape[0] == 1024 else _COMPILED["add"]
        add(a, b, c, a.shape[0])
        return c

    _cute_add.register_fake(lambda a, b: torch.empty_like(a))

    _TMA_GEMMS = {}

    @torch.library.custom_op("host_trace_test::tma_gemm", mutates_args=(), device_types="cuda")
    def _tma_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # (m, k, 1) and (n, k, 1), K-major: the compile is picked from m
        c = a.new_empty(1, a.shape[0], b.shape[0]).permute(1, 2, 0)
        (_TMA_GEMMS[True] if a.shape[0] >= 512 else _TMA_GEMMS[False])(a, b, c)
        return c

    _tma_gemm.register_fake(lambda a, b: a.new_empty(1, a.shape[0], b.shape[0]).permute(1, 2, 0))

    _GEMMS = {}

    def _gemm_out(a, b):
        return a.new_empty(1, a.shape[0], b.shape[0]).permute(1, 2, 0)

    @torch.library.custom_op("host_trace_test::tma_choice", mutates_args=(), device_types="cuda")
    def _tma_choice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a library's kernel choice on the row count: a cluster of 2 CTAs past 256
        c = _gemm_out(a, b)
        _GEMMS[(1, 1) if a.shape[0] <= 256 else (2, 1)](a, b, c)
        return c

    _tma_choice.register_fake(_gemm_out)

    @torch.library.custom_op("host_trace_test::tma_choice_into", mutates_args=("c",), device_types="cuda")
    def _tma_choice_into(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        _GEMMS[(1, 1) if a.shape[0] <= 256 else (2, 1)](a, b, c)

    _EXAMPLES = os.environ.get(
        "CUTE_DSL_EXAMPLES",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "third_party/cutlass/examples/python/CuTeDSL",
        ),
    )

    def _compile_gemms():
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")
        for cluster in ((1, 1), (2, 1)):
            if cluster not in _GEMMS:
                op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), cluster, True)
                args = [from_dlpack(_half(1, 256, 128).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1) for _ in range(3)]
                _GEMMS[cluster] = cute.compile(op, *args, make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")

    def _example(path, name):
        path = os.path.join(_EXAMPLES, path)
        if not os.path.exists(path):
            raise unittest.SkipTest(f"the CuTe DSL example {path} is not found")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _cute_tensor(t):
    # SGLang's TGV GEMM's arguments: compiled without TVM-FFI
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()


def _half(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16)


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@unittest.skipIf(not HAS_CUTE, "requires the CuTe DSL")
@requires_cuda_python_bindings
class TestHostTraceCute(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.add = _compile_add()
        cls.add_explicit = _compile_add(env_stream=False)
        _COMPILED.update(add=cls.add, add_1024=_compile_add(shape=(1024,)))

    def test_a_shared_size_symbol_is_guarded(self):
        # self.add's three tensors share one symbol, which TVM-FFI's binder checks equal: a call at
        # unequal sizes misses and raises eager's error, where a replay would read past b's end
        add = self.add

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        ref = torch.empty_like(a)
        add(a, b, ref, 1000)
        self.assertIsNotNone(htc._compiles[add].spec)
        r = HostTraceReplay(f)
        for _ in range(2):
            self.assertEqual(r(a, b), ref, atol=0, rtol=0)
        replays = r.replays
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            r(a, b[:500])
        self.assertEqual(r.replays, replays)
        self.assertEqual(r(a, b), ref, atol=0, rtol=0)
        self.assertEqual(r.replays, replays + 1)

    def test_the_spec_guards_divisibility_and_alignment(self):
        # sizes divisible by 8 and 16-byte addresses, which TVM-FFI's binder checks: a call that breaks
        # either misses, and eager raises
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int32(divisibility=8),), assumed_align=16)
        add = cute.compile(_add, t, t, t, cutlass.Int32(0), make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        big = torch.randn(2000, device="cuda")
        r = HostTraceReplay(f)
        for _ in range(2):
            self.assertEqual(r(big[:1000], big[1000:]), big[:1000] + big[1000:], atol=0, rtol=0)
        replays = r.replays
        for msg, (a, b) in {"divisible by 8": (big[:1004], big[996:]), "Misaligned": (big[1:1001], big[1000:])}.items():
            with self.assertRaisesRegex(ValueError, msg):
                r(a, b)
        self.assertEqual(r.replays, replays)

    def test_add_is_a_launch(self):
        add = self.add

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        tape = trace(f, (a, b))
        ((_, launch),) = tape.launches
        self.assertIsInstance(launch, KernelLaunch)
        self.assertEqual(launch.layout, ((0, 16), (16, 16), (32, 16), (48, 4)))
        # each memref: its address, then its one dynamic size; the stride is static
        self.assertEqual(
            launch.fields,
            (
                (0, 0, 8),
                (0, 8, 4),
                (1, 0, 8),
                (1, 8, 4),
                (2, 0, 8),
                (2, 8, 4),
                (3, 0, 4),
            ),
        )
        self.assertEqual(launch.block, (128, 1, 1))
        r = HostTraceReplay(f)
        for n in (1000, 4096, 777, 64, 100000, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            ref = torch.empty_like(a)
            add(a, b, ref, n)
            self.assertEqual(r(a, b), ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 5, 0))

    def _pdl_entry(self):
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int32(),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        add = cute.compile(_add_pdl32, t, t, t, cutlass.Int32(0), stream, options="--enable-tvm-ffi")

        def f(a, b):
            c = torch.empty_like(a)
            add(a + 1, b, c, a.shape[0])
            return c

        def check(r, sizes):
            for n in sizes:
                a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                self.assertEqual(r(a, b), a + 1 + b, atol=0, rtol=0)

        return f, check

    @mock.patch.object(torch.cuda._host_trace, "trace_pdl", True)
    def test_a_programmatic_launch_with_trace_pdl_is_a_launch(self):
        f, check = self._pdl_entry()
        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        (_, plain), (_, launch) = trace(f, (a, b)).launches
        self.assertIsInstance(launch, KernelLaunch)
        self.assertTrue(launch.programmatic)
        self.assertFalse(plain.programmatic)
        r = HostTraceReplay(f)
        check(r, (1000, 4096, 777, 100000, 1000))
        self.assertEqual((r.traces, r.replays, r.eager), (1, 4, 0))

    def test_explicit_stream(self):
        add = self.add_explicit

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0], torch.cuda.current_stream())
            d = torch.empty_like(a)
            add(
                c,
                b,
                d,
                a.shape[0],
                cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream),
            )
            return d

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        tape = trace(f, (a, b))
        self.assertTrue(all(isinstance(r, KernelLaunch) for _, r in tape.launches))
        r = HostTraceReplay(f)
        for n in (1000, 4096, 333):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.eager), (1, 0))

    def test_declines(self):
        add, explicit = self.add, self.add_explicit
        other = torch.cuda.Stream()
        real = torch.randn(1000, device="cuda")
        cases = {
            "a stream other than the trace's": lambda a: explicit(a, a, a, 1000, other),
            "a tensor the trace does not track": lambda a: add(a, real, a, 1000),
            "a stream as a int": lambda a: explicit(
                a, a, a, 1000, torch.cuda.current_stream().cuda_stream
            ),
        }
        for msg, g in cases.items():

            def f(a):
                g(a)
                return a

            with self.assertRaisesRegex(Declined, msg):
                trace(f, (torch.randn(1000, device="cuda"),))

        def dlpack(a):
            add(from_dlpack(a, assumed_align=16), a, a, 1000)
            return a

        with self.assertRaisesRegex(Declined, "from_dlpack tensor of a traced tensor passed to a TVM-FFI function"):
            trace(dlpack, (torch.randn(1000, device="cuda"),), warm_up=False)

        def on_side_stream(a):
            with torch.cuda.stream(other):
                add(a, a, a, 1000)
            return a

        with self.assertRaisesRegex(Declined, "stream other than the trace's"):
            trace(on_side_stream, (torch.randn(1000, device="cuda"),), warm_up=False)

    def test_dtype_is_checked(self):
        add = _compile_add()
        x = torch.randn(1000, device="cuda")
        add(x, x, torch.empty_like(x), 1000)

        def f(a):
            c = torch.empty_like(a)
            add(a, a, c, a.shape[0])
            return c

        ((_, call),) = trace(f, (x.half(),), warm_up=False).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("torch.float16 tensor for a 1-d float32", call.reason)
        r = HostTraceReplay(f)
        self.assertEqual(r(x), x + x, atol=0, rtol=0)
        self.assertEqual(r(x), x + x, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 1, 0))
        with self.assertRaisesRegex(ValueError, "Mismatched Tensor"):
            r(x.half())

    def test_an_unsupported_op_is_an_eager_call_naming_it(self):
        t = make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int32(),), assumed_align=16
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        add = cute.compile(
            _add_shifted, t, t, t, cutlass.Int32(0), stream, options="--enable-tvm-ffi"
        )

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        ((_, call),) = trace(f, (a, b)).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("unsupported host op arith.shrsi", call.reason)
        r = HostTraceReplay(f)
        self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
        # nothing on the tape captures: the call runs eagerly
        self.assertEqual((r.replays, r.uncaptured, r.eager), (0, 1, 1))

    def test_a_launch_without_a_stream_is_an_eager_call(self):
        t = make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int32(),), assumed_align=16
        )
        n = cutlass.Int32(0)
        options = "--enable-tvm-ffi"
        add = cute.compile(_add_on_the_null_stream, t, t, t, n, options=options)
        a, b = torch.zeros(1000, device="cuda"), torch.ones(1000, device="cuda")
        add(a, b, a, 1000)
        torch.cuda.synchronize()
        self.assertEqual(a, b, atol=0, rtol=0)

        def f(a, b):
            add(a, b, a, a.shape[0])
            return a

        ((_, call),) = trace(f, (a, b), warm_up=False).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("a launch on a stream the call does not pass", call.reason)

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires two GPUs")
    def test_witness_on_another_device(self):
        add = _compile_add()
        x = torch.randn(1000, device="cuda:0")
        add(x, x, torch.empty_like(x), 1000)

        def f(a):
            c = torch.empty_like(a)
            add(a, a, c, a.shape[0])
            return c

        y = x.cuda(1)
        with torch.cuda.device(1):
            ((_, launch),) = trace(f, (y,), warm_up=False).launches
            self.assertIsInstance(launch, KernelLaunch)
            r = HostTraceReplay(f)
            for n in (1000, 2048):
                y = torch.randn(n, device="cuda:1")
                self.assertEqual(r(y), y + y, atol=0, rtol=0)
            self.assertEqual((r.traces, r.eager), (1, 0))

    def test_integer_results_wrap(self):
        i32, i64 = htc._Builtin("int", 32, "i32"), htc._Builtin("int", 64, "i64")
        self.assertEqual(htc._wrap(i32, 2**31 + 5, []), -(2**31) + 5)
        self.assertEqual(htc._wrap(i32, -(2**31) - 1, []), 2**31 - 1)
        self.assertEqual(htc._wrap(i64, 2**31 + 5, []), 2**31 + 5)
        ty = '!cute.int_tuple<"(?,?{i64})">'
        self.assertEqual(htc._wrap(ty, (2**32 + 1, 2**32 + 1), []), (1, 2**32 + 1))

    def test_a_ceil_div_by_a_tile_of_fewer_modes(self):
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int32(),), assumed_align=16)
        tiles = cute.compile(_plus_one_tiles, t, t, cutlass.Int32(0), make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")

        def f(a):
            c = torch.empty_like(a)
            tiles(a, c, a.shape[0])
            return c

        ((_, launch),) = trace(f, (torch.randn(1000, device="cuda"),)).launches
        self.assertIsInstance(launch, KernelLaunch)
        r = HostTraceReplay(f)
        for n in (1000, 2000, 300):
            a = torch.randn(n, device="cuda")
            self.assertEqual(r(a), a + 1, atol=0, rtol=0)
        self.assertEqual((r.traces, r.eager, r.declines), (1, 0, []))

    def test_repeated_launches_parse_types_once(self):
        # every call of a compiled function re-evaluates its host program; the
        # MLIR type strings it parses are the same each time
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        add = cute.compile(_add, t, t, t, cutlass.Int32(0), stream, options="--enable-tvm-ffi")

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            d = torch.empty_like(a)
            add(c, b, d, a.shape[0])
            return d

        htc._int_widths.cache_clear()
        r = HostTraceReplay(f)
        for n in (1000, 4096, 777, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))
        self.assertGreater(htc._int_widths.cache_info().hits, 0)

    def test_an_i64_size_and_a_programmatic_launch_are_a_launch(self):
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(64),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        add = cute.compile(_add_pdl, t, t, t, cutlass.Int64(1), stream, options="--enable-tvm-ffi")

        def f(a, b):
            c = torch.empty_like(a)
            add(a + 1, b, c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        (_, plain), (_, launch) = trace(f, (a, b)).launches
        self.assertIsInstance(launch, KernelLaunch)
        self.assertTrue(launch.programmatic)
        self.assertFalse(plain.programmatic)
        r = HostTraceReplay(f)
        for n in (1000, 4096, 777, 100000, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            ref = torch.empty_like(a)
            add(a + 1, b, ref, n)
            self.assertEqual(r(a, b), ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 4, 0))

    @mock.patch.object(torch.cuda._host_trace, "trace_pdl", False)
    def test_a_programmatic_launch_with_trace_pdl_off_is_an_eager_call(self):
        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(64),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        add = cute.compile(_add_pdl, t, t, t, cutlass.Int64(1), stream, options="--enable-tvm-ffi")

        def f(a, b):
            c = torch.empty_like(a)
            add(a + 1, b, c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        (_, plain), (_, call) = trace(f, (a, b)).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("programmatic_stream_serialization_allowed", call.reason)
        r = HostTraceReplay(f)
        for n in (1000, 777, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            ref = torch.empty_like(a)
            add(a + 1, b, ref, n)
            self.assertEqual(r(a, b), ref, atol=0, rtol=0)

    def test_static_leaf_change_retraces(self):
        add, add_1024 = _compile_add(), _compile_add(shape=(1024,))

        def f(a, b):
            c = torch.empty_like(a)
            (add_1024 if a.shape[0] == 1024 else add)(a, b, c, a.shape[0])
            return c

        r = HostTraceReplay(f)
        for n in (1024, 1000, 1000, 1024, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (2, 3, 0))
        self.assertEqual(len(r.variants), 2)

        # the compile's unit stride is a guard: a strided view misses it,
        # and its retrace raises as eager does
        def g(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        r = HostTraceReplay(g)
        x = torch.randn(2000, device="cuda")
        self.assertEqual(r(x[:1000], x[1000:]), x[:1000] + x[1000:], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "Mismatched mA.strides"):
            r(x[::2], x[1::2])
        self.assertEqual(r.traces, 2)

    def test_a_calls_specialization_is_its_own(self):
        # the compile's align and unit stride guard the call, not the graph
        add = self.add

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        a = torch.randn(1000, device="cuda")
        (op,) = trace(f, (a, a)).ops
        self.assertEqual(op.kind, "traced")
        self.assertTrue(op.guards)

    def test_a_custom_ops_compile_choice_dispatches_again_into_one_variant(self):
        # test_static_leaf_change_retraces's choice, made inside a custom op's
        # kernel: the op's own, an entry of the one variant
        r = HostTraceReplay(_cute_add)
        for n in (1024, 1000, 1000, 1024, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.folds, r.redispatches, len(r.variants), r.eager), (1, 0, 1, 1, 0))

    def test_a_torch_native_routers_dlpack_impl_is_traced(self):
        # _fused_rms_norm's router is its host, its condition the op's guards;
        # quack's adapter hands the input to cute.runtime.from_dlpack, whose
        # tensors the trace follows, so the override's launch is on the tape
        def f(x, w):
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), w, 1e-6)

        def bf16(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        tape = trace(f, (bf16(64, 1024), bf16(1024)))
        (op,) = tape.ops
        self.assertEqual(op.kind, "traced")
        self.assertTrue(op.guards)
        self.assertEqual([type(c) for _, c in tape.launches], [KernelLaunch])
        r = HostTraceReplay(f)
        # a retrace at a size quack has not compiled compiles inside the router
        for m, n in ((64, 1024), (128, 1024), (1000, 1024), (64, 1024), (64, 1096), (64, 1160)):
            x, w = bf16(m, n), bf16(n)
            self.assertEqual(r(x, w), f(x, w), atol=0, rtol=0)
        self.assertEqual(r.eager, 0)
        self.assertGreater(r.replays, 0)

    def test_witness_mismatch_is_an_eager_call(self):
        add, traced = _compile_add(env_stream=False), self.add

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0], torch.cuda.current_stream())
            d = torch.empty_like(a)
            traced(c, b, d, a.shape[0])
            return d

        pack = htc.pack_params

        def flipped(launch, slots, pointers):
            out = pack(launch, slots, pointers)
            if launch.owner is add:
                out[0][0] ^= 1
            return out

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        with mock.patch.object(htc, "pack_params", flipped):
            tape = trace(f, (a, b))
        (_, call), (_, launch) = tape.launches
        self.assertIsInstance(call, EagerCall)
        self.assertIsInstance(launch, KernelLaunch)
        self.assertIn("holds other bytes than the evaluation's", call.reason)
        self.assertEqual(call.name, f"CuTe function {add.function_name}")
        r = HostTraceReplay(f)
        s = torch.cuda.Stream()
        for n in (1000, 4096, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            s.wait_stream(torch.cuda.current_stream())
            # the eager call launches on the replay's stream
            with torch.cuda.stream(s):
                out = r(a, b)
            torch.cuda.current_stream().wait_stream(s)
            self.assertEqual(out, a + b + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 2, 0))

    def test_unobserved_compile_is_an_eager_call(self):
        add = _compile_add()
        del htc._compiles[add]

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        ((_, call),) = trace(f, (a, b)).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("its compile was not observed", call.reason)
        r = HostTraceReplay(f)
        for n in (1000, 2048):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
        self.assertEqual((r.replays, r.eager), (0, 2))

    def _export(self, compiled, d):
        path = os.path.join(d, "add.o")
        compiled.export_to_c(object_file_path=path, function_name="func")
        return path

    def test_a_loaded_object_is_a_launch_in_a_warm_process(self):
        script = """if True:
            import sys, torch, cutlass
            from torch.cuda import _host_trace_cute as htc
            from torch.cuda._host_trace_replay import HostTraceReplay
            from torch.cuda._host_trace_tape import trace
            htc.install()
            add = cutlass.cute.runtime.load_module(sys.argv[1], enable_tvm_ffi=True).func
            def f(a, b):
                c = torch.empty_like(a)
                add(a, b, c, a.shape[0])
                return c
            a = torch.randn(1000, device="cuda")
            ((_, launch),) = trace(f, (a, a)).launches
            r = HostTraceReplay(f)
            ok = []
            for n in (1000, 4096, 777, 1000):
                a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                ok.append(torch.equal(r(a, b), a + b))
            print(type(launch).__name__, all(ok), r.traces, r.replays, r.eager)
        """
        with tempfile.TemporaryDirectory() as d:
            path = self._export(self.add, d)
            self.assertTrue(os.path.exists(path + htc.SIDECAR))
            out = subprocess.run(
                [sys.executable, "-c", script, path],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        # the first call runs eagerly
        self.assertEqual(out.split(), ["KernelLaunch", "True", "1", "2", "1"])

    def test_a_loaded_object_declines(self):
        other = torch.cuda.Stream()
        real = torch.randn(1000, device="cuda")
        with tempfile.TemporaryDirectory() as d:
            add = cute.runtime.load_module(self._export(self.add_explicit, d), enable_tvm_ffi=True).func
        cases = {
            "a stream other than the trace's": lambda a: add(a, a, a, 1000, other),
            "a tensor the trace does not track": lambda a: add(a, real, a, 1000, torch.cuda.current_stream()),
        }
        for msg, g in cases.items():

            def f(a):
                g(a)
                return a

            with self.assertRaisesRegex(Declined, msg):
                trace(f, (torch.randn(1000, device="cuda"),))

    def test_a_loaded_object_without_its_host_function_is_an_eager_call(self):
        def run(add):
            def f(a, b):
                c = torch.empty_like(a)
                add(a, b, c, a.shape[0])
                return c

            a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
            ((_, call),) = trace(f, (a, b)).launches
            r = HostTraceReplay(f)
            for n in (1000, 2048):
                a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
            return call, r

        def check(add, reason):
            call, r = run(add)
            self.assertIsInstance(call, EagerCall)
            self.assertIn(reason, call.reason)
            self.assertEqual((r.replays, r.eager), (0, 2))

        with tempfile.TemporaryDirectory() as d:
            path = self._export(self.add, d)
            os.remove(path + htc.SIDECAR)
            # the copy under the object's content hash still has it
            call, r = run(cute.runtime.load_module(path, enable_tvm_ffi=True).func)
            self.assertIsInstance(call, KernelLaunch)
            self.assertEqual((r.traces, r.replays, r.eager), (1, 1, 0))
            os.remove(htc._by_content(path))
            check(cute.runtime.load_module(path, enable_tvm_ffi=True).func, "without its host function")
            with open(path + htc.SIDECAR, "w") as f:
                json.dump({"error": "its host function did not load: X", "version": htc._SIDECAR_VERSION, "object": htc._digest(path)}, f)
            check(cute.runtime.load_module(path, enable_tvm_ffi=True).func, "did not load: X")
            # an export of before TVM-FFI's spec was kept: its binder's checks are unknown
            with open(path + htc.SIDECAR, "w") as f:
                json.dump({"name": htc._compiles[self.add].name, "text": htc._compiles[self.add].text}, f)
            check(cute.runtime.load_module(path, enable_tvm_ffi=True).func, "exported without TVM-FFI's argument spec")

    def test_an_object_renamed_after_its_export_keeps_its_host_function(self):
        # upstream quack's jit_cache exports to `<sha>.o.tmp.<pid>` and renames it
        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "add.o")
            self.add.export_to_c(object_file_path=path + ".tmp.123", function_name="func")
            os.replace(path + ".tmp.123", path)
            add = cute.runtime.load_module(path, enable_tvm_ffi=True).func
            a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
            ((_, rec),) = trace(f, (a, b)).launches
            self.assertIsInstance(rec, KernelLaunch)
            r = HostTraceReplay(f)
            for n in (1000, 4096, 777, 1000):
                a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
            self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))

    def test_a_tvm_ffi_function_from_outside_cute_dsl_is_an_eager_call(self):
        # a raw TVM-FFI function, as FlashInfer or a JIT op hands out
        raw = self.add._tvm_ffi_function

        def f(a, b):
            c = torch.empty_like(a)
            raw(a * 2, b, c, a.shape[0])
            return c + 1

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        calls = [r for _, r in trace(f, (a, b)).launches if isinstance(r, EagerCall)]
        self.assertEqual(len(calls), 1)
        self.assertIn("a TVM-FFI function from outside CuTe DSL", calls[0].reason)
        r = HostTraceReplay(f)
        for n in (1000, 4096, 777, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a * 2 + b + 1, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))
        other = torch.randn(1000, device="cuda")
        with self.assertRaisesRegex(Declined, "a tensor the trace does not track"):
            trace(lambda a: raw(a, other, a, 1000) or a, (a,))

    def test_a_namedtuple_compile_argument_keeps_its_fields(self):
        # quack's GEMM passes its epilogue arguments to cute.compile as a
        # NamedTuple; rebuilt from a generator, its first field was the generator
        Args = collections.namedtuple("Args", "alpha beta")
        got = htc._concrete(Args(1.0, (2, [3])))
        self.assertIs(type(got), Args)
        self.assertEqual(got, Args(1.0, (2, [3])))
        self.assertEqual(htc._concrete([1, (2, 3)]), [1, (2, 3)])

    def test_fastdivmod_host_setup_is_symbolic(self):
        # quack's tile scheduler builds FastDivmods on the host: ctlz, shifts,
        # masks and unsigned casts; cute.assume marks a divisibility. The magic
        # number and shift are expressions of d, so no redispatch
        from torch._vendor.quack.fast_math import FastDivmod

        @cute.kernel
        def kernel(gQ: cute.Tensor, gR: cute.Tensor, fd: FastDivmod, n: cutlass.Int32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            bdim, _, _ = cute.arch.block_dim()
            i = bidx * bdim + tidx
            if i < n:
                gQ[i], gR[i] = divmod(i, fd)

        @cute.jit
        def host(mQ: cute.Tensor, mR: cute.Tensor, d: cutlass.Int32, stream: cuda_driver.CUstream):
            n = cute.size(mQ, mode=[0])
            fd = FastDivmod(cute.assume(d, divby=2))
            kernel(mQ, mR, fd, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

        def f(x, y):
            q, r = (torch.empty(x.shape[0], dtype=torch.int32, device="cuda") for _ in range(2))
            divmod_(q, r, y.shape[0])
            return q, r

        t = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int32(),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        divmod_ = cute.compile(host, t, t, cutlass.Int32(2), stream, options="--enable-tvm-ffi")
        r = HostTraceReplay(f)
        for n, d in ((1000, 6), (4096, 8), (777, 6), (1000, 100), (3000, 2), (5000, 4098)):
            i = torch.arange(n, device="cuda", dtype=torch.int32)
            got = r(torch.empty(n, device="cuda"), torch.empty(d, device="cuda"))
            self.assertEqual(got, (i // d, i % d), atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager, r.redispatches), (1, 5, 0, 0))

    def test_a_host_branch_over_a_traced_value_is_a_select(self):
        # an scf.if whose branches yield only integers: both are evaluated and
        # the launch takes a select; the else branch's i32 range (d * 600000 <
        # 2**31) holds only where it is taken, so (6000, 5000) replays
        @cute.kernel
        def kernel(gO: cute.Tensor, k: cutlass.Int32, n: cutlass.Int32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            bdim, _, _ = cute.arch.block_dim()
            i = bidx * bdim + tidx
            if i < n:
                gO[i] = k

        @cute.jit
        def host(mO: cute.Tensor, d: cutlass.Int32, stream: cuda_driver.CUstream):
            n = cute.size(mO, mode=[0])
            k = cutlass.Int32(0)
            if n > d:
                k = n * 3
            else:
                k = d * 600000
            kernel(mO, k, n).launch(grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

        def f(x, y):
            o = torch.empty(x.shape[0], dtype=torch.int32, device="cuda")
            branch(o, y.shape[0])
            return o

        t = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int32(),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        branch = cute.compile(host, t, cutlass.Int32(2), stream, options="--enable-tvm-ffi")
        r = HostTraceReplay(f)
        for n, d in ((1000, 6), (6000, 5000), (100, 600), (1000, 6), (50, 3000)):
            got = r(torch.empty(n, device="cuda"), torch.empty(d, device="cuda"))
            want = torch.full((n,), n * 3 if n > d else d * 600000, dtype=torch.int32, device="cuda")
            self.assertEqual(got, want, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager, r.redispatches), (1, 4, 0, 0))
        self.assertTrue(all(isinstance(rec, KernelLaunch) for v in r.variants for _, rec in v.tape.launches))

    def test_a_namedtuple_call_argument_is_flattened(self):
        # quack passes its epilogue arguments as a NamedTuple; TVM-FFI flattens
        # it into its fields, a None field into none
        class Args(NamedTuple):
            b: cute.Tensor
            scale: Optional[cute.Tensor]

        @cute.jit
        def host(mA: cute.Tensor, args: Args, mC: cute.Tensor, n: cutlass.Int32, stream: cuda_driver.CUstream):
            m = cute.size(mA, mode=[0])
            _add_kernel(mA, args.b, mC, n).launch(grid=[(m + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream)

        t = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int32(),), assumed_align=16)
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        add = cute.compile(host, t, Args(t, None), t, cutlass.Int32(0), stream, options="--enable-tvm-ffi")

        def f(a, b):
            c = torch.empty_like(a)
            add(a, Args(b, None), c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        ((_, rec),) = trace(f, (a, b)).launches
        self.assertIsInstance(rec, KernelLaunch)
        r = HostTraceReplay(f)
        for n in (1000, 4096, 777, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))
        other = torch.randn(1000, device="cuda")
        with self.assertRaisesRegex(Declined, "argument 1 holds a tensor the trace does not track"):
            trace(lambda a: f(a, other), (a,))

    def test_a_compile_cache_key_is_a_specialization(self):
        # quack's jit_cache keys its compile by x.size(-1) and the kernel is compiled
        # for a static weight shape: guards on N and w.size(0), M stays symbolic
        from torch._vendor.quack.rmsnorm import rmsnorm_fwd

        add = self.add

        def f(x, w, a):
            c = torch.empty_like(a)
            add(a, a, c, a.shape[0])
            return rmsnorm_fwd(x, w)[0], c

        def bf16(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        a = torch.randn(1000, device="cuda")
        tape = trace(f, (bf16(64, 1024), bf16(1024), a))
        self.assertTrue(all(isinstance(r, KernelLaunch) for _, r in tape.launches), tape.launches)
        self.assertEqual(sum("1024" in str(g) for g in tape.guards), 2)
        r = HostTraceReplay(f)
        for m, n in ((64, 1024), (128, 1024), (1000, 1024), (64, 512)):
            x, w = bf16(m, n), bf16(n)
            # compiled eagerly first: a compile inside a later trace gets N as a SymInt
            ref = rmsnorm_fwd(x, w)[0]
            self.assertEqual(r(x, w, a), (ref, a + a), atol=0, rtol=0)
        # N=512 fails the guard: a second trace
        self.assertEqual((r.traces, r.replays, r.eager), (2, 2, 0))

    def test_a_compile_under_a_trace_declines(self):
        # a new N compiled inside a trace without a warm-up reaches quack's
        # RMSNorm.N as a SymInt; each trace's warm-up compiles it first
        import torch._vendor.quack.cache as quack_cache
        from torch._vendor.quack.rmsnorm import rmsnorm_fwd

        def f(x, w):
            return rmsnorm_fwd(x, w)[0]

        def bf16(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        r = HostTraceReplay(f)
        with mock.patch.object(quack_cache, "CACHE_ENABLED", False):
            for m, n in ((64, 1024), (64, 1792), (64, 1792), (32, 1792)):
                x, w = bf16(m, n), bf16(n)
                self.assertEqual(r(x, w), f(x, w), atol=0, rtol=0)
            self.assertEqual((r.traces, r.replays, r.eager, r.declines), (2, 2, 0, []))
            with self.assertRaisesRegex(Declined, "CuTe DSL compile under the trace"):
                trace(f, (bf16(64, 2048), bf16(2048)), warm_up=False)

    def test_a_compile_under_every_trace_is_retried_once(self):
        # quack's rmsnorm_bwd keys its compile cache on T_hint, 0 under a
        # SymInt: every trace compiles it with N symbolic, which raises. One
        # retry per replay, then the op is an eager step of the variant
        import torch._vendor.quack.cache as quack_cache
        from torch._vendor.quack.rmsnorm import rmsnorm_bwd

        def f(x, w, dout, rstd):
            return rmsnorm_bwd(x, w, dout, rstd)[0]

        x, w, dout = (torch.randn(*s, device="cuda", dtype=torch.bfloat16) for s in ((300, 4096), (4096,), (300, 4096)))
        rstd = torch.rand(300, device="cuda") + 0.5
        with mock.patch.object(quack_cache, "CACHE_ENABLED", False):
            for r in (HostTraceReplay(f), HostTraceReplay(f)):
                for _ in range(5):
                    self.assertEqual(r(x, w, dout, rstd), f(x, w, dout, rstd), atol=0, rtol=0)
                self.assertEqual((r.traces, r.replays, r.eager), (2, 3, 1))
                ((variant,),) = r._families.values()
                (why,) = [rec.reason for _, rec in variant.tape.launches if isinstance(rec, EagerCall)]
                self.assertIn("a CuTe DSL compile under the trace raised", why)

    @parametrize("sidecar", ["missing", "older"])
    def test_a_disk_cache_entry_without_its_host_function_recompiles(self, sidecar):
        # an object quack's jit_cache exported before its compile was observed
        # has no host function beside it (nor under its content hash), or one
        # without TVM-FFI's argument spec: loading it is a cache miss
        import torch._vendor.quack.cache as quack_cache
        from torch._vendor.quack.rmsnorm import _compile_rmsnorm_fwd, rmsnorm_fwd

        def f(x, w):
            return rmsnorm_fwd(x, w)[0]

        def bf16(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as d, quack_cache.cache_dir_override(d), mock.patch.object(
            quack_cache, "CACHE_ENABLED", True
        ):
            _compile_rmsnorm_fwd.cache_clear()
            f(bf16(64, 1536), bf16(1536))
            sidecars = [os.path.join(root, n) for root, _, names in os.walk(d) for n in names if n.endswith(htc.SIDECAR)]
            self.assertEqual(len(sidecars), 2)
            for path in sidecars:
                if sidecar == "missing":
                    os.remove(path)
                else:
                    with open(path, "w") as out:
                        json.dump({"name": "func", "text": ""}, out)
            _compile_rmsnorm_fwd.cache_clear()
            r = HostTraceReplay(f)
            for m in (64, 32):
                x, w = bf16(m, 1536), bf16(1536)
                self.assertEqual(r(x, w), f(x, w), atol=0, rtol=0)
            self.assertEqual((r.traces, r.replays, r.eager), (1, 1, 0))
            self.assertTrue(all(json.load(open(path))["version"] == htc._SIDECAR_VERSION for path in sidecars))

    def test_a_compile_inside_a_trace_specializes(self):
        # a per-N compile cache: hashing N, the static shape and the Int32 argument are guards on N
        compiled = {}
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)

        def f(a, b):
            n = a.shape[0]
            if n not in compiled:
                t = make_fake_compact_tensor(cutlass.Float32, (n,), assumed_align=16)
                compiled[n] = cute.compile(_add, t, t, t, n, stream, options="--enable-tvm-ffi")
            c = torch.empty_like(a)
            compiled[n](a, b, c, n)
            return c

        r = HostTraceReplay(f)
        for n in (1024, 2048, 2048, 1024, 4096):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
        # each N is compiled by its trace's eager warm-up
        self.assertEqual((r.traces, r.replays, r.eager), (3, 2, 0))
        a = torch.randn(8192, device="cuda")
        tape = trace(f, (a, a), warm_up=False)
        ((_, launch),) = tape.launches
        self.assertIsInstance(launch, KernelLaunch)
        self.assertTrue(any("8192" in str(g) for g in tape.guards))

    def test_a_replay_on_another_stream(self):
        add = self.add

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c * 2

        r = HostTraceReplay(f)
        a, b = torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda")
        self.assertEqual(r(a, b), (a + b) * 2)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            out = r(a, b)
        torch.cuda.current_stream().wait_stream(s)
        self.assertEqual(out, (a + b) * 2, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 1, 0))

    def test_ampere_gemm(self):
        tg = _example("cute/ampere/kernel/dense_gemm/tensorop_gemm.py", "tensorop_gemm")
        a, b, c = _half(1, 256, 128), _half(1, 128, 256), _half(1, 256, 256)
        cute_args = tg.mark_dynamic_layout(
            a, b, c, 2, 2, 2, cutlass.Float16, cutlass.Float16
        )
        op = tg.TensorOpGemm(
            cutlass.Float16, cutlass.Float16, cutlass.Float32, (2, 2, 1), False
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm = cute.compile(tg.bmm, op, *cute_args, stream, options="--enable-tvm-ffi")

        def f(a, b):
            c = a.new_empty(a.shape[0], a.shape[1], b.shape[2])
            gemm(a, b, c)
            return c

        ((_, launch),) = trace(f, (a, b)).launches
        self.assertIsInstance(launch, KernelLaunch)
        self.assertEqual([size for _, size in launch.layout], [40, 40, 40, 4])
        # N tiles of 128 pick the raster factor: 1 tile 1, 2 tiles 2, 3 to 5
        # tiles 4, more 8; a select of them
        sizes = [
            (256, 256, 128),
            (128, 128, 64),
            (256, 200, 128),
            (128, 512, 64),
            (64, 1024, 256),
            (256, 256, 128),
            (192, 384, 96),
            (128, 136, 64),
        ]
        r = HostTraceReplay(f)
        for m, n, k in sizes:
            a, b = _half(1, m, k), _half(1, k, n)
            ref = a.new_empty(1, m, n)
            gemm(a, b, ref)
            self.assertEqual(r(a, b), ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.redispatches, r.replays, r.eager), (1, 0, 7, 0))

    def test_witness_at_m_not_n(self):
        tg = _example("cute/ampere/kernel/dense_gemm/tensorop_gemm.py", "tensorop_gemm")
        a, b, c = _half(1, 256, 128), _half(1, 128, 256), _half(1, 256, 256)
        cute_args = tg.mark_dynamic_layout(
            a, b, c, 2, 2, 2, cutlass.Float16, cutlass.Float16
        )
        op = tg.TensorOpGemm(
            cutlass.Float16, cutlass.Float16, cutlass.Float32, (2, 2, 1), False
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm = cute.compile(tg.bmm, op, *cute_args, stream, options="--enable-tvm-ffi")
        memref = htc._memref

        # an evaluation that reads C's M and N swapped agrees with the
        # launch only where M == N
        def swapped(ty, t, what):
            view = memref(ty, t, what)
            if what == "arg2":
                l, m, n = view["shape"]
                view["shape"] = (l, n, m)
            return view

        def f(a, b):
            c = a.new_empty(a.shape[0], a.shape[1], b.shape[2])
            gemm(a, b, c)
            return c

        with mock.patch.object(htc, "_memref", swapped):
            ((_, launch),) = trace(f, (_half(1, 256, 128), _half(1, 128, 256))).launches
            self.assertIsInstance(launch, KernelLaunch)
            ((_, call),) = trace(f, (_half(1, 256, 128), _half(1, 128, 384))).launches
        self.assertIsInstance(call, EagerCall)

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    @parametrize("cluster", [(1, 1), (2, 1), (2, 2)])
    def test_blackwell_tma_gemm(self, cluster):
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")

        def operands(m, n, k, pad=0):
            # (m, k, l), (n, k, l), (m, n, l), each K- or N-major, rows padded
            return tuple(
                _half(1, r, c + pad)[:, :, :c].permute(1, 2, 0)
                for r, c in ((m, k), (n, k), (m, n))
            )

        op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), cluster, True)
        compile_args = [
            from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            for t in operands(256, 256, 128)
        ]
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")

        def f(a, b, c):
            gemm(a, b, c)
            return c

        ((_, launch),) = trace(f, operands(256, 256, 128)).launches
        self.assertIsInstance(launch, KernelLaunch)
        self.assertEqual(launch.cluster, (*cluster, 1))
        # A is multicast across the cluster's N CTAs, B across its M CTAs
        boxes = [d.box for d in launch.descriptors]
        m, n = cluster
        self.assertEqual(boxes, [(64, 128 // n, 1), (64, 128 // m, 1), (32, 128, 1)])
        r = HostTraceReplay(f)
        for m, n, k, pad in (
            (256, 256, 128, 0),
            (512, 384, 256, 0),
            (1024, 128, 64, 8),
            (384, 640, 192, 24),
            (256, 256, 128, 0),
        ):
            a, b, c = operands(m, n, k, pad)
            ref = c.clone()
            gemm(a, b, ref)
            self.assertEqual(r(a, b, c), ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 4, 0))

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    @unittest.skipIf(not getattr(torch._C, "_host_trace_tma_raw", False), "needs the native encode of the driver's map as is")
    @mock.patch.object(torch.cuda._host_trace, "tma_library_bits", False)
    def test_a_tma_gemm_launches_the_drivers_maps(self):
        # the stand-in compare checks the DSL's maps (with its bit); the
        # replay launches the driver's, without it
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")

        def operands(m, n, k):
            return tuple(_half(1, r, c).permute(1, 2, 0) for r, c in ((m, k), (n, k), (m, n)))

        op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
        compile_args = [from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1) for t in operands(256, 256, 128)]
        gemm = cute.compile(op, *compile_args, make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")

        def f(a, b, c):
            gemm(a, b, c)
            return c

        r = HostTraceReplay(f)
        for m, n, k in ((256, 256, 128), (512, 384, 256), (256, 256, 128)):
            a, b, c = operands(m, n, k)
            ref = c.clone()
            gemm(a, b, ref)
            self.assertEqual(r(a, b, c), ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 2, 0))
        ((_, launch),) = r.variants[0].tape.launches
        ((_, held),) = torch._C._host_trace_held_images(r.variants[0].native)
        for d in launch.descriptors:
            self.assertFalse(d.edits)
            self.assertEqual(held[d.param][8] & 0x02, 0)

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    def test_a_tma_over_a_unit_batch_mode_is_a_launch(self):
        # quack's memrefs are (?,?,1):(?,1,0): the box covers modes 0 and 1
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")

        def operands(m, n, k):
            return tuple(_half(r, c).as_strided((r, c, 1), (c, 1, 0)) for r, c in ((m, k), (n, k), (m, n)))

        def unit(t):
            t = from_dlpack(t, assumed_align=16)
            t = t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
            return t.mark_compact_shape_dynamic(mode=0, stride_order=(2, 0, 1))

        op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), (2, 1), True)
        compile_args = [unit(_half(1, 256, 128).permute(1, 2, 0)) for _ in range(3)]
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")

        def f(a, b, c):
            gemm(a, b, c)
            return c

        ((_, rec),) = trace(f, operands(256, 256, 128)).launches
        self.assertIsInstance(rec, KernelLaunch)
        self.assertEqual([d.box for d in rec.descriptors], [(64, 128), (64, 64), (32, 128)])
        r = HostTraceReplay(f)
        for m, n, k in ((256, 256, 128), (512, 384, 256), (1024, 128, 64), (256, 256, 128)):
            a, b, c = operands(m, n, k)
            ref = c.clone()
            gemm(a, b, ref)
            self.assertEqual(r(a, b, c), ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    def test_a_tma_kernel_choice_traces_its_class(self):
        # the op's TMA kernel and cluster change with m, which no entry holds:
        # its class traces
        _compile_gemms()

        def f(a, b):
            return _tma_choice(a, b) * 2

        r = HostTraceReplay(f)
        for m in (256, 512, 128, 1024, 256):
            a, b = (_half(1, rows, 128).permute(1, 2, 0) for rows in (m, 256))
            self.assertEqual(r(a, b), f(a, b), atol=0, rtol=0)
        self.assertEqual((r.traces, len(r.variants), r.replays, r.eager), (2, 2, 3, 0))
        clusters = [[lo.launch.cluster for lo in v.captured.lowered.launches if isinstance(lo.launch, KernelLaunch) and lo.launch.descriptors] for v in r.variants]
        self.assertEqual(clusters, [[(1, 1, 1)], [(2, 1, 1)]])

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    @parametrize("free", [True, False])
    def test_a_tma_descriptor_of_a_size_1_view_dim(self, free):
        # A = qkv[:, :128].view(m, 128): at m=1 eager's row stride is 128, the
        # free one qkv's 136; A's descriptor encodes it, TMA reads only row 0
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")
        op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
        compile_args = [
            from_dlpack(_half(1, 256, 128).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
            for _ in range(3)
        ]
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")
        qkv, b = _half(256, 136), _half(1, 128, 128).permute(1, 2, 0)

        def f(qkv, b):
            m = qkv.shape[0]
            c = qkv.new_empty(1, m, 128).permute(1, 2, 0)
            gemm(qkv[:, :128].view(1, m, 128).permute(1, 2, 0), b, c)
            return c

        r = HostTraceReplay(f)
        with mock.patch("torch.cuda._host_trace.free_size_one_strides", free):
            for m in (256, 1, 130, 1, 2):
                self.assertEqual(r(qkv[:m], b), f(qkv[:m], b), atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 4, 0) if free else (2, 3, 0))

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    def test_a_tma_over_a_size_1_mode_of_any_stride(self):
        # unsqueeze(-1)'s batch stride is 2 bytes: the DSL's inline encode
        # rounds a byte stride down to a multiple of 16, as a replay encodes it
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")
        op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
        compile_args = [
            from_dlpack(_half(1, 256, 128).permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
            for _ in range(3)
        ]
        gemm = cute.compile(op, *compile_args, make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        b = _half(1, 128, 128).permute(1, 2, 0)

        def f(x, b):
            c = x.new_empty(x.shape[0], 128, 1)
            gemm(x.unsqueeze(-1), b, c)
            return c

        r = HostTraceReplay(f)
        for m in (256, 512, 128, 256):
            x = _half(m, 128)
            self.assertEqual(r(x, b), f(x, b), atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))

    def test_witness_rejects_unmodelled_bytes(self):
        # an evaluation without the memrefs' dynamic extents leaves the
        # captured n unmodelled, past the fields' alignment padding; a replay
        # would pack zeros there
        add = self.add

        def f(a, b):
            c = torch.empty_like(a)
            add(a, b, c, a.shape[0])
            return c

        a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
        with mock.patch.object(htc, "_leaf_fields", lambda pat, vals: []):
            ((_, call),) = trace(f, (a, b)).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("kernel parameter 0 is 16 bytes; its fields 8", call.reason)

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    def test_blackwell_tma_descriptor_is_witnessed(self):
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")
        a, b, c = (_half(1, 256, 128).permute(1, 2, 0) for _ in range(3))
        op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
        compile_args = [
            from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            for t in (a, b, c)
        ]
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")

        def f(a, b, c):
            gemm(a, b, c)
            return c

        pack = htc.pack_params

        def flipped(launch, slots, pointers):
            out = pack(launch, slots, pointers)
            out[launch.descriptors[0].param][56] ^= 1  # a box extent
            return out

        with mock.patch.object(htc, "pack_params", flipped):
            ((_, call),) = trace(f, (a, b, c)).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("parameter 1 holds other bytes", call.reason)

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    def test_tma_descriptors_are_reencoded_only_on_change(self):
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")
        a, b, c = (_half(1, 128, 64).permute(1, 2, 0) for _ in range(3))
        op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), (1, 1), True)
        compile_args = [
            from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            for t in (a, b, c)
        ]
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        gemm = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")
        calls = 350  # 1050 descriptors, past the 1024 entries an LRU of encodes held

        def f(a, b):
            out = []
            for _ in range(calls):
                c = a.new_empty(1, 128, 128).permute(1, 2, 0)
                gemm(a, b, c)
                out.append(c)
            return out

        r = HostTraceReplay(f)
        a, b = (_half(1, 128, 64).permute(1, 2, 0) for _ in range(2))
        r(a, b)
        r(a, b)
        encodes, replaces = torch._C._host_trace_tma_counts()
        a, b = (_half(1, 128, 64).permute(1, 2, 0) for _ in range(2))
        out = r(a, b)
        # every descriptor of A and B moved and none was encoded again
        after = torch._C._host_trace_tma_counts()
        self.assertEqual(after[0] - encodes, 0)
        self.assertGreaterEqual(after[1] - replaces, 2 * calls)
        ref = a.new_empty(1, 128, 128).permute(1, 2, 0)
        gemm(a, b, ref)
        self.assertEqual(out[-1], ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.eager), (1, 0))

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    @parametrize("entry_descriptors", [True, False])
    def test_a_tma_compile_choice_dispatches_again_into_one_variant(self, entry_descriptors):
        # _tma_gemm's N tile (128 from m 512, else 64) is its op's own choice:
        # an entry of the one variant, with its TMA descriptors; without entry
        # descriptors native add_entry refuses them and the choice's class traces
        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")
        a, b, c = (_half(1, 256, 128).permute(1, 2, 0) for _ in range(3))
        compile_args = [
            from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            for t in (a, b, c)
        ]
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        for big, tile in ((True, (128, 128)), (False, (128, 64))):
            op = bw.DenseGemmKernel(cutlass.Float32, False, tile, (1, 1), True)
            _TMA_GEMMS[big] = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")
        torch._C._host_trace_entry_descriptors(entry_descriptors)
        try:
            r = HostTraceReplay(_tma_gemm)
            encodes = torch._C._host_trace_tma_counts()[0]
            for m in (512, 256, 1024, 384, 512, 256):
                a, b = _half(1, m, 128).permute(1, 2, 0), _half(1, 256, 128).permute(1, 2, 0)
                self.assertEqual(r(a, b), _tma_gemm(a, b), atol=0, rtol=0)
            # a call whose rows repeat packs no kernel, so no descriptor is compared
            r(a, b)
            counts = torch._C._host_trace_kernel_packs(), torch._C._host_trace_tma_counts()
            r(a, b)
            self.assertEqual((torch._C._host_trace_kernel_packs(), torch._C._host_trace_tma_counts()), counts)
        finally:
            torch._C._host_trace_entry_descriptors(True)
        self.assertGreater(torch._C._host_trace_tma_counts()[0], encodes)
        if entry_descriptors:
            self.assertEqual((r.traces, r.redispatches, len(r.variants), r.eager), (1, 1, 1, 0))
            self.assertEqual(r.redispatch_refusals, {})
        else:
            self.assertEqual((r.traces, r.redispatches, len(r.variants), r.eager), (2, 0, 2, 0))
            self.assertEqual(list(r.redispatch_refusals), ["a TMA launch in host_trace_test.tma_gemm.default"])

    @unittest.skipIf(
        not TEST_CUDA or torch.cuda.get_device_capability() < (10, 0),
        "requires Blackwell",
    )
    @parametrize("entry_descriptors", [True, False])
    def test_a_tma_compile_choice_folds_into_one_variant(self, entry_descriptors):
        # as above with no redispatch (a trusted replay's way): the second
        # tile's trace folds its TMA launch into the variant as an entry
        from torch.cuda._host_trace_lower_tape import FoldRefused

        bw = _example("cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm")
        a, b, c = (_half(1, 256, 128).permute(1, 2, 0) for _ in range(3))
        compile_args = [
            from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
            for t in (a, b, c)
        ]
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        for big, tile in ((True, (128, 128)), (False, (128, 64))):
            op = bw.DenseGemmKernel(cutlass.Float32, False, tile, (1, 1), True)
            _TMA_GEMMS[big] = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")
        torch._C._host_trace_entry_descriptors(entry_descriptors)
        try:
            with mock.patch.object(_host_trace_replay, "redispatch", side_effect=FoldRefused("no redispatch")):
                r = HostTraceReplay(_tma_gemm)
                for m in (512, 256, 1024, 384, 512, 256):
                    a, b = _half(1, m, 128).permute(1, 2, 0), _half(1, 256, 128).permute(1, 2, 0)
                    self.assertEqual(r(a, b), _tma_gemm(a, b), atol=0, rtol=0)
        finally:
            torch._C._host_trace_entry_descriptors(True)
        self.assertGreater(r.replays, 0)
        if entry_descriptors:
            self.assertEqual((r.traces, r.folds, len(r.variants), r.eager), (2, 1, 1, 0))
            self.assertEqual(r.fold_refusals, {})
        else:
            self.assertEqual((r.traces, r.folds, len(r.variants), r.eager), (2, 0, 2, 0))
            self.assertEqual(list(r.fold_refusals), ["an RNG, TMA or CPU scalar launch in host_trace_test.tma_gemm.default"])

    @parametrize(
        "kernel",
        [
            "add",
            "add_explicit",
            "add_static",
            "ampere",
            "blackwell_1x1",
            "blackwell_2x1",
            "blackwell_2x2",
        ],
    )
    def test_witness_agrees_on_every_kernel(self, kernel):
        if kernel.startswith("add"):
            add = (
                _compile_add(shape=(1024,))
                if kernel == "add_static"
                else getattr(self, kernel)
            )
            args = tuple(torch.randn(1024, device="cuda") for _ in range(3))

            def fn(a, b, c):
                streams = (
                    [torch.cuda.current_stream()] if kernel == "add_explicit" else []
                )
                add(a, b, c, a.shape[0], *streams)
        elif kernel == "ampere":
            tg = _example(
                "cute/ampere/kernel/dense_gemm/tensorop_gemm.py", "tensorop_gemm"
            )
            args = (_half(1, 256, 128), _half(1, 128, 256), _half(1, 256, 256))
            cute_args = tg.mark_dynamic_layout(
                *args, 2, 2, 2, cutlass.Float16, cutlass.Float16
            )
            op = tg.TensorOpGemm(
                cutlass.Float16, cutlass.Float16, cutlass.Float32, (2, 2, 1), False
            )
            stream = make_fake_stream(use_tvm_ffi_env_stream=True)
            fn = cute.compile(
                tg.bmm, op, *cute_args, stream, options="--enable-tvm-ffi"
            )
        else:
            if torch.cuda.get_device_capability() < (10, 0):
                raise unittest.SkipTest("requires Blackwell")
            bw = _example(
                "cute/blackwell/kernel/dense_gemm/dense_gemm.py", "bw_dense_gemm"
            )
            cluster = tuple(int(x) for x in kernel.split("_")[1].split("x"))
            args = tuple(_half(1, 256, c).permute(1, 2, 0) for c in (128, 128, 256))
            op = bw.DenseGemmKernel(cutlass.Float32, False, (128, 128), cluster, True)
            compile_args = [
                from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                for t in args
            ]
            stream = make_fake_stream(use_tvm_ffi_env_stream=True)
            fn = cute.compile(op, *compile_args, stream, options="--enable-tvm-ffi")

        def f(*xs):
            fn(*xs)
            return xs[-1]

        pack, witnessed = htc.pack_params, []

        def counted(launch, slots, pointers):
            witnessed.append(launch.name)
            return pack(launch, slots, pointers)

        with mock.patch.object(htc, "pack_params", counted):
            launches = [r for _, r in trace(f, args).launches]
        self.assertTrue(launches)
        self.assertTrue(all(isinstance(r, KernelLaunch) for r in launches), launches)
        self.assertEqual(witnessed, [r.name for r in launches])

    def _compile_without_tvm_ffi(self, jit):
        x = torch.randn(1000, device="cuda")
        args = [_cute_tensor(x) for _ in range(3)]
        return cute.compile(jit, *args, cutlass.Int32(0), cuda_driver.CUstream(0))

    def test_a_from_dlpack_on_another_thread_during_a_trace(self):
        # the trace's from_dlpack hook is process-wide: an untraced thread's
        # from_dlpack makes cutlass's own tensor, and its call runs (on a
        # stream of its own, allocating nothing: the trace holds a capture)
        add, cute_tensor_class = self._compile_without_tvm_ffi(_add), cute.runtime._Tensor
        a, c, side = torch.randn(1000, device="cuda"), torch.zeros(1000, device="cuda"), torch.cuda.Stream()
        seen = []

        def other():
            seen.append((cute.runtime._Tensor is htc._cute_tensor, isinstance(from_dlpack(a), cute_tensor_class)))
            add(_cute_tensor(a), _cute_tensor(a), _cute_tensor(c), 1000, cuda_driver.CUstream(side.cuda_stream))

        def f(x):
            thread = threading.Thread(target=other)
            thread.start()
            thread.join()
            return x + 1

        trace(f, (a,), warm_up=False)
        side.synchronize()
        self.assertEqual(seen, [(True, True)])
        self.assertEqual(c, a + a)

    def test_a_from_dlpack_call_without_tvm_ffi_is_a_launch(self):
        add = self._compile_without_tvm_ffi(_add)

        def f(a, b):
            c = torch.empty_like(a)
            stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
            add(_cute_tensor(a), _cute_tensor(b), _cute_tensor(c), a.shape[0], stream)
            return c

        x = torch.randn(1000, device="cuda")
        ((_, launch),) = trace(f, (x, x)).launches
        self.assertIsInstance(launch, KernelLaunch)
        r = HostTraceReplay(f)
        for n in (1000, 4096, 777, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))

        def reads_shape(a):
            from_dlpack(a).shape
            return a

        with self.assertRaisesRegex(Declined, "reads shape of a from_dlpack tensor of a traced tensor"):
            trace(reads_shape, (x,), warm_up=False)

    def test_an_undescribed_from_dlpack_call_is_an_eager_call(self):
        add, traced = self._compile_without_tvm_ffi(_add_shifted), self.add

        def f(a, b):
            c = torch.empty_like(a)
            stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
            add(_cute_tensor(a), _cute_tensor(b), _cute_tensor(c), a.shape[0], stream)
            d = torch.empty_like(a)
            traced(c, b, d, a.shape[0])
            return d

        x = torch.randn(1000, device="cuda")
        (_, call), (_, launch) = trace(f, (x, x)).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIsInstance(launch, KernelLaunch)
        self.assertIn("unsupported host op arith.shrsi", call.reason)
        self.assertFalse(any(isinstance(a, htc._DLPackArg) for a in call.args))
        # the replay makes the CuTe tensors of its own tensors
        r = HostTraceReplay(f)
        for n in (1000, 4096, 1000):
            a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
            self.assertEqual(r(a, b), a + b + b, atol=0, rtol=0)
        self.assertEqual((r.traces, r.replays, r.eager), (1, 2, 0))

    def test_a_from_dlpack_bound_before_the_hook_is_a_launch(self):
        # SGLang imports its TGV GEMM, which binds from_dlpack by name, at
        # model load: before torch.cuda._host_trace_cute installs its hooks
        script = """if True:
            import sys
            from cutlass.cute.runtime import from_dlpack
            hooked = "torch.cuda._host_trace_cute" in sys.modules
            sys.path.insert(0, sys.argv[1])
            import cuda.bindings.driver as cuda_driver, torch, test_cuda_host_trace_cute as t
            add = t.TestHostTraceCute._compile_without_tvm_ffi(None, t._add)
            def f(a, b):
                c = torch.empty_like(a)
                stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
                add(*[from_dlpack(x, assumed_align=16).mark_layout_dynamic() for x in (a, b, c)], a.shape[0], stream)
                return c
            r, ok = t.HostTraceReplay(f), []
            for n in (1000, 4096, 1000):
                a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                ok.append(torch.equal(r(a, b), a + b))
            print(hooked, all(ok), r.traces, r.replays, r.eager)
        """
        out = subprocess.run(
            [sys.executable, "-c", script, os.path.dirname(os.path.abspath(__file__))],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(out.split(), ["False", "True", "1", "2", "0"])

    def test_cute_tensor_construction_is_a_launch(self):
        # from_dlpack's options, the changes libraries make to its tensor, and
        # a torch tensor passed as it (cutlass's TensorAdapter)
        def fp8(t):
            a = from_dlpack(t, assumed_align=16)
            a.element_type = cutlass.Float8E4M3FN
            return a.mark_layout_dynamic()

        def tf32(t, **kwargs):
            a = from_dlpack(t, force_tf32=True, **kwargs)
            self.assertEqual(a.element_type, cutlass.TFloat32)
            return a.mark_layout_dynamic()

        f32 = functools.partial(torch.randn, device="cuda")
        fp8_bytes = lambda n: f32(n).to(torch.float8_e4m3fn).view(torch.uint8)  # noqa: E731
        cases = {
            "force_tf32": (functools.partial(tf32, assumed_align=16), f32, lambda x: x + 1),
            "use_32bit_stride": (lambda t: from_dlpack(t, use_32bit_stride=True).mark_layout_dynamic(0), f32, lambda x: x + 1),
            "compact": (lambda t: from_dlpack(t).mark_compact_shape_dynamic(0, stride_order=t.dim_order(), divisibility=8), f32, lambda x: x + 1),
            "fp8 over bytes": (fp8, fp8_bytes, lambda u: u.view(torch.float8_e4m3fn).float() + 1),
            "torch tensor": (lambda t: t, f32, lambda x: x + 1),
            "tvm_ffi force_tf32": (functools.partial(tf32, enable_tvm_ffi=True), f32, lambda x: x + 1),
        }
        for name, (arg, make, ref) in cases.items():
            with self.subTest(name):
                tvm_ffi = name.startswith("tvm_ffi")
                x = make(1024)
                compile_arg = from_dlpack(x).mark_layout_dynamic() if name == "torch tensor" else arg(x)
                out = from_dlpack(torch.empty(1024, device="cuda"), enable_tvm_ffi=tvm_ffi).mark_layout_dynamic()
                options = "--enable-tvm-ffi" if tvm_ffi else ""
                fn = cute.compile(_plus_one, compile_arg, out, cutlass.Int32(0), cuda_driver.CUstream(0), options=options)

                def f(a):
                    c = torch.empty(a.shape, device="cuda")
                    fn(arg(a), c, a.shape[0], cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))
                    return c

                ((_, launch),) = trace(f, (x,)).launches
                self.assertIsInstance(launch, KernelLaunch)
                r = HostTraceReplay(f)
                for n in (1024, 4096, 2048, 1024):
                    x = make(n)
                    self.assertEqual(r(x), ref(x), atol=0, rtol=0)
                self.assertEqual((r.traces, r.replays, r.eager), (1, 3, 0))
        # outside a trace, cutlass's tensors are its own
        hooks = {cute.runtime._FakeTensor.__init__, cute.runtime._Tensor, ReadOnlyTensorWrapper.__new__}
        self.assertFalse(hooks & {htc._fake_tensor, htc._cute_tensor, htc._read_only})

    def test_cute_tensor_construction_declines(self):
        add, x = self._compile_without_tvm_ffi(_add), torch.randn(1000, device="cuda")

        def stream():
            return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

        def by_keyword(a):
            add(a, a, mC=a, n=1000, stream=stream())
            return a

        def jit_call(a):
            _add(_cute_tensor(a), _cute_tensor(a), _cute_tensor(a), 1000, stream())
            return a

        def by_pointer(a):
            _add(make_ptr(cutlass.Float32, a.data_ptr(), cute.AddressSpace.gmem), a, a, 1000, stream())
            return a

        cases = ((by_keyword, "passed by keyword"), (jit_call, "of a from_dlpack tensor of a traced tensor"), (by_pointer, "make_ptr from a traced tensor's data_ptr"))
        for g, msg in cases:
            with self.assertRaisesRegex(Declined, msg):
                trace(g, (x,), warm_up=False)

        def tvm_ffi_by_keyword(a):
            self.add(a, a, mC=a, n=1000)
            return a

        ((_, call),) = trace(tvm_ffi_by_keyword, (x,), warm_up=False).launches
        self.assertIsInstance(call, EagerCall)
        self.assertIn("keyword arguments", call.reason)

instantiate_parametrized_tests(TestHostTraceCute)

def setUpModule():
    from torch.cuda import _host_trace_hint_audit
    import torch.cuda._host_trace_capture as capture

    _host_trace_hint_audit.enable_for_tests()
    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
