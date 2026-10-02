# Owner(s): ["module: cuda graphs"]

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
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

    _EXAMPLES = os.environ.get(
        "CUTE_DSL_EXAMPLES",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "third_party/cutlass/examples/python/CuTeDSL",
        ),
    )

    def _example(path, name):
        path = os.path.join(_EXAMPLES, path)
        if not os.path.exists(path):
            raise unittest.SkipTest(f"the CuTe DSL example {path} is not found")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


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

        with self.assertRaisesRegex(Declined, "DLPack export of a traced tensor"):
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
        self.assertIn("float16 where the compile has !cute.memref<f32", call.reason)
        r = HostTraceReplay(f)
        self.assertEqual(r(x), x + x, atol=0, rtol=0)
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
        self.assertEqual(htc._wrap("i32", 2**31 + 5, []), -(2**31) + 5)
        self.assertEqual(htc._wrap("i32", -(2**31) - 1, []), 2**31 - 1)
        self.assertEqual(htc._wrap("i64", 2**31 + 5, []), 2**31 + 5)
        ty = '!cute.int_tuple<"(?,?{i64})">'
        self.assertEqual(htc._wrap(ty, (2**32 + 1, 2**32 + 1), []), (1, 2**32 + 1))

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
        def check(add, reason):
            def f(a, b):
                c = torch.empty_like(a)
                add(a, b, c, a.shape[0])
                return c

            a, b = torch.randn(1000, device="cuda"), torch.randn(1000, device="cuda")
            ((_, call),) = trace(f, (a, b)).launches
            self.assertIsInstance(call, EagerCall)
            self.assertIn(reason, call.reason)
            r = HostTraceReplay(f)
            for n in (1000, 2048):
                a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
                self.assertEqual(r(a, b), a + b, atol=0, rtol=0)

        with tempfile.TemporaryDirectory() as d:
            path = self._export(self.add, d)
            os.remove(path + htc.SIDECAR)
            check(cute.runtime.load_module(path, enable_tvm_ffi=True).func, "without its host function")
            with open(path + htc.SIDECAR, "w") as f:
                json.dump({"error": "its host function did not load: X"}, f)
            check(cute.runtime.load_module(path, enable_tvm_ffi=True).func, "did not load: X")

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

    def test_a_disk_cache_entry_without_its_host_function_recompiles(self):
        # an object quack's jit_cache exported before its compile was observed
        # has no host function beside it: loading it is a cache miss
        import torch._vendor.quack.cache as quack_cache
        from torch._vendor.quack.rmsnorm import _compile_rmsnorm_fwd, rmsnorm_fwd

        def f(x, w):
            return rmsnorm_fwd(x, w)[0]

        def bf16(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as d, quack_cache.cache_dir_override(d), mock.patch.object(quack_cache, "CACHE_ENABLED", True):
            _compile_rmsnorm_fwd.cache_clear()
            f(bf16(64, 1536), bf16(1536))
            sidecars = [os.path.join(root, n) for root, _, names in os.walk(d) for n in names if n.endswith(htc.SIDECAR)]
            self.assertEqual(len(sidecars), 1)
            os.remove(sidecars[0])
            _compile_rmsnorm_fwd.cache_clear()
            r = HostTraceReplay(f)
            for m in (64, 32):
                x, w = bf16(m, 1536), bf16(1536)
                self.assertEqual(r(x, w), f(x, w), atol=0, rtol=0)
            self.assertEqual((r.traces, r.replays, r.eager), (1, 1, 0))
            self.assertTrue(os.path.exists(sidecars[0]))

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
        r = HostTraceReplay(f)
        # N tiles of 128 pick the raster factor: 1 tile 1, 2 tiles 2, 3 to 5
        # tiles 4, more 8; each is the kernel's host dispatched again
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
        for m, n, k in sizes:
            a, b = _half(1, m, k), _half(1, k, n)
            ref = a.new_empty(1, m, n)
            gemm(a, b, ref)
            self.assertEqual(r(a, b), ref, atol=0, rtol=0)
        self.assertEqual((r.traces, r.redispatches, r.replays, r.eager), (1, 3, 7, 0))

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
            view, predicates = memref(ty, t, what)
            if what == "arg2":
                l, m, n = view["shape"]
                view["shape"] = (l, n, m)
            return view, predicates

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


instantiate_parametrized_tests(TestHostTraceCute)

def setUpModule():
    import torch.cuda._host_trace_capture as capture

    capture.raise_trace_disagreements = True
    torch.cuda._host_trace.raise_unexpected = True


if __name__ == "__main__":
    run_tests()
