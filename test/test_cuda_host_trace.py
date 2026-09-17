# Owner(s): ["module: cuda"]

import gc
import json
import math
import operator
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfRocm,
    TestCase,
)


if torch.cuda.is_available():
    from torch.cuda import _host_trace as ht

# the op whose CUDA host is traced: it returns the output and the two statistics
layer_norm = torch.ops.aten.native_layer_norm.default
# the source tree, for the tests that read the recorder's sources and run its
# lint (torch.__file__ is site-packages in CI); a wheel-only environment skips them
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTrace(TestCase):
    N = 4096

    def _inputs(self, M, dtype=torch.bfloat16, offset=0):
        flat = torch.randn(M * self.N + offset, device="cuda", dtype=dtype)
        x = flat[offset : offset + M * self.N].view(M, self.N)
        w = torch.randn(self.N, device="cuda", dtype=dtype)
        b = torch.randn(self.N, device="cuda", dtype=dtype)
        return x, w, b

    def _args(self, x, w, b, eps=1e-5):
        return (x, [self.N], w, b, eps)

    def _trace(self, M=64, **kwargs):
        args = self._args(*self._inputs(M, **kwargs))
        return ht.trace(layer_norm, args), args

    def _check(self, out, x, w, b, M):
        self.assertEqual(
            out[0], F.layer_norm(x, (self.N,), w, b, 1e-5), atol=2e-2, rtol=2e-2
        )
        xf = x.float()
        self.assertEqual(out[1].view(M), xf.mean(-1), atol=1e-3, rtol=1e-3)
        self.assertEqual(
            out[2].view(M),
            torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_trace_records_the_forward(self):
        tape, _ = self._trace()
        self.assertEqual(tape.num_launches, 1)
        self.assertEqual(tape.num_allocations, 3)
        self.assertGreater(tape.num_guards, 0)
        parsed = json.loads(tape.to_json())
        self.assertEqual(len(parsed["launches"]), 1)
        self.assertEqual(len(parsed["outputs"]), 3)
        launch = parsed["launches"][0]
        # the grid is the row count: the size symbol of the input's first dim,
        # not a number
        self.assertEqual(launch["grid"][0], parsed["inputs"][0]["sizes"][0])
        pointers = [p for p in launch["params"] if p["kind"] == "ptr"]
        self.assertEqual(len(pointers), 6)
        self.assertTrue(all(not p["const"] for p in pointers))
        # the kernel's signature says which pointers it may write
        self.assertEqual(
            [p["access"] for p in pointers], ["r", "r", "r", "rw", "rw", "rw"]
        )
        # this kernel's block and shared memory come from device properties,
        # not from the shape
        self.assertTrue(
            all(isinstance(d, int) and d >= 1 for d in launch["block_expr"])
        )
        self.assertIsInstance(launch["smem"], int)

    def test_replay_matches_eager_at_other_shapes(self):
        tape, args = self._trace()
        variant = ht.build(tape, layer_norm, args)
        for M in (64, 48, 5, 128):
            x, w, b = self._inputs(M)
            self._check(variant.replay(self._args(x, w, b)), x, w, b, M)

    def test_trace_at_batch_one_serves_other_batches(self):
        # decode traces are taken at batch 1; the contiguity and view checks
        # must not pin the trace there
        tape, args = self._trace(1)
        variant = ht.build(tape, layer_norm, args)
        for M in (8, 1, 3):
            x, w, b = self._inputs(M)
            self._check(variant.replay(self._args(x, w, b)), x, w, b, M)

    def test_misaligned_input_misses_on_the_alignment_guard(self):
        tape, args = self._trace()
        variant = ht.build(tape, layer_norm, args)
        # the host reads the address of every buffer through an alignment test,
        # so a storage offset that keeps 8-byte alignment replays and one that
        # does not is a named miss
        self.assertIsNotNone(
            variant.try_replay(self._args(*self._inputs(64, offset=8)))
        )
        odd = self._inputs(64, offset=1)
        self.assertIsNone(variant.try_replay(self._args(*odd)))
        with self.assertRaisesRegex(ht.Miss, "guard failed"):
            variant.replay(self._args(*odd))

    def test_raw_data_ptr_on_a_traced_input_raises(self):
        x, w, b = self._inputs(8)

        def read_pointer(t, shape, weight, bias, eps):
            return t.data_ptr()

        with self.assertRaisesRegex(RuntimeError, "sym_const_data_ptr"):
            ht.trace(read_pointer, self._args(x, w, b))
        # the failed trace left nothing behind
        self.assertFalse(torch._C._host_trace_tracing())
        tape, _ = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

    def test_views_of_the_output_replay_at_the_new_shape(self):
        # a view taken at the top level of the traced function is symbolic too:
        # the replayed slice has the new batch, not the traced one
        def fn(t, shape, weight, bias, eps):
            return layer_norm(t, shape, weight, bias, eps)[0][:, :8]

        tape, args = self._trace(16)
        tape = ht.trace(fn, args)
        variant = ht.build(tape, fn, args)
        x, w, b = self._inputs(24)
        (out,) = variant.replay(self._args(x, w, b))
        self.assertEqual(tuple(out.shape), (24, 8))
        self.assertEqual(
            out, F.layer_norm(x, (self.N,), w, b, 1e-5)[:, :8], atol=2e-2, rtol=2e-2
        )

    def test_unconverted_ops_decline_by_name(self):
        x, w, b = self._inputs(8)

        def after(t, shape, weight, bias, eps):
            return torch.sigmoid(layer_norm(t, shape, weight, bias, eps)[0])

        def before(t, shape, weight, bias, eps):
            return layer_norm(torch.relu(t), shape, weight, bias, eps)

        with self.assertRaisesRegex(ht.Declined, "aten.sigmoid.default"):
            ht.trace(after, self._args(x, w, b))
        with self.assertRaisesRegex(ht.Declined, "aten.relu.default"):
            ht.trace(before, self._args(x, w, b))
        # a non-contiguous input makes the host copy it: a named decline, not
        # an internal error
        xt = torch.randn(self.N, 8, device="cuda", dtype=torch.bfloat16).t()
        with self.assertRaisesRegex(ht.Declined, "aten.clone.default"):
            ht.trace(layer_norm, self._args(xt, w, b))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_outputs_survive_later_replays(self):
        tape, args = self._trace(32)
        variant = ht.build(tape, layer_norm, args)
        held = []
        for M in (32, 48, 64, 80):
            x, w, b = self._inputs(M)
            out = variant.replay(self._args(x, w, b))
            held.append((out[0], F.layer_norm(x, (self.N,), w, b, 1e-5)))
        torch.cuda.synchronize()
        for got, want in held:
            self.assertEqual(got, want, atol=2e-2, rtol=2e-2)

    def test_input_contract(self):
        tape, args = self._trace(16)
        variant = ht.build(tape, layer_norm, args)
        # dtype, and any non-tensor argument, are part of the variant
        half = self._inputs(16, dtype=torch.float16)
        self.assertIsNone(variant.try_replay(self._args(*half)))
        self.assertIsNone(variant.try_replay(self._args(*self._inputs(16), eps=2e-5)))
        # a transposed input misses on the stride guards
        y = torch.randn(self.N, 16, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(variant.try_replay(self._args(y.t(), *self._inputs(16)[1:])))

    def test_mixed_dtypes_raise_in_ordinary_mode_and_in_a_trace(self):
        x, w, b = self._inputs(16)
        bad = self._args(x, w.float(), b)
        message = "expected scalar type BFloat16 but found Float"
        with self.assertRaisesRegex(RuntimeError, message):
            layer_norm(*bad)
        with self.assertRaisesRegex(RuntimeError, message):
            ht.trace(layer_norm, bad)
        self.assertFalse(torch._C._host_trace_tracing())
        tape, _ = self._trace(16)
        self.assertEqual(tape.num_launches, 1)

    def test_exception_in_the_traced_call_closes_the_capture(self):
        x, w, b = self._inputs(16)

        def fn(t, shape, weight, bias, eps):
            layer_norm(t, shape, weight, bias, eps)
            raise ValueError("after the launch")

        # the context manager keeps the traceback, and with it the recorder,
        # alive: the capture must be closed anyway
        with self.assertRaises(ValueError) as cm:
            ht.trace(fn, self._args(x, w, b))
        self.assertFalse(torch._C._host_trace_tracing())
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite((x.float() + 1).sum()).item())
        tape, _ = self._trace(16)
        self.assertEqual(tape.num_launches, 1)
        del cm

    def test_build_survives_eager_work_on_another_thread(self):
        # eager work on its own stream, with fresh allocations and stream
        # syncs, while this thread traces, builds and replays
        stop = threading.Event()
        errors = []
        iterations = [0]

        def worker():
            stream = torch.cuda.Stream()
            try:
                with torch.cuda.stream(stream):
                    k = 1
                    while not stop.is_set():
                        a = torch.randn(1024 * k, 1024, device="cuda")
                        (a[:256, :256] @ a[:256, :256]).tanh()
                        stream.synchronize()
                        iterations[0] += 1
                        k = k % 8 + 1
            except Exception as e:
                errors.append(e)

        interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-3)
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            for M in range(16, 36, 2):
                tape, args = self._trace(M)
                variant = ht.build(tape, layer_norm, args)
                x, w, b = self._inputs(M + 8)
                out = variant.replay(self._args(x, w, b))
                self.assertEqual(
                    out[0], F.layer_norm(x, (self.N,), w, b, 1e-5), atol=2e-2, rtol=2e-2
                )
                time.sleep(0.005)  # let the worker run between builds
        finally:
            stop.set()
            thread.join()
            sys.setswitchinterval(interval)
        self.assertEqual(errors, [])
        self.assertGreater(iterations[0], 10)

    @unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")
    def test_variant_on_a_second_device(self):
        # the first device is used first, so a process-wide capture stream would
        # be bound to it
        tape, args = self._trace(16)
        ht.build(tape, layer_norm, args)
        with torch.cuda.device(1):
            tape1, args1 = self._trace(16)
            variant = ht.build(tape1, layer_norm, args1)
            x, w, b = self._inputs(40)
            self.assertEqual(x.device.index, 1)
            out = variant.replay(self._args(x, w, b))
            torch.cuda.synchronize(1)
            self.assertEqual(
                out[0], F.layer_norm(x, (self.N,), w, b, 1e-5), atol=2e-2, rtol=2e-2
            )

    def test_argument_contract_on_the_python_surface(self):
        tape, args = self._trace(16)
        variant = ht.build(tape, layer_norm, args)
        x, w, b = self._inputs(16)
        # a tuple or a torch.Size is the same constant as the traced list
        self.assertIsNotNone(variant.try_replay((x, (self.N,), w, b, 1e-5)))
        self.assertIsNotNone(variant.try_replay((x, torch.Size([self.N]), w, b, 1e-5)))
        # a tensor position that is no longer a tensor is a miss
        self.assertIsNone(variant.try_replay((x, [self.N], None, b, 1e-5)))
        with self.assertRaisesRegex(ht.Declined, "only CUDA tensors"):
            ht.trace(layer_norm, self._args(x.cpu(), w.cpu(), b.cpu()))

    def test_device_class_is_part_of_the_contract(self):
        tape, args = self._trace(16)
        keys = [k for k, _ in tape.device_identity]
        for prop in (
            "name",
            "multi_processor_count",
            "shared_memory_per_block",
            "warp_size",
        ):
            self.assertIn(prop, keys)
        # a tape from a device of another class (same SM count, another model)
        # misses at build before any GPU work
        other = tuple(
            (k, ("Other GPU" if k == "name" else v)) for k, v in tape.device_identity
        )
        tape.device_identity = other
        with self.assertRaisesRegex(ht.Miss, "name=Other GPU"):
            ht.build(tape, layer_norm, args)

    def test_tape_json_is_reproducible(self):
        x, w, b = self._inputs(16)
        a = self._args(x, w, b)
        first = ht.trace(layer_norm, a).to_json()
        second = ht.trace(layer_norm, a).to_json()
        self.assertEqual(first, second)

    def test_float_guards_keep_the_program_order(self):
        # a float branch is replayed as the host computed it, not as sympy
        # would re-associate it: (x*0.1)*3 is not 0.30000000000000004*x
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.utils._sympy.printers import PythonPrinter

        def branch(flag):
            env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
            env.exact_float_arithmetic = flag
            src = ht._Src("s")
            s = env.create_symintnode(
                env.create_unspecified_symbol(7, src), hint=7, source=src
            )
            y = (torch.sym_float(s) / 10.0 * 0.1) * 3
            taken = bool(y > 0.21)
            guard = PythonPrinter().doprint(env.guards[-1].expr)
            return taken, guard, str(s.node.expr)

        host = ((7 / 10.0) * 0.1) * 3 > 0.21
        for flag in (True, False):
            taken, guard, name = branch(flag)
            self.assertEqual(taken, host)
            # the recorded guard is the condition that held; re-evaluated at the
            # traced value it must hold again, which the re-associated form
            # (0.30000000000000004*x) does not
            holds = eval(guard, {"math": math, "torch": torch, name: 7})
            self.assertEqual(holds, flag)

    def test_size_one_view_branches_are_guarded_on_the_traced_value(self):
        # a squeeze of the batch dim depends on whether it is 1: traced at 1 it
        # squeezes and the tape pins to 1, traced at 8 it does not and the
        # tape misses at 1. Never the other shape.
        def fn(t, shape, weight, bias, eps):
            return layer_norm(t, shape, weight, bias, eps)[0].squeeze(0)

        for traced_at, other in ((1, 8), (8, 1)):
            args = self._args(*self._inputs(traced_at))
            variant = ht.build(ht.trace(fn, args), fn, args)
            (out,) = variant.replay(args)
            self.assertEqual(tuple(out.shape), tuple(fn(*args).shape))
            self.assertEqual(out, fn(*args), atol=2e-2, rtol=2e-2)
            self.assertIsNone(variant.try_replay(self._args(*self._inputs(other))))
        # the plain forward does not depend on it: the batch-1 tape serves 8
        tape, args = self._trace(1)
        variant = ht.build(tape, layer_norm, args)
        x, w, b = self._inputs(8)
        self._check(variant.replay(self._args(x, w, b)), x, w, b, 8)

    def _compiles(self, body):
        # syntax-only compile of a host TU against the public headers
        from torch.utils.cpp_extension import CUDA_HOME, include_paths

        cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
        if cxx is None or CUDA_HOME is None:
            self.skipTest("no host compiler or CUDA_HOME")
        src = (
            "#include <ATen/cuda/host_trace/Launch.h>\n"
            "namespace ht = at::cuda::host_trace;\n"
            "int64_t f(const c10::SymInt& m) {\n" + body + "\n}\n"
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "host.cpp")
            with open(path, "w") as f:
                f.write(src)
            cmd = [cxx, "-std=c++20", "-fsyntax-only", "-D_GLIBCXX_USE_CXX11_ABI=1"]
            cmd += [f"-I{p}" for p in include_paths(device_type="cuda")]
            cmd += ["-I" + os.path.join(CUDA_HOME, "include"), path]
            r = subprocess.run(cmd, capture_output=True, text=True)
            return r.returncode == 0, r.stderr

    def test_recorder_hint_reads_are_not_host_facing(self):
        # a host reads a symbolic int with guard_int (a recorded guard) or
        # expect_int (raises); the recorder's guard-free hint read is a private
        # member a host TU cannot name, and the hooks header is not installed
        # behind the public headers
        ok, err = self._compiles("return m.guard_int(__FILE__, __LINE__);")
        self.assertTrue(ok, err)
        ok, err = self._compiles("return ht::Hints::of(m);")
        self.assertFalse(ok)
        self.assertIn("private", err)
        ok, _ = self._compiles("return ht::detail::hint_of(m);")
        self.assertFalse(ok)
        root = os.path.join(_REPO_ROOT, "aten", "src", "ATen", "cuda", "host_trace")
        if os.path.isdir(root):
            for name in ("Recorder.h", "Field.h", "Launch.h", "Tape.h", "Exec.h"):
                with open(os.path.join(root, name)) as f:
                    self.assertNotIn("Hooks.h", f.read(), name)
        # and the hooks header is not installed at all: an out-of-tree host
        # cannot include it
        installed = os.path.join(
            os.path.dirname(torch.__file__), "include", "ATen", "cuda", "host_trace"
        )
        if os.path.isdir(installed):
            self.assertTrue(os.path.exists(os.path.join(installed, "Recorder.h")))
            self.assertFalse(os.path.exists(os.path.join(installed, "Hooks.h")))

    def test_synchronous_cuda_work_inside_a_trace_declines(self):
        # a synchronous CUDA call would run at the trace and never in the
        # replayed graph; the trace capture is thread-local so it fails there
        def fn(t, shape, weight, bias, eps):
            out = layer_norm(t, shape, weight, bias, eps)
            torch.cuda.synchronize()
            return out

        x, w, b = self._inputs(8)
        with self.assertRaisesRegex(
            ht.Declined, "not permitted inside a stream capture"
        ):
            ht.trace(fn, self._args(x, w, b))
        self.assertFalse(torch._C._host_trace_tracing())
        torch.cuda.synchronize()
        tape, _ = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

    def test_capture_errors_are_classified_by_code(self):
        # an event or stream query inside the trace is a decline named after
        # the CUDA error, from the error code, not from the message text; a
        # user error that merely mentions capture propagates as it was raised
        x, w, b = self._inputs(8)

        def event_query(t, shape, weight, bias, eps):
            out = layer_norm(t, shape, weight, bias, eps)
            ev = torch.cuda.Event()
            ev.record()
            ev.query()
            return out

        def stream_query(t, shape, weight, bias, eps):
            out = layer_norm(t, shape, weight, bias, eps)
            torch.cuda.current_stream().query()
            return out

        def user_error(t, shape, weight, bias, eps):
            layer_norm(t, shape, weight, bias, eps)
            raise RuntimeError("my own message during capture")

        with self.assertRaisesRegex(ht.Declined, "cudaErrorCapturedEvent"):
            ht.trace(event_query, self._args(x, w, b))
        with self.assertRaisesRegex(ht.Declined, "cudaErrorStreamCapture"):
            ht.trace(stream_query, self._args(x, w, b))
        with self.assertRaises(RuntimeError) as cm:
            ht.trace(user_error, self._args(x, w, b))
        self.assertNotIsInstance(cm.exception, ht.Declined)
        self.assertEqual(str(cm.exception), "my own message during capture")
        self.assertFalse(torch._C._host_trace_tracing())

        # the completeness declines from the recorder are the same type
        def memset_inside(t, shape, weight, bias, eps):
            out = layer_norm(t, shape, weight, bias, eps)
            out[1].zero_()
            return out

        with self.assertRaises(ht.Declined):
            ht.trace(memset_inside, self._args(x, w, b))
        self.assertFalse(torch._C._host_trace_tracing())
        tape, _ = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

    def test_unconverted_hosts_that_read_pointers_decline_by_name(self):
        # an op whose CUDA entry reads a raw pointer of a traced input is a
        # decline naming the op, not a bare RuntimeError; rms norm's entry on
        # this base does exactly that, so it is not in the traceable set
        x, w, _ = self._inputs(8)
        with self.assertRaisesRegex(ht.Declined, "_fused_rms_norm"):
            ht.trace(torch.ops.aten._fused_rms_norm.default, (x, [self.N], w, 1e-5))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_view_guards_never_divide_by_zero(self):
        # x.view(M // 4, -1): traced at M=8 the tape guards that M // 4 is
        # nonzero before it divides by it; a replay at M=3 is a miss, not an
        # exception out of the guard evaluation
        def fn(t, shape, weight, bias, eps):
            out = layer_norm(t, shape, weight, bias, eps)[0]
            return out.view(out.shape[0] // 4, -1)

        args = self._args(*self._inputs(8))
        variant = ht.build(ht.trace(fn, args), fn, args)
        (out,) = variant.replay(self._args(*self._inputs(16)))
        self.assertEqual(tuple(out.shape), (4, 4 * self.N))
        for M in (3, 1):
            with self.assertRaisesRegex(ht.Miss, "guard failed"):
                variant.replay(self._args(*self._inputs(M)))

        def zero(t, shape, weight, bias, eps):
            return layer_norm(t, shape, weight, bias, eps)[0].view(0, -1)

        # eager raises for this view; the trace does too, without an internal
        # ZeroDivisionError
        with self.assertRaises(RuntimeError):
            zero(*args)
        with self.assertRaises(RuntimeError) as cm:
            ht.trace(zero, args)
        self.assertNotIsInstance(cm.exception, ZeroDivisionError)
        self.assertFalse(torch._C._host_trace_tracing())

    def test_views_between_ops_do_not_pin_the_batch(self):
        # a decode-style function: two layer norms with an unsqueeze, a
        # no-op expand, a transpose pair and a slice between them. None of
        # those depends on the batch being 1, so the batch-1 tape serves
        # batch 8 and the batch-8 tape serves batch 1. A squeeze does depend
        # on it and pins.
        def fn(t, shape, weight, bias, eps):
            h = layer_norm(t, shape, weight, bias, eps)[0]
            h = h.unsqueeze(1).expand(h.shape[0], 1, h.shape[1]).squeeze(1)
            h = h.transpose(0, 1).transpose(0, 1)[:, : self.N]
            h = h.expand(h.shape[0], self.N)
            return layer_norm(h, shape, weight, bias, eps)[0]

        def pins(t, shape, weight, bias, eps):
            h = layer_norm(t, shape, weight, bias, eps)[0]
            return layer_norm(h.squeeze(0).unsqueeze(0), shape, weight, bias, eps)[0]

        served = {}
        for traced_at in (1, 8):
            args = self._args(*self._inputs(traced_at))
            variant = ht.build(ht.trace(fn, args), fn, args)
            for M in (1, 8, 3):
                x, w, b = self._inputs(M)
                out = variant.try_replay(self._args(x, w, b))
                served[(traced_at, M)] = out is not None
                if out is not None:
                    self.assertEqual(
                        out[0], fn(x, [self.N], w, b, 1e-5), atol=2e-2, rtol=2e-2
                    )
        self.assertEqual(served, dict.fromkeys(served, True))
        for traced_at, other in ((1, 8), (8, 1)):
            args = self._args(*self._inputs(traced_at))
            variant = ht.build(ht.trace(pins, args), pins, args)
            self.assertIsNotNone(variant.try_replay(args))
            self.assertIsNone(variant.try_replay(self._args(*self._inputs(other))))

    def test_build_with_inputs_that_fail_the_guards_misses_before_gpu_work(self):
        # a build whose inputs fail the tape's own guards is a miss, not a
        # tape mismatch after a capture
        tape, _ = self._trace(64)
        odd = self._args(*self._inputs(64, offset=1))
        with self.assertRaisesRegex(ht.Miss, "guard failed"):
            ht.build(tape, layer_norm, odd)
        variant = ht.build(tape, layer_norm, self._args(*self._inputs(64)))
        self.assertEqual(variant.calls, 0)

    @unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")
    def test_build_uses_the_tapes_device(self):
        tape, args = self._trace(16)
        with torch.cuda.device(1):
            variant = ht.build(tape, layer_norm, args)
            x, w, b = self._inputs(24)
        self.assertEqual(x.device.index, 1)
        self.assertEqual(variant.device, 0)
        x, w, b = self._inputs(24)
        self._check(variant.replay(self._args(x, w, b)), x, w, b, 24)

    def test_builds_and_traces_on_other_threads_do_not_interfere(self):
        # a build synchronizes its own stream only: traces in flight on other
        # threads survive it, and several threads can trace, build and replay
        # at once
        stop = threading.Event()
        errors: list = []

        def builder():
            try:
                args = self._args(*self._inputs(8))
                tape = ht.trace(layer_norm, args)
                while not stop.is_set():
                    ht.build(tape, layer_norm, args)
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=builder)
        thread.start()
        try:
            time.sleep(0.2)
            for M in range(9, 19):
                tape, _ = self._trace(M)
                self.assertEqual(tape.num_launches, 1)
        finally:
            stop.set()
            thread.join()
        self.assertEqual(errors, [])

        results: list = []

        def worker(k):
            try:
                args = self._args(*self._inputs(8 + k))
                variant = ht.build(ht.trace(layer_norm, args), layer_norm, args)
                for j in range(5):
                    x, w, b = self._inputs(8 + k + j)
                    out = variant.replay(self._args(x, w, b))
                    results.append((out[0], F.layer_norm(x, (self.N,), w, b, 1e-5)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        torch.cuda.synchronize()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 15)
        for got, want in results:
            self.assertEqual(got, want, atol=2e-2, rtol=2e-2)

    def test_non_tensor_results_and_foreign_traced_tensors_are_declined(self):
        x, w, b = self._inputs(8)
        leaked: list = []

        def returns_a_size(t, shape, weight, bias, eps):
            leaked.append(layer_norm(t, shape, weight, bias, eps)[0])
            return t.shape[0]

        with self.assertRaisesRegex(ht.Declined, "returned SymInt"):
            ht.trace(returns_a_size, self._args(x, w, b))
        # the warm-up call leaked a real tensor first, the symbolic run a traced one
        self.assertEqual(len(leaked), 2)
        with self.assertRaisesRegex(ht.Declined, "traced tensor of another trace"):
            ht.trace(layer_norm, self._args(leaked[-1], w, b))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_trace_warms_up_once_before_the_symbolic_run(self):
        # one-time initializations happen in the warm-up call on the real
        # inputs, never inside the trace; warm_up=False skips it
        calls = []

        def fn(t, shape, weight, bias, eps):
            calls.append(isinstance(t, ht._TracedTensor))
            return layer_norm(t, shape, weight, bias, eps)

        args = self._args(*self._inputs(8))
        ht.trace(fn, args)
        self.assertEqual(calls, [False, True])
        calls.clear()
        ht.trace(fn, args, warm_up=False)
        self.assertEqual(calls, [True])

    def test_sync_memory_api_lint_catches_a_converted_host(self):
        # the host contract's synchronous memory calls are invisible to a
        # capture; the lint reports them in every translation unit under the
        # CUDA host directories (however the host reaches the recorder's
        # headers) and in any file that includes them, and nothing else
        root = _REPO_ROOT
        linter = os.path.join(
            root, "tools", "linter", "adapters", "host_trace_sync_api_linter.py"
        )
        if not os.path.isfile(linter):
            self.skipTest("source tree not available")
        offender = (
            "#include <ATen/cuda/host_trace/Recorder.h>\n"
            "void host(void* d, const void* s) {\n"
            "  // cudaMemcpy( in a comment does not count\n"
            "  cudaMemcpyAsync(d, s, 4, cudaMemcpyDeviceToDevice, 0);\n"
            "  cudaMemcpy(d, s, 4, cudaMemcpyDeviceToDevice);\n"
            "  AT_CUDA_CHECK(cudaMemcpyToSymbol(d, s, 4));\n"
            "  cudaMemset(d, 0, 4);\n"
            "  cudaMemcpyPeer(d, 0, s, 1, 4);\n"
            "  cuMemcpyHtoD_v2(0, s, 4);\n"
            "  cudaMemset3D(make_cudaPitchedPtr(d, 4, 4, 1), 0, make_cudaExtent(4, 1, 1));\n"
            "  auto fp = &cudaMemcpy;\n"
            "  cuMemcpyHtoDAsync(0, s, 4, 0);\n"
            "  cudaMemcpy\n"
            "      (d, s, 4, cudaMemcpyDeviceToDevice);\n"
            "}\n"
        )
        bystander = offender.replace(
            "#include <ATen/cuda/host_trace/Recorder.h>", "#include <cuda_runtime.h>"
        )
        transitive = bystander.replace("<cuda_runtime.h>", '"my_host_header.h"')
        private = "struct at::cuda::host_trace::HintsInternal { };\n"

        def run(text, subdir=""):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, subdir, "host.cu")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(text)
                r = subprocess.run(
                    [sys.executable, linter, path], capture_output=True, text=True
                )
                self.assertEqual(r.returncode, 0, r.stderr)
                return [
                    json.loads(line) for line in r.stdout.splitlines() if line.strip()
                ]

        expected = [5, 6, 7, 8, 9, 10, 11, 13]
        found = run(offender)
        self.assertEqual(sorted(m["line"] for m in found), expected)
        self.assertTrue(all(m["code"] == "HOSTTRACE_SYNC_API" for m in found))
        self.assertEqual(run(bystander), [])
        found = run(transitive, "aten/src/ATen/native/cuda")
        self.assertEqual(sorted(m["line"] for m in found), expected)
        self.assertEqual(
            [m["name"] for m in run(private)], ["recorder-private-interface"]
        )
        for rel in (
            ("aten", "src", "ATen", "native", "cuda", "layer_norm_kernel.cu"),
            ("aten", "src", "ATen", "cuda", "detail", "BLASConstants.cu"),
            ("torch", "csrc", "cuda", "CUDAPluggableAllocator.cpp"),
        ):
            r = subprocess.run(
                [sys.executable, linter, os.path.join(root, *rel)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.stdout.strip(), "", (rel, r.stdout))

    @unittest.skipIf(torch.cuda.device_count() < 2, "needs two GPUs")
    def test_variant_built_on_a_second_device_from_a_trace_on_the_first(self):
        # the variant's device is where its exec runs and where its
        # allocations and events live: a tape traced on cuda:0 and built with
        # device=1 replays cuda:1 inputs entirely on cuda:1
        tape, _ = self._trace(16)
        with torch.cuda.device(1):
            x1, w1, b1 = self._inputs(16)
        variant = ht.build(tape, layer_norm, self._args(x1, w1, b1), device=1)
        self.assertEqual(variant.device, 1)
        with torch.cuda.device(1):
            x, w, b = self._inputs(24)
        out = variant.replay(self._args(x, w, b))
        self.assertTrue(all(o.device == torch.device("cuda", 1) for o in out))
        self._check(out, x, w, b, 24)

    def test_symint_indices_do_not_pin_the_batch(self):
        # Tensor.__getitem__ turns an integer index into a plain int through
        # SymInt.__index__ (a guard_int, a pin to the traced batch); on a
        # traced tensor an index built from a SymInt is routed to select /
        # slice / unsqueeze on the symbolic value instead
        N = self.N

        def last_row(x, w, b):
            return layer_norm(x, [N], w, b, 1e-5)[0][x.shape[0] - 1]

        def half_index(x, w, b):
            return layer_norm(x, [N], w, b, 1e-5)[0][x.shape[0] // 2]

        def middle_rows(x, w, b):
            return layer_norm(x, [N], w, b, 1e-5)[0][1 : x.shape[0] - 1]

        def right_half(x, w, b):
            h = layer_norm(x, [N], w, b, 1e-5)[0]
            return h[:, h.shape[1] // 2 :]

        def mixed(x, w, b):
            h = layer_norm(x, [N], w, b, 1e-5)[0]
            return h[None, ..., x.shape[0] - 1 :, 0]

        for fn in (last_row, half_index, middle_rows, right_half, mixed):
            x, w, b = self._inputs(16)
            tape = ht.trace(fn, (x, w, b))
            variant = ht.build(tape, fn, (x, w, b))
            for M in (8, 3, 16, 5):
                x, w, b = self._inputs(M)
                out = variant.replay((x, w, b))[0]
                ref = fn(x, w, b)
                self.assertEqual(out.shape, ref.shape, (fn.__name__, M))
                self.assertEqual(out.stride(), ref.stride(), (fn.__name__, M))
                self.assertEqual(out, ref, atol=2e-2, rtol=2e-2)

        def row_eight(x, w, b):
            return layer_norm(x, [N], w, b, 1e-5)[0][8]

        x, w, b = self._inputs(16)
        variant = ht.build(ht.trace(row_eight, (x, w, b)), row_eight, (x, w, b))
        x, w, b = self._inputs(3)
        with self.assertRaises(ht.Miss):
            variant.replay((x, w, b))

        # a mask index takes the stock path even beside a SymInt (and there
        # declines by name: masked indexing is not a traceable host)
        mask = torch.zeros(N, dtype=torch.bool, device="cuda")
        mask[::2] = True

        def masked(x, w, b):
            return layer_norm(x, [N], w, b, 1e-5)[0][x.shape[0] - 1, mask]

        x, w, b = self._inputs(16)
        with self.assertRaisesRegex(ht.Declined, "index"):
            ht.trace(masked, (x, w, b))

    def test_a_nested_capture_inside_a_trace_declines(self):
        # a second capture begun on this thread while the trace capture is
        # open fails with a non-capture code (cudaErrorIllegalState); it is a
        # decline by name, not a raw error, and the next trace is clean
        def nested(x, w, b):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(torch.cuda.Stream()):
                g.capture_begin()
                g.capture_end()
            return layer_norm(x, [self.N], w, b, 1e-5)

        x, w, b = self._inputs(8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(
                ht.Declined, "cudaErrorIllegalState|cudaErrorStreamCapture"
            ):
                ht.trace(nested, (x, w, b))
        tape, args = self._trace(8)
        x, w, b = self._inputs(4)
        self._check(
            ht.build(tape, layer_norm, args).replay(self._args(x, w, b)), x, w, b, 4
        )

    def test_non_contiguous_inputs_of_any_rank_trace_in_bounded_time(self):
        # a contiguity query on a non-contiguous rank-4 input used to feed the
        # negation of a rank-sized conjunction to the ShapeEnv, whose CNF pass
        # is exponential in rank: a channels-last input never finished tracing.
        # Term-wise guards keep it linear; the answer is either a served tape
        # or a decline at the copy the host makes, never a hang.
        def fn(t):
            return F.layer_norm(t.contiguous(), (t.shape[-1],))

        cases = [
            torch.randn(2, 8, 5, 8, device="cuda").to(
                memory_format=torch.channels_last
            ),
            torch.randn(3, 4, 5, 6, 8, device="cuda").transpose(0, 4),
            torch.randn(2, 3, 4, 5, 6, 8, device="cuda").permute(5, 0, 1, 2, 3, 4),
        ]
        for t in cases:
            t0 = time.time()
            try:
                tape = ht.trace(fn, (t,))
            except ht.Declined:
                tape = None
            self.assertLess(time.time() - t0, 30.0)
            self.assertFalse(torch._C._host_trace_tracing())
            if tape is None:
                continue
            variant = ht.build(tape, fn, (t,))
            out = variant.try_replay((torch.randn_like(t),))
            self.assertIsNotNone(out)
        # the True path records one guard per dim, the False path only the
        # failing dim's, both shape-generic: a permuted view served at a new
        # shape and missed where the layout no longer matches
        x = torch.randn(4, 8, 6, 16, device="cuda")
        fn2 = lambda t: F.layer_norm(t, (16,))  # noqa: E731
        tape = ht.trace(fn2, (x,))
        variant = ht.build(tape, fn2, (x,))
        y = torch.randn(3, 5, 7, 16, device="cuda")
        self.assertEqual(variant.try_replay((y,))[0], fn2(y))
        self.assertIsNone(variant.try_replay((y.transpose(1, 2),)))

    def test_ops_that_refuse_inside_the_trace_decline_by_name(self):
        # an op the mode runs during the symbolic run can raise (a host's own
        # check on the traced inputs, a copy the capture forbids): that is a
        # decline naming the op, with the op's error as the cause, not a raw
        # RuntimeError. A user's own exception still propagates unchanged.
        x, w, b = self._inputs(8)
        bad = self._args(x, w.float(), b)
        with self.assertRaisesRegex(
            ht.Declined, "native_layer_norm.*expected scalar type BFloat16"
        ) as cm:
            ht.trace(layer_norm, bad, warm_up=False)
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertFalse(torch._C._host_trace_tracing())

        class UserError(RuntimeError):
            pass

        def user_raises(*args):
            raise UserError("my own failure during capture")

        with self.assertRaises(UserError):
            ht.trace(user_raises, self._args(x, w, b), warm_up=False)
        self.assertFalse(torch._C._host_trace_tracing())

        # a tensor factory copies onto the device below the mode (the Python
        # dispatch keys are excluded around that copy), so its refusal under
        # the capture is the copy's own error; the recorder is clean after it
        def materializes(x, shape, w, b, eps):
            return layer_norm(x + torch.tensor(2.0, device="cuda"), shape, w, b, eps)

        with self.assertRaisesRegex(RuntimeError, "Cannot copy between CPU and CUDA"):
            ht.trace(materializes, self._args(x, w, b))
        self.assertFalse(torch._C._host_trace_tracing())
        tape, _ = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

    def test_cyclic_collections_do_not_invalidate_the_capture(self):
        # An earlier variant reachable only through a reference cycle is freed
        # by the cyclic collector; finalizing its graph on the capturing thread
        # during a trace would invalidate the thread-local capture. trace()
        # holds collections until it is over.
        thresholds = gc.get_threshold()
        enabled = gc.isenabled()
        try:
            gc.enable()
            for _ in range(4):
                tape, args = self._trace(16)
                variant = ht.build(tape, layer_norm, args)
                variant.replay(args)
                cycle = [variant, tape]
                cycle.append(cycle)
                del variant, tape, cycle
            gc.set_threshold(1, 1, 1)
            for M in (16, 24, 32):
                tape, args = self._trace(M)
                self.assertEqual(len(tape.launches), 1)
                self.assertTrue(gc.isenabled())
        finally:
            gc.set_threshold(*thresholds)
            if not enabled:
                gc.disable()

    def test_launch_on_another_stream_declines_without_executing(self):
        # a launch on a stream that is not the capturing one is not captured:
        # it would run for real with the traced tensors' placeholder addresses
        side = torch.cuda.Stream()
        x, w, b = self._inputs(4)

        def on_side(x, w, b):
            with torch.cuda.stream(side):
                return layer_norm(x, [self.N], w, b, 1e-5)

        on_side(x, w, b)
        side.synchronize()
        with self.assertRaisesRegex(ht.Declined, "not the trace's capturing stream"):
            ht.trace(on_side, (x, w, b), warm_up=False)
        side.synchronize()  # nothing ran outside the capture
        tape, args = self._trace(8)
        x, w, b = self._inputs(4)
        self._check(
            ht.build(tape, layer_norm, args).replay(self._args(x, w, b)), x, w, b, 4
        )

    def test_argument_contract_is_part_of_the_tape(self):
        def head(x, w, b, k):
            return layer_norm(x, [self.N], w, b, 1e-5)[0][:k]

        x, w, b = self._inputs(8)
        tape = ht.trace(head, (x, w, b, 2))
        self.assertEqual(tape.nargs, 4)
        self.assertEqual(tape.positions, [0, 1, 2])
        # a constant that changed, or changed type, between the trace and the
        # build is a miss before any capture
        for k in (3, 2.0, True):
            with self.assertRaisesRegex(ht.Miss, "non-tensor arguments"):
                ht.build(tape, head, (x, w, b, k))
        variant = ht.build(tape, head, (x, w, b, 2))
        self.assertEqual(variant.replay((x, w, b, 2))[0], head(x, w, b, 2))
        self.assertIsNone(variant.try_replay((x, w, b, 3)))
        # a tensor where the trace had a constant
        with self.assertRaisesRegex(ht.Miss, "tensor arguments at"):
            variant.replay((x, w, b, torch.tensor(2)))

    def test_float_constants_compare_by_bits(self):
        # a float argument is a constant of the tape by its bits: 0.0 and -0.0
        # are different constants (the value is baked into the launch as
        # written), a fresh nan is the traced nan, ints and bools are unchanged
        x, w, b = self._inputs(16)
        tape = ht.trace(layer_norm, self._args(x, w, b, eps=0.0))
        variant = ht.build(tape, layer_norm, self._args(x, w, b, eps=0.0))
        self.assertIsNotNone(variant.try_replay(self._args(x, w, b, eps=0.0)))
        self.assertIsNone(variant.try_replay(self._args(x, w, b, eps=-0.0)))
        with self.assertRaisesRegex(ht.Miss, "non-tensor arguments"):
            ht.build(tape, layer_norm, self._args(x, w, b, eps=-0.0))
        nan = float("nan")
        tape = ht.trace(layer_norm, self._args(x, w, b, eps=nan))
        variant = ht.build(tape, layer_norm, self._args(x, w, b, eps=float("nan")))
        self.assertIsNotNone(variant.try_replay(self._args(x, w, b, eps=float("nan"))))
        self.assertIsNone(variant.try_replay(self._args(x, w, b, eps=-nan)))
        self.assertNotEqual(ht._constant(0.0), ht._constant(-0.0))
        self.assertNotEqual(ht._constant([0.0]), ht._constant([-0.0]))
        self.assertEqual(ht._constant([0.0]), ht._constant((0.0,)))
        self.assertEqual(ht._constant(nan), ht._constant(float("nan")))
        self.assertNotEqual(ht._constant(nan), ht._constant(math.copysign(nan, -1.0)))
        self.assertEqual(
            len({ht._constant(2), ht._constant(2.0), ht._constant(True)}), 3
        )
        self.assertEqual(ht._constant(0), ht._constant(-0))
        self.assertEqual(ht._constant(False), ht._constant(False))

    def test_empty_outputs_take_eagers_strides(self):
        # an empty tensor is contiguous whatever its strides (c10's numel == 0
        # rule), so contiguous() is the identity on x[:, :0] as in eager, and
        # an allocation with a zero size takes c10::contiguous_strides' (1, 1)
        x = torch.randn(16, self.N, device="cuda")
        y = torch.randn(12, self.N, device="cuda")
        for fn in (
            lambda t: t[:, :0].contiguous(),
            lambda t: t[:0].contiguous(),
            lambda t: t[:, :0].t().contiguous(),
        ):
            want = fn(x)
            tape = ht.trace(fn, (x,))
            self.assertEqual(tape.num_launches, 0)
            variant = ht.build(tape, fn, (x,))
            (got,) = variant.replay((x,))
            self.assertEqual((got.shape, got.stride()), (want.shape, want.stride()))
            self.assertEqual(got.storage_offset(), want.storage_offset())
            (got,) = variant.replay((y,))
            self.assertEqual((got.shape, got.stride()), (fn(y).shape, fn(y).stride()))
        self.assertTrue(ht._guard_each(ht._contiguous_terms([16, 0], [self.N, 1])))
        self.assertFalse(ht._guard_each(ht._contiguous_terms([16, 2], [self.N, 1])))
        self.assertEqual(ht._contiguous_strides([16, 0]), [1, 1])
        self.assertEqual(ht._contiguous_strides([0, 16]), [16, 1])
        self.assertEqual(ht._contiguous_strides([3, 0, 5]), [5, 5, 1])
        self.assertEqual(ht._contiguous_strides([3, 4]), [4, 1])

    def test_host_value_reads_of_a_traced_tensor_decline(self):
        # bool(), is_nonzero(), tolist(), numpy(), __array__ and a 0-dim
        # format() read the value on the host without dispatching an op: they
        # decline like item(), float(), int() and __index__ (which reach
        # _local_scalar_dense) instead of raising PyTorch's own errors
        x = torch.randn(4, 8, device="cuda")
        y = torch.tensor(0.5, device="cuda")
        yi = torch.tensor(3, device="cuda")
        # numpy(force=True) works on a CUDA tensor in eager; __array__ goes
        # through numpy(), which eager refuses on a CUDA tensor, so it is
        # traced without the warm-up and declines by name
        reads = {
            "bool()": (y, lambda t, u: t * bool(u)),
            "is_nonzero()": (y, lambda t, u: t * u.is_nonzero()),
            "tolist()": (y, lambda t, u: t * u.reshape(-1).tolist()[0]),
            "numpy()": (y, lambda t, u: t * float(u.numpy(force=True))),
            "__array__": (y, lambda t, u: t * float(u.__array__())),
            "format()": (y, lambda t, u: t * float(f"{u}")),
        }
        for what, (u, fn) in reads.items():
            with self.subTest(read=what):
                warm_up = what != "__array__"
                if warm_up:
                    fn(x, u)
                else:
                    with self.assertRaisesRegex(TypeError, "can't convert cuda"):
                        fn(x, u)
                expected = "numpy" if what == "__array__" else re.escape(what)
                with self.assertRaisesRegex(ht.Declined, expected):
                    ht.trace(fn, (x, u), warm_up=warm_up)
                self.assertFalse(torch._C._host_trace_tracing())
        for fn in (
            lambda t, u: t * u.item(),
            lambda t, u: t * float(u),
            lambda t, u: t * int(u),
            lambda t, u: t * complex(u).real,
            lambda t, u: t * operator.index(u),
        ):
            fn(x, yi)
            with self.assertRaisesRegex(ht.Declined, "_local_scalar_dense"):
                ht.trace(fn, (x, yi))
        # len() and iter() of a 0-dim tensor raise eager's TypeError in the
        # trace too
        for fn in (lambda t, u: t * len(u), lambda t, u: t * next(iter(u))):
            with self.assertRaisesRegex(TypeError, "0-d tensor"):
                fn(x, y)
            with self.assertRaisesRegex(TypeError, "0-d tensor"):
                ht.trace(fn, (x, y), warm_up=False)
        self.assertFalse(torch._C._host_trace_tracing())

    def test_math_bit_inputs_decline_and_miss(self):
        x, w, b = self._inputs(8)
        with self.assertRaisesRegex(ht.Declined, "negative view"):
            ht.trace(layer_norm, self._args(torch._neg_view(x), w, b))
        tape, args = self._trace(8)
        variant = ht.build(tape, layer_norm, args)
        self.assertIsNone(variant.try_replay(self._args(torch._neg_view(x), w, b)))
        self.assertIsNotNone(variant.try_replay(self._args(x, w, b)))

        def real_view(z, w, b):
            return layer_norm(torch.view_as_real(z), [2], w, b, 1e-5)[0]

        z = torch.randn(8, 16, device="cuda", dtype=torch.complex64)
        w2, b2 = torch.randn(2, device="cuda"), torch.randn(2, device="cuda")
        variant = ht.build(ht.trace(real_view, (z, w2, b2)), real_view, (z, w2, b2))
        self.assertIsNone(variant.try_replay((z.conj(), w2, b2)))
        self.assertEqual(variant.replay((z, w2, b2))[0], real_view(z, w2, b2))

    def test_dtype_changing_views_keep_their_units_and_dtype(self):
        # view_as_real: the input's offset is in complex elements, the view's
        # in floats; the address must use the view's element size
        def real_view(z, w, b):
            return layer_norm(torch.view_as_real(z), [2], w, b, 1e-5)[0]

        first = torch.randn(8 * 16 + 4, device="cuda", dtype=torch.complex64)
        w, b = torch.randn(2, device="cuda"), torch.randn(2, device="cuda")
        args = (first[: 8 * 16].view(8, 16), w, b)
        variant = ht.build(ht.trace(real_view, args), real_view, args)
        second = torch.randn_like(first)
        # an odd complex offset (8 bytes) may miss on the host's alignment
        # guard; an even one keeps the traced alignment and must serve
        for start in (1, 2, 4):
            changed = (second[start : start + 8 * 16].view(8, 16), w, b)
            out = variant.try_replay(changed)
            if start % 2 == 0:
                self.assertIsNotNone(out)
            if out is not None:
                self.assertEqual(out[0], real_view(*changed))

        # view_as_complex: the output has a dtype of its own over the root
        def complex_out(x, w, b):
            y = layer_norm(x, [self.N], w, b, 1e-5)[0]
            return torch.view_as_complex(y.view(x.shape[0], self.N // 2, 2))

        x, w, b = self._inputs(8, dtype=torch.float32)
        variant = ht.build(ht.trace(complex_out, (x, w, b)), complex_out, (x, w, b))
        for M in (8, 3):
            x, w, b = self._inputs(M, dtype=torch.float32)
            out = variant.replay((x, w, b))[0]
            expected = complex_out(x, w, b)
            self.assertEqual(out.dtype, expected.dtype)
            self.assertEqual(out.stride(), expected.stride())
            self.assertEqual(out, expected)

    def test_expand_of_a_leading_singleton_keeps_atens_stride(self):
        # ATen gives an added leading dim of size 1 the following dim's
        # size * stride, not 0; an as_strided over that stride reads past the
        # first half only if the stride is right
        def fn(x, w, b):
            y = layer_norm(x, [self.N], w, b, 1e-5)[0]
            half = y[: x.shape[0] // 2]
            expanded = half.expand(1, half.shape[0], self.N)
            return expanded.as_strided((2, half.shape[0], self.N), expanded.stride())

        tape, args = self._trace(8, dtype=torch.float32)
        x, w, b = args[0], args[2], args[3]
        variant = ht.build(ht.trace(fn, (x, w, b)), fn, (x, w, b))
        for M in (8, 6, 2):
            x, w, b = self._inputs(M, dtype=torch.float32)
            out = variant.replay((x, w, b))[0]
            expected = fn(x, w, b)
            self.assertEqual(out.stride(), expected.stride())
            self.assertEqual(out, expected)

    def test_view_strides_follow_compute_stride(self):
        # a view of a non-contiguous source takes at::detail::computeStride's
        # strides (a size-1 dim gets the chunk's stride, not the source's), so
        # a kernel whose image carries strides sees eager's bytes; each view
        # is taken inside the trace and its strides compared with eager's
        B, H, DH = 4, 12, 64
        row = torch.randn(B, 1, 3 * H * DH, device="cuda", dtype=torch.bfloat16)
        wide = torch.randn(6, 5, 8, device="cuda", dtype=torch.bfloat16)

        def split_head(r):
            return r.narrow(2, H * DH, H * DH).view(B, 1, H, DH).permute(0, 2, 1, 3)

        cases = [
            (split_head, (row,)),
            (lambda r: r.narrow(2, 0, H * DH).view(B, 1, H, DH), (row,)),
            (lambda r: r.narrow(2, 0, H * DH).view(B, H * DH), (row,)),
            (lambda r: r.narrow(2, 0, H * DH).view(B, 1, 1, H, DH, 1), (row,)),
            (lambda t: t[:, :, :4].view(6, 5, 2, 2), (wide,)),
            (lambda t: t[:, :, :4].view(6, 5, 4, 1), (wide,)),
            (lambda t: t[:, :, :4].view(6, 1, 5, 4), (wide,)),
            (lambda t: t.transpose(1, 2).view(6, 8, 5, 1), (wide,)),
            (lambda t: t[:, 2:3, :].view(6, 8), (wide,)),
            (lambda t: t[:, 2:3, :].view(6, 2, 4), (wide,)),
            (lambda t: t.unsqueeze(1).view(6, 1, 5, 8), (wide,)),
            (lambda t: t[:1].view(5, 8), (wide,)),
            (lambda t: t[:, :1].view(6, 8).view(6, 8, 1), (wide,)),
        ]
        for fn, args in cases:
            with self.subTest(fn=fn(*args).shape, strides=fn(*args).stride()):
                tape = ht.trace(fn, args)
                self.assertEqual(tape.num_launches, 0)
                (rec,) = tape.outputs
                want = fn(*args)
                self.assertEqual(
                    tuple(ht._hint(v) for v in rec.sizes), tuple(want.shape)
                )
                self.assertEqual(tuple(ht._hint(v) for v in rec.strides), want.stride())
                (out,) = ht.build(tape, fn, args).replay(args)
                self.assertEqual(out.stride(), want.stride())
                self.assertTrue(torch.equal(out, want))
        # a view no stride table can express raises the CUDA op's text
        for bad in (
            lambda t: t.transpose(1, 2).view(-1),
            lambda t: t.transpose(0, 2).view(48, 5),
        ):
            with self.assertRaisesRegex(RuntimeError, "at least one dimension spans"):
                bad(wide)
            with self.assertRaisesRegex(RuntimeError, "at least one dimension spans"):
                ht.trace(bad, (wide,), warm_up=False)
        self.assertFalse(torch._C._host_trace_tracing())

    def test_split_and_split_with_sizes_are_narrows(self):
        # split, chunk and split_with_sizes are the sequence of narrows eager
        # makes: no launch, the piece count a guard, the remainder in the
        # last piece; the pieces replay at other sizes of the other dims
        def split3(t):
            return t.split(768, dim=2)

        def remainder(t):
            return t.split(700, -1)

        def chunk(t):
            return t.chunk(3, dim=2)

        def sizes(t):
            return t.split([1000, 304, 1000], dim=2)

        def dim0(t):
            return t.split(3, 0)

        def symbolic(t):
            return t.split(t.shape[-1] // 3, dim=-1)

        row = torch.randn(4, 1, 2304, device="cuda", dtype=torch.bfloat16)
        for fn in (split3, remainder, chunk, sizes, dim0, symbolic):
            with self.subTest(fn=fn.__name__):
                tape = ht.trace(fn, (row,))
                self.assertEqual(tape.num_launches, 0)
                want = fn(row)
                self.assertEqual(len(tape.outputs), len(want))
                variant = ht.build(tape, fn, (row,))
                for other in (
                    row,
                    torch.randn(5, 2, 2304, device="cuda", dtype=torch.bfloat16),
                ):
                    got = variant.replay((other,))
                    want = fn(other)
                    self.assertEqual(len(got), len(want))
                    for g, w in zip(got, want):
                        self.assertEqual(g.stride(), w.stride())
                        self.assertTrue(torch.equal(g, w))
        # a split size that gives another piece count is a miss, not a wrong count
        tape = ht.trace(split3, (row,))
        variant = ht.build(tape, split3, (row,))
        self.assertIsNone(
            variant.try_replay(
                (torch.randn(4, 1, 3072, device="cuda", dtype=torch.bfloat16),)
            )
        )
        short = torch.randn(4, 1, 2000, device="cuda", dtype=torch.bfloat16)
        got = variant.try_replay((short,))
        self.assertIsNotNone(got)
        self.assertEqual(
            [tuple(g.shape) for g in got], [tuple(w.shape) for w in split3(short)]
        )
        self.assertTrue(all(torch.equal(g, w) for g, w in zip(got, split3(short))))
        # eager's checks keep their texts
        with self.assertRaisesRegex(RuntimeError, "sum exactly to"):
            ht.trace(lambda t: t.split([100, 100], dim=2), (row,), warm_up=False)
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            ht.trace(lambda t: t.split(-1, dim=2), (row,), warm_up=False)
        # pieces feed the traced host like any other view
        x, w, b = self._inputs(8)

        def halves(t, shape, weight, bias, eps):
            a, c = t.split(4, 0)
            return layer_norm(a, shape, weight, bias, eps)[0], layer_norm(
                c, shape, weight, bias, eps
            )[0]

        args = self._args(x, w, b)
        tape = ht.trace(halves, args)
        self.assertEqual(tape.num_launches, 2)
        variant = ht.build(tape, halves, args)
        x2, w2, b2 = self._inputs(8)
        got = variant.replay(self._args(x2, w2, b2))
        want = halves(*self._args(x2, w2, b2))
        for g, w_ in zip(got, want):
            self.assertEqual(g, w_, atol=2e-2, rtol=2e-2)
        self.assertFalse(torch._C._host_trace_tracing())

    def test_failed_recorder_construction_leaves_no_trace_active(self):
        self.assertFalse(torch._C._host_trace_tracing())
        with self.assertRaisesRegex(RuntimeError, "out of index range"):
            torch._C._HostTraceRecorder(torch.cuda.device_count())
        self.assertFalse(torch._C._host_trace_tracing())
        tape, args = self._trace(8)
        x, w, b = self._inputs(4)
        self._check(
            ht.build(tape, layer_norm, args).replay(self._args(x, w, b)), x, w, b, 4
        )

    def test_shape_derived_normalized_shape_pins_by_name(self):
        # decode code writes normalized_shape from the tensor; the CUDA kernel
        # is registered on the int signature, so a symbolic element used to
        # assert in the dispatcher's wrapper. It is converted through a guard:
        # the trace serves the traced width and misses another one by name.
        def fn(x, w, b):
            return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)

        x, w, b = self._inputs(32)
        tape = ht.trace(fn, (x, w, b))
        variant = ht.build(tape, fn, (x, w, b))
        y, _, _ = self._inputs(48)
        out = variant.replay((y, w, b))[0]
        self.assertEqual(out, fn(y, w, b), atol=2e-2, rtol=2e-2)
        z = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ht.Miss, str(self.N)):
            variant.replay((z, w2, w2))

    def test_a_failed_push_does_not_leave_stale_node_state(self):
        # the exec's node state (_last) is what the exec holds only after a
        # push succeeded: after a push that raised, the next call with the
        # same bindings pushes again instead of running the previous state
        tape, args = self._trace(8)
        variant = ht.build(tape, layer_norm, args)
        real_exec = variant.exec

        class FailingOnce:
            def __init__(self, inner):
                self.inner, self.failed = inner, False

            def run(self, *a, **k):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("simulated failure inside exec.run")
                return self.inner.run(*a, **k)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        x, w, b = self._inputs(4)
        variant.exec = FailingOnce(real_exec)
        try:
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                variant.replay(self._args(x, w, b))
        finally:
            variant.exec = real_exec
        dirty = real_exec.dirty_nodes
        out = variant.replay(self._args(x, w, b))
        self.assertGreater(real_exec.dirty_nodes, dirty)
        self._check(out, x, w, b, 4)

    def test_a_copy_on_write_input_stays_lazy(self):
        # The converted host reads its inputs through the const accessor, which
        # is const_data_ptr in ordinary mode: a lazy clone must not materialize
        # through an ordinary call, through the trace's warm-up, or through the
        # replay's build. Outputs go through the mutable accessor and do.
        x, w, b = self._inputs(8)
        lazy = torch._lazy_clone(x)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        out = layer_norm(*self._args(lazy, w, b))
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        self.assertEqual(out, layer_norm(*self._args(x, w, b)))
        lazy_w = torch._lazy_clone(w)
        args = self._args(lazy, lazy_w, b)
        tape = ht.trace(layer_norm, args)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        self.assertTrue(torch._C._is_cow_tensor(lazy_w))
        variant = ht.build(tape, layer_norm, args)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        replayed = variant.replay(args)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        self.assertEqual(replayed, layer_norm(*self._args(x, w, b)))

    def test_ordinary_path_is_unchanged(self):
        for M in (64, 5):
            x, w, b = self._inputs(M)
            out, mean, rstd = layer_norm(x, [self.N], w, b, 1e-5)
            self.assertEqual(
                out, F.layer_norm(x, (self.N,), w, b, 1e-5), atol=2e-2, rtol=2e-2
            )
            xf = x.float()
            self.assertEqual(mean.view(M), xf.mean(-1), atol=1e-3, rtol=1e-3)
            self.assertEqual(
                rstd.view(M),
                torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5),
                atol=1e-3,
                rtol=1e-3,
            )

    def test_a_forked_and_joined_side_stream_is_captured(self):
        # a side stream forked from the current stream with an event, and
        # joined back, is part of the capture (same capture id): the host's
        # launch on it is recorded, and the replay reproduces the dependency
        side = torch.cuda.Stream()

        def forked(x, shape, w, b, eps):
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                out = layer_norm(x, shape, w, b, eps)
            cur.wait_stream(side)
            return out

        x, w, b = self._inputs(8)
        args = self._args(x, w, b)
        tape = ht.trace(forked, args)
        variant = ht.build(tape, forked, args)
        x, w, b = self._inputs(4)
        self._check(variant.replay(self._args(x, w, b)), x, w, b, 4)

    def test_a_single_stream_trace_is_all_on_capture_stream(self):
        # the tape says whether every launch was issued on the trace's own
        # capturing stream (typed: the stream argument; verbatim: the stream
        # current at the conversion): a consumer that replays on one stream
        # reads it before lowering, the raw capture accepts fork and join
        # either way
        tape, _ = self._trace(8)
        self.assertIs(tape.all_on_capture_stream, True)
        self.assertIs(json.loads(tape.to_json())["all_on_capture_stream"], True)

    def test_a_forked_side_stream_clears_all_on_capture_stream(self):
        side = torch.cuda.Stream()

        def forked(x, shape, w, b, eps):
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                out = layer_norm(x, shape, w, b, eps)
            cur.wait_stream(side)
            return out

        tape = ht.trace(forked, self._args(*self._inputs(8)))
        self.assertIs(tape.all_on_capture_stream, False)
        self.assertIs(json.loads(tape.to_json())["all_on_capture_stream"], False)

    def test_isomorphic_branches_on_two_side_streams_replay_bitwise(self):
        # two side streams forked from the current stream, the same kernel at
        # the same shapes on each, joined: the graph holds two nodes nothing
        # tells apart, and each launch record names the node the capture
        # created for it (the launching stream's frontier right after the
        # launch), so each branch's arguments bind to its own node whatever
        # order the driver lists the nodes in
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()

        def two_branches(x1, x2, shape, w, b, eps):
            cur = torch.cuda.current_stream()
            s1.wait_stream(cur)
            s2.wait_stream(cur)
            with torch.cuda.stream(s1):
                o1 = layer_norm(x1, shape, w, b, eps)
            with torch.cuda.stream(s2):
                o2 = layer_norm(x2, shape, w, b, eps)
            cur.wait_stream(s1)
            cur.wait_stream(s2)
            return (*o1, *o2)

        def inputs(M):
            x1, w, b = self._inputs(M)
            return (x1, torch.randn_like(x1), [self.N], w, b, 1e-5)

        args = inputs(8)
        tape = ht.trace(two_branches, args)
        self.assertEqual(tape.num_launches, 2)
        variant = ht.build(tape, two_branches, args)
        for M in (4, 8, 16):
            a = inputs(M)
            got = variant.replay(a)
            want = (*layer_norm(a[0], *a[2:]), *layer_norm(a[1], *a[2:]))
            self.assertEqual(len(got), 6)
            for g, e in zip(got, want):
                self.assertTrue(torch.equal(g, e))

    def test_pairing_does_not_depend_on_the_node_order(self):
        # a launch record is paired with the node the recorder read from the
        # capture frontier, not with the node at the record's position in
        # cudaGraphGetNodes (creation order on the drivers measured, not a
        # documented one). With the recorder reading the nodes back reversed,
        # two layer norms whose N picks different kernels (the vectorized one,
        # then the rowwise-moments pair) still record in host order and
        # replay bitwise; the positional pairing would have declined this
        # trace ("launches out of order").
        N2 = 1001  # not a multiple of the vector width: the two-kernel path

        def two_norms(x1, x2, w1, b1, w2, b2):
            o1 = layer_norm(x1, [self.N], w1, b1, 1e-5)
            o2 = layer_norm(x2, [N2], w2, b2, 1e-5)
            return (*o1, *o2)

        def inputs(M):
            x1, w1, b1 = self._inputs(M)
            x2 = torch.randn(M, N2, device="cuda", dtype=torch.bfloat16)
            w2 = torch.randn(N2, device="cuda", dtype=torch.bfloat16)
            b2 = torch.randn(N2, device="cuda", dtype=torch.bfloat16)
            return (x1, x2, w1, b1, w2, b2)

        args = inputs(8)
        torch._C._host_trace_test_reverse_node_order(True)
        try:
            tape = ht.trace(two_norms, args)
        finally:
            torch._C._host_trace_test_reverse_node_order(False)
        kernels = [L["kernel"] for L in tape.launches]
        self.assertEqual(len(kernels), 3)
        self.assertIn("vectorized_layer_norm_kernel", kernels[0])
        self.assertIn("RowwiseMomentsCUDAKernel", kernels[1])
        self.assertIn("LayerNormForwardCUDAKernel", kernels[2])
        variant = ht.build(tape, two_norms, args)
        for M in (4, 16):
            a = inputs(M)
            got = variant.replay(a)
            want = (
                *layer_norm(a[0], [self.N], a[2], a[3], 1e-5),
                *layer_norm(a[1], [N2], a[4], a[5], 1e-5),
            )
            for g, e in zip(got, want):
                self.assertTrue(torch.equal(g, e))

    def test_an_external_event_cannot_fork_a_stream_into_the_trace(self):
        # streams facts, fact 2: an external event recorded on the capturing
        # stream is a node, and a wait on it from a never-forked stream fails
        # with cudaErrorIllegalState. A host that lets the error propagate
        # declines by that code; a host that swallows it and launches on the
        # stream declines before the kernel reaches the driver, since the
        # stream is not in the capture; a bare external record declines as a
        # node the tape does not describe. The recorder is clean after each.
        side = torch.cuda.Stream()
        x, w, b = self._inputs(4)

        def propagate(t, shape, weight, bias, eps):
            ev = torch.cuda.Event(external=True)
            ev.record(torch.cuda.current_stream())
            side.wait_event(ev)
            with torch.cuda.stream(side):
                return layer_norm(t, shape, weight, bias, eps)

        def swallow(t, shape, weight, bias, eps):
            ev = torch.cuda.Event(external=True)
            ev.record(torch.cuda.current_stream())
            try:
                side.wait_event(ev)
            except torch.AcceleratorError:
                pass
            with torch.cuda.stream(side):
                return layer_norm(t, shape, weight, bias, eps)

        def bare_record(t, shape, weight, bias, eps):
            out = layer_norm(t, shape, weight, bias, eps)
            torch.cuda.Event(external=True).record(torch.cuda.current_stream())
            return out

        for fn, message in (
            (propagate, "cudaErrorIllegalState"),
            (swallow, "not the trace's capturing stream"),
            (bare_record, "event record node"),
        ):
            with self.assertRaisesRegex(ht.Declined, message):
                ht.trace(fn, self._args(x, w, b))
            self.assertFalse(torch._C._host_trace_tracing())
            side.synchronize()
        tape, args = self._trace(8)
        x, w, b = self._inputs(4)
        self._check(
            ht.build(tape, layer_norm, args).replay(self._args(x, w, b)), x, w, b, 4
        )

    def test_a_verbatim_launch_naming_another_stream_never_touches_real_memory(self):
        # outside the contract: a verbatim launch that names a never-forked
        # stream explicitly cannot be seen before it runs. It must not read or
        # write real memory: the traced tensors' addresses are placeholders, so
        # the escaped launch faults (a loud error in this child process) or, if
        # the driver tolerates it, the real tensor is untouched and the
        # completeness rule declines the trace. Built as an extension in a
        # subprocess because a fault is sticky.
        from torch.utils.cpp_extension import CUDA_HOME

        nvcc = shutil.which("nvcc") or (
            CUDA_HOME and os.path.join(CUDA_HOME, "bin", "nvcc")
        )
        if not nvcc or not os.path.exists(nvcc):
            self.skipTest("no nvcc")
        cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Field.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
namespace ht = at::cuda::host_trace;
struct P { float* out; long long n; };
namespace at::cuda::host_trace {
template <> struct Traced<P> : TracedBase {
  P pod{};
  PtrField<0> out{this, "out"};
  IntField<long long, 8> n{this, "n"};
  Traced() : TracedBase(&pod, sizeof(P)) {}
  operator P&() { on_convert(); return pod; }
};
}  // namespace at::cuda::host_trace
__global__ void fill_ones(P p) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < p.n) p.out[i] = 1.0f;
}
// a host that launches verbatim on a stream it was handed (never forked)
at::Tensor escaped(const at::Tensor& x, int64_t side_stream) {
  c10::cuda::CUDAGuard g(x.device());
  ht::Traced<P> params;
  params.n = x.sym_numel();
  params.out = ht::sym_mutable_data_ptr(x);
  ht::Grid grid((x.sym_numel() + 255) / 256);
  cudaStream_t s = reinterpret_cast<cudaStream_t>(side_stream);
  fill_ones<<<grid, 256, 0, s>>>(params);
  return x;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("escaped", &escaped); }
"""
        script = r"""
import sys, torch
from torch.utils.cpp_extension import load_inline
import torch.cuda._host_trace as ht
src = open(sys.argv[1]).read()
ext = load_inline("ht_escaped_probe", cpp_sources="", cuda_sources=src,
                  functions=None, with_cuda=True, verbose=False,
                  extra_cuda_cflags=["-std=c++20"], extra_cflags=["-std=c++20"])
x = torch.full((4096,), 7.0, device="cuda")
side = torch.cuda.Stream()
def host(x):
    return ext.escaped(x, side.cuda_stream)
try:
    ht.trace(host, (x,), warm_up=False)
    print("RESULT trace-served")
except ht.Declined as e:
    print("RESULT declined", str(e).splitlines()[0])
torch.cuda.synchronize()
print("RESULT untouched", bool((x == 7.0).all().item()))
"""
        with tempfile.TemporaryDirectory() as d:
            cu = os.path.join(d, "probe.cu")
            py = os.path.join(d, "run.py")
            with open(cu, "w") as f:
                f.write(cuda_src)
            with open(py, "w") as f:
                f.write(script)
            env = dict(os.environ, TORCH_EXTENSIONS_DIR=os.path.join(d, "ext"))
            r = subprocess.run(
                [sys.executable, py, cu],
                capture_output=True,
                text=True,
                env=env,
                timeout=900,
            )
        out = r.stdout + r.stderr
        served = "RESULT trace-served" in out
        untouched = "RESULT untouched True" in out
        faulted = (
            r.returncode != 0
            or "illegal" in out.lower()
            or "an illegal memory access" in out
        )
        # never: a served trace with the real tensor overwritten by the escaped launch
        self.assertFalse(served and not untouched, out[-3000:])
        # loud (fault) or clean (untouched and declined); silent success is the only failure
        self.assertTrue(faulted or untouched, out[-3000:])

    def test_verbatim_launches_pair_by_frontier_under_fork_and_join(self):
        # a verbatim <<<>>> is issued by the host after its Grid or proxy
        # conversion opened the record, so the recorder reads the node from
        # the frontier of the stream that was current at the conversion when
        # the next recorder event closes the record. By then that stream may
        # have joined another: a verbatim launch on a forked side stream that
        # the current stream joins before the host returns, and a verbatim
        # launch on the current stream followed by a join of a side stream
        # that ran a typed launch (both nodes on the frontier; the typed one
        # is already claimed) both record the right node and replay bitwise.
        from torch.utils.cpp_extension import CUDA_HOME

        nvcc = shutil.which("nvcc") or (
            CUDA_HOME and os.path.join(CUDA_HOME, "bin", "nvcc")
        )
        if not nvcc or not os.path.exists(nvcc):
            self.skipTest("no nvcc")
        cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Field.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
namespace ht = at::cuda::host_trace;
struct P { float* out; long long n; };
namespace at::cuda::host_trace {
template <> struct Traced<P> : TracedBase {
  P pod{};
  PtrField<0> out{this, "out"};
  IntField<long long, 8> n{this, "n"};
  Traced() : TracedBase(&pod, sizeof(P)) {}
  operator P&() { on_convert(); return pod; }
};
}  // namespace at::cuda::host_trace
__global__ void fill_ones(P p) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < p.n) p.out[i] = 1.0f;
}
// a verbatim launch on the current stream, as a converted host writes it
at::Tensor fill(const at::Tensor& x) {
  c10::cuda::CUDAGuard g(x.device());
  ht::Traced<P> params;
  params.n = x.sym_numel();
  params.out = ht::sym_mutable_data_ptr(x);
  ht::Grid grid((x.sym_numel() + 255) / 256);
  fill_ones<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(params);
  return x;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fill", &fill); }
"""
        script = r"""
import sys, torch
from torch.utils.cpp_extension import load_inline
import torch.cuda._host_trace as ht
src = open(sys.argv[1]).read()
ext = load_inline("ht_verbatim_fork_join", cpp_sources="", cuda_sources=src,
                  functions=None, with_cuda=True, verbose=False,
                  extra_cuda_cflags=["-std=c++20"], extra_cflags=["-std=c++20"])
layer_norm = torch.ops.aten.native_layer_norm.default
side = torch.cuda.Stream()
N = 1024

def on_side(x):
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        ext.fill(x)
    cur.wait_stream(side)
    return x

def then_join(x, y, w, b):
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        o = layer_norm(y, [N], w, b, 1e-5)
    ext.fill(x)
    cur.wait_stream(side)
    return (x, *o)

def fill_args(M):
    return (torch.zeros(M * 256, device="cuda"),)

def join_args(M):
    return (torch.zeros(M * 256, device="cuda"), torch.randn(M, N, device="cuda"),
            torch.randn(N, device="cuda"), torch.randn(N, device="cuda"))

for label, fn, args_of in (("on_side", on_side, fill_args), ("then_join", then_join, join_args)):
    args = args_of(4)
    try:
        tape = ht.trace(fn, args)
        variant = ht.build(tape, fn, args)
    except ht.Declined as e:
        print("RESULT", label, "declined", str(e).splitlines()[0])
        continue
    for M in (2, 8):
        a = args_of(M)
        want = fn(*[t.clone() for t in a])
        want = (want,) if isinstance(want, torch.Tensor) else tuple(want)
        got = variant.replay(a)
        torch.cuda.synchronize()
        ok = len(got) == len(want) and all(torch.equal(g, w) for g, w in zip(got, want))
        print("RESULT", label, M, "bitwise" if ok else "MISMATCH", tape.num_launches)
"""
        with tempfile.TemporaryDirectory() as d:
            cu = os.path.join(d, "probe.cu")
            py = os.path.join(d, "run.py")
            with open(cu, "w") as f:
                f.write(cuda_src)
            with open(py, "w") as f:
                f.write(script)
            env = dict(os.environ, TORCH_EXTENSIONS_DIR=os.path.join(d, "ext"))
            r = subprocess.run(
                [sys.executable, py, cu],
                capture_output=True,
                text=True,
                env=env,
                timeout=900,
            )
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        for line in (
            "RESULT on_side 2 bitwise 1",
            "RESULT on_side 8 bitwise 1",
            "RESULT then_join 2 bitwise 2",
            "RESULT then_join 8 bitwise 2",
        ):
            self.assertIn(line, out, out[-3000:])


_VERBATIM_FORMALS_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace ht = at::cuda::host_trace;

struct P { int64_t value; };
struct Wide { int64_t first; int64_t second; };
static_assert(sizeof(Wide) == 2 * sizeof(P));
static_assert(sizeof(P) == sizeof(int64_t));
static_assert(offsetof(P, value) == 0);

namespace at::cuda::host_trace {
template <> struct Traced<P> : TracedBase {
  P pod{};
  IntField<int64_t, offsetof(P, value)> value{this, "value"};
  Traced() : TracedBase(&pod, sizeof(P)) {}
  operator P&() { on_convert(); return pod; }
};
} // namespace at::cuda::host_trace

__device__ int64_t observed_values[2];

__global__ void write_pair(P fixed, P dynamic) {
  observed_values[0] = fixed.value;
  observed_values[1] = dynamic.value;
}

__global__ void write_single(P dynamic) {
  observed_values[0] = 11;
  observed_values[1] = dynamic.value;
}

__global__ void write_wide(Wide fixed) {
  observed_values[0] = fixed.first;
  observed_values[1] = fixed.second;
}

at::Tensor verbatim_pair(const at::Tensor& x, int64_t fixed_value) {
  c10::cuda::CUDAGuard guard(x.device());
  P fixed{fixed_value};
  ht::Traced<P> dynamic;
  dynamic.value = x.sym_size(0);
  ht::Grid grid(1);
  write_pair<<<grid, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(fixed, dynamic);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}

at::Tensor typed_pair(const at::Tensor& x, int64_t fixed_value) {
  c10::cuda::CUDAGuard guard(x.device());
  P fixed{fixed_value};
  ht::Traced<P> dynamic;
  dynamic.value = x.sym_size(0);
  ht::launch(write_pair, 1, 1, 0, c10::cuda::getCurrentCUDAStream(), fixed, dynamic);
  return x;
}

at::Tensor verbatim_single(const at::Tensor& x, bool alter) {
  c10::cuda::CUDAGuard guard(x.device());
  ht::Traced<P> dynamic;
  dynamic.value = x.sym_size(0);
  ht::Grid grid(1);
  if (alter) {
    P changed = dynamic;
    changed.value += 1;
    write_single<<<grid, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(changed);
  } else {
    write_single<<<grid, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(dynamic);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}

at::Tensor unmatched_proxy(const at::Tensor& x) {
  c10::cuda::CUDAGuard guard(x.device());
  ht::Traced<P> dynamic;
  dynamic.value = x.sym_size(0);
  P ignored = dynamic;
  (void)ignored;
  ht::Grid grid(1);
  Wide fixed{0, 0};
  write_wide<<<grid, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(fixed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}

std::vector<int64_t> read_observed() {
  TORCH_CHECK(ht::active() == nullptr, "observation is outside the traced host");
  std::vector<int64_t> result(2);
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(
      result.data(), observed_values, sizeof(int64_t) * 2));
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("verbatim_pair", &verbatim_pair);
  m.def("typed_pair", &typed_pair);
  m.def("verbatim_single", &verbatim_single);
  m.def("unmatched_proxy", &unmatched_proxy);
  m.def("read_observed", &read_observed);
}
"""


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceVerbatimFormals(TestCase):
    # the runtime team's regression tests for the proxy-to-formal rule of a
    # verbatim launch (verbatim_formals_fix review): a converted proxy is
    # passed by value exactly once and unchanged, so its formal is the one
    # size-compatible parameter; the bytes only verify that slot
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from torch.utils.cpp_extension import CUDA_HOME, load_inline

        nvcc = CUDA_HOME and os.path.join(CUDA_HOME, "bin", "nvcc")
        if not nvcc or not os.path.isfile(nvcc):
            raise unittest.SkipTest("requires nvcc for the test extension")
        cls.extension = load_inline(
            "hosttrace_verbatim_formals",
            cpp_sources="",
            cuda_sources=_VERBATIM_FORMALS_SOURCE,
            functions=None,
            with_cuda=True,
            extra_cflags=["-std=c++20"],
            extra_cuda_cflags=["-std=c++20"],
        )

    @parametrize("fixed_value", [7, 4])
    def test_multiple_size_compatible_formals_decline(self, device, fixed_value):
        def host(x):
            return self.extension.verbatim_pair(x, fixed_value)

        x = torch.empty(7, device=device)
        with self.assertRaisesRegex(ht.Declined, "multiple size-compatible parameters"):
            ht.trace(host, (x,))
        self.assertFalse(torch._C._host_trace_tracing())

        def unique_host(t):
            return self.extension.verbatim_single(t, False)

        self.assertIsNotNone(ht.trace(unique_host, (x,)))

    @parametrize("fixed_value", [7, 4])
    def test_typed_formals_rebind_positionally(self, device, fixed_value):
        def host(x):
            return self.extension.typed_pair(x, fixed_value)

        x = torch.empty(7, device=device)
        tape = ht.trace(host, (x,))
        symbolic = [
            p
            for p in tape.launches[0]["params"]
            if isinstance(p["value"], torch.SymInt)
        ]
        self.assertEqual([p["offset"] for p in symbolic], [8])
        variant = ht.build(tape, host, (x,))
        y = torch.empty(8, device=device)
        variant.replay((y,))
        torch.cuda.synchronize()
        self.assertEqual(self.extension.read_observed(), [fixed_value, 8])

    def test_unique_formal_rebinds(self, device):
        def host(x):
            return self.extension.verbatim_single(x, False)

        x = torch.empty(7, device=device)
        tape = ht.trace(host, (x,))
        variant = ht.build(tape, host, (x,))
        variant.replay((torch.empty(8, device=device),))
        torch.cuda.synchronize()
        self.assertEqual(self.extension.read_observed(), [11, 8])

    def test_changed_proxy_bytes_decline(self, device):
        def host(x):
            return self.extension.verbatim_single(x, True)

        with self.assertRaisesRegex(
            ht.Declined, "bytes differ from its selected parameter"
        ):
            ht.trace(host, (torch.empty(7, device=device),))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_no_size_compatible_formal_declines(self, device):
        with self.assertRaisesRegex(ht.Declined, "no size-compatible parameter"):
            ht.trace(self.extension.unmatched_proxy, (torch.empty(7, device=device),))
        self.assertFalse(torch._C._host_trace_tracing())


instantiate_device_type_tests(
    TestCudaHostTraceVerbatimFormals, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()
