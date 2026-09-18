# Owner(s): ["module: cuda"]

import contextlib
import gc
import inspect
import json
import math
import operator
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from unittest import mock

from host_trace_testing import (
    assert_no_disabled_memset,
    load_test_extension,
    needs_two_gpus,
    REPO_ROOT,
    require_nvcc,
)

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    skipIfRocm,
    TEST_NUMPY,
    TestCase,
)


if torch.cuda.is_available():
    import host_trace_two_hint as two_hint

    from torch.cuda import _host_trace as ht

# the op whose CUDA host is traced: it returns the output and the two statistics
layer_norm = torch.ops.aten.native_layer_norm.default
layer_norm_backward = torch.ops.aten.native_layer_norm_backward.default


def _device_work(fn, args):
    # the device work one call issues, in order: kernels by name, memsets and
    # memcpys as such (the profiler's device events)
    from torch.profiler import profile, ProfilerActivity

    fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(*args)
        torch.cuda.synchronize()
    work = []
    for e in p.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        low = e.name.lower()
        kind = next((k for k in ("memset", "memcpy") if low.startswith(k)), "kernel")
        work.append((kind, e.name if kind == "kernel" else None))
    return work


# a traced sibling launches eager's kernel template with a named functor in
# place of eager's lambda: the template, its leading integer arguments and
# the operand array's size identify the launch (the ti suite's rule; a
# strided launch names no operand array on either side)
_TEMPLATE = re.compile(
    r"^(?:void )?(?:\(anonymous namespace\)::|[A-Za-z_]\w*::)*(\w+)<((?:\d+, )*)"
)
_OPERANDS = re.compile(r"std::array<char\*, (\d+)ul>")


def _launch_shape(name):
    m = _TEMPLATE.match(name)
    if m is None:
        return name
    arr = _OPERANDS.search(name)
    return (m.group(1), m.group(2), arr.group(1) if arr else None)


# a typed in-place kernel: x += 1.0f per execution (test helper _add_one)
_ADD_ONE_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
namespace ht = at::cuda::host_trace;
struct P { float* data; long long n; };
namespace at::cuda::host_trace {
template <> struct Traced<P> : TracedBase {
  P pod{};
  PtrField<0> data{this, "data"};
  IntField<long long, 8> n{this, "n"};
  Traced() : TracedBase(&pod, sizeof(P)) {}
  operator P&() { on_convert(); return pod; }
};
}  // namespace at::cuda::host_trace
__global__ void add_one(P p) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < p.n) p.data[i] += 1.0f;
}
at::Tensor add_one_(const at::Tensor& x) {
  c10::cuda::CUDAGuard g(x.device());
  ht::Traced<P> params;
  params.n = x.sym_numel();
  params.data = ht::sym_mutable_data_ptr(x);
  ht::Grid grid((x.sym_numel() + 255) / 256);
  ht::launch(add_one, grid, 256, 0, c10::cuda::getCurrentCUDAStream(), params);
  return x;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_one_", &add_one_); }
"""


# a typed launch of two int64 fields the host computes in Python (test helper
# write_fields)
_INT_FIELDS_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
namespace ht = at::cuda::host_trace;
struct F { int64_t* data; int64_t k; int64_t m; };
namespace at::cuda::host_trace {
template <> struct Traced<F> : TracedBase {
  F pod{};
  PtrField<offsetof(F, data)> data{this, "data"};
  IntField<int64_t, offsetof(F, k)> k{this, "k"};
  IntField<int64_t, offsetof(F, m)> m{this, "m"};
  Traced() : TracedBase(&pod, sizeof(F)) {}
  operator F&() { on_convert(); return pod; }
};
}  // namespace at::cuda::host_trace
__global__ void write_fields(F f) { f.data[0] = f.k; f.data[1] = f.m; }
at::Tensor write_fields_(const at::Tensor& out, c10::SymInt k, c10::SymInt m) {
  c10::cuda::CUDAGuard g(out.device());
  ht::Traced<F> params;
  params.data = ht::sym_mutable_data_ptr(out);
  params.k = k;
  params.m = m;
  ht::launch(write_fields, ht::Grid(1), 1, 0, c10::cuda::getCurrentCUDAStream(), params);
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("write_fields", &write_fields_); }
"""


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

    def _assert_capture_streams_free(self, streams):
        # each stream a trace captured on is free again: not capturing, and a
        # capture of its own begins and ends on it
        self.assertGreaterEqual(len(streams), 1)
        y = torch.zeros(8, device="cuda")
        graphs = []
        for stream in streams:
            with torch.cuda.stream(stream):
                self.assertFalse(torch.cuda.is_current_stream_capturing())
                g = torch.cuda.CUDAGraph()
                g.capture_begin(capture_error_mode="thread_local")
                y.add_(1)
                g.capture_end()
            g.replay()
            graphs.append(g)
        torch.cuda.synchronize()
        self.assertEqual(y, torch.full((8,), float(len(streams)), device="cuda"))

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

    def _add_one(self):
        return load_test_extension("ht_exactly_once", _ADD_ONE_SOURCE).add_one_

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

    @requires_cuda_python_bindings
    def test_built_exec_holds_no_disabled_node(self):
        # the exec holds exactly the capture's nodes and disables none (a
        # kernel node behind a disabled memset node does not wait for the
        # stream's prior work on driver 580.126.20; a node chain the exec
        # does not hold is another exec of the same tape, never an inserted
        # or disabled node). The shared setUp of the family suites applies the
        # memset rule to every build; this is the rule on the first exec,
        # after the build and after a replay at another shape
        tape, args = self._trace()
        variant = ht.build(tape, layer_norm, args)
        states = assert_no_disabled_memset(self, variant, "layer norm build")
        self.assertTrue(all(kind == "kernel" for _, kind, _ in states), states)
        x, w, b = self._inputs(5)
        variant.replay(self._args(x, w, b))
        states = assert_no_disabled_memset(self, variant, "layer norm replay")
        self.assertTrue(all(enabled for _, _, enabled in states), states)

    def test_trace_at_batch_one_serves_other_batches(self):
        # decode traces are taken at batch 1; the contiguity and view checks
        # must not pin the trace there
        tape, args = self._trace(1)
        variant = ht.build(tape, layer_norm, args)
        for M in (8, 1, 3):
            x, w, b = self._inputs(M)
            self._check(variant.replay(self._args(x, w, b)), x, w, b, M)

    def test_two_hints_trace_the_same_layer_norm(self):
        # the second assignment moves the batch (the sizes the call pins stay)
        # and the two tapes agree; the first is the one returned and built
        args = self._args(*self._inputs(64))
        log: list = []
        tape = two_hint.trace_twice(layer_norm, args, log=log)
        moved = [m[0] for m in log[0].moved]
        self.assertTrue(any("arg0.size(0)" in m for m in moved), moved)
        self.assertEqual(log[0].held, [])
        variant = ht.build(tape, layer_norm, args)
        for M in (5, 128):
            x, w, b = self._inputs(M)
            self._check(variant.replay(self._args(x, w, b)), x, w, b, M)

    def test_two_struct_parameters_of_one_type_trace_the_same_under_other_hints(self):
        # kernel(P fixed, P dynamic) with fixed = P{7} traced at n = 7: the
        # constant's bytes equal the symbolic value's, so nothing in the
        # captured image tells the two formals apart. A parameter is bound by
        # formal identity, never by matching bytes: the verbatim launch, whose
        # two same-sized formals it cannot tell apart, declines by name before
        # any tape exists, and the typed launch of the same kernel (positional
        # formals) traces the same program under both hints and replays [7, 8].
        ext = load_test_extension(
            "hosttrace_verbatim_formals", _VERBATIM_FORMALS_SOURCE
        )

        def verbatim(x):
            return ext.verbatim_pair(x, 7)

        def typed(x):
            return ext.typed_pair(x, 7)

        x = torch.empty(7, device="cuda")
        with self.assertRaisesRegex(ht.Declined, "multiple size-compatible parameters"):
            two_hint.trace_twice(verbatim, (x,))
        log: list = []
        tape = two_hint.trace_twice(typed, (x,), log=log)
        moved = [m[0] for m in log[0].moved]
        self.assertTrue(any("arg0.size(0)" in m for m in moved), moved)
        variant = ht.build(tape, typed, (x,))
        variant.replay((torch.empty(8, device="cuda"),))
        torch.cuda.synchronize()
        self.assertEqual(ext.read_observed(), [7, 8])

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
            return torch.atan(layer_norm(t, shape, weight, bias, eps)[0])

        def before(t, shape, weight, bias, eps):
            return layer_norm(torch.erf(t), shape, weight, bias, eps)

        with self.assertRaisesRegex(ht.Declined, "aten.atan.default"):
            ht.trace(after, self._args(x, w, b))
        with self.assertRaisesRegex(ht.Declined, "aten.erf.default"):
            ht.trace(before, self._args(x, w, b))
        # a non-contiguous input makes the host copy it; the copy has a traced
        # sibling (torch/cuda/_host_trace_ti.py), so the trace holds two launches
        xt = torch.randn(self.N, 8, device="cuda", dtype=torch.bfloat16).t()
        tape = ht.trace(layer_norm, self._args(xt, w, b))
        self.assertEqual(tape.num_launches, 2)
        variant = ht.build(tape, layer_norm, self._args(xt, w, b))
        yt = torch.randn(self.N, 12, device="cuda", dtype=torch.bfloat16).t()
        _, w2, b2 = self._inputs(12)
        self._check(variant.replay(self._args(yt, w2, b2)), yt, w2, b2, 12)
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

    @needs_two_gpus
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

    def test_root_facts_declare_every_input_size_positive(self):
        # the tape's declared facts, not guards: every input size symbol is a
        # positive integer (a replay misses on an empty input before any
        # guard), one ("domain", symbol, 1, None) row per size ahead of the
        # root identity pairs; the JSON carries the rows (DOMAIN_GUARDS.md
        # follow-up 2, the consumer's contract table)
        tape, _ = self._trace()
        names = [ht._symbol_name(s) for i in tape.inputs for s in i.sizes]
        self.assertEqual(len(names), 4)  # x (M, N), w (N,), b (N,)
        self.assertTrue(all(names))
        rows = [("domain", n, 1, None) for n in names]
        self.assertEqual(tape.root_facts, rows)
        self.assertEqual(
            json.loads(tape.to_json())["root_facts"], [list(r) for r in rows]
        )

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

    def test_guards_are_the_expressions_the_host_evaluated(self):
        # the trace's ShapeEnv decides every branch by its hint and records
        # the expression as evaluated: an int read is `Eq(s, 1024)`, never a
        # replacement of s; the same relation is recorded once; a guard the
        # earlier guards imply is dropped; the order the host asked in stays
        import sympy

        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.utils._sympy.printers import PythonPrinter

        def program(env):
            def sym(v, name, size=False):
                src = ht._Src(name)
                dyn = ht.DimDynamic.DYNAMIC
                if size:
                    kw = {"positive": True, "do_not_specialize_zero_one": True}
                    e = env.create_symbol(v, src, dyn, None, **kw)
                else:
                    e = env.create_unspecified_symbol(v, src, dyn)
                return env.create_symintnode(e, hint=v, source=src)

            a, b = sym(8, "a", True), sym(8, "b", True)
            c, st = sym(1, "c", True), sym(8, "st")
            taken = [
                bool(a == b),
                bool(4 * a == 4 * b),  # the same fact in another form
                bool(a == b),  # a repeat
                bool(c == 1),  # a specialization
                bool((c == 1) | (st == 8)),  # implied by the specialization
                int(a * b),  # an int read
                bool(st != 0),  # the zero-divisor guard ...
                bool(a * b % st == 0),  # ... before the division that needs it
                bool(a == a),  # a constant: no guard
            ]
            return taken, (a, b, c, st)

        plain = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        env = ht._TraceShapeEnv()
        taken, syms = program(env)
        self.assertEqual(taken, program(plain)[0])
        self.assertFalse(env.replacements)
        self.assertTrue(plain.replacements)
        A, B, C, ST = (x.node.expr for x in syms)
        p = PythonPrinter()
        guards, pins, _ = env.tape_guards()
        got = [p.doprint(g) for g in guards]
        # after `Eq(a, b)` the later symbol b reads as a, after `Eq(c, 1)` c
        # reads as 1: the pins, read into the guards that follow them
        want = [sympy.Eq(A, B), sympy.Eq(C, 1), sympy.Eq(A * A, 64)]
        want += [sympy.Ne(ST, 0), sympy.Eq(sympy.Mod(A * A, ST), 0)]
        self.assertEqual(got, [p.doprint(g) for g in want])
        self.assertEqual(pins, {B: A, C: 1})
        # the raw record keeps the two dropped forms, as evaluated
        self.assertEqual(len(env.guards), len(want) + 2)
        self.assertEqual(env.guards[4].expr, sympy.Eq(A * B, 64))
        env.evaluate_expr(sympy.Lt(C * ST, 64), True)
        self.assertEqual(str(env.tape_guards()[0][-1]), str(sympy.Lt(ST, 64)))

    def test_a_tape_holds_every_guard_at_its_own_hints(self):
        tape, _ = self._trace()
        self.assertFalse(tape.shape_env.replacements)
        guards, pins, _ = tape.shape_env.tape_guards()
        self.assertEqual(tape.guards, guards)
        prog = ht._Program(tape)
        hints = {str(k): int(v) for k, v in tape.shape_env.backed_var_to_val.items()}
        self.assertTrue(all(prog.ev(g, hints) for g in tape.guards))
        # a pinned symbol (the normalized width) is a number in every launch
        # field and allocation size; its pin guard names it
        self.assertTrue(pins)
        named = {str(s) for s in pins}
        self.assertTrue(any(ht._free_symbols(g) & named for g in tape.guards))
        values = [x for a in tape.allocs for x in (*a.sizes, *a.strides)]
        for L in tape.launches:
            values += [p["value"] for p in L["params"]] + L["grid"]
        self.assertFalse(any(ht._free_symbols(v) & named for v in values))
        self.assertTrue(any(isinstance(v, int) for v in values))

    def test_partial_operations_record_their_domain_when_created(self):
        # a floor division, a modulo or a true division of traced values is
        # defined only for a nonzero divisor: the trace's ShapeEnv records
        # `Ne(divisor, 0)` when the operation is created, so the ordered guard
        # list meets it before any guard or value built on the operation,
        # whether or not the host tests the divisor itself; a domain sympy
        # decides from a size's declared positivity or from a literal is no
        # guard; a pin established earlier makes it trivially true and the
        # tape's dedupe drops it, a pin established later leaves it in place
        import sympy

        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.utils._sympy.functions import FloorDiv, Mod, PythonMod
        from torch.utils._sympy.printers import PythonPrinter

        p = PythonPrinter()

        def symbols(env):
            def sym(v, name, size=False):
                src = ht._Src(name)
                dyn = ht.DimDynamic.DYNAMIC
                if size:
                    kw = {"positive": True, "do_not_specialize_zero_one": True}
                    e = env.create_symbol(v, src, dyn, None, **kw)
                else:
                    e = env.create_unspecified_symbol(v, src, dyn)
                if isinstance(v, float):
                    return env.create_symfloatnode(e, hint=v, source=src)
                return env.create_symintnode(e, hint=v, source=src)

            return sym(8, "a", True), sym(6, "b", True), sym(2, "st"), sym(0.5, "zf")

        def program(env, order):
            a, b, st, zf = symbols(env)
            taken = []
            if order == "div-test":  # a relation over the quotient
                q = a // st
                taken.append(bool(q > 0))
            elif order == "test-div":  # the host's own test first
                taken.append(bool(st != 0))
                taken.append(bool(a % st == 0))
            elif order == "pin-div":  # the payload: never a guard itself
                taken.append(bool(st == 2))
                torch.sym_min(a // st, 0)
            elif order == "div-pin":
                a // st
                taken.append(bool(st == 2))
            else:  # no guard: sizes, literals, a shift, torch's Mod by sizes
                for v in (a % b, (a - 1) % b, a // 4, (a // 2) % b, a >> st):
                    taken.append(v.node.hint)
                taken.append((a / st).node.hint)  # IntTrueDiv: the one guard
                taken.append((1.0 / zf).node.hint)  # a float division too
            return taken, (a, b, st, zf)

        def want(order, A, ST, ZF):
            Ne, Eq = sympy.Ne, sympy.Eq
            return {
                "div-test": [Ne(ST, 0), sympy.Gt(FloorDiv(A, ST), 0)],
                "test-div": [Ne(ST, 0), Eq(PythonMod(A, ST), 0)],
                "pin-div": [Eq(ST, 2), Ne(ST, 0)],
                "div-pin": [Ne(ST, 0), Eq(ST, 2)],
                "none": [Ne(ST, 0), Ne(ZF, 0)],
            }[order]

        def texts(guards):
            return [p.doprint(g) for g in guards]

        for order in ("div-test", "test-div", "pin-div", "div-pin", "none"):
            env = ht._TraceShapeEnv()
            plain = ShapeEnv(duck_shape=False, specialize_zero_one=False)
            taken, (a, b, st, zf) = program(env, order)
            self.assertEqual(taken, program(plain, order)[0], order)
            raw = want(order, a.node.expr, st.node.expr, zf.node.expr)
            self.assertEqual(texts(g.expr for g in env.guards), texts(raw), order)
            kept, pins, _ = env.tape_guards()
            tape = raw[:1] if order == "pin-div" else raw
            self.assertEqual(texts(kept), texts(tape), order)
            self.assertEqual(pins, {st.node.expr: 2} if "pin" in order else {}, order)
        # torch's Mod is defined for nonnegative operands only; sym_node.py
        # builds it when it knows both are, from the symbols' declared
        # properties (no guard) or from their value ranges: the domain rule
        # then records the sign it relied on
        env = ht._TraceShapeEnv()
        a, b, st, zf = symbols(env)
        A, ST = a.node.expr, st.node.expr
        self.assertIsInstance(((a // 2) % b).node.expr, Mod)
        self.assertIsInstance((a % st).node.expr, PythonMod)
        self.assertEqual([g.expr for g in env.guards], [sympy.Ne(ST, 0)])
        env.domain(ST, A, Mod(ST, A))
        env.domain(A, FloorDiv(A, ST), FloorDiv(A, FloorDiv(A, ST)))
        got = [g.expr for g in env.guards]
        self.assertEqual(got[1:], [sympy.Ge(ST, 0), sympy.Ne(FloorDiv(A, ST), 0)])

    def _opaque_alloc_extension(self):
        # a host whose allocation size divides by an opaque rebind result (an
        # eighth of the length; 0 below 8, where eager's own division faults)
        src = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
namespace ht = at::cuda::host_trace;
struct A { int64_t* data; int64_t r; int64_t count; };
namespace at::cuda::host_trace {
template <> struct Traced<A> : TracedBase {
  A pod{};
  PtrField<offsetof(A, data)> data{this, "data"};
  IntField<int64_t, offsetof(A, r)> r{this, "r"};
  IntField<int64_t, offsetof(A, count)> count{this, "count"};
  Traced() : TracedBase(&pod, sizeof(A)) {}
  operator A&() { on_convert(); return pod; }
};
}  // namespace at::cuda::host_trace
__global__ void fill_r(A a) { for (int i = threadIdx.x; i < a.count; i += blockDim.x) a.data[i] = a.r; }
int64_t eighth(const std::vector<int64_t>& v) { return v[0] / 8; }
at::Tensor alloc_by_eighth(const at::Tensor& x) {
  c10::cuda::CUDAGuard g(x.device());
  c10::SymInt n = x.sym_size(0);
  c10::SymInt r = ht::opaque("eighth", {n}, &eighth, "rebind");
  at::Tensor out = at::empty_symint({n / r}, x.options().dtype(at::kLong));
  ht::Traced<A> params;
  params.data = ht::sym_mutable_data_ptr(out);
  params.r = r;
  params.count = out.sym_size(0);
  ht::launch(fill_r, ht::Grid(1), 32, 0, c10::cuda::getCurrentCUDAStream(), params);
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("alloc_by_eighth", &alloc_by_eighth); }
"""
        return load_test_extension("hosttrace_opaque_alloc", src)

    def test_a_late_domain_guard_is_checked_before_the_allocation_that_divides(self):
        # the division's domain over an opaque result (Ne(r, 0)) is a late
        # guard, closed by the opaque event: the interim checks it right after
        # that event, before the allocation size computes with it, so a call
        # whose result is zero misses on the guard and never reaches a
        # ZeroDivisionError inside the size (DOMAIN_GUARDS.md follow-up 3)
        import sympy

        ext = self._opaque_alloc_extension()

        def fn(x):
            return ext.alloc_by_eighth(x)

        x = torch.randn(16, device="cuda")
        tape = ht.trace(fn, (x,))
        r = next(o for o in tape.opaque if o["fn"] == "eighth")["sym"].node.expr
        self.assertIn(sympy.Ne(r, 0), tape.guards)
        variant = ht.build(tape, fn, (x,))
        (out,) = variant.replay((torch.randn(40, device="cuda"),))
        self.assertEqual(out.tolist(), [5] * 8)
        self.assertEqual(fn(torch.randn(40, device="cuda")).tolist(), [5] * 8)
        calls = variant.calls
        with self.assertRaisesRegex(
            ht.Miss, r"guard failed: \S+ != 0 is not true"
        ) as cm:
            variant.replay((torch.randn(4, device="cuda"),))
        self.assertNotIn("undefined", str(cm.exception))
        self.assertEqual(variant.calls, calls)

    def test_a_zero_stride_misses_on_the_divisions_domain_before_the_payload(self):
        # the runtime team's case: a launch field min(size // stride, 0) is
        # never a guard, and folds to 0 under `stride >= 0`; at stride 0 the
        # folded field accepts a call whose evaluation is undefined (eager
        # raises). The tape carries `Ne(stride, 0)` from the division's
        # creation, before any other guard over the stride and before the
        # field: a replay at stride 0 misses on it, never raises inside the
        # evaluation and never launches; a modulo field the same
        import sympy

        from torch.utils._sympy.functions import FloorDiv, Min, PythonMod

        ext = load_test_extension("hosttrace_int_fields", _INT_FIELDS_SOURCE)

        def fn(x, out):
            n, st = x.size(0), x.stride(0)
            return ext.write_fields(out, torch.sym_min(n // st, 0), n % st)

        def inputs(size, stride):
            if stride == 0:
                x = torch.zeros((), device="cuda").expand(size)
            else:
                x = torch.randn(size * stride, device="cuda")[::stride]
            self.assertEqual((x.size(0), x.stride(0)), (size, stride))
            return x, torch.full((2,), -1, device="cuda", dtype=torch.int64)

        args = inputs(16, 2)
        tape = ht.trace(fn, args)
        rec = tape.inputs[0]
        N, ST = rec.sizes[0].node.expr, rec.strides[0].node.expr
        domain = sympy.Ne(ST, 0)
        self.assertIn(domain, tape.guards)
        at = tape.guards.index(domain)
        self.assertFalse(any(ST in g.free_symbols for g in tape.guards[:at]))
        fields = {q["name"]: q["value"] for L in tape.launches for q in L["params"]}
        self.assertEqual(fields["k"].node.expr, Min(0, FloorDiv(N, ST)))
        self.assertEqual(fields["m"].node.expr, PythonMod(N, ST))
        variant = ht.build(tape, fn, args)
        for size, stride in ((16, 3), (15, 2), (7, 5)):
            x, out = inputs(size, stride)
            (got,) = variant.replay((x, out))
            self.assertEqual(got.tolist(), [min(size // stride, 0), size % stride])
            self.assertTrue(torch.equal(got, fn(*inputs(size, stride))))
        x, out = inputs(16, 0)
        with self.assertRaises(ZeroDivisionError):
            fn(x, out.clone())
        self.assertFalse(variant.matches((x, out)))
        calls = variant.calls
        with self.assertRaisesRegex(ht.Miss, r"\S+ != 0 is not true") as cm:
            variant.replay((x, out))
        self.assertNotIn("undefined", str(cm.exception))
        self.assertEqual((variant.calls, out.tolist()), (calls, [-1, -1]))

    def test_a_pinned_divisors_domain_is_its_pin(self):
        # a host that branches on the stride pins it (`Eq(st, 2)`); the
        # division's domain is recorded raw after the pin and the tape's
        # dedupe drops it (the pin substitutes, `Ne(2, 0)` is true), so a
        # replay at any other stride, zero included, misses on the pin
        import sympy

        ext = load_test_extension("hosttrace_int_fields", _INT_FIELDS_SOURCE)

        def fn(x, out):
            n, st = x.size(0), x.stride(0)
            if st == 2:
                return ext.write_fields(out, n // st, n % st)
            return ext.write_fields(out, n, n)

        def inputs(size, stride):
            if stride == 0:
                x = torch.zeros((), device="cuda").expand(size)
            else:
                x = torch.randn(size * stride, device="cuda")[::stride]
            return x, torch.full((2,), -1, device="cuda", dtype=torch.int64)

        args = inputs(16, 2)
        tape = ht.trace(fn, args)
        ST = tape.inputs[0].strides[0].node.expr
        raw = [g.expr for g in tape.shape_env.guards]
        self.assertLess(raw.index(sympy.Eq(ST, 2)), raw.index(sympy.Ne(ST, 0)))
        self.assertIn(sympy.Eq(ST, 2), tape.guards)
        self.assertNotIn(sympy.Ne(ST, 0), tape.guards)
        # the pin is read into the fields: no field mentions the stride
        values = [q["value"] for L in tape.launches for q in L["params"]]
        self.assertFalse(any(str(ST) in ht._free_symbols(v) for v in values))
        variant = ht.build(tape, fn, args)
        x, out = inputs(15, 2)
        (got,) = variant.replay((x, out))
        self.assertEqual(got.tolist(), [7, 1])
        for stride in (3, 0):
            x, out = inputs(16, stride)
            with self.assertRaisesRegex(ht.Miss, r"guard failed: \S+ == 2 is not true"):
                variant.replay((x, out))
            self.assertEqual(out.tolist(), [-1, -1])

    def test_a_len_pin_names_its_frame_in_the_miss(self):
        # len(tensor) returns the first dim through CPython's __index__, a
        # guard_int: the pin eager code writes to test emptiness (transformers'
        # DynamicCache). The pin is a guard like any other; the tape notes the
        # frame that called len() beside it, so a Miss on that guard names it,
        # and nothing walks a stack at replay
        import sympy

        def fn(t, shape, weight, bias, eps):
            if len(t) == 0:
                return t
            return layer_norm(t, shape, weight, bias, eps)

        args = self._args(*self._inputs(8))
        tape = ht.trace(fn, args)
        M = tape.inputs[0].sizes[0].node.expr
        pin = sympy.Eq(M, 8)
        self.assertIn(pin, tape.guards)
        note = tape.guard_notes[pin]
        self.assertRegex(note, r"^len\(\) of p0 at test_cuda_host_trace\.py:\d+$")
        lines, start = inspect.getsourcelines(fn)
        at = start + next(i for i, line in enumerate(lines) if "len(t)" in line)
        self.assertEqual(int(note.rsplit(":", 1)[1]), at)
        self.assertEqual(self._trace(8)[0].guard_notes, {})
        variant = ht.build(tape, fn, args)
        x, w, b = self._inputs(8)
        self._check(variant.replay(self._args(x, w, b)), x, w, b, 8)
        with self.assertRaisesRegex(
            ht.Miss,
            r"guard failed: \S+ == 8 is not true \(len\(\) of p0 at test_cuda_host_trace\.py:\d+\)",
        ):
            variant.replay(self._args(*self._inputs(3)))

    def test_the_deterministic_fill_of_allocations_declines_by_name(self):
        # use_deterministic_algorithms(True) with fill_uninitialized_memory
        # (the default) makes eager's empty*() launch a fill per allocation
        # inside the allocation itself (TensorFactories.h
        # fill_empty_deterministic_), eager's own kernel by name, which no
        # launch of the trace stands in for: the trace declines by name at
        # the first allocation instead of describing an allocation eager
        # fills. With the fill off the flag changes nothing the tape describes.
        with DeterministicGuard(True):
            with self.assertRaisesRegex(ht.Declined, "fill_uninitialized_memory"):
                self._trace(8)
        with DeterministicGuard(True, fill_uninitialized_memory=False):
            tape, args = self._trace(8)
            variant = ht.build(tape, layer_norm, args)
            x, w, b = self._inputs(8)
            got = variant.replay(self._args(x, w, b))
            want = layer_norm(*self._args(x, w, b))
            self.assertTrue(all(torch.equal(g, w_) for g, w_ in zip(got, want)))

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
        root = os.path.join(REPO_ROOT, "aten", "src", "ATen", "cuda", "host_trace")
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

        # the declines from inside the trace are the same type (an op with no
        # host after the converted one; zero_ was the example before it got
        # its fill sibling)
        def unconverted_inside(t, shape, weight, bias, eps):
            out = layer_norm(t, shape, weight, bias, eps)
            out[1].erf_()
            return out

        with self.assertRaises(ht.Declined):
            ht.trace(unconverted_inside, self._args(x, w, b))
        self.assertFalse(torch._C._host_trace_tracing())
        tape, _ = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

    def _composite_ops(self):
        # two custom ops with a CompositeImplicit body: one with a CUDA kernel
        # of its own beside it (the shape of _fused_rms_norm), one without
        def body(t):
            return t * 2.0 + 1.0

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        lib = stack.enter_context(
            torch.library._scoped_library("host_trace_test_composite", "DEF")
        )
        lib.define("kernel_and_body(Tensor x) -> Tensor")
        lib.impl("kernel_and_body", body, "CompositeImplicitAutograd")
        lib.impl("kernel_and_body", lambda t: t * 2.0 + 2.0, "CUDA")
        lib.define("composite_only(Tensor x) -> Tensor")
        lib.impl("composite_only", body, "CompositeImplicitAutograd")
        ops = torch.ops.host_trace_test_composite
        return ops.kernel_and_body.default, ops.composite_only.default

    def test_unconverted_hosts_decline_by_name(self):
        # an op eager serves with a CUDA kernel of its own is never decomposed
        # under the trace, whatever CompositeImplicit body it registers beside
        # the kernel: it is traced by a converted host or declined, and the
        # decline names the op and that route (DECISIONS A190, E34). The class
        # is the dispatcher's registrations, not a list of names: a custom op
        # of _fused_rms_norm's shape declines the same way, at any depth
        kernel_and_body, composite_only = self._composite_ops()
        aten = torch.ops.aten
        own = ht._decomposes_only_off_cuda
        fused = aten._fused_rms_norm.default
        silu, mish = aten.silu_backward.default, aten.mish_backward.default
        for op in (kernel_and_body, fused, silu, mish):
            self.assertTrue(own(op), op)
        promote_types, reshape = aten.promote_types.default, aten.reshape.default
        for op in (composite_only, promote_types, reshape, aten.linear.default):
            self.assertFalse(own(op), op)
        route = "runs its own CUDA kernel.*converted host for {} is the way to trace it"
        x, w, _ = self._inputs(8)
        message = "_fused_rms_norm.*" + route.format("aten::_fused_rms_norm")
        with self.assertRaisesRegex(ht.Declined, message):
            ht.trace(fused, (x, [self.N], w, 1e-5))
        with self.assertRaisesRegex(ht.Declined, message):
            ht.trace(lambda t, g: F.rms_norm(t, (self.N,), g, 1e-5), (x, w))

        def below_autograd(t):
            with torch._C._AutoDispatchBelowAutograd():
                return kernel_and_body(t)

        custom = "host_trace_test_composite::kernel_and_body"
        message = "kernel_and_body.*" + route.format(custom)
        for fn in (kernel_and_body, below_autograd):
            with self.assertRaisesRegex(ht.Declined, message):
                ht.trace(fn, (x,))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_composite_ops_follow_eager_dispatch(self):
        # an op with a CompositeImplicit kernel and none of its own is what
        # eager itself decomposes. At top level the dispatcher decomposes it
        # at the autograd key, above the mode (reshape, linear: the mode never
        # sees the op). Below the autograd keys (a host's own at::reshape, no
        # tensor argument as promote_types) it reaches the mode intact and
        # the fallback runs the same CompositeImplicit kernel eager runs there.
        # Either way the tape is eager's own sequence: the replay's device work
        # is eager's kernel by kernel (the profiler's names, a traced sibling's
        # twin counted as its kernel) and the output is bitwise eager's (E34)
        _, composite_only = self._composite_ops()
        aten = torch.ops.aten
        x, _, _ = self._inputs(64)

        def reshaped(t):
            return (t.reshape(2, -1) * 2.0).reshape(-1)

        def promoted(t):
            return (t * 2.0).to(torch.promote_types(t.dtype, t.dtype))

        def below_autograd(t):
            with torch._C._AutoDispatchBelowAutograd():
                return composite_only(t.reshape(2, -1)).reshape(-1)

        own = ht._decomposes_only_off_cuda
        cases = (
            (reshaped, set()),
            (composite_only, set()),
            (promoted, {aten.promote_types.default}),
            (below_autograd, {aten.reshape.default, composite_only}),
        )
        for fn, want in cases:
            with mock.patch.object(ht, "_decomposes_only_off_cuda", wraps=own) as asked:
                tape = ht.trace(fn, (x,))
            self.assertEqual({c.args[0] for c in asked.call_args_list}, want)
            self._assert_replay_matches_eager(tape, fn, x)
        self.assertFalse(torch._C._host_trace_tracing())

    def _assert_replay_matches_eager(self, tape, fn, x):
        # the replay's device work is eager's kernel by kernel (the profiler's
        # kinds and names; a traced sibling's twin counted as its kernel) and
        # the output is bitwise eager's
        variant = ht.build(tape, fn, (x,))
        replayed = _device_work(lambda t: variant.replay((t,)), (x,))
        eager = _device_work(fn, (x,))
        self.assertEqual(len(replayed), len(eager), (replayed, eager))
        for (kind, name), (kind_e, name_e) in zip(replayed, eager):
            self.assertEqual(kind, kind_e, (replayed, eager))
            if kind == "kernel" and "host_trace" in name:
                self.assertEqual(_launch_shape(name), _launch_shape(name_e))
            elif kind == "kernel":
                self.assertEqual(name, name_e)
        self.assertTrue(torch.equal(variant.replay((x,))[0], fn(x)))

    def _explicit_ops(self):
        # custom ops of slice_backward's shape: a CompositeExplicit body under
        # either alias key and no kernel of their own; one whose body reads
        # its input on the host; one with a CUDA kernel beside the body
        def body(t):
            return t * 2.0 + 1.0

        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        lib = stack.enter_context(
            torch.library._scoped_library("host_trace_test_explicit", "DEF")
        )
        lib.define("body(Tensor x) -> Tensor")
        lib.impl("body", body, "CompositeExplicitAutograd")
        lib.define("nonfunctional(Tensor x) -> Tensor")
        lib.impl("nonfunctional", body, "CompositeExplicitAutogradNonFunctional")
        lib.define("host_read(Tensor x) -> Tensor")
        lib.impl("host_read", lambda t: t * float(t[0, 0]), "CompositeExplicitAutograd")
        lib.define("body_and_kernel(Tensor x) -> Tensor")
        lib.impl("body_and_kernel", body, "CompositeExplicitAutograd")
        lib.impl("body_and_kernel", lambda t: t * 2.0 + 2.0, "CUDA")
        ops = torch.ops.host_trace_test_explicit
        names = ("body", "nonfunctional", "host_read", "body_and_kernel")
        return tuple(getattr(ops, name).default for name in names)

    def test_explicit_bodies_run_as_eager_does(self):
        # an op with no kernel of its own whose entry at the tensors' key is a
        # CompositeExplicit body (slice_backward's shape, either alias key)
        # runs eager's own body under the mode, at top level and below
        # autograd: its pieces are traced as eager launches them and the
        # replay's device work is eager's kernel by kernel. The route is the
        # dispatcher's registrations, not a list of names: a kernel of the
        # op's own beside the body declines by name as before, and a host
        # read inside the body declines where it occurs (DECISIONS E38)
        body, nonfunctional, host_read, body_and_kernel = self._explicit_ops()
        aten = torch.ops.aten
        key = ht._explicit_body_key
        backward = (aten.slice_backward.default, aten.select_backward.default)
        for op in (body, nonfunctional, *backward):
            self.assertIsNotNone(key(op, "CUDA"), op)
        scalar, linear = aten._local_scalar_dense.default, aten.linear.default
        for op in (body_and_kernel, scalar, linear, aten.reshape.default):
            self.assertIsNone(key(op, "CUDA"), op)
        self.assertIsNone(key(aten.full.default, "Undefined"))
        x, _, _ = self._inputs(64)

        def below_autograd(t):
            with torch._C._AutoDispatchBelowAutograd():
                return body(t)

        cases = ((body, body), (nonfunctional, nonfunctional), (below_autograd, body))
        for fn, want in cases:
            with mock.patch.object(ht, "_explicit_body_key", wraps=key) as asked:
                tape = ht.trace(fn, (x,))
            self.assertEqual({c.args[0] for c in asked.call_args_list}, {want})
            self.assertEqual(tape.num_launches, 2)
            self._assert_replay_matches_eager(tape, fn, x)
        by_name = "body_and_kernel.*is not a traceable CUDA host"
        with self.assertRaisesRegex(ht.Declined, by_name):
            ht.trace(body_and_kernel, (x,))
        message = "_local_scalar_dense.*not a traceable CUDA host.*from .*host_read"
        with self.assertRaisesRegex(ht.Declined, message):
            ht.trace(host_read, (x,))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_kernel_less_backward_bodies_are_a_memset_and_a_copy(self):
        # slice_backward (CompositeExplicitAutograd) and select_backward
        # (CompositeExplicitAutogradNonFunctional) have no kernel of their own:
        # eager runs a C++ body at the CUDA key, zeros(input_sizes) then a
        # copy into a view of it. Under the mode that body runs as eager's own
        # and its pieces are the tape, the zeros' memset and one copy launch
        # (a strided copy: eager's kernel, not a memcpy): eager's device work
        # by the profiler, replayed bitwise (E38)
        aten = torch.ops.aten
        M, N = 64, self.N
        g = torch.randn(M, N - 1, device="cuda", dtype=torch.bfloat16)
        h = torch.randn(M, device="cuda", dtype=torch.bfloat16)
        cases = (
            (aten.slice_backward.default, g, (1, 0, N - 1, 1)),
            (aten.select_backward.default, h, (1, 3)),
        )
        key = ht._explicit_body_key
        for op, grad, rest in cases:

            def fn(t, op=op, rest=rest):
                return op(t, [M, N], *rest)

            with mock.patch.object(ht, "_explicit_body_key", wraps=key) as asked:
                tape = ht.trace(fn, (grad,))
            self.assertEqual({c.args[0] for c in asked.call_args_list}, {op})
            self.assertEqual((tape.num_launches, len(tape.memsets)), (1, 1))
            self._assert_replay_matches_eager(tape, fn, grad)
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

    @needs_two_gpus
    def test_build_uses_the_tapes_device(self):
        tape, args = self._trace(16)
        with torch.cuda.device(1):
            variant = ht.build(tape, layer_norm, args)
            x, w, b = self._inputs(24)
        self.assertEqual(x.device.index, 1)
        self.assertEqual(variant.device, 0)
        x, w, b = self._inputs(24)
        self._check(variant.replay(self._args(x, w, b)), x, w, b, 24)

    def _layer_norm_autograd_function(self, threads):
        # a differentiable op whose backward is itself a layer norm launch (a
        # gradient in form only); the engine runs it on its device worker thread
        N = self.N

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w, b):
                ctx.save_for_backward(w, b)
                return layer_norm(x, [N], w, b, 1e-5)[0]

            @staticmethod
            def backward(ctx, g):
                w, b = ctx.saved_tensors
                threads.append(threading.get_ident())
                return layer_norm(g, [N], w, b, 1e-5)[0], None, None

        return Fn

    def test_backward_launches_on_the_engines_worker_thread_are_recorded(self):
        # the autograd engine runs backward nodes on its device worker thread,
        # on the forward op's stream (the capturing one), under a copy of the
        # tracing thread's dispatch-mode stack; the trace mode brings the
        # recorder along for each op it routes there (Recorder.h ThreadScope),
        # so the backward host's launch is recorded like the forward's and the
        # tape replays both
        threads: list = []
        Fn = self._layer_norm_autograd_function(threads)

        def fwd_bwd(x, w, b, cot):
            leaf = x.detach().requires_grad_(True)
            out = Fn.apply(leaf, w, b)
            (gx,) = torch.autograd.grad(out, leaf, grad_outputs=cot)
            return out.detach(), gx

        x, w, b = self._inputs(8)
        cot = torch.randn_like(x)
        tape = ht.trace(fwd_bwd, (x, w, b, cot))
        self.assertEqual(tape.num_launches, 2)
        self.assertGreaterEqual(len(threads), 1)
        self.assertNotIn(threading.get_ident(), threads)
        self.assertFalse(torch._C._host_trace_tracing())
        variant = ht.build(tape, fwd_bwd, (x, w, b, cot))
        for M in (4, 16):
            x, w, b = self._inputs(M)
            cot = torch.randn_like(x)
            out, gx = variant.replay((x, w, b, cot))
            want_out, want_gx = fwd_bwd(x, w, b, cot)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out, want_out))
            self.assertTrue(torch.equal(gx, want_gx))
            self.assertEqual(gx, F.layer_norm(cot, (self.N,), w, b, 1e-5))

    def test_a_launch_from_an_unrelated_thread_is_not_part_of_the_trace(self):
        # a thread the trace mode never reached keeps ordinary mode. Its launch
        # on its own stream runs for real, outside the capture, and the trace
        # records only its own launches; a launch it issues on the trace's
        # capturing stream is a captured node without a record, which the
        # completeness check declines.
        inside = threading.Event()
        done = threading.Event()
        capture_streams: list = []
        results: list = []
        errors: list = []

        def traced(x, w, b):
            capture_streams.append(torch.cuda.current_stream())
            inside.set()
            self.assertTrue(done.wait(30))
            return layer_norm(x, [self.N], w, b, 1e-5)

        def other(stream_of):
            try:
                self.assertTrue(inside.wait(30))
                x, w, b = self._inputs(4)
                # the inputs are made on this thread's default stream; the
                # launch below is on another stream, which must not read
                # them early (a stream wait would touch the capturing stream
                # in the second round, so the default stream is drained)
                torch.cuda.current_stream().synchronize()
                with torch.cuda.stream(stream_of()):
                    out = layer_norm(x, [self.N], w, b, 1e-5)[0]
                results.append((out, F.layer_norm(x, (self.N,), w, b, 1e-5)))
            except Exception as e:
                errors.append(e)
            finally:
                done.set()

        x, w, b = self._inputs(8)
        layer_norm(x, [self.N], w, b, 1e-5)
        torch.cuda.synchronize()
        own = torch.cuda.Stream()
        thread = threading.Thread(target=other, args=(lambda: own,))
        thread.start()
        try:
            tape = ht.trace(traced, (x, w, b), warm_up=False)
        finally:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(tape.num_launches, 1)
        torch.cuda.synchronize()
        self.assertEqual(results[0][0], results[0][1], atol=2e-2, rtol=2e-2)

        inside.clear()
        done.clear()
        capture_streams.clear()
        thread = threading.Thread(target=other, args=(lambda: capture_streams[0],))
        thread.start()
        try:
            with self.assertRaisesRegex(ht.Declined, "has no launch record"):
                ht.trace(traced, (x, w, b), warm_up=False)
        finally:
            thread.join()
        self.assertEqual(errors, [])
        self.assertFalse(torch._C._host_trace_tracing())
        tape, _ = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

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

    def test_the_build_log_is_keyed_by_the_captures_pool_not_by_a_stream_query(self):
        # The build's allocation log is read off the capture's private pool
        # (the allocator's trace entry names the pool a block came from), and
        # the tracker makes no CUDA call: a capture-status query from the
        # allocating thread while the build thread ends its capture faults
        # inside the driver (580.126.20 / CUDA 13.0; the window is inside
        # cudaStreamEndCapture, so no barrier can force it). While another
        # thread allocates throughout, this thread opens, fills and ends
        # captures, and the log holds the captured call's allocations in
        # order, a forked side stream's included, and none of the other
        # thread's.
        C = torch._C
        stop = threading.Event()
        errors: list = []
        counted = [0]

        def allocator():
            try:
                with torch.cuda.stream(torch.cuda.Stream()):
                    while not stop.is_set():
                        torch.empty(96, device="cuda")
                        counted[0] += 1
            except Exception as e:
                errors.append(e)

        stream, side = torch.cuda.Stream(), torch.cuda.Stream()
        thread = threading.Thread(target=allocator)
        thread.start()
        try:
            for _ in range(200):
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                pool = torch.cuda.graph_pool_handle()
                C._host_trace_alloc_log_begin(0, pool)
                try:
                    with torch.cuda.stream(stream):
                        graph.capture_begin(pool=pool, capture_error_mode="relaxed")
                        try:
                            a = torch.empty(1024, device="cuda")
                            side.wait_stream(stream)
                            with torch.cuda.stream(side):
                                b = torch.empty(2048, device="cuda").fill_(2.0)
                            stream.wait_stream(side)
                            a.fill_(1.0)
                        finally:
                            graph.capture_end()
                finally:
                    log = C._host_trace_alloc_log_end()
                self.assertEqual([nbytes for _, nbytes in log], [4096, 8192])
                del graph, a, b
        finally:
            stop.set()
            thread.join()
        self.assertEqual(errors, [])
        self.assertGreater(counted[0], 0)

    def test_the_build_ends_its_capture_on_the_build_stream_whatever_is_current(self):
        # the build captures the ordinary host on its own stream; a host that
        # returns with another stream current must not leave that capture open
        side = torch.cuda.Stream()
        x, w, b = self._inputs(4)

        def returns_with_side_current(x, w, b):
            out = layer_norm(x, [self.N], w, b, 1e-5)
            torch.cuda.set_stream(side)
            return out

        layer_norm(x, [self.N], w, b, 1e-5)
        torch.cuda.synchronize()
        try:
            tape = ht.trace(returns_with_side_current, (x, w, b), warm_up=False)
            variant = ht.build(tape, returns_with_side_current, (x, w, b))
        finally:
            torch.cuda.set_stream(torch.cuda.default_stream())
        x2, w2, b2 = self._inputs(4)
        self._check(variant.replay((x2, w2, b2)), x2, w2, b2, 4)

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

    def test_a_warm_up_that_resizes_an_out_declines_by_name(self):
        # eager's out= of another shape is resized in place (resize_output,
        # a deprecation warning) and the warm-up is the call's execution: the
        # out stays as eager left it, and the trace declines before a symbolic
        # run would describe the resized call; a growing resize replaces the
        # storage, one of the same numel keeps it
        def fn(x, y, o):
            torch.add(x, y, out=o)
            return o

        for shape, kept in (((16, 32), True), ((8, 32), False)):
            with self.subTest(out=shape):
                x, y = torch.randn(2, 8, 64, device="cuda")
                o = torch.randn(shape, device="cuda")
                address = torch._C._host_trace_storage_address(o)
                expected = rf"warm-up changed the metadata of arg2 \(sizes {re.escape(str(shape))} -> \(8, 64\)"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with self.assertRaisesRegex(ht.Declined, expected) as cm:
                        ht.trace(fn, (x, y, o))
                self.assertIn("eager resized it", str(cm.exception))
                self.assertEqual("storage replaced" in str(cm.exception), not kept)
                self.assertIsNone(cm.exception.partial)
                self.assertEqual(o.shape, (8, 64))
                self.assertTrue(torch.equal(o, x + y))
                same = torch._C._host_trace_storage_address(o) == address
                self.assertEqual(same, kept)
                self.assertFalse(torch._C._host_trace_tracing())
        # without the warm-up the out is untouched and the symbolic run decides
        o = torch.randn(16, 32, device="cuda")
        with self.assertRaises(ht.Declined):
            ht.trace(fn, (x, y, o), warm_up=False)
        self.assertEqual(o.shape, (16, 32))
        # an entry remembers the declined class by the inputs as made, not as
        # the warm-up left them: the next such call runs the ordinary host
        entry = ht.Entry(fn)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                x, y = torch.randn(2, 8, 64, device="cuda")
                o = torch.randn(16, 32, device="cuda")
                self.assertTrue(torch.equal(entry(x, y, o)[0], x + y))
        self.assertEqual((entry.traces, entry.ordinary), (1, 2))
        declined = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(declined), 1)
        self.assertIn("warm-up changed the metadata of arg2", str(declined[0].message))

    def test_a_warm_up_that_replaces_an_inputs_storage_declines(self):
        # set_ keeps the sizes and strides and swaps the storage: the symbolic
        # run would bind the new storage's address as the input's root
        x, w, b = self._inputs(8)

        def fn(t, shape, weight, bias, eps):
            t.set_(torch.empty_like(t))
            return layer_norm(t, shape, weight, bias, eps)

        with self.assertRaisesRegex(
            ht.Declined, r"warm-up changed the metadata of arg0 \(storage replaced\)"
        ):
            ht.trace(fn, self._args(x, w, b))
        self.assertEqual(x.shape, (8, self.N))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_a_warm_up_that_changes_values_still_traces(self):
        # the warm-up is the call's execution: an in-place kernel leaves its
        # input's values changed and its metadata as it found them, and the
        # trace goes on to the symbolic run
        add_one = self._add_one()
        x = torch.randn(8, device="cuda")
        start = x.clone()
        tape = ht.trace(lambda t: add_one(t), (x,))
        self.assertTrue(torch.equal(x, start + 1))
        self.assertEqual(tape.num_launches, 1)

    def test_a_warm_up_that_materializes_a_copy_on_write_input_still_traces(self):
        # an in-place kernel's mutable read materializes a lazily cloned input
        # (eager's first call does the same): the storage address moves and
        # nothing else, the trace goes on and describes the materialized
        # tensor; the clone's source keeps its values
        add_one = self._add_one()
        x = torch.randn(8, device="cuda")
        start = x.clone()
        lazy = torch._lazy_clone(x)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        tape = ht.trace(lambda t: add_one(t), (lazy,))
        self.assertFalse(torch._C._is_cow_tensor(lazy))
        self.assertTrue(torch.equal(lazy, start + 1))
        self.assertTrue(torch.equal(x, start))
        self.assertEqual(tape.num_launches, 1)

    def test_a_replay_materializes_a_copy_on_write_input_in_a_written_position(self):
        # the tape writes its argument (the host's mutable read names the root
        # in Tape.written_roots): a replay bound to a fresh lazy clone must not
        # write through the storage the clone still shares, so the binding
        # materializes it first, where eager's mutable read would, and the
        # clone's source keeps its values; a read-only position stays lazy
        # (test_a_copy_on_write_input_stays_lazy)
        add_one = self._add_one()

        def fn(t):
            return add_one(t)

        args = (torch.randn(8, device="cuda"),)
        tape = ht.trace(fn, args)
        self.assertEqual(tape.written_roots, ["p0"])
        self.assertEqual(tape.written_inputs, (0,))
        records = json.loads(tape.to_json())
        self.assertEqual(records["written_roots"], ["p0"])
        self.assertEqual(records["written_inputs"], [0])
        variant = ht.build(tape, fn, args)
        src = torch.randn(8, device="cuda")
        start = src.clone()
        lazy = torch._lazy_clone(src)
        self.assertTrue(torch._C._is_cow_tensor(lazy))
        (out,) = variant.replay((lazy,))
        torch.cuda.synchronize()
        self.assertIs(out, lazy)
        self.assertFalse(torch._C._is_cow_tensor(lazy))
        self.assertTrue(torch.equal(lazy, start + 1))
        self.assertTrue(torch.equal(src, start))
        # a call outside the argument contract misses before any of this
        with self.assertRaisesRegex(ht.Miss, "0 arguments, the trace had 1"):
            variant.replay(())
        # eager on a fresh lazy clone of the same source: the same result
        lazy = torch._lazy_clone(src)
        add_one(lazy)
        self.assertTrue(torch.equal(lazy, start + 1))
        self.assertTrue(torch.equal(src, start))

    def test_materialize_reads_the_named_positions_through_the_mutable_accessor(self):
        # the one call both replays make before they bind a tape's written
        # inputs: a copy-on-write tensor at a named position materializes, a
        # tensor elsewhere stays lazy, a non-tensor at a named position is
        # left alone, a position outside the sequence is the caller's error
        src = torch.randn(8, device="cuda")
        a, b = torch._lazy_clone(src), torch._lazy_clone(src)
        torch._C._host_trace_materialize((a, 3, b), (0, 1))
        self.assertFalse(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertTrue(torch.equal(a, src))
        with self.assertRaises(IndexError):
            torch._C._host_trace_materialize((a,), (1,))

    def test_an_entry_torn_down_on_another_thread_leaves_a_live_trace_intact(self):
        # a variant's teardown is its CUDAGraph's (the graph and the exec
        # destroyed, the pool released; on CUDA no device-wide synchronize),
        # which the driver permits beside a thread-local capture on another
        # thread: variants with replays behind them dropped by refcount and
        # by an explicit gc.collect() on one thread while another traces
        stop = threading.Event()
        errors: list = []
        drops = [0]

        def dropper():
            try:
                args = self._args(*self._inputs(8))
                tape = ht.trace(layer_norm, args)
                while not stop.is_set():
                    variant = ht.build(tape, layer_norm, args)
                    variant.replay(args)
                    del variant
                    drops[0] += 1
                    if drops[0] % 2 == 0:
                        gc.collect()
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=dropper)
        thread.start()
        try:
            for M in range(9, 21):
                tape, _ = self._trace(M)
                self.assertEqual(tape.num_launches, 1)
        finally:
            stop.set()
            thread.join()
        self.assertEqual(errors, [])
        self.assertGreater(drops[0], 0)

    def test_an_entry_torn_down_inside_the_traced_call_declines_by_name(self):
        # the other side of the rule: a CUDAGraph destroyed on the tracing
        # thread under the capture is a call the capture forbids, so the
        # driver invalidates it; the trace holds the cyclic collector off
        # (the collector cannot do this), so the drop is the traced
        # function's own, named at its next launch; the thread traces again
        # afterwards
        args = self._args(*self._inputs(8))
        held = [ht.build(ht.trace(layer_norm, args), layer_norm, args)]
        held[0].replay(args)

        def fn(x, shape, w, b, eps):
            if torch._C._host_trace_tracing() and held:
                held.pop()
                gc.collect()
            return layer_norm(x, shape, w, b, eps)

        with self.assertRaisesRegex(
            ht.Declined, "after the trace's capture was invalidated"
        ):
            ht.trace(fn, self._args(*self._inputs(12)))
        self.assertEqual(held, [])
        tape, args = self._trace(13)
        variant = ht.build(tape, layer_norm, args)
        x, w, b = self._inputs(13)
        out = variant.replay(self._args(x, w, b))[0]
        self.assertEqual(
            out, F.layer_norm(x, (self.N,), w, b, 1e-5), atol=2e-2, rtol=2e-2
        )

    def test_sync_memory_api_lint_catches_a_converted_host(self):
        # the host contract's synchronous memory calls are invisible to a
        # capture; the lint reports them in every translation unit under the
        # CUDA host directories (however the host reaches the recorder's
        # headers) and in any file that includes them, and nothing else
        root = REPO_ROOT
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

    def test_a_build_under_the_cudamallocasync_backend_declines_by_name(self):
        # the build's allocation log is the native caching allocator's trace
        # tracker, which the cudaMallocAsync backend does not offer: a trace
        # still records, the build is a Declined naming the backend, and an
        # entry serves such calls through the ordinary host with the one
        # warning; the backend is chosen at process start, so in a subprocess
        script = r"""
import warnings, torch, torch.nn.functional as F
import torch.cuda._host_trace as ht
print("RESULT backend", torch.cuda.get_allocator_backend())
x, w, b = (torch.randn(8, 64, device="cuda"), torch.randn(64, device="cuda"), torch.randn(64, device="cuda"))
def layer_norm(x, w, b):
    return F.layer_norm(x, (64,), w, b, 1e-5)
tape = ht.trace(layer_norm, (x, w, b))
print("RESULT traced", tape.num_launches)
try:
    ht.build(tape, layer_norm, (x, w, b))
    print("RESULT built")
except ht.Declined as e:
    print("RESULT declined", str(e).splitlines()[0])
entry = ht.Entry(layer_norm)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    first = entry(x, w, b)[0]
    second = entry(x, w, b)[0]
warned = sum("the ordinary host serves such calls" in str(c.message) for c in caught)
print("RESULT entry", entry.traces, entry.ordinary, len(entry.variants), warned, torch.equal(first, layer_norm(x, w, b)), torch.equal(second, first))
"""
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync")
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out[-3000:])
        self.assertIn("RESULT backend cudaMallocAsync", out)
        self.assertIn("RESULT traced 1", out)
        self.assertNotIn("RESULT built", out)
        self.assertRegex(
            out,
            r"RESULT declined host_trace: a build reads .*cudaMallocAsync allocator backend",
        )
        self.assertIn("RESULT entry 1 2 0 1 True True", out)

    @needs_two_gpus
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

    def test_a_capture_end_that_fails_after_the_capture_ended_is_not_ended_twice(self):
        # capture_end can throw after the driver ended the capture and the
        # allocator's capture count was decremented (a generator or node
        # readback failure): an unclassified error, propagated as is. The
        # scope closing behind it must not end the capture a second time,
        # which would decrement the count again: a capture open on another
        # thread would then fail at its own end with "no active capture".
        opened = threading.Event()
        release = threading.Event()
        errors: list = []

        def other_capture():
            try:
                g = torch.cuda.CUDAGraph()
                y = torch.ones(8, device="cuda")
                with torch.cuda.stream(torch.cuda.Stream()):
                    g.capture_begin(capture_error_mode="thread_local")
                    try:
                        y.add_(1)
                        opened.set()
                        release.wait(30)
                    finally:
                        g.capture_end()
            except Exception as e:
                errors.append(e)
            finally:
                opened.set()

        thread = threading.Thread(target=other_capture)
        thread.start()
        self.assertTrue(opened.wait(30))
        x, w, b = self._inputs(8)
        torch._C._host_trace_test_fail_capture_end(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "test hook: capture_end failed"):
                ht.trace(layer_norm, self._args(x, w, b))
        finally:
            torch._C._host_trace_test_fail_capture_end(False)
            release.set()
            thread.join()
        self.assertFalse(torch._C._host_trace_tracing())
        self.assertEqual(errors, [])
        tape, args = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

    def test_non_contiguous_inputs_of_any_rank_trace_in_bounded_time(self):
        # a contiguity query on a non-contiguous input is guarded term by term:
        # the negation of a rank-sized conjunction would go through the
        # ShapeEnv's CNF pass, exponential in rank. The answer is either a
        # served tape or a decline at the copy the host makes, never a hang.
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

        def fn2(t):
            return F.layer_norm(t, (16,))

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

    def test_the_collector_stays_held_until_the_last_trace_on_any_thread_ends(self):
        # trace() holds the cyclic collector off; the hold is one process-wide
        # count, so a trace that ends on one thread does not re-enable the
        # collector under a trace still running on another (a collection there
        # would invalidate that thread's capture)
        enabled = gc.isenabled()
        main_in = threading.Event()
        other_in = threading.Event()
        main_done = threading.Event()
        seen: list = []
        errors: list = []

        def outer(x, shape, w, b, eps):
            main_in.set()
            self.assertTrue(other_in.wait(30))
            return layer_norm(x, shape, w, b, eps)

        def inner(x, shape, w, b, eps):
            other_in.set()
            self.assertTrue(main_done.wait(30))
            seen.append(gc.isenabled())
            return layer_norm(x, shape, w, b, eps)

        def other():
            try:
                args = self._args(*self._inputs(8))
                layer_norm(*args)
                torch.cuda.current_stream().synchronize()
                self.assertTrue(main_in.wait(30))
                ht.trace(inner, args, warm_up=False)
            except Exception as e:
                errors.append(e)
            finally:
                other_in.set()

        gc.enable()
        thread = threading.Thread(target=other)
        thread.start()
        try:
            args = self._args(*self._inputs(8))
            layer_norm(*args)
            torch.cuda.current_stream().synchronize()
            tape = ht.trace(outer, args, warm_up=False)
            held_after_first = gc.isenabled()
        finally:
            main_done.set()
            thread.join()
            enabled_after_last = gc.isenabled()
            if not enabled:
                gc.disable()
        self.assertEqual(errors, [])
        self.assertFalse(held_after_first)
        # one entry per trace of inner: the two-hint family traces it again
        self.assertEqual(set(seen), {False})
        self.assertTrue(enabled_after_last)
        self.assertEqual(tape.num_launches, 1)

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

    def test_a_return_with_another_stream_current_still_ends_the_capture(self):
        # CUDAGraph ends a capture only on the stream that began it; the
        # recorder makes the capture stream current again for capture_end, so
        # a host that left another stream current does not leave its capture
        # open (the pool stream stuck capturing, the allocator's capture count
        # never decremented)
        side = torch.cuda.Stream()
        x, w, b = self._inputs(4)
        capture_streams = []

        def returns_with_side_current(x, w, b):
            capture_streams.append(torch.cuda.current_stream())
            out = layer_norm(x, [self.N], w, b, 1e-5)
            torch.cuda.set_stream(side)
            return out

        layer_norm(x, [self.N], w, b, 1e-5)
        torch.cuda.synchronize()
        try:
            tape = ht.trace(returns_with_side_current, (x, w, b), warm_up=False)
        finally:
            torch.cuda.set_stream(torch.cuda.default_stream())
        self.assertEqual(tape.num_launches, 1)
        self._assert_capture_streams_free(capture_streams)
        tape, args = self._trace(8)
        x, w, b = self._inputs(4)
        self._check(
            ht.build(tape, layer_norm, args).replay(self._args(x, w, b)), x, w, b, 4
        )

    def test_a_decline_with_another_stream_current_still_ends_the_capture(self):
        # the decline is raised inside the host with its stream current; the
        # abandoned capture ends on the capture stream all the same
        side = torch.cuda.Stream()
        x, w, b = self._inputs(4)
        capture_streams = []

        def launches_on_side(x, w, b):
            capture_streams.append(torch.cuda.current_stream())
            torch.cuda.set_stream(side)
            return layer_norm(x, [self.N], w, b, 1e-5)

        layer_norm(x, [self.N], w, b, 1e-5)
        torch.cuda.synchronize()
        with self.assertRaisesRegex(ht.Declined, "not the trace's capturing stream"):
            try:
                ht.trace(launches_on_side, (x, w, b), warm_up=False)
            finally:
                torch.cuda.set_stream(torch.cuda.default_stream())
        self.assertFalse(torch._C._host_trace_tracing())
        self._assert_capture_streams_free(capture_streams)
        tape, _ = self._trace(8)
        self.assertEqual(tape.num_launches, 1)

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
        # is registered on the int signature, so a symbolic element is
        # converted through a guard before the redispatch: the trace serves
        # the traced width and misses another one by name.
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
        # the host writes its allocations only: no input position is written
        self.assertEqual(set(tape.written_roots), {a.root.name for a in tape.allocs})
        self.assertEqual(tape.written_inputs, ())
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
        # replay bitwise.
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
        # an external event recorded on the capturing stream is a node, and a
        # wait on it from a never-forked stream fails
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
        require_nvcc()
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
        require_nvcc()
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

    def test_a_call_executes_the_user_function_exactly_once(self):
        # the interim entry (a trace, hits, a miss traced again) executes the
        # user's function once per call, as eager does: the ordinary call
        # whose outputs the caller receives is the warm-up, and the trace and
        # the build that follow with warm_up=False execute nothing; an
        # in-place function shows every extra execution in its input
        add_one = self._add_one()

        def fn(x):
            return add_one(x)

        def executions(x, start):
            # add_one adds 1.0f per execution, in sequence: the count is the
            # number of eager += 1 steps that reproduce x from its start
            torch.cuda.synchronize()
            t = start.clone()
            for n in range(8):
                if torch.equal(x, t):
                    return n
                t += 1
            self.fail("x is not a whole number of executions from its start")

        variants = []

        def entry(x):
            for v in variants:
                out = v.try_replay((x,))
                if out is not None:
                    return out[0]
            out = fn(x)
            tape = ht.trace(fn, (x,), warm_up=False)
            variants.append(ht.build(tape, fn, (x,), warm_up=False))
            return out

        # a trace, a hit, and a miss (another rank) traced again: one
        # execution per call, as eager, and eager's outputs
        calls = [torch.randn(8, device="cuda") for _ in range(2)]
        calls.append(torch.randn(3, 4, device="cuda"))
        served = [c.clone() for c in calls]
        for k, (c, s, n_variants) in enumerate(zip(calls, served, (1, 1, 2))):
            want = fn(c.clone())
            got = entry(s)
            self.assertEqual(executions(s, c), 1, f"call {k}")
            self.assertTrue(torch.equal(got, want), f"call {k}")
            self.assertEqual(len(variants), n_variants, f"call {k}")
        self.assertEqual(variants[0].calls, 1)
        # no later call touched an earlier input
        self.assertEqual([executions(s, c) for c, s in zip(calls, served)], [1, 1, 1])
        # the count reads the defaults' executions too: the trace's warm-up
        # (1), then the build's two ordinary calls and its instantiating
        # launch (3)
        x = torch.randn(8, device="cuda")
        start = x.clone()
        tape = ht.trace(fn, (x,))
        self.assertEqual(executions(x, start), 1)
        ht.build(tape, fn, (x,))
        self.assertEqual(executions(x, start), 4)

    def test_addresses_of_two_roots_are_decided_by_root_identity(self):
        # a host's `dst != src` over two live tensors of different roots is
        # answered by identity when either is an allocation (an allocation
        # never shares memory with another live root): no guard, whatever the
        # address hints, and the pair is a root fact of the tape. Two inputs
        # may be one storage passed twice, so their comparison is an address
        # guard on the real addresses. An ordering of two roots declines.
        def address(t):
            if isinstance(t, ht._TracedTensor):
                return t._root.sym + t._sym_offset * t.element_size()
            return t.data_ptr()

        seen = []

        def fn(x, shape, w, b, eps):
            y, z = torch.empty_like(x), torch.empty_like(x)
            seen.append(
                (
                    bool(address(y) == address(z)),
                    bool(address(y) != address(z)),
                    bool(address(y) == address(x[1:])),
                    bool(address(w) == address(b)),
                )
            )
            return layer_norm(x, shape, w, b, eps)

        args = self._args(*self._inputs(8))
        tape = two_hint.trace_twice(fn, args)
        self.assertEqual(seen[-1], (False, True, False, False))
        # after the declared domain rows of the input sizes
        self.assertEqual(tape.root_facts[-2:], [("a0", "a1"), ("a0", "p0")])
        self.assertEqual(
            json.loads(tape.to_json())["root_facts"][-2:], [["a0", "a1"], ["a0", "p0"]]
        )
        self.assertTrue(all(r[0] == "domain" for r in tape.root_facts[:-2]))
        alloc_syms = {a.q.node.expr for a in tape.allocs}
        self.assertFalse(any(g.free_symbols & alloc_syms for g in tape.guards))
        by_position = {i.position: i for i in tape.inputs}
        w_sym, b_sym = (by_position[k].root.sym.node.expr for k in (2, 3))
        self.assertTrue(any({w_sym, b_sym} <= g.free_symbols for g in tape.guards))
        # the allocations' hints are distinct placeholders under their own top
        q0, q1 = (a.q.node.hint for a in tape.allocs[:2])
        self.assertNotEqual(q0, q1)
        self.assertEqual({(256 * q) >> 52 for q in (q0, q1)}, {ht._ALLOC_TAG >> 52})
        variant = ht.build(tape, fn, args)
        x, w, b = self._inputs(8)
        self._check(variant.replay(self._args(x, w, b)), x, w, b, 8)

        # one storage at two positions: equal under the trace, an address
        # guard the replay checks
        def same(x, shape, w, b, eps):
            seen.append(bool(address(w) == address(b)))
            return layer_norm(x, shape, w, b, eps)

        x, w, b = self._inputs(8)
        args = self._args(x, w, w)
        tape = ht.trace(same, args)
        self.assertTrue(seen[-1])
        self.assertEqual([r for r in tape.root_facts if r[0] != "domain"], [])
        variant = ht.build(tape, same, args)
        self._check(variant.replay(args), x, w, w, 8)
        self.assertIsNone(variant.try_replay(self._args(x, w, b)))

        def ordered(x, shape, w, b, eps):
            bool(address(torch.empty_like(x)) < address(x))
            return layer_norm(x, shape, w, b, eps)

        with self.assertRaisesRegex(ht.Declined, "ordered the addresses of two roots"):
            ht.trace(ordered, args)

    def test_an_empty_view_has_a_null_address_like_eager(self):
        # TensorImpl::data() is null for a tensor with no elements whatever its
        # storage, and a host passes that null on (cat's metadata for an empty
        # piece); the recorder's accessors yield the constant 0 for an empty
        # traced view, a guard on the sizes when they are symbolic. Read
        # through the accessor a converted host reads, in a test-built host
        ext = load_test_extension(
            "hosttrace_verbatim_formals", _VERBATIM_FORMALS_SOURCE
        )
        x, w, b = self._inputs(8)
        self.assertEqual(x[:0].data_ptr(), 0)
        self.assertEqual(ext.address_of(x[:0]), 0)
        self.assertEqual(ext.address_of(x), x.data_ptr())
        addresses = []

        def fn(x, shape, w, b, eps):
            views = (x[:0], x[:, :0], x[3:3], x.split([0, x.shape[0]])[0], x[8:], x[:1])
            addresses.extend(ext.address_of(v) for v in views)
            return layer_norm(x, shape, w, b, eps)

        args = self._args(x, w, b)
        tape = two_hint.trace_twice(fn, args)
        traced = addresses[-6:]
        self.assertEqual(traced[:5], [0] * 5)
        self.assertIsInstance(traced[5], torch.SymInt)
        # x[8:] is empty at the traced batch only: the null is guarded on the
        # batch, so the trace serves that batch and misses others by name
        variant = ht.build(tape, fn, args)
        self._check(variant.replay(args), x, w, b, 8)
        self.assertIsNone(variant.try_replay(self._args(*self._inputs(16))))

    def test_a_closure_tensor_under_a_symbolic_view_declines_by_name(self):
        # matmul expands a weight the trace does not own (captured by the
        # closure, not an argument) to the traced batch: a symbolic size
        # inside expand's size list is a typed decline naming the op, not the
        # composite's SymIntArrayRef error
        w = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        cases = (
            (lambda x: F.linear(x.transpose(1, 2), w), (2, 64, 8)),
            (lambda x: F.linear(x[:, 1:], w), (2, 8, 64)),
        )
        for fn, shape in cases:
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            with self.assertRaisesRegex(
                ht.Declined, "aten.expand.default with a symbolic size on a tensor"
            ):
                ht.trace(fn, (x,))

    def test_outputs_that_are_arguments_or_earlier_outputs_keep_their_identity(self):
        # eager returns the argument object itself and one object for a
        # repeated output; the tape records both by identity, read from the
        # objects the host returned, and the replay returns those objects. A
        # view is a distinct object in eager and stays one.
        def fn(x, w, b):
            y, mean, rstd = layer_norm(x, [self.N], w, b, 1e-5)
            return y, x, y, y.view(-1), mean, rstd

        x, w, b = self._inputs(8)
        tape = ht.trace(fn, (x, w, b))
        self.assertEqual(
            [o.identity for o in tape.outputs],
            [None, ("argument", 0), ("output", 0), None, None, None],
        )
        self.assertEqual(
            [o["identity"] for o in json.loads(tape.to_json())["outputs"]],
            [None, ["argument", 0], ["output", 0], None, None, None],
        )
        variant = ht.build(tape, fn, (x, w, b))
        x2, w2, b2 = self._inputs(16)
        eager = fn(x2, w2, b2)
        self.assertIs(eager[1], x2)
        self.assertIs(eager[2], eager[0])
        outs = variant.replay((x2, w2, b2))
        self.assertIs(outs[1], x2)
        self.assertIs(outs[2], outs[0])
        self.assertIsNot(outs[3], outs[0])
        self.assertEqual(outs[3].data_ptr(), outs[0].data_ptr())
        for got, want in zip(outs, eager):
            self.assertTrue(torch.equal(got, want))

    def test_float_subclass_constants_compare_by_bits(self):
        # a float subclass reaches the kernel as its double: eager sees neither
        # the subclass's __eq__ nor anything else it carries, so its constant is
        # its type and its bits like a float's. Two tagged values with
        # equal bits are one constant; a signed zero change misses, also for a
        # numpy float64, which is a float subclass whose == treats 0.0 and -0.0
        # as equal.
        class Tagged(float):
            def __new__(cls, value, tag):
                self = super().__new__(cls, value)
                self.tag = tag
                return self

            def __eq__(self, other):
                return type(other) is Tagged and self.tag == other.tag

            __hash__ = float.__hash__

        self.assertEqual(ht._constant(Tagged(1.0, "a")), ht._constant(Tagged(1.0, "b")))
        self.assertNotEqual(
            ht._constant(Tagged(0.0, "a")), ht._constant(Tagged(-0.0, "a"))
        )
        if not TEST_NUMPY:
            self.skipTest("numpy")
        import numpy as np

        zero, minus_zero = np.float64(0.0), np.float64(-0.0)
        self.assertTrue(zero == minus_zero)
        self.assertNotEqual(ht._constant(zero), ht._constant(minus_zero))
        x, w, b = self._inputs(16)
        args = self._args(x, w, b, eps=zero)
        variant = ht.build(ht.trace(layer_norm, args), layer_norm, args)
        self.assertIsNotNone(variant.try_replay(self._args(x, w, b, eps=zero)))
        self.assertIsNone(variant.try_replay(self._args(x, w, b, eps=minus_zero)))

    def test_float32_hint_and_python_execution(self):
        # _round_float32 narrows as a float formal does (the float64 to float32
        # cast), keeps the symbol under a Float32 node with the narrowed hint,
        # records no guard, and the program evaluates it back to the narrowed
        # value
        import sympy

        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def bits(v):
            return struct.pack("<d", v)

        program = ht._Program(self._trace(8)[0])
        for value in (0.0, -0.0, 1.0 + 2**-24, 1e300, -1e300, math.inf, math.nan):
            expected = torch.tensor([value], dtype=torch.float64).float().item()
            self.assertEqual(bits(ht._round_float32(value)), bits(expected))
            env = ShapeEnv()
            symbol = sympy.Symbol("scale", real=True)
            original = torch.SymFloat(SymNode(symbol, env, float, value))
            rounded = ht._round_float32(original)
            self.assertIs(rounded.node.shape_env, env)
            self.assertEqual(rounded.node._expr, ht.Float32(symbol))
            self.assertEqual(bits(rounded.node.hint), bits(expected))
            self.assertEqual(env.guards, [])
            evaluated = program.ev(rounded, {"scale": value})
            self.assertEqual(bits(evaluated), bits(expected))

    def test_float32_keeps_original_symbol_and_fx_provenance(self):
        import sympy

        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        env = ShapeEnv()
        symbol = sympy.Symbol("scale", real=True)
        original = torch.SymFloat(SymNode(symbol, env, float, 1.1))
        env._set_replacement(symbol, sympy.Float(1.1), "test replacement")
        with mock.patch.object(
            env, "_create_fx_call_function", return_value=(None, True)
        ) as fx:
            rounded = ht._round_float32(original)
        self.assertEqual(rounded.node._expr, ht.Float32(symbol))
        fx.assert_called_once_with(ht._round_float32, (original.node.fx_node,))
        self.assertEqual(env.guards, [])

    def test_a_float_member_narrows_before_it_is_read_back(self):
        # a float member of a parameter struct narrows on assignment, and a
        # host that reads it back computes with the narrowed value (flash:
        # scale_softmax_log2 = softmax_scale * M_LOG2E after the scale went
        # through a float). The recorder records the member as Float32 of its
        # symbolic value, so the replay's bits equal eager's at every head dim;
        # the double product narrowed once differs in the last bit at some.
        cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
namespace ht = at::cuda::host_trace;
struct Scale { float scale; float log2_scale; };
namespace at::cuda::host_trace {
template <> struct Traced<Scale> : TracedBase {
  Scale pod{};
  FloatField<float, offsetof(Scale, scale)> scale{this, "scale"};
  FloatField<float, offsetof(Scale, log2_scale)> log2_scale{this, "log2_scale"};
  Traced() : TracedBase(&pod, sizeof(Scale)) {}
  operator Scale&() { on_convert(); return pod; }
};
}  // namespace at::cuda::host_trace
__global__ void write_scale(Scale s, float* out) {
  out[0] = s.scale;
  out[1] = s.log2_scale;
}
at::Tensor scale_of(const at::Tensor& x) {
  c10::cuda::CUDAGuard g(x.device());
  at::Tensor out = at::empty({2}, x.options().dtype(at::kFloat));
  ht::Traced<Scale> s;
  const c10::SymFloat head_dim = x.sym_size(-1);
  s.scale = c10::SymFloat(1.0) / head_dim.sqrt();
  s.log2_scale = s.scale.sym() * M_LOG2E;
  ht::launch(write_scale, 1, 1, 0, c10::cuda::getCurrentCUDAStream(), s, ht::sym_mutable_data_ptr(out));
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scale_of", &scale_of);
}
"""
        ext = load_test_extension("ht_float_member_narrowing", cuda_src)

        def fn(x):
            return ext.scale_of(x)

        def narrowed(d, once):
            # the two float values as int32 bits: the scale, and the product
            # with log2(e) taken from the double scale (once) or from the
            # narrowed scale (as a float member reads back)
            scale = torch.tensor([1.0 / math.sqrt(d)], dtype=torch.float64)
            log2 = scale if once else scale.float().double()
            both = torch.cat([scale.float(), (log2 * math.log2(math.e)).float()])
            return both.view(torch.int32).tolist()

        x = torch.empty(1, 48, device="cuda")
        tape = ht.trace(fn, (x,))
        self.assertIn("Float32(", tape.to_json())
        variant = ht.build(tape, fn, (x,))
        differing = 0
        for d in (48, 120, 200, 64, 17, 3):
            y = torch.empty(1, d, device="cuda")
            eager = fn(y).cpu().view(torch.int32).tolist()
            self.assertEqual(eager, narrowed(d, once=False), f"d={d}")
            out = variant.replay((y,))[0]
            torch.cuda.synchronize()
            self.assertEqual(out.cpu().view(torch.int32).tolist(), eager, f"d={d}")
            differing += narrowed(d, once=True) != eager
        self.assertGreater(differing, 0)

    def test_every_case_traces_the_same_program_under_other_hints(self):
        # the recorder never reads a hint: every trace this class makes, made
        # again under other hints, is the same program (host_trace_two_hint)
        two_hint.assert_family(
            self,
            exclude={
                "test_trace_warms_up_once_before_the_symbolic_run": "counts the calls of the traced function; the second run adds one",
            },
        )

    def _after_layer_norm(self, t, w, b):
        # an unsupported op after layer norm's launch: the trace declines
        # with layer norm's guards recorded
        return torch.atan(layer_norm(t, [self.N], w, b, 1e-5)[0])

    def test_a_decline_carries_the_guards_so_far(self):
        # Declined.partial: the argument contract, the inputs bound to their
        # symbols and the guards recorded up to the decline; matches() says
        # whether a call is in the class of calls that reach it
        x, w, b = self._inputs(8)
        with self.assertRaisesRegex(ht.Declined, "aten.atan.default") as cm:
            ht.trace(self._after_layer_norm, (x, w, b))
        p = cm.exception.partial
        self.assertIs(p.op, torch.ops.aten.atan.default)
        self.assertEqual(p.reason, str(cm.exception))
        self.assertEqual((p.nargs, p.positions, p.constants), (3, [0, 1, 2], ()))
        self.assertEqual([i.name for i in p.inputs], ["arg0", "arg1", "arg2"])
        # layer norm's contiguity and alignment guards are over the inputs
        # alone and hold at every batch: the class is size-independent
        self.assertGreater(len(p.guards), 0)
        self.assertTrue(p.matches((x, w, b)))
        for M in (1, 5, 64):
            self.assertTrue(p.matches(self._inputs(M)))
        self.assertTrue(p.matches(self._inputs(8, offset=8)))
        # outside the class: an input that fails a guard (misaligned), another
        # dtype, another rank, another arity or a constant where a tensor was
        self.assertFalse(p.matches(self._inputs(8, offset=1)))
        self.assertFalse(p.matches(self._inputs(8, dtype=torch.float32)))
        self.assertFalse(p.matches((x.view(-1), w, b)))
        self.assertFalse(p.matches((x, w)))
        self.assertFalse(p.matches((x, w, b, 1e-5)))
        self.assertFalse(p.matches((x, w, 2.0)))
        # a guard over a symbol the inputs do not bind cannot be decided
        # without running the host: no call matches, not even the traced one
        import dataclasses

        import sympy

        foreign = dataclasses.replace(p, guards=(*p.guards, sympy.Symbol("s0") > 0))
        self.assertFalse(foreign.matches((x, w, b)))
        # the reason names the declining op; a Python-side host read of a
        # traced tensor (outside any op) has no op
        with self.assertRaisesRegex(ht.Declined, "tolist") as cm:
            ht.trace(lambda t, u: t * u.tolist()[0], (x, x[0, :1]))
        self.assertIsNone(cm.exception.partial.op)
        self.assertEqual(cm.exception.partial.guards, ())
        self.assertFalse(torch._C._host_trace_tracing())

    def test_a_decline_before_the_inputs_are_bound_carries_no_partial_trace(self):
        x, w, b = self._inputs(8)
        for args, message in (
            ((torch.empty(0, self.N, device="cuda", dtype=x.dtype), w, b), "empty"),
            ((x.cpu(), w, b), "pageable"),
            ((torch._neg_view(x), w, b), "negative view"),
            (([self.N], 1e-5), "no tensor arguments"),
        ):
            with self.assertRaisesRegex(ht.Declined, message) as cm:
                ht.trace(self._after_layer_norm, args)
            self.assertIsNone(cm.exception.partial)
        self.assertFalse(torch._C._host_trace_tracing())

    def _host_trace_warnings(self, caught):
        return [str(c.message) for c in caught if "host_trace" in str(c.message)]

    def test_a_declined_class_is_remembered_across_sizes(self):
        # a decline whose guards so far do not involve the batch: one trace,
        # one warning, every later size runs the ordinary host at once
        entry = ht.Entry(self._after_layer_norm)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for M in (8, 16, 5):
                x, w, b = self._inputs(M)
                self.assertEqual(entry(x, w, b), [self._after_layer_norm(x, w, b)])
        self.assertEqual((entry.traces, entry.ordinary, len(entry.variants)), (1, 3, 0))
        self.assertEqual((len(entry.declined), len(entry.declined_exact)), (1, 1))
        warned = self._host_trace_warnings(caught)
        self.assertEqual(len(warned), 1, warned)
        self.assertIn("aten.atan.default", warned[0])

    def test_a_size_dependent_decline_is_remembered_within_its_guards(self):
        # the decline is behind a branch on the batch: a call on the declining
        # side of that guard runs the ordinary host without another trace, a
        # call on the other side is traced and served by a variant
        def sized(t, w, b):
            if t.shape[0] > 4:
                return torch.atan(t)
            return layer_norm(t, [self.N], w, b, 1e-5)[0]

        entry = ht.Entry(sized)

        def call(M):
            x, w, b = self._inputs(M)
            self.assertEqual(entry(x, w, b), [sized(x, w, b)])
            return entry.traces, entry.ordinary, len(entry.variants)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(call(8), (1, 1, 0))  # traced; declines behind s > 4
            (p,) = entry.declined
            self.assertEqual(len(p.guards), 1)
            self.assertEqual(call(16), (1, 2, 0))  # in the class: no trace
            self.assertEqual(call(2), (2, 2, 1))  # outside it: traced, served
            self.assertEqual(call(3), (2, 2, 1))  # the variant serves
            self.assertEqual(call(32), (2, 3, 1))  # the class again
        self.assertEqual(len(self._host_trace_warnings(caught)), 1)

    def test_an_early_decline_is_remembered_by_its_exact_inputs(self):
        # a decline before the inputs are bound (an empty input) carries no
        # guards: the entry remembers exactly those inputs, and a call of any
        # other exact class is traced (and declines, and is warned) once more
        entry = ht.Entry(self._after_layer_norm)
        _, w, b = self._inputs(1)

        def call(x):
            self.assertEqual(entry(x, w, b), [self._after_layer_norm(x, w, b)])
            return (
                entry.traces,
                entry.ordinary,
                len(entry.declined),
                len(entry.declined_exact),
            )

        empty = torch.empty(0, self.N, device="cuda", dtype=torch.bfloat16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(call(empty), (1, 1, 0, 1))
            self.assertEqual(call(empty), (1, 2, 0, 1))
            self.assertEqual(call(torch.empty_like(empty)), (1, 3, 0, 1))
            self.assertEqual(call(empty.float()), (2, 4, 0, 2))
        warned = self._host_trace_warnings(caught)
        self.assertEqual(len(warned), 2, warned)
        self.assertTrue(all("empty" in m for m in warned), warned)

    def test_an_entry_serves_misses_by_new_variants_and_never_falls_back(self):
        # a miss is traced and served by a new variant; the outputs are
        # the replay's; the cap raises instead of running the ordinary host
        entry = ht.Entry(layer_norm, max_variants=1)
        x, w, b = self._inputs(8)
        args = self._args(x, w, b)
        self._check(entry(*args), x, w, b, 8)
        y, w2, b2 = self._inputs(3)
        self._check(entry(*self._args(y, w2, b2)), y, w2, b2, 3)
        self.assertEqual((entry.traces, entry.ordinary, len(entry.variants)), (1, 0, 1))
        with self.assertRaisesRegex(RuntimeError, "max_variants"):
            entry(*self._args(*self._inputs(8, offset=1)))
        self.assertEqual((entry.traces, entry.ordinary), (1, 0))

    def test_an_entry_takes_its_variants_from_its_builder(self):
        # the policy is the entry's and the variants its builder's: a backend
        # that matches by the batch, decides the offset only when called
        # (Miss) and serves eagerly is dispatched, traced and capped like the
        # interim replay, with no interim build
        class Served:
            def __init__(self, tape, args):
                self.key = (args[0].shape[0], args[0].storage_offset())
                self.calls = 0

            def matches(self, args):
                return args[0].shape[0] == self.key[0]

            def __call__(self, args):
                if args[0].storage_offset() != self.key[1]:
                    raise ht.Miss("the offset, decided by the call")
                self.calls += 1
                return list(layer_norm(*args))

        entry = ht.Entry(layer_norm, build_variant=Served, max_variants=3)
        for M, offset in ((8, 0), (8, 0), (3, 0), (8, 8), (8, 0)):
            args = self._args(*self._inputs(M, offset=offset))
            self.assertEqual(entry(*args), list(layer_norm(*args)))
        served = [(v.key, v.calls) for v in entry.variants]
        self.assertEqual(served, [((8, 0), 3), ((3, 0), 1), ((8, 8), 1)])
        self.assertEqual((entry.traces, entry.ordinary), (3, 0))
        with self.assertRaisesRegex(RuntimeError, "max_variants"):
            entry(*self._args(*self._inputs(5)))
        self.assertEqual((entry.traces, entry.ordinary), (3, 0))

    def test_an_entry_rebuilds_the_same_tape_on_a_topology_miss(self):
        # a TopologyMiss says the tape's guards held and only the call's
        # class (a closed region's node chain) is not the variant's: the
        # entry builds the same tape again at the call's inputs, without a
        # trace, and keeps the variant beside the first; a later variant of
        # that tape that holds the class serves before anything is built
        built = []

        class ByOffset:
            def __init__(self, tape, args):
                self.tape = tape
                self.key = (args[0].shape[0], args[0].storage_offset())
                self.calls = 0
                built.append(self)

            def matches(self, args):
                return args[0].shape[0] == self.key[0]

            def __call__(self, args):
                if args[0].storage_offset() != self.key[1]:
                    raise ht.TopologyMiss("another chain at this offset", self.tape)
                self.calls += 1
                return list(layer_norm(*args))

        entry = ht.Entry(layer_norm, build_variant=ByOffset, max_variants=3)
        for M, offset in ((8, 0), (8, 8), (8, 0), (8, 8), (3, 0)):
            args = self._args(*self._inputs(M, offset=offset))
            self.assertEqual(entry(*args), list(layer_norm(*args)))
        served = [(v.key, v.calls) for v in built]
        self.assertEqual(served, [((8, 0), 2), ((8, 8), 2), ((3, 0), 1)])
        self.assertIs(built[1].tape, built[0].tape)  # rebuilt, not re-traced
        self.assertIsNot(built[2].tape, built[0].tape)
        self.assertEqual((entry.traces, entry.ordinary), (2, 0))
        with self.assertRaisesRegex(RuntimeError, "max_variants"):
            entry(*self._args(*self._inputs(8, offset=16)))
        self.assertEqual((entry.traces, len(entry.variants)), (2, 3))


_VERBATIM_FORMALS_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/Recorder.h>
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

__global__ void write_with_scalar(P dynamic, int scalar) {
  observed_values[0] = scalar;
  observed_values[1] = dynamic.value;
}

at::Tensor verbatim_with_scalar(const at::Tensor& x) {
  // a bare scalar beside the proxy: outside the verbatim contract
  c10::cuda::CUDAGuard guard(x.device());
  ht::Traced<P> dynamic;
  dynamic.value = x.sym_size(0);
  ht::Grid grid(1);
  write_with_scalar<<<grid, 1, 0, c10::cuda::getCurrentCUDAStream()>>>(
      dynamic, static_cast<int>(x.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return x;
}

// the recorder's const accessor as a converted host reads it: the address a
// kernel image would carry for the tensor (a SymInt under a trace)
c10::SymInt address_of(const at::Tensor& x) {
  return ht::sym_const_data_ptr(x);
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
  m.def("verbatim_with_scalar", &verbatim_with_scalar);
  m.def("read_observed", &read_observed);
  m.def("address_of", &address_of);
}
"""


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceVerbatimFormals(TestCase):
    # the proxy-to-formal rule of a verbatim launch: a converted proxy is
    # passed by value exactly once and unchanged, so its formal is the one
    # size-compatible parameter; the bytes only verify that slot
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.extension = load_test_extension(
            "hosttrace_verbatim_formals", _VERBATIM_FORMALS_SOURCE
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

    def test_a_formal_beside_the_proxy_declines(self, device):
        # a bare scalar (or a by-value struct) beside the proxy would be a
        # constant of the variant read from the captured bytes; the verbatim
        # contract is one formal, the proxy
        with self.assertRaisesRegex(ht.Declined, "exactly one converted proxy struct"):
            ht.trace(
                self.extension.verbatim_with_scalar, (torch.empty(7, device=device),)
            )
        self.assertFalse(torch._C._host_trace_tracing())


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceBuildPairing(TestCase):
    # what the build checks when it pairs the tape's records with the
    # capture's nodes and allocations
    N = 1024

    def _layer_norm(self):
        w = torch.randn(self.N, device="cuda")
        b = torch.randn(self.N, device="cuda")

        def fn(x):
            return F.layer_norm(x, (self.N,), w, b, 1e-5)

        return fn, (torch.randn(8, self.N, device="cuda"),)

    def test_kernel_name_must_be_equal_not_a_substring(self):
        fn, args = self._layer_norm()
        tape = ht.trace(fn, args)
        ht.build(tape, fn, args)
        name = tape.launches[0]["kernel"]
        tape.launches[0]["kernel"] = name[:-1]
        with self.assertRaisesRegex(ht.TapeMismatch, "launch 0 is "):
            ht.build(tape, fn, args)
        tape.launches[0]["kernel"] = name
        ht.build(tape, fn, args)

    def test_identical_launches_with_different_tables_decline(self):
        # two launches the capture cannot tell apart at the build inputs pair
        # by the graph's order when it orders their nodes; incomparable nodes
        # (parallel branches) pair by creation order, exact only when the
        # symbolic tables agree
        fn, args = self._layer_norm()
        tape = ht.trace(fn, args)
        variant = ht.build(tape, fn, args)
        state = (tape.launches[0]["kernel"], b"image", (1, 1, 1), (1, 1, 1), 0)
        chain = [0, 1]  # node 1 depends on node 0
        parallel = [0, 0]
        seen: dict = {}
        variant._check_distinct(0, 0, state, seen, parallel)
        variant._check_distinct(0, 1, state, seen, parallel)  # the same table: fine
        table = seen[state][2]
        seen[state] = (0, 0, (table[0][1:], *table[1:]))
        variant._check_distinct(0, 1, state, seen, chain)  # ordered by the graph: fine
        with self.assertRaisesRegex(ht.TapeMismatch, "differ symbolically"):
            variant._check_distinct(0, 1, state, seen, parallel)

    def test_pairing_follows_the_graphs_edges_not_the_node_array(self):
        # the build pairs host order with the topological order of the
        # capture's edges: with the test hook reversing cudaGraphGetNodes'
        # array the pairing, the byte check and the replay are unchanged
        w = torch.randn(self.N, device="cuda")
        b = torch.randn(self.N, device="cuda")

        def fn(x, y):
            return (
                F.layer_norm(x, (self.N,), w, b, 1e-5),
                F.layer_norm(y, (self.N,), w, b, 1e-5),
            )

        args = (
            torch.randn(8, self.N, device="cuda"),
            torch.randn(4, self.N, device="cuda"),
        )
        tape = ht.trace(fn, args)
        self.assertEqual(len(tape.launches), 2)
        torch._C._host_trace_test_reverse_node_order(True)
        try:
            variant = ht.build(tape, fn, args)
        finally:
            torch._C._host_trace_test_reverse_node_order(False)
        new = (
            torch.randn(3, self.N, device="cuda"),
            torch.randn(5, self.N, device="cuda"),
        )
        got = variant.replay(new)
        for g, want in zip(got, fn(*new)):
            self.assertEqual(g, want, atol=0, rtol=0)

    def test_allocation_log_entries_are_checked_by_size_and_exhausted(self):
        fn, args = self._layer_norm()
        tape = ht.trace(fn, args)
        self.assertGreater(len(tape.allocs), 0)
        real = torch._C._host_trace_alloc_log_end

        def with_extra():
            return real() + [(256, 4096)]

        def resized():
            log = real()
            addr, nbytes = log[0]
            return [(addr, nbytes * 2)] + log[1:]

        with mock.patch.object(torch._C, "_host_trace_alloc_log_end", with_extra):
            with self.assertRaisesRegex(ht.TapeMismatch, "allocation"):
                ht.build(tape, fn, args)
        with mock.patch.object(torch._C, "_host_trace_alloc_log_end", resized):
            with self.assertRaisesRegex(
                ht.TapeMismatch, "bytes at the build, the tape says"
            ):
                ht.build(tape, fn, args)
        ht.build(tape, fn, args)


instantiate_device_type_tests(
    TestCudaHostTraceVerbatimFormals, globals(), only_for="cuda"
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@skipIfRocm(msg="host tracing is CUDA-only in this version")
class TestCudaHostTraceLayerNormBackward(TestCase):
    """The layer norm backward host, converted beside the forward: traced at
    one shape, replayed at others with new addresses, bitwise against eager on
    grad_input, grad_weight and grad_bias; the tape launches the two kernels
    eager launches; the tile-size and alignment choices are guards."""

    N = 4096

    def _inputs(self, M, device, dtype=torch.bfloat16, offset=0, N=None):
        N = N or self.N
        flat = torch.randn(2 * M * N + offset, device=device, dtype=dtype)
        x = flat[offset : offset + M * N].view(M, N)
        dy = flat[M * N + offset : 2 * M * N + offset].view(M, N)
        w = torch.randn(N, device=device, dtype=dtype)
        b = torch.randn(N, device=device, dtype=dtype)
        _, mean, rstd = layer_norm(x, [N], w, b, 1e-5)
        return dy, x, mean, rstd, w, b

    def _args(self, t, mask=(True, True, True), N=None):
        dy, x, mean, rstd, w, b = t
        return (dy, x, [N or self.N], mean, rstd, w, b, list(mask))

    def _bits(self, t):
        return t.contiguous().view(
            {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
                t.element_size()
            ]
        )

    def _assert_bitwise(self, got, want):
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            self.assertEqual(g.dtype, w.dtype)
            self.assertTrue(torch.equal(self._bits(g), self._bits(w)))

    def _roundtrip(self, fn, base, news):
        tape = ht.trace(fn, base)
        variant = ht.build(tape, fn, base)
        served = []
        for args in news:
            out = variant.try_replay(args)
            if out is None:
                served.append(False)
                continue
            want = fn(*args)
            torch.cuda.synchronize()
            self._assert_bitwise(out, want)
            served.append(True)
        return tape, served

    def _kernels(self, tape):
        return [torch._C._demangle(L["kernel"]) for L in tape.launches]

    def test_backward_replays_at_other_row_counts(self, device):
        # M = 100 traces the [64, 128) tile of the gamma / beta kernel (its
        # unaligned-grid instantiation: 100 is no multiple of the tile's 64
        # rows) and the vectorized grad-input kernel: the other row counts of
        # that tile serve with new addresses, bitwise on all three gradients;
        # M = 64 is the aligned instantiation, a named miss
        base = self._args(self._inputs(100, device))
        news = [self._args(self._inputs(M, device)) for M in (65, 127, 96, 100)]
        tape, served = self._roundtrip(layer_norm_backward, base, news)
        self.assertEqual(served, [True] * len(news))
        _, served = self._roundtrip(
            layer_norm_backward, base, [self._args(self._inputs(64, device))]
        )
        self.assertEqual(served, [False])
        self.assertEqual(tape.num_launches, 2)
        # the three gradients are allocations of the trace
        self.assertEqual(tape.num_allocations, 3)
        names = self._kernels(tape)
        self.assertIn("layer_norm_grad_input_kernel_vectorized", names[0])
        self.assertIn("GammaBetaBackwardCUDAKernelTemplate", names[1])
        # the grid of the grad-input kernel is the row count, a symbol
        launch = json.loads(tape.to_json())["launches"][0]
        self.assertNotIsInstance(launch["grid"][0], int)

    def test_tile_choices_are_guards(self, device):
        # the gamma / beta kernel's tile is chosen by M (< 64, < 128, < 256,
        # else) and its aligned-grid instantiation by M % rows and N % 32: a
        # replay in another class is a miss, never a wrong kernel
        for M0, serve, miss in (
            (100, (65, 127), (64, 32, 200)),
            (300, (257, 511, 1000), (256, 100)),
            (12, (4, 63), (16, 64)),
            (512, (256, 1024), (384,)),
        ):
            with self.subTest(M=M0):
                base = self._args(self._inputs(M0, device))
                news = [self._args(self._inputs(M, device)) for M in (*serve, *miss)]
                _, served = self._roundtrip(layer_norm_backward, base, news)
                self.assertEqual(served, [True] * len(serve) + [False] * len(miss))

    def test_alignment_selects_the_grad_input_kernel(self, device):
        # a misaligned input row takes the unvectorized grad-input kernel; the
        # choice is a guard on the address: another misaligned input serves,
        # an aligned one misses
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                base = self._args(self._inputs(64, device, dtype, offset=1))
                tape, served = self._roundtrip(
                    layer_norm_backward,
                    base,
                    [
                        self._args(self._inputs(64, device, dtype, offset=3)),
                        self._args(self._inputs(64, device, dtype)),
                    ],
                )
                self.assertEqual(served, [True, False])
                name = self._kernels(tape)[0]
                self.assertIn("layer_norm_grad_input_kernel", name)
                self.assertNotIn("vectorized", name)

    def test_partial_masks_and_dtypes(self, device):
        # grad_input_mask picks the kernels: only the grad-input kernel, only
        # the gamma / beta kernel, or both; every dtype the host dispatches. An
        # undefined gradient is not a tensor the trace can return, so the
        # traced call drops it
        def defined(*args):
            return [t for t in layer_norm_backward(*args) if t is not None]

        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            for mask, launches in (
                ((True, False, False), 1),
                ((False, True, True), 1),
                ((False, True, False), 1),
                ((True, True, True), 2),
            ):
                with self.subTest(dtype=dtype, mask=mask):
                    base = self._args(self._inputs(100, device, dtype), mask)
                    news = [self._args(self._inputs(80, device, dtype), mask)]
                    tape, served = self._roundtrip(defined, base, news)
                    self.assertEqual(served, [True])
                    self.assertEqual(tape.num_launches, launches)

    def test_huge_row_count_path(self, device):
        # M > 64K over a narrow N takes the M-parallel gamma / beta kernel
        # with a per-block partial buffer and a sum(0) through the reduction
        # sibling; traced and replayed like any other path
        N = 256
        base = self._args(self._inputs(70000, device, N=N), N=N)
        news = [self._args(self._inputs(66000, device, N=N), N=N)]
        tape, served = self._roundtrip(layer_norm_backward, base, news)
        self.assertEqual(served, [True])
        names = self._kernels(tape)
        self.assertGreaterEqual(len(names), 3)
        self.assertTrue(any("GammaBetaBackwardCUDAKernelTemplate" in n for n in names))

    def test_two_hints_trace_the_same_backward(self, device):
        two_hint.trace_twice(layer_norm_backward, self._args(self._inputs(64, device)))


instantiate_device_type_tests(
    TestCudaHostTraceLayerNormBackward, globals(), only_for="cuda"
)

if __name__ == "__main__":
    run_tests()
