# Owner(s): ["module: inductor"]
"""CuTe DSL kernels on the tape (torch/cuda/_host_trace_cute.py), lowered through the
runtime's CuTe binder into the shared native replay (hosttrace_cute._HostTraceCuTeModule):
a user kernel invoked from a traced function through the runtime's registered entry, every
launch kind of this tree in one call, the declines."""

import gc
import os
import sys
import unittest

import torch
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.dlpack import ReadOnlyTensorWrapper


try:
    from torch._inductor.runtime._cudagraph import _sdk

    _sdk.activate()
    import cutlass

    from cuda.bindings import driver
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

    HAS_CUTE = True
except Exception:  # the DSL is absent or another version
    HAS_CUTE = False


if HAS_CUTE:

    @cute.kernel
    def affine(source: cute.Tensor, destination: cute.Tensor, bias: cutlass.Int32):
        column, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        destination[row, column] = source[row, column] * 2.0 + bias

    @cute.jit
    def launch_affine(
        source: cute.Tensor,
        destination: cute.Tensor,
        bias: cutlass.Int32,
        stream: driver.CUstream,
    ):
        affine(source, destination, bias).launch(
            grid=(source.shape[0], 1, 1), block=(128, 1, 1), smem=0, stream=stream
        )

    @cute.jit
    def launch_dispatch(
        source: cute.Tensor,
        destination: cute.Tensor,
        bias: cutlass.Int32,
        stream: driver.CUstream,
    ):
        # a two-arm host dispatch on a dynamic size: one launch per arm (the arms
        # differ in their dynamic shared request; the kernel uses none)
        if source.shape[0] < 16:
            affine(source, destination, bias).launch(
                grid=(source.shape[0], 1, 1), block=(128, 1, 1), smem=0, stream=stream
            )
        else:
            affine(source, destination, bias).launch(
                grid=(source.shape[0], 1, 1), block=(128, 1, 1), smem=256, stream=stream
            )

    def convert_arguments(source, destination, bias):
        values = tuple(
            from_dlpack(t, assumed_align=16, use_32bit_stride=False)
            for t in (source, destination)
        )
        for value in values:
            value.mark_compact_shape_dynamic(0, stride_order=(0, 1), divisibility=1)
        return (*values, cutlass.Int32(bias))


def make_entry(host=None):
    """The runtime's registered observed entry over the kernel: what a traced function invokes."""
    from torch._inductor.runtime._cudagraph.api import (
        DirectCuTe,
        ObservedOrdinaryEntry,
        PythonEntry,
        SignaturePolicy,
    )

    owner = ObservedOrdinaryEntry(
        PythonEntry(launch_affine if host is None else host),
        affine,
        policy=SignaturePolicy(32, 64, 16, "stream"),
        conversion=convert_arguments,
    )
    return DirectCuTe(owner), owner


CUTE = None  # the DirectCuTe of the running test
EXT = None

try:
    import triton
    import triton.language as tl

    @triton.jit
    def scale_add(x_ptr, y_ptr, n, shift, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, x * 2.0 + shift, mask=mask)

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def cute_host(x):
    # an ATen host, the CuTe kernel (its bias the row count: a value of the tape), an ATen host
    h = torch.nn.functional.silu(x)
    y = torch.empty_like(h)
    CUTE(h, y, h.shape[0])
    return torch.add(y, x)


def data_host(x):
    y = torch.empty_like(x)
    CUTE(x, y, int(x[0, 0].item()))
    return y


def plain_host(x):
    # the DSL launched as written, outside a registered entry
    y = torch.empty_like(x)
    source, destination = (from_dlpack(t, assumed_align=16) for t in (x, y))
    launch_affine(
        source,
        destination,
        cutlass.Int32(3),
        driver.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return y


# a user CUDA C++ kernel through the typed launch helper (the host-tracing bar:
# SymInt on sizes, the launch as written)
_SCALE_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/host_trace/Launch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
namespace ht = at::cuda::host_trace;
struct S { const float* x; float* out; long long n; };
namespace at::cuda::host_trace {
template <> struct Traced<S> : TracedBase {
  S pod{};
  PtrField<0> x{this, "x"};
  PtrField<8> out{this, "out"};
  IntField<long long, 16> n{this, "n"};
  Traced() : TracedBase(&pod, sizeof(S)) {}
  operator S&() { on_convert(); return pod; }
};
}  // namespace at::cuda::host_trace
__global__ void scale_two_plus_one(S p) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < p.n) p.out[i] = p.x[i] * 2.0f + 1.0f;
}
at::Tensor scale_two_plus_one_(const at::Tensor& x) {
  c10::cuda::CUDAGuard g(x.device());
  at::Tensor out = at::empty_like(x);
  ht::Traced<S> params;
  params.x = ht::sym_const_data_ptr(x);
  params.out = ht::sym_mutable_data_ptr(out);
  params.n = x.sym_numel();
  ht::Grid grid((x.sym_numel() + 255) / 256);
  ht::launch(scale_two_plus_one, grid, 256, 0, c10::cuda::getCurrentCUDAStream(), params);
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scale_two_plus_one_", &scale_two_plus_one_); }
"""


def five_kind_host(x, w):
    # every launch kind in one call: an ATen converted host, a cuBLAS region, a user CUDA
    # C++ kernel (typed launch helper), the CuTe kernel through the registered entry, a
    # user Triton kernel (torch/cuda/_host_trace_triton.py), an ATen host
    import triton

    h = torch.nn.functional.silu(x)
    g = torch.mm(h, w)
    t = EXT.scale_two_plus_one_(g)
    y = torch.empty_like(t)
    CUTE(t, y, t.shape[0])
    z = torch.empty_like(y)
    n = y.numel()
    scale_add[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](y, z, n, 3, BLOCK=1024)
    return torch.add(z, g)


def mixed_host(x, w):
    # one call, every launch kind of this tree: an ATen converted host, a cuBLAS
    # region, a user CUDA C++ kernel (typed launch helper), the CuTe kernel, an ATen host
    h = torch.nn.functional.silu(x)
    g = torch.mm(h, w)
    t = EXT.scale_two_plus_one_(g)
    y = torch.empty_like(t)
    CUTE(t, y, t.shape[0])
    return torch.add(y, g)


def _kernel_nodes(graph):
    """(kernel name, function handle) per kernel node of a raw CUDA graph."""
    _, count = _check_cuda_bindings(driver.cuGraphGetNodes(graph, 0))
    nodes, _ = _check_cuda_bindings(driver.cuGraphGetNodes(graph, count))
    out = []
    for node in nodes:
        kind = _check_cuda_bindings(driver.cuGraphNodeGetType(node))
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            out.append((kind.name, 0))
            continue
        params = _check_cuda_bindings(driver.cuGraphKernelNodeGetParams(node))
        handle = int(params.kern) or int(params.func)
        name = _check_cuda_bindings(
            driver.cuKernelGetName(params.kern)
            if int(params.kern)
            else driver.cuFuncGetName(params.func)
        )
        out.append((name.decode() if isinstance(name, bytes) else str(name), handle))
    return out


@unittest.skipUnless(
    torch.cuda.is_available() and HAS_CUTE and torch.version.hip is None,
    "CUDA and the CuTe DSL required",
)
class TestHostTraceCuTe(TestCase):
    def setUp(self):
        super().setUp()
        from torch._inductor.runtime._cudagraph import direct_hosttrace
        from torch.cuda import _host_trace

        self.module, self.ht = direct_hosttrace, _host_trace
        global CUTE
        CUTE, self.owner = make_entry()
        self.addCleanup(self.owner.close)

    def _replay(self, fn, args):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args)
        self.addCleanup(replay.close)
        return replay

    def test_user_kernel_replays_natively_with_a_rebind(self):
        x = torch.randn(8, 128, device="cuda")
        replay = self._replay(cute_host, (x,))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (3, 0))
        silu, launch, add = tape.launches
        self.assertLess(silu["seq"], launch["seq"])
        self.assertLess(launch["seq"], add["seq"])
        self.assertIn("affine", launch["kernel"])
        record = launch["cute"]
        self.assertIs(record.invocation.adapter, CUTE)
        self.assertEqual(
            (record.invocation.arm, len(record.invocation.sites)), (None, 1)
        )
        # pointer operands as roots of the tape (written), integers as values
        params = {p["name"]: p for p in launch["params"]}
        self.assertEqual(params["source.data_ptr"]["kind"], "ptr")
        self.assertEqual(params["destination.data_ptr"]["access"], "rw")
        self.assertIsInstance(params["bias"]["value"], torch.SymInt)
        self.assertIsInstance(params["source.shape[0]"]["value"], torch.SymInt)
        # the column count the conversion marked static is baked into the kernel: no
        # field of the launch, a guard of the tape (its symbol pinned to 128)
        self.assertNotIn("source.shape[1]", params)
        self.assertTrue(
            any(
                str(g).startswith("Eq(s") and str(g).endswith(", 128)")
                for g in tape.guards
            ),
            tape.guards,
        )
        self.assertEqual(list(launch["grid"]), [8, 1, 1])
        self.assertEqual(tuple(launch["block"]), (128, 1, 1))
        # the grid is the row count: a symbolic launch bound of the lowering
        (call,) = [
            c
            for c in replay.lowered.calls
            if type(c.module).__name__ == "_HostTraceCuTeModule"
        ]
        self.assertEqual(call.module.block, (128, 1, 1))
        graph, _ = replay.variants[0].lowered.capture_handles
        names = [name for name, _ in _kernel_nodes(graph)]
        self.assertEqual(len(names), 3)
        self.assertIn("affine", names[1])
        self.assertEqual(names[1], launch["kernel"])
        for rows in (5, 7, 35, 96, 8):
            other = torch.randn(rows, 128, device="cuda")
            self.assertEqual(replay(other), cute_host(other), atol=0, rtol=0)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.declines, replay.ordinary),
            (0, 1, [], 0),
        )
        held = replay(x)
        expected = cute_host(x)
        replay.close()
        # the receipt closed with the replay: the ordinary owner is free to close
        self.assertEqual(held, expected, atol=0, rtol=0)
        self.owner.close()

    def test_a_two_arm_host_dispatch_is_a_guard(self):
        # the host's dispatch predicate selects the launch site at the traced values
        # and guards the tape: the other arm misses, re-traces and serves its own
        # variant; both variants stay
        global CUTE
        CUTE, owner = make_entry(launch_dispatch)
        self.addCleanup(owner.close)
        x = torch.randn(8, 128, device="cuda")
        replay = self._replay(cute_host, (x,))
        record = replay.tape.launches[1]["cute"]
        self.assertEqual(
            (record.invocation.arm, len(record.invocation.artifact.sites)), (True, 2)
        )
        self.assertEqual(replay.tape.launches[1]["smem"], 0)
        for rows in (5, 15):
            other = torch.randn(rows, 128, device="cuda")
            self.assertEqual(replay(other), cute_host(other), atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (0, 1))
        other = torch.randn(35, 128, device="cuda")
        self.assertEqual(replay(other), cute_host(other), atol=0, rtol=0)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.declines), (1, 2, [])
        )
        second = replay.variants[1].tape.launches[1]["cute"]
        self.assertEqual(
            (second.invocation.arm, replay.variants[1].tape.launches[1]["smem"]),
            (False, 256),
        )
        for rows in (16, 7, 96):
            other = torch.randn(rows, 128, device="cuda")
            self.assertEqual(replay(other), cute_host(other), atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (1, 2))

    def test_a_view_in_the_conversion_and_an_offset_input(self):
        # the user's conversion may view its operands; an input at a storage offset
        # keeps its offset in the pointer (the tape's root plus the view's bytes)
        storage = torch.randn(8 * 128 + 32, device="cuda")
        x = storage[32:].view(8, 128)
        replay = self._replay(cute_host, (x,))
        self.assertEqual(replay.tape.num_launches, 3)
        for rows, offset in ((5, 16), (35, 0), (8, 48)):
            other = torch.randn(rows * 128 + offset, device="cuda")[offset:].view(
                rows, 128
            )
            self.assertEqual(replay(other), cute_host(other), atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (0, 1))

    def test_declines_by_name(self):
        x = torch.randn(8, 128, device="cuda")
        # a Python selection before the invocation that reads tensor data
        with self.assertRaisesRegex(self.ht.Declined, "_local_scalar_dense"):
            self.ht.trace(data_host, (x,))
        # the DSL launched outside a registered entry: its from_dlpack reads a traced tensor
        self.assertEqual(plain_host(x), x * 2.0 + 3, atol=0, rtol=0)
        with self.assertRaisesRegex(
            self.ht.Declined,
            "CuTe DSL program the recorder does not hook.*not recorded on the tape",
        ):
            self.ht.trace(plain_host, (x,))
        # an entry that has not run once (a trace without a warm-up on a fresh entry)
        global CUTE
        fresh, owner = make_entry()
        self.addCleanup(owner.close)
        CUTE = fresh
        with self.assertRaisesRegex(
            self.ht.Declined,
            "CuTe DSL kernel launch_affine: the entry has not run once",
        ):
            self.ht.trace(cute_host, (x,), warm_up=False)
        # a declined trace serves the ordinary host, by class
        replay = self._replay(plain_host, (x,))
        self.assertEqual(replay(x), plain_host(x), atol=0, rtol=0)
        self.assertEqual(len(replay.declines), 1)

    def test_a_direct_triton_proxy_declines_by_name(self):
        from torch.testing._internal.inductor_utils import HAS_TRITON

        if not HAS_TRITON:
            self.skipTest("Triton required")
        import triton
        import triton.language as tl

        from torch._inductor.runtime._cudagraph.api import DirectTriton

        @triton.jit
        def add_one(source, destination, count, BLOCK: tl.constexpr):
            index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            value = tl.load(source + index, index < count, other=0)
            tl.store(destination + index, value + 1, index < count)

        add = DirectTriton(add_one)
        self.addCleanup(add.close)

        def proxy_host(x):
            y = torch.empty_like(x)
            count = x.numel()
            add[lambda meta: (triton.cdiv(count, meta["BLOCK"]),)](
                x, y, count, BLOCK=128
            )
            return y

        x = torch.randn(1024, device="cuda")
        with self.assertRaisesRegex(
            self.ht.Declined,
            "Triton kernel add_one: launched through a DirectTriton proxy",
        ):
            self.ht.trace(proxy_host, (x,))

    def test_the_hook_leaves_the_runtime_as_it_found_it(self):
        from torch._inductor.runtime._cudagraph.direct_cute import DirectCuTe
        from torch._inductor.runtime._cudagraph.direct_invocation import ACTIVE

        invoke = DirectCuTe.invoke
        x = torch.randn(8, 128, device="cuda")
        tape = self.ht.trace(cute_host, (x,))
        self.assertEqual(tape.num_launches, 3)
        self.assertIsNone(ACTIVE.get())
        self.assertIs(DirectCuTe.invoke, invoke)
        # the runtime's own symbolic view of the same entry still constructs
        from torch._inductor.runtime._cudagraph.cute_adapter import make_cute_trace_view
        from torch._subclasses.fake_tensor import FakeTensorMode

        view = make_cute_trace_view(CUTE, lambda event: None, FakeTensorMode())
        view.check()

    def test_one_tape_every_launch_kind_replays_as_one_graph(self):
        # ATen converted hosts, a cuBLAS region, a user CUDA C++ kernel, the CuTe kernel and
        # a user Triton kernel in one call: one tape in host order, one native graph, a
        # shape change served by rebinds (the region by a harvest of its new key), no re-trace
        if not HAS_TRITON:
            self.skipTest("Triton required")
        global EXT
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        from host_trace_testing import load_test_extension

        EXT = load_test_extension("ht_cute_capture_scale", _SCALE_SOURCE)
        w = torch.randn(256, 128, device="cuda")
        x = torch.randn(64, 256, device="cuda")
        replay = self._replay(five_kind_host, (x, w))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (5, 1))
        names = [L["kernel"] for L in tape.launches]
        self.assertIn("silu", names[0].lower())
        self.assertIn("scale_two_plus_one", names[1])
        self.assertIn("affine", names[2])
        self.assertEqual(names[3], "scale_add")
        self.assertIn("add", names[4].lower())
        seqs = [L["seq"] for L in tape.launches]
        self.assertEqual(seqs, sorted(seqs))
        self.assertLess(seqs[0], tape.regions[0].seq)
        self.assertLess(tape.regions[0].seq, seqs[1])
        kinds = [type(call.module).__name__ for call in replay.lowered.calls]
        self.assertEqual(
            kinds,
            [
                "_HostTraceKernelModule",
                "_HostTraceKernelModule",
                "_HostTraceCuTeModule",
                "_HostTraceTritonModule",
                "_HostTraceKernelModule",
            ],
        )
        graph, _ = replay.variants[0].lowered.capture_handles
        graph_names = [name for name, _ in _kernel_nodes(graph)]
        for name in names[1:4]:
            self.assertIn(name, graph_names)
        for rows in (48, 7, 64, 33):
            other = torch.randn(rows, 256, device="cuda")
            self.assertEqual(replay(other, w), five_kind_host(other, w), atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines, replay.ordinary),
            (1, 1, [], 0),
        )

    def test_one_tape_every_launch_kind_of_this_tree_replays_as_one_graph(self):
        # ATen converted hosts, a cuBLAS region, a user CUDA C++ kernel and the CuTe
        # kernel in one call: one tape in host order, one native graph, a shape change
        # served by rebinds (the region by a harvest of its new key), no re-trace
        global EXT
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        from host_trace_testing import load_test_extension

        EXT = load_test_extension("ht_cute_capture_scale", _SCALE_SOURCE)
        w = torch.randn(256, 128, device="cuda")
        x = torch.randn(64, 256, device="cuda")
        replay = self._replay(mixed_host, (x, w))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (4, 1))
        names = [L["kernel"] for L in tape.launches]
        self.assertIn("silu", names[0].lower())
        self.assertIn("scale_two_plus_one", names[1])
        self.assertIn("affine", names[2])
        self.assertEqual(tape.regions[0].op, "mm")
        self.assertLess(tape.launches[0]["seq"], tape.regions[0].seq)
        self.assertLess(tape.regions[0].seq, tape.launches[1]["seq"])
        self.assertLess(tape.launches[1]["seq"], tape.launches[2]["seq"])
        self.assertLess(tape.launches[2]["seq"], tape.launches[3]["seq"])
        kinds = [type(call.module).__name__ for call in replay.lowered.calls]
        self.assertEqual(kinds.count("_HostTraceCuTeModule"), 1)
        self.assertEqual(kinds.count("_HostTraceKernelModule"), 3)
        graph, _ = replay.variants[0].lowered.capture_handles
        graph_names = [name for name, _ in _kernel_nodes(graph)]
        self.assertIn(names[2], graph_names)
        for rows in (48, 7, 64):
            other = torch.randn(rows, 256, device="cuda")
            self.assertEqual(replay(other, w), mixed_host(other, w), atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines, replay.ordinary),
            (1, 1, [], 0),
        )


if HAS_CUTE:

    def compile_affine(rows_dynamic=True):
        """The affine host compiled against fake tensors (dynamic rows, 128 static
        columns) and a fake stream: what a user calls directly, no registered entry."""
        fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(), 128), stride=(128, 1), assumed_align=16
        )
        other = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(), 128), stride=(128, 1), assumed_align=16
        )
        return cute.compile(
            launch_affine,
            fake,
            other,
            cutlass.Int32(0),
            cute.runtime.make_fake_stream(),
        )

    def marked(t, read_only=False):
        source = ReadOnlyTensorWrapper(t) if read_only else t
        value = from_dlpack(source, assumed_align=16, use_32bit_stride=False)
        return value.mark_compact_shape_dynamic(0, stride_order=(0, 1), divisibility=1)


if HAS_CUTE:

    @cute.kernel
    def scale(source: cute.Tensor, destination: cute.Tensor, factor: cutlass.Float32):
        column, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        destination[row, column] = source[row, column] * factor

    @cute.jit
    def launch_scale(
        source: cute.Tensor,
        destination: cute.Tensor,
        factor: cutlass.Float32,
        stream: driver.CUstream,
    ):
        scale(source, destination, factor).launch(
            grid=(source.shape[0], 1, 1), block=(128, 1, 1), smem=0, stream=stream
        )

    def compile_scale():
        fake = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(), 128), stride=(128, 1), assumed_align=16
        )
        other = cute.runtime.make_fake_tensor(
            cutlass.Float32, (cute.sym_int(), 128), stride=(128, 1), assumed_align=16
        )
        return cute.compile(
            launch_scale,
            fake,
            other,
            cutlass.Float32(0.0),
            cute.runtime.make_fake_stream(),
        )


COMPILED = None  # the compiled affine program of the running test
SCALE = None  # the compiled scale program (a float formal)
FACTOR = {"value": 2.0}


def program_host(x):
    # the compiled program called directly on the input (read-only) between two
    # ATen hosts, its bias the row count: a value of the tape
    h = torch.nn.functional.silu(x)
    y = torch.empty_like(h)
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    COMPILED(marked(x, read_only=True), marked(y), x.shape[0], stream)
    return y + h


def constant_host(x):
    # the float factor is a constant of the call, baked into the synthesized entry
    h = torch.nn.functional.silu(x)
    y = torch.empty_like(h)
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    SCALE(marked(h), marked(y), FACTOR["value"], stream)
    return y


def rms_norm_host(x, w):
    return torch.nn.functional.rms_norm(x, (2048,), w, eps=1e-6)


def rms_norm_sized_host(x, w):
    # a normalized shape written from a traced size: ATen's rms_norm composite
    # reinterprets it as ints unchecked (layer_norm.cpp rms_norm_symint), so
    # _fused_rms_norm receives the size's pointer bits
    return torch.nn.functional.rms_norm(x, (w.shape[0],), w, eps=1e-6)


def rms_norm_half_host(x, w):
    # a non-contiguous input: the override copies it before its kernel
    return torch.nn.functional.rms_norm(x[:, :1024], (1024,), w[:1024], eps=1e-6)


def rms_norm_backward_host(g, x, rstd, w):
    return torch.ops.aten._fused_rms_norm_backward(g, x, [2048], rstd, w, [True, True])


def topk_host(x):
    # fp32, k = 16, N = 1024: torch._native's register kernel on this box
    return torch.topk(x, 16)


def scatter_host(dst, idx, src):
    return dst.scatter_add(0, idx, src)


def scatter_inplace_host(dst, idx, src):
    return dst.scatter_add_(0, idx, src)


def scatter_args(rows):
    # distinct target rows: two sources into one row would add in atomic order,
    # which is not reproducible in eager either
    dst = torch.zeros(256, 1024, device="cuda")
    idx = torch.randperm(256, device="cuda")[:rows].view(rows, 1).expand(rows, 1024)
    return dst, idx, torch.randn(rows, 1024, device="cuda")


class RMSBlock(torch.nn.Module):
    """A torch-native block: RMSNorm (eager's CuTe override on this box), a linear
    (a cuBLAS region), an activation and the residual."""

    def __init__(self, width):
        super().__init__()
        self.norm = torch.nn.RMSNorm(width, eps=1e-6)
        self.proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, x):
        return x + torch.nn.functional.silu(self.proj(self.norm(x)))


@unittest.skipUnless(
    torch.cuda.is_available() and HAS_CUTE and torch.version.hip is None,
    "CUDA and the CuTe DSL required",
)
class TestHostTraceCuTeDSL(TestCase):
    """CuTe DSL programs at the DSL's launch level (torch/cuda/_host_trace_cute_dsl.py):
    a compiled program called directly, registered entry or not, and torch._native's
    CuTe overrides."""

    def setUp(self):
        super().setUp()
        from torch._inductor.runtime._cudagraph import direct_hosttrace
        from torch.cuda import _host_trace, _host_trace_cute_desc, _host_trace_cute_dsl

        self.module, self.ht, self.dsl, self.desc = (
            direct_hosttrace,
            _host_trace,
            _host_trace_cute_dsl,
            _host_trace_cute_desc,
        )
        global COMPILED, SCALE
        COMPILED = compile_affine()
        SCALE = compile_scale()

    def _replay(self, fn, args):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args)
        self.addCleanup(replay.close)
        return replay

    def test_a_compiled_program_called_directly_is_recorded(self):
        x = torch.randn(8, 128, device="cuda")
        replay = self._replay(program_host, (x,))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (3, 0))
        silu, launch, add = tape.launches
        self.assertLess(silu["seq"], launch["seq"])
        self.assertLess(launch["seq"], add["seq"])
        self.assertIn("affine", launch["kernel"])
        record = launch["cute"]
        self.assertEqual(record.invocation.read_only, frozenset({"source"}))
        access = {
            p["name"]: p["access"] for p in launch["params"] if p["kind"] == "ptr"
        }
        self.assertEqual(access, {"source.data_ptr": "r", "destination.data_ptr": "rw"})
        names = [p["name"] for p in launch["params"]]
        self.assertIn("source.shape[0]", names)
        self.assertIn("bias", names)
        self.assertEqual(launch["grid"], [8, 1, 1])
        # the read-only input's root is not written; the destination's is
        source_root = record.invocation.origins[id(record.invocation.operands[0])][0]
        destination_root = record.invocation.origins[id(record.invocation.operands[1])][
            0
        ]
        self.assertEqual(source_root.name, "p0")
        self.assertNotIn("p0", tape.written_roots)
        self.assertIn(destination_root.name, tape.written_roots)
        for rows in (8, 5, 33):
            other = torch.randn(rows, 128, device="cuda")
            self.assertEqual(replay(other), program_host(other), atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (0, 1))

    def test_the_program_runs_as_compiled_outside_a_trace(self):
        x = torch.randn(4, 128, device="cuda")
        before = program_host(x)
        self._replay(program_host, (x,))
        self.assertIsNone(getattr(self.dsl._state, "phase", None))
        self.assertEqual(program_host(x), before, atol=0, rtol=0)

    def test_a_compile_the_recorder_did_not_observe_declines_by_name(self):
        x = torch.randn(4, 128, device="cuda")
        self.dsl._compiles.pop(COMPILED, None)
        with self.assertRaisesRegex(
            self.ht.Declined, "its compile was not observed by the recorder"
        ):
            self.ht.trace(program_host, (x,))

    def test_a_float_formal_is_a_constant_of_the_call(self):
        x = torch.randn(4, 128, device="cuda")
        FACTOR["value"] = 2.0
        replay = self._replay(constant_host, (x,))
        launch = next(L for L in replay.tape.launches if L.get("cute"))
        self.assertIn("scale", launch["kernel"])
        other = torch.randn(6, 128, device="cuda")
        self.assertEqual(replay(other), constant_host(other), atol=0, rtol=0)
        FACTOR["value"] = 3.0
        with self.assertRaisesRegex(
            self.ht.Declined, "the entry was synthesized with 2.0"
        ):
            self.ht.trace(constant_host, (x,))

    def _quack_in_process(self):
        # QuACK's on-disk cache serves a loaded module (no jit callable to re-select);
        # the in-process compile is the recordable form
        from torch._vendor.quack import cache

        previous = cache.CACHE_ENABLED
        cache.CACHE_ENABLED = False
        self.addCleanup(setattr, cache, "CACHE_ENABLED", previous)

    def _eager_nodes(self, fn, args):
        # the kernel nodes of eager's own capture of the call
        g = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(g):
            fn(*args)
        return torch._C._host_trace_harvest_nodes(g.raw_cuda_graph())

    def _templates(self, op):
        return [t for t in self.ht._gemm_templates.values() if t.key[2] == op]

    def _descriptor_launches(self, tape):
        # the launch records made from a program's persisted descriptor
        # (torch/cuda/_host_trace_cute_desc.py): eager's own kernel handle
        return [L for L in tape.launches if L.get("cute_desc") is not None]

    def _cache_dir(self):
        import tempfile

        from torch._vendor.quack import cache

        tmp = tempfile.mkdtemp(prefix="host_trace_cute_desc_")
        scope = cache.cache_dir_override(tmp)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        return tmp

    def test_f_rms_norm_is_recorded_from_its_descriptor_with_eager_s_kernel(self):
        # eager's route on this box is torch._native's QuACK override, whose program
        # QuACK's jit_cache loads from its on-disk object (the default): the call is
        # one launch record of the tape, made from the descriptor persisted beside
        # the object and the call's arguments, with the function handle an eager
        # capture's node holds (E36); the grid is symbolic, so another row count is
        # a rebind of the one variant, not a re-trace and not a harvest
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        (eager,) = self._eager_nodes(rms_norm_host, (x, w))
        self.assertIn("quackrmsnormRMSNorm", eager["name"])
        replay = self._replay(rms_norm_host, (x, w))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (1, 0))
        (record,) = self._descriptor_launches(tape)
        self.assertEqual(record["func"], eager["func"])
        self.assertEqual(record["kernel"], eager["name"])
        self.assertEqual(record["param_layout"], [tuple(x) for x in eager["layout"]])
        self.assertEqual(
            [(p["name"], p["kind"], p["access"]) for p in record["params"]],
            [
                ("mX.data_ptr", "ptr", "r"),
                ("mX.shape[0]", "i32", ""),
                ("mX.stride[0]", "i64", ""),
                ("mW.data_ptr", "ptr", "r"),
                ("mO.data_ptr", "ptr", "rw"),
                ("mO.shape[0]", "i32", ""),
                ("mO.stride[0]", "i64", ""),
                ("mRstd.data_ptr", "ptr", "rw"),
                ("mRstd.shape[0]", "i32", ""),
                ("eps", "f32", ""),
            ],
        )
        self.assertIsInstance(record["grid"][0], torch.SymInt)
        self.assertEqual(sorted(tape.written_roots), ["a0", "a1"])
        self.assertEqual(self._templates("_fused_rms_norm"), [])
        for rows in (4, 7, 64):
            other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(replay(other, w), rms_norm_host(other, w), atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines, replay.ordinary),
            (1, 1, [], 0),
        )
        # the descriptor's guards: the static N and the assumed alignment
        guards = [str(g) for g in tape.guards]
        self.assertTrue(
            any(g == "Eq(s75, 2048)" or g.endswith(", 2048)") for g in guards), guards
        )
        self.assertTrue(
            any("PythonMod(" in g and ", 16), 0)" in g for g in guards), guards
        )

    def test_the_program_compiled_in_process_records_the_same_kernel(self):
        # the cold cache: QuACK compiles in-process, the recorder's compile hook
        # builds the descriptor at the DSL's finalize, and the record holds the
        # in-process program's own function handle (eager's capture holds it too)
        self._quack_in_process()
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        (eager,) = self._eager_nodes(rms_norm_host, (x, w))
        replay = self._replay(rms_norm_host, (x, w))
        (record,) = self._descriptor_launches(replay.tape)
        self.assertEqual(
            (record["func"], record["kernel"]), (eager["func"], eager["name"])
        )
        for rows in (4, 9):
            other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(replay(other, w), rms_norm_host(other, w), atol=0, rtol=0)
        self.assertEqual((replay.traces, len(replay.variants)), (1, 1))

    def test_the_descriptor_is_persisted_beside_the_object_and_an_object_without_one_is_a_miss(
        self,
    ):
        # QuACK's jit_cache writes the descriptor next to the .o; an object without
        # one (an older cache, an async worker's export) is a cache miss: the key is
        # recompiled once in-process, eager's own cold-cache behaviour, and both
        # files are written back (A411: never a stand-in capture)
        import glob
        import os

        from torch._native.ops.norm import norms
        from torch._vendor.quack.cache import jit as qjit

        tmp = self._cache_dir()
        compile_fn = norms._instrumented_rmsnorm_fwd()
        compile_fn.cache_clear()
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        rms_norm_host(x, w)
        (obj,) = glob.glob(os.path.join(tmp, "*", "*.o"))
        desc = obj[: -len(".o")] + qjit.DESCRIPTOR_SUFFIX
        self.assertTrue(os.path.exists(desc), desc)
        descriptor = self.desc.Descriptor.from_json(open(desc).read())
        self.assertIsNone(descriptor.declined)
        self.assertEqual(len(descriptor.launches), 1)
        self.assertEqual(descriptor.recipe["options"], "--enable-tvm-ffi")
        self.assertEqual(
            [f.name for f in descriptor.formals if f.kind == "tensor"],
            ["mX", "mW", "mO", "mRstd"],
        )
        self.assertEqual(descriptor.formals[0].shape[1], 2048)
        self.assertEqual(
            descriptor.symbols[descriptor.formals[0].stride[0]].divisibility, 8
        )
        # a warm load returns the program with its descriptor
        compile_fn.cache_clear()
        program = compile_fn(*self._rms_key(), per_head=False)
        self.assertIs(type(program), self.dsl.LoadedProgram)
        self.assertEqual(compile_fn.cache_info().hits, 1)
        # the object without its descriptor: a miss, recompiled, written back
        os.rename(desc, desc + ".moved")
        compile_fn.cache_clear()
        program = compile_fn(*self._rms_key(), per_head=False)
        self.assertEqual(compile_fn.cache_info().misses, 1)
        self.assertTrue(os.path.exists(desc))
        self.assertIsNot(type(program), self.dsl.LoadedProgram)
        # and the traced call is served either way, with the same function handle
        (eager,) = self._eager_nodes(rms_norm_host, (x, w))
        replay = self._replay(rms_norm_host, (x, w))
        (record,) = self._descriptor_launches(replay.tape)
        self.assertEqual(record["func"], eager["func"])
        self.assertEqual(replay(x, w), rms_norm_host(x, w), atol=0, rtol=0)

    def _rms_key(self):
        from torch._vendor.quack.cute_dsl_utils import torch2cute_dtype_map as m

        # the override's own call of the compile function (norms.quack_rmsnorm_fwd)
        bf16 = m[torch.bfloat16]
        return (bf16, bf16, None, bf16, None, None, 2048, True, False, False)

    def test_another_normalized_size_misses_on_the_descriptor_s_guard(self):
        # N is static in QuACK's compile (one program per N): a call at another N
        # misses the variant's guard and is traced anew through its own program
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        replay = self._replay(rms_norm_host, (x, w))
        self.assertEqual(replay(x, w), rms_norm_host(x, w), atol=0, rtol=0)

        def rms_1024(a, b):
            return torch.nn.functional.rms_norm(a, (1024,), b, eps=1e-6)

        y = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
        other = self._replay(rms_1024, (y, v))
        (record,) = self._descriptor_launches(other.tape)
        (eager,) = self._eager_nodes(rms_1024, (y, v))
        self.assertEqual(record["func"], eager["func"])
        self.assertEqual(other(y, v), rms_1024(y, v), atol=0, rtol=0)

    def test_a_normalized_shape_from_a_traced_size_declines_by_name(self):
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            self.ht.Declined, "a traced size's pointer bits.*write it as ints"
        ):
            self.ht.trace(rms_norm_sized_host, (x, w))

    def test_an_input_the_override_copies_first_is_eager_s_copy_and_the_kernel(self):
        # a non-contiguous input: the override copies it before its kernel, and
        # under the trace that copy is ATen's converted host, a launch of the tape
        # ahead of the kernel's record (the closed region declined this by name)
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        replay = self._replay(rms_norm_half_host, (x, w))
        tape = replay.tape
        self.assertEqual(tape.num_regions, 0)
        self.assertEqual(len(self._descriptor_launches(tape)), 1)
        self.assertGreaterEqual(tape.num_launches, 2)
        self.assertTrue(tape.launches[-1].get("cute_desc"))
        for rows in (4, 6):
            other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(
                replay(other, w), rms_norm_half_host(other, w), atol=0, rtol=0
            )

    def test_with_the_override_off_rms_norm_is_aten_s_launches(self):
        # eager's route with torch._native's override deregistered is ATen's converted
        # host: launches on the tape, no region, a shape change a rebind
        from torch._native.registry import (
            deregister_op_overrides,
            reenable_op_overrides,
        )

        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        deregister_op_overrides(disable_op_symbols="_fused_rms_norm")
        try:
            names = [n["name"] for n in self._eager_nodes(rms_norm_host, (x, w))]
            self.assertFalse(any("quack" in n for n in names), names)
            replay = self._replay(rms_norm_host, (x, w))
            tape = replay.tape
            self.assertEqual(tape.num_regions, 0)
            self.assertEqual(tape.num_launches, len(names))
            for rows in (4, 7):
                other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
                self.assertEqual(
                    replay(other, w), rms_norm_host(other, w), atol=0, rtol=0
                )
            self.assertEqual(
                (replay.traces, len(replay.variants), replay.declines), (1, 1, [])
            )
            replay.close()
        finally:
            reenable_op_overrides(enable_op_symbols="_fused_rms_norm")

    def test_a_block_with_rms_norm_is_one_tape(self):
        # the parameters as inputs (functional_call), as the recorder's regions take
        # them: the RMSNorm a launch record of eager's CuTe kernel (from its
        # descriptor; QuACK compiled in-process here, the cold cache), the linear a
        # cuBLAS region, the rest converted hosts, one tape
        self._quack_in_process()
        block = RMSBlock(2048).cuda().to(torch.bfloat16)
        names, params = zip(*block.named_parameters())
        params = tuple(p.detach() for p in params)

        def step(x, *values):
            return torch.func.functional_call(block, dict(zip(names, values)), (x,))

        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            replay = self._replay(step, (x, *params))
            tape = replay.tape
            self.assertEqual([r.op for r in tape.regions], ["mm"])
            self.assertEqual(sum(1 for L in tape.launches if L.get("cute")), 0)
            self.assertEqual(len(self._descriptor_launches(tape)), 1)
            self.assertEqual(tape.num_launches, 3)
            for rows in (4, 9):
                other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
                self.assertEqual(
                    replay(other, *params), step(other, *params), atol=0, rtol=0
                )
            self.assertEqual(
                (replay.traces, len(replay.variants), replay.declines), (1, 1, [])
            )

    def test_a_captured_tensor_declines_by_name(self):
        # a tensor the trace does not own (a parameter the host's closure holds): the
        # regions take inputs and allocations only, and so does a CuTe operand
        weight = torch.randn(4, 128, device="cuda")

        def host(x):
            y = torch.empty_like(x)
            stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
            COMPILED(marked(weight, read_only=True), marked(y), x.shape[0], stream)
            return y + x

        x = torch.randn(4, 128, device="cuda")
        with self.assertRaisesRegex(
            self.ht.Declined, "a tensor the trace does not own"
        ):
            self.ht.trace(host, (x,))

    def test_a_host_that_catches_the_decline_does_not_publish_without_the_launch(self):
        # the warm-up ran the program as written; under the trace its recording
        # declines after its observation is consumed (the captured tensor above),
        # and a host that catches the RuntimeError and falls back must not yield
        # a tape without the launch (the runtime team's R36 Triton finding)
        weight = torch.randn(4, 128, device="cuda")

        def host(x):
            y = torch.empty_like(x)
            stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
            try:
                COMPILED(marked(weight, read_only=True), marked(y), x.shape[0], stream)
            except RuntimeError:
                return x * 2
            return y + x

        x = torch.randn(4, 128, device="cuda")
        with self.assertRaisesRegex(
            self.ht.Declined,
            "recording failed under the trace.*a tensor the trace does not own",
        ):
            self.ht.trace(host, (x,))
        self.assertFalse(torch._C._host_trace_tracing())

    def test_topk_is_recorded_from_its_descriptor_with_eager_s_kernel(self):
        # eager's route is torch._native's register kernel; the override's condition
        # runs on the traced tensors (N against its kernel table and the row count
        # against the SM count are guards), the launch is recorded from the
        # program's descriptor with eager's own function handle; another row count
        # is a rebind (the grid is a ceil_div of the rows), no re-trace
        x = torch.randn(256, 1024, device="cuda")
        (eager,) = self._eager_nodes(topk_host, (x,))
        self.assertIn("nativeopstopkcutedsl_kernels", eager["name"])
        replay = self._replay(topk_host, (x,))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (1, 0))
        (record,) = self._descriptor_launches(tape)
        self.assertEqual(
            (record["func"], record["kernel"]), (eager["func"], eager["name"])
        )
        self.assertEqual(
            [p["name"] for p in record["params"]],
            [
                "mX.data_ptr",
                "mX.shape[0]",
                "mX.stride[0]",
                "mValues.data_ptr",
                "mValues.shape[0]",
                "mValues.stride[0]",
                "mIndices.data_ptr",
                "mIndices.shape[0]",
                "mIndices.stride[0]",
            ],
        )
        for rows in (256, 300):
            other = torch.randn(rows, 1024, device="cuda")
            for got, expected in zip(replay(other), topk_host(other)):
                self.assertEqual(got, expected, atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines), (1, 1, [])
        )
        self.assertEqual(self._templates("topk"), [])

    def test_a_topk_eager_serves_from_the_aot_kernel_declines_by_name(self):
        # k = 64 at this N is covered by the AOT-embedded kernel (the router's ATen
        # fallback), not the Python override
        x = torch.randn(256, 1024, device="cuda")
        with self.assertRaisesRegex(self.ht.Declined, "AOT-embedded kernel"):
            self.ht.trace(lambda x: torch.topk(x, 64), (x,))

    def test_scatter_add_is_the_tape_s_copy_and_a_region_of_eager_s_kernel(self):
        # eager's functional override clones self (a device memcpy) and scatters into
        # the clone in place: the tape records the copy as its own and the region's
        # template is the in-place override's kernel, eager's own function. The
        # condition (TensorIterator's analysis, in Python) runs on the traced tensors
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        args = scatter_args(128)
        (eager,) = self._eager_nodes(scatter_inplace_host, (args[0].clone(), *args[1:]))
        self.assertIn("kernel_cutlass", eager["name"])
        replay = self._replay(scatter_host, args)
        tape = replay.tape
        self.assertEqual(
            (tape.num_launches, len(tape.memcpys), tape.num_regions), (0, 1, 1)
        )
        self.assertEqual(tape.memcpys[0]["kind"], "d2d")
        # the closed region serves because the program's descriptor does not
        # express its launch: the host builds a TMA descriptor over src
        from torch._native.ops.scatter_add import tma_kernel

        program = tma_kernel._compile_tma_scatter(torch.float32)
        record = self.dsl._compiles.get(program)
        self.assertIsNotNone(record.descriptor)
        self.assertIsNotNone(record.descriptor.declined)
        self.assertRegex(
            record.descriptor.declined, "identity_layout|tma|over runtime values"
        )
        (region,) = tape.regions
        self.assertEqual((region.op, region.scalars), ("scatter_add_", (0,)))
        self.assertEqual(
            [o.name for o in (*region.inputs, *region.outputs)],
            ["index", "src", "self"],
        )
        self.assertEqual(tape.written_inputs, ())
        self.assertEqual(replay(*args), scatter_host(*args), atol=0, rtol=0)
        other = scatter_args(64)
        self.assertEqual(replay(*other), scatter_host(*other), atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines), (1, 1, [])
        )
        for t in self._templates("scatter_add_"):
            self.assertEqual([n["func"] for n in t.nodes], [eager["func"]])

    def test_scatter_add_s_decision_is_its_condition_s_comparisons(self):
        # E40: the condition runs on the traced tensors, each comparison it makes a
        # guard (the coalescing's relations, the alignment reads), no size pinned; a
        # layout that flips its decision (a dense index) misses at replay into
        # eager's ATen route, whose scatter_add is not a traced host: served ordinarily
        args = scatter_args(128)
        tape = self.ht.trace(scatter_host, args)
        guards = [str(g) for g in tape.guards]
        self.assertTrue(any("Mod(" in g for g in guards), guards)
        self.assertFalse(
            any(g == f"Eq({s}, 128)" for g in guards for s in ("s26", "s75")), guards
        )
        replay = self._replay(scatter_host, args)
        dst, _, src = args
        dense = (
            torch.randperm(256, device="cuda")[:128]
            .view(128, 1)
            .expand(128, 1024)
            .contiguous()
        )
        names = [
            n["name"]
            for n in self._eager_nodes(scatter_inplace_host, (dst.clone(), dense, src))
        ]
        self.assertFalse(any("cutlass" in n for n in names), names)
        self.assertEqual(
            replay(dst, dense, src), scatter_host(dst, dense, src), atol=0, rtol=0
        )
        self.assertEqual(
            (replay.misses, replay.traces, len(replay.declines)), (1, 2, 1)
        )
        self.assertIn("scatter_add", replay.declines[0])

    def test_scatter_add__writes_the_input_through_the_region(self):
        args = scatter_args(128)
        tape = self.ht.trace(scatter_inplace_host, (args[0].clone(), *args[1:]))
        self.assertEqual(
            (tape.num_launches, len(tape.memcpys), tape.num_regions), (0, 0, 1)
        )
        self.assertEqual(tape.written_inputs, (0,))
        replay = self._replay(scatter_inplace_host, (args[0].clone(), *args[1:]))
        got, expected = args[0].clone(), args[0].clone()
        replay(got, *args[1:])
        scatter_inplace_host(expected, *args[1:])
        self.assertEqual(got, expected, atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines), (1, 1, [])
        )

    def test_rms_norm_backward_is_its_descriptor_s_record_and_aten_s_two_launches(self):
        # the backward override launches QuACK's kernel (recorded from its
        # descriptor: the persistent grid is the sm_count formal), then ATen's
        # reduction of dw_partial and a cast into the returned grad_weight, both
        # converted hosts: three launches of one tape, eager's three function
        # objects, dw_partial an allocation of the tape
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        _, rstd = torch.ops.aten._fused_rms_norm(x, [2048], w, 1e-6)
        g = torch.randn_like(x)
        eager = self._eager_nodes(rms_norm_backward_host, (g, x, rstd, w))
        self.assertIn("RMSNormBackward", eager[0]["name"])
        replay = self._replay(rms_norm_backward_host, (g, x, rstd, w))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (3, 0))
        (record,) = self._descriptor_launches(tape)
        self.assertIs(record, tape.launches[0])
        # the descriptor's record holds the CUfunction eager's node holds; ATen's
        # two launches record their host symbols (the runtime resolves them)
        self.assertEqual(
            (record["func"], record["kernel"]), (eager[0]["func"], eager[0]["name"])
        )
        self.assertEqual(
            [L["kernel"] for L in tape.launches[1:]], [n["name"] for n in eager[1:]]
        )
        # the persistent grid is the sm_count formal, a value of the call
        self.assertEqual(list(record["grid"]), list(eager[0]["grid"]))
        self.assertEqual(
            [p["name"] for p in record["params"]][:3],
            ["mX.data_ptr", "mX.shape[0]", "mX.stride[0]"],
        )
        for rows in (4, 9):
            other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            _, other_rstd = torch.ops.aten._fused_rms_norm(other, [2048], w, 1e-6)
            args = (torch.randn_like(other), other, other_rstd, w)
            for got, expected in zip(replay(*args), rms_norm_backward_host(*args)):
                self.assertEqual(got, expected, atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines), (1, 1, [])
        )
        self.assertEqual(self._templates("_fused_rms_norm_backward"), [])

    def test_without_a_warm_up_the_program_declines_by_name(self):
        x = torch.randn(4, 128, device="cuda")
        with self.assertRaisesRegex(self.ht.Declined, "was not met at the warm-up"):
            self.ht.trace(program_host, (x,), warm_up=False)


if __name__ == "__main__":
    run_tests()
