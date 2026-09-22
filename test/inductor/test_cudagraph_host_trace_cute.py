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
            self.ht.Declined, "CuTe DSL program the recorder does not hook.*not recorded on the tape"
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
        # the interim replay declines a tape with a CuTe record by name
        with self.assertRaisesRegex(
            self.ht.Declined,
            "interim replay does not serve a tape with a CuTe DSL launch",
        ):
            self.ht.Variant(tape, cute_host, (x,), warm_up=False)

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
    # the normalized shape as a module holds it (ints); one written from a traced
    # shape declines by name (the override's condition sees the router's copy)
    return torch.nn.functional.rms_norm(x, (2048,), w, eps=1e-6)


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
        from torch.cuda import _host_trace, _host_trace_cute_dsl

        self.module, self.ht, self.dsl = (
            direct_hosttrace,
            _host_trace,
            _host_trace_cute_dsl,
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

    def test_f_rms_norm_through_the_quack_override(self):
        self._quack_in_process()
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        replay = self._replay(rms_norm_host, (x, w))
        tape = replay.tape
        launches = [L for L in tape.launches if L.get("cute")]
        self.assertEqual(len(launches), 1)
        (launch,) = launches
        self.assertIn("rmsnormRMSNorm", launch["kernel"])
        record = launch["cute"]
        self.assertEqual(record.invocation.read_only, frozenset({"mX", "mW"}))
        access = {
            p["name"]: p["access"] for p in launch["params"] if p["kind"] == "ptr"
        }
        self.assertEqual(
            access,
            {
                "mX.data_ptr": "r",
                "mW.data_ptr": "r",
                "mO.data_ptr": "rw",
                "mRstd.data_ptr": "rw",
            },
        )
        for rows in (4, 7, 64):
            other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(replay(other, w), rms_norm_host(other, w), atol=0, rtol=0)
        self.assertEqual((replay.misses, len(replay.variants)), (0, 1))

    def test_a_block_with_rms_norm_is_one_tape(self):
        # the parameters as inputs (functional_call), as the recorder's regions
        # take them; the RMSNorm weight is then a traced tensor of the CuTe launch
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
            self.assertEqual(sum(1 for L in tape.launches if L.get("cute")), 1)
            self.assertEqual(tape.num_regions, 1)
            for rows in (4, 9):
                other = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
                self.assertEqual(
                    replay(other, *params), step(other, *params), atol=0, rtol=0
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

    def test_the_scatter_add_override_declines_by_name(self):
        self._quack_in_process()
        dst = torch.zeros(256, 1024, device="cuda")
        idx = torch.randint(0, 256, (128, 1), device="cuda").expand(128, 1024)
        src = torch.randn(128, 1024, device="cuda")

        def scatter(dst, idx, src):
            return dst.scatter_add(0, idx, src)

        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        with self.assertRaisesRegex(
            self.ht.Declined,
            "the warm-up called the CuTe DSL program .*_launch here.*TensorIterator",
        ):
            self.ht.trace(scatter, (dst, idx, src))

    def test_without_a_warm_up_the_program_declines_by_name(self):
        x = torch.randn(4, 128, device="cuda")
        with self.assertRaisesRegex(self.ht.Declined, "was not met at the warm-up"):
            self.ht.trace(program_host, (x,), warm_up=False)


if __name__ == "__main__":
    run_tests()
