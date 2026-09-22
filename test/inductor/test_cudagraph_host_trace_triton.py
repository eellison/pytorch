# Owner(s): ["module: inductor"]
"""Python-launched Triton kernels on the tape (torch/cuda/_host_trace_triton.py), lowered
into the shared native replay (direct_hosttrace._HostTraceTritonModule): a user kernel called
from a traced function, torch._native's K = 1 bmm override, the declines."""

import gc
import os
import unittest

import torch
from torch.cuda._utils import _check_cuda_bindings
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_TRITON


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def scale_add(x_ptr, y_ptr, n, shift, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, x * 2.0 + shift, mask=mask)

    @triton.jit
    def scale_float(x_ptr, y_ptr, n, alpha, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * alpha, mask=mask)

    @triton.autotune(
        configs=[triton.Config({"BLOCK": 256}), triton.Config({"BLOCK": 512})],
        key=["n"],
    )
    @triton.jit
    def tuned(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)


def scale_add_host(x):
    y = torch.empty_like(x)
    n = x.numel()
    scale_add[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, y, n, 3, BLOCK=1024)
    return y


def data_grid_host(x):
    y = torch.empty_like(x)
    n = x.numel()
    scale_add[lambda meta: (int(x[0].item()),)](x, y, n, 3, BLOCK=1024)
    return y


def float_host(x):
    y = torch.empty_like(x)
    n = x.numel()
    scale_float[(triton.cdiv(n, 1024),)](x, y, n, 1.5, BLOCK=1024)
    return y


def tuned_host(x):
    y = torch.empty_like(x)
    n = x.numel()
    tuned[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, y, n)
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

EXT = None


def mixed_host(x, w):
    # one call, every launch kind: an ATen converted host, a cuBLAS region, a
    # user CUDA C++ kernel (typed launch helper), a user Triton kernel, an ATen host
    h = torch.nn.functional.silu(x)
    g = torch.mm(h, w)
    t = EXT.scale_two_plus_one_(g)
    y = torch.empty_like(t)
    n = t.numel()
    scale_add[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](t, y, n, 3, BLOCK=1024)
    return torch.add(y, h)


def cute_host(x):
    import cutlass

    from cuda.bindings import driver
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack

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

    y = torch.empty_like(x)
    source, destination = (from_dlpack(t, assumed_align=16) for t in (x, y))
    launch_affine(
        source,
        destination,
        cutlass.Int32(3),
        driver.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return y


def _kernel_nodes(graph):
    """(kernel name, function handle) per kernel node of a raw CUDA graph."""
    from cuda.bindings import driver

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
    torch.cuda.is_available() and HAS_TRITON and torch.version.hip is None,
    "CUDA and Triton required",
)
class TestHostTraceTriton(TestCase):
    def setUp(self):
        super().setUp()
        from torch._inductor.runtime._cudagraph import direct_hosttrace
        from torch.cuda import _host_trace

        self.module, self.ht = direct_hosttrace, _host_trace

    def _replay(self, fn, args):
        gc.collect()
        replay = self.module.HostTraceReplay(fn, args)
        self.addCleanup(replay.close)
        return replay

    def _eager_nodes(self, fn, args):
        # a plain capture of eager's call: the function objects eager launches
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(*args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            fn(*args)
        torch.cuda.synchronize()
        self.addCleanup(lambda: graph)  # alive until the test ends
        return _kernel_nodes(graph.raw_cuda_graph())

    def test_user_kernel_replays_natively_with_a_rebind(self):
        x = torch.randn(4096, device="cuda")
        replay = self._replay(scale_add_host, (x,))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (1, 0))
        (launch,) = tape.launches
        self.assertEqual(launch["kernel"], "scale_add")
        self.assertEqual(
            launch["triton"].formals,
            (
                ("x_ptr", "ptr"),
                ("y_ptr", "ptr"),
                ("n", "i32"),
                ("shift", "i32"),
                ("BLOCK", "constexpr"),
            ),
        )
        kinds = [(p["name"], p["kind"], p["access"]) for p in launch["params"]]
        self.assertEqual(
            kinds[:4],
            [
                ("x_ptr", "ptr", "rw"),
                ("y_ptr", "ptr", "rw"),
                ("n", "i32", ""),
                ("shift", "i32", ""),
            ],
        )
        # an unwrapped pointer argument may be written: its root is a written root of the tape
        self.assertEqual(tape.written_inputs, (0,))
        (call,) = replay.lowered.calls
        self.assertIsInstance(call.module, self.module._HostTraceTritonModule)
        # the prepared node launches eager's own function object (E36)
        graph, _ = replay.variants[0].lowered.capture_handles
        self.assertEqual(_kernel_nodes(graph), self._eager_nodes(scale_add_host, (x,)))
        for n in (16000, 64, 2**20):
            other = torch.randn(n, device="cuda")
            self.assertEqual(replay(other), scale_add_host(other), atol=0, rtol=0)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.declines), (0, 1, [])
        )
        # a size outside the divisible-by-16 class Triton compiled for misses:
        # the entry traces at it and serves natively (E24), never the ordinary host
        odd = torch.randn(1000, device="cuda")
        self.assertEqual(replay(odd), scale_add_host(odd), atol=0, rtol=0)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.declines, replay.ordinary),
            (1, 2, [], 0),
        )
        self.assertEqual(replay.variants[1].tape.launches[0]["kernel"], "scale_add")

    def test_bmm_k1_takes_eagers_triton_override(self):
        from torch._native import registry

        nodes = registry._graphs.get(("bmm", "CUDA"), ())
        self.assertTrue(
            any(node.active and node.dsl_name == "triton" for node in nodes)
        )
        a = torch.randn(4, 32, 1, device="cuda")
        b = torch.randn(4, 1, 16, device="cuda")
        replay = self._replay(torch.bmm, (a, b))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (1, 0))
        (launch,) = tape.launches
        self.assertEqual(launch["kernel"], "_bmm_outer_product_kernel")
        access = {p["name"]: p["access"] for p in launch["params"]}
        # the override wraps its inputs read-only (ConstTensorWrapper); the output is written
        self.assertEqual(
            (access["A_ptr"], access["B_ptr"], access["OUT_ptr"]), ("r", "r", "rw")
        )
        self.assertEqual(tape.written_inputs, ())
        graph, _ = replay.variants[0].lowered.capture_handles
        self.assertEqual(_kernel_nodes(graph), self._eager_nodes(torch.bmm, (a, b)))
        for batch in (2, 7, 4):
            args = (
                torch.randn(batch, 32, 1, device="cuda"),
                torch.randn(batch, 1, 16, device="cuda"),
            )
            self.assertEqual(replay(*args), torch.bmm(*args), atol=0, rtol=0)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.declines), (0, 1, [])
        )
        # another BLOCK_N (N = 20 rounds to 32): another compilation, another variant
        args = (
            torch.randn(4, 32, 1, device="cuda"),
            torch.randn(4, 1, 20, device="cuda"),
        )
        self.assertEqual(replay(*args), torch.bmm(*args), atol=0, rtol=0)
        self.assertEqual(
            (replay.misses, len(replay.variants), replay.declines), (1, 2, [])
        )

    def test_bmm_with_the_override_off_is_a_closed_region(self):
        from torch._native.registry import (
            deregister_op_overrides,
            reenable_op_overrides,
        )

        a = torch.randn(4, 32, 1, device="cuda")
        b = torch.randn(4, 1, 16, device="cuda")
        # the registry's symbol is the op name without its namespace
        deregister_op_overrides(disable_op_symbols="bmm")
        try:
            tape = self.ht.trace(torch.bmm, (a, b))
        finally:
            reenable_op_overrides(enable_op_symbols="bmm")
        self.assertEqual((tape.num_launches, tape.num_regions), (0, 1))
        tape = self.ht.trace(torch.bmm, (a, b))
        self.assertEqual((tape.num_launches, tape.num_regions), (1, 0))

    def test_trace_without_a_warm_up_selects_the_compilation(self):
        # a native entry's miss path traces without a warm-up: the compilation
        # is selected from the traced values and guarded, the same record
        x = torch.randn(4096, device="cuda")
        warm = self.ht.trace(scale_add_host, (x,))
        cold = self.ht.trace(scale_add_host, (x,), warm_up=False)
        self.assertEqual(cold.launches[0]["func"], warm.launches[0]["func"])
        self.assertEqual([str(g) for g in cold.guards], [str(g) for g in warm.guards])

    def test_declines_by_name(self):
        x = torch.randn(4096, device="cuda")
        with self.assertRaisesRegex(
            self.ht.Declined, "the grid of Triton kernel scale_add.*_local_scalar_dense"
        ):
            self.ht.trace(data_grid_host, (x,))
        with self.assertRaisesRegex(
            self.ht.Declined, "Triton kernel scale_float: .*no supported compiler ABI"
        ):
            self.ht.trace(float_host, (x,))
        with self.assertRaisesRegex(
            self.ht.Declined, "Triton Autotuner tuned: an autotuned launch"
        ):
            self.ht.trace(tuned_host, (x,))
        # a declined trace serves the ordinary host, by class (E24 applies to misses, not declines)
        replay = self._replay(tuned_host, (x,))
        self.assertEqual(replay(x), tuned_host(x), atol=0, rtol=0)
        self.assertEqual(len(replay.declines), 1)

    def test_one_tape_every_launch_kind_replays_as_one_graph(self):
        # ATen converted hosts, a cuBLAS region, a user CUDA C++ kernel and a user
        # Triton kernel in one call: one tape in host order, one native graph
        # holding eager's function objects, a shape change served by rebinds
        # (the region by a harvest of its new key), no re-trace
        global EXT
        import sys

        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        from host_trace_testing import load_test_extension

        EXT = load_test_extension("ht_triton_capture_scale", _SCALE_SOURCE)
        w = torch.randn(256, 256, device="cuda")
        x = torch.randn(64, 256, device="cuda")
        replay = self._replay(mixed_host, (x, w))
        tape = replay.tape
        self.assertEqual((tape.num_launches, tape.num_regions), (4, 1))
        # host order: silu, the region, the C++ kernel (its demangled symbol), the
        # Triton kernel (its cubin name), add
        names = [L["kernel"] for L in tape.launches]
        self.assertIn("silu", names[0].lower())
        self.assertIn("scale_two_plus_one", names[1])  # the mangled symbol
        self.assertEqual(names[2], "scale_add")
        self.assertEqual(tape.regions[0].op, "mm")
        self.assertLess(tape.launches[0]["seq"], tape.regions[0].seq)
        self.assertLess(tape.regions[0].seq, tape.launches[1]["seq"])
        kinds = [type(call.module).__name__ for call in replay.lowered.calls]
        self.assertEqual(kinds.count("_HostTraceTritonModule"), 1)
        self.assertEqual(kinds.count("_HostTraceKernelModule"), 3)
        graph, _ = replay.variants[0].lowered.capture_handles
        self.assertEqual(_kernel_nodes(graph), self._eager_nodes(mixed_host, (x, w)))
        for rows in (48, 7, 64):
            other = torch.randn(rows, 256, device="cuda")
            self.assertEqual(replay(other, w), mixed_host(other, w), atol=0, rtol=0)
        self.assertEqual(
            (replay.traces, len(replay.variants), replay.declines, replay.ordinary),
            (1, 1, [], 0),
        )

    def test_a_cute_dsl_kernel_declines_by_name(self):
        try:
            import cutlass  # noqa: F401
        except ImportError:
            self.skipTest("the CuTe DSL is not installed")
        x = torch.randn(8, 128, device="cuda")
        self.assertEqual(cute_host(x), x * 2.0 + 3, atol=0, rtol=0)
        with self.assertRaisesRegex(
            self.ht.Declined,
            "CuTe DSL program the recorder does not hook.*not recorded on the tape",
        ):
            self.ht.trace(cute_host, (x,))

    def test_the_hook_leaves_triton_as_it_found_it(self):
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import JITFunction, KernelInterface

        before = (KernelInterface.__getitem__, JITFunction.run, Autotuner.run)
        x = torch.randn(4096, device="cuda")
        self.ht.trace(scale_add_host, (x,))
        self.assertEqual(
            (KernelInterface.__getitem__, JITFunction.run, Autotuner.run), before
        )
        self.assertEqual(scale_add_host(x), x * 2.0 + 3, atol=0, rtol=0)


if __name__ == "__main__":
    run_tests()
