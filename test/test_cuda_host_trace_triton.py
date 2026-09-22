# Owner(s): ["module: cuda"]
"""Python-launched Triton kernels on the tape (torch/cuda/_host_trace_triton.py) through the
replay a test drives (host_trace_testing.build): the record's node is eager's kernel, and its
parameters are updated per call like any launch's."""

import unittest

from host_trace_testing import graph_nodes, HostTraceTestCase

import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_TRITON


if torch.cuda.is_available():
    from torch.cuda import _host_trace as ht

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
    def add_inplace(x_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)


def scale_add_host(x):
    y = torch.empty_like(x)
    n = x.numel()
    scale_add[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, y, n, 3, BLOCK=1024)
    return y


def scale_add_const_host(x):
    from torch._native.const_tensor_wrapper import ConstTensorWrapper

    y = torch.empty_like(x)
    n = x.numel()
    scale_add[(triton.cdiv(n, 1024),)](ConstTensorWrapper(x), y, n, 3, BLOCK=1024)
    return y


def add_inplace_host(x):
    n = x.numel()
    add_inplace[(triton.cdiv(n, 512),)](x, n, BLOCK=512)
    return x


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


def every_kind_host(x, w):
    # an ATen converted host, a cuBLAS region, a user CUDA C++ kernel (typed
    # launch helper), a user Triton kernel, an ATen host: one call
    h = torch.nn.functional.silu(x)
    g = torch.mm(h, w)
    t = EXT.scale_two_plus_one_(g)
    y = torch.empty_like(t)
    n = t.numel()
    scale_add[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](t, y, n, 3, BLOCK=1024)
    return torch.add(y, h)


def mixed_host(x):
    # an ATen launch, the Triton launch, an ATen launch: one tape, host order
    y = torch.mul(x, 2.0)
    z = scale_add_host(y)
    return torch.add(z, y)


@unittest.skipUnless(
    torch.cuda.is_available() and HAS_TRITON and torch.version.hip is None,
    "CUDA and Triton required",
)
class TestHostTraceTriton(HostTraceTestCase):
    def test_user_kernel_replays_in_its_class_and_misses_outside(self):
        x = torch.randn(4096, device="cuda")
        tape, variant, cases = self._replay_cases(
            scale_add_host,
            (x,),
            [(torch.randn(n, device="cuda"),) for n in (4096, 16000, 64, 1000)],
            msg=lambda args: f"n={args[0].numel()}",
        )
        self.assertEqual((tape.num_launches, tape.num_regions), (1, 0))
        self.assertEqual(tape.launches[0]["kernel"], "scale_add")
        self.assertEqual([c.miss for c in cases[:3]], [None] * 3)
        # 1000 is not a multiple of 16: Triton compiles another kernel for it
        self.assertIsNotNone(cases[3].miss)
        self.assertIn("% 16", cases[3].miss)
        if variant.graph is not None:
            # the replay's graph holds eager's own kernel: one node, the Triton function
            names = [k[0] for k in graph_nodes(variant.graph)[0]]
            self.assertEqual(names, ["scale_add"])

    def test_bmm_k1_takes_the_override(self):
        a = torch.randn(4, 32, 1, device="cuda")
        b = torch.randn(4, 1, 16, device="cuda")
        tape, variant, cases = self._replay_cases(
            torch.bmm,
            (a, b),
            [
                (
                    torch.randn(n, 32, 1, device="cuda"),
                    torch.randn(n, 1, 16, device="cuda"),
                )
                for n in (4, 2, 7)
            ],
            msg=lambda args: f"B={args[0].shape[0]}",
        )
        self.assertEqual((tape.num_launches, tape.num_regions), (1, 0))
        self.assertEqual(tape.launches[0]["kernel"], "_bmm_outer_product_kernel")
        self.assertEqual([c.miss for c in cases], [None] * 3)
        if variant.graph is not None:
            names = [k[0] for k in graph_nodes(variant.graph)[0]]
            self.assertEqual(names[0], "_bmm_outer_product_kernel")

    def test_written_roots_follow_the_pointer_declarations(self):
        x = torch.randn(2048, device="cuda")
        # an unwrapped pointer may be written: the input is a written position
        self.assertEqual(ht.trace(scale_add_host, (x,)).written_inputs, (0,))
        # a ConstTensorWrapper declares a read-only argument
        self.assertEqual(ht.trace(scale_add_const_host, (x,)).written_inputs, ())
        # an in-place kernel on the input: the tape writes it and returns it
        tape, variant, cases = self._replay_cases(
            add_inplace_host, (x.clone(),), [(torch.randn(2048, device="cuda"),)]
        )
        self.assertEqual(tape.written_inputs, (0,))
        self.assertEqual(tape.outputs[0].identity, ("argument", 0))
        self.assertEqual([c.miss for c in cases], [None])

    def test_one_tape_every_launch_kind_through_the_replay(self):
        global EXT
        from host_trace_testing import load_test_extension

        EXT = load_test_extension("ht_triton_capture_scale", _SCALE_SOURCE)
        w = torch.randn(256, 256, device="cuda")
        x = torch.randn(64, 256, device="cuda")
        tape, variant, cases = self._replay_cases(
            every_kind_host,
            (x, w),
            [(torch.randn(rows, 256, device="cuda"), w) for rows in (64, 48)],
            msg=lambda args: f"rows={args[0].shape[0]}",
        )
        self.assertEqual((tape.num_launches, tape.num_regions), (4, 1))
        names = [L["kernel"] for L in tape.launches]
        self.assertIn("scale_two_plus_one", names[1])  # the mangled symbol
        self.assertEqual(names[2], "scale_add")
        self.assertEqual([c.miss for c in cases], [None, None])
        if variant.graph is not None:
            # the native replay's kernel nodes, in the tape's order, with the
            # region's in between
            kinds = [k[0] for k in graph_nodes(variant.graph)[0]]
            cpp = next(i for i, k in enumerate(kinds) if "scale_two_plus_one" in k)
            self.assertLess(cpp, kinds.index("scale_add"))

    def test_triton_launch_sits_in_host_order_among_aten_launches(self):
        x = torch.randn(4096, device="cuda")
        tape, variant, cases = self._replay_cases(
            mixed_host, (x,), [(torch.randn(8192, device="cuda"),)]
        )
        self.assertEqual(tape.num_launches, 3)
        self.assertEqual(tape.launches[1]["kernel"], "scale_add")
        self.assertEqual([c.miss for c in cases], [None])
        if variant.graph is not None:
            names = [k[0] for k in graph_nodes(variant.graph)[0]]
            self.assertEqual(names[1], "scale_add")


if __name__ == "__main__":
    run_tests()
