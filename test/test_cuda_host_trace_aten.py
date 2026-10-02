# Owner(s): ["module: cuda graphs"]

import contextlib
import functools
import gc
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch.cuda import _host_trace_replay, _host_trace_tape
from torch.cuda._host_trace_capture import (
    capture_kernel_nodes,
    KernelNode,
    MemcpyNode,
    MemsetNode,
)
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_tape import _hint, EagerCall, Memcpy, Memset, trace, TrustedInputs
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
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


@contextlib.contextmanager
def _zero_init():
    # the harvest flag, as a harvest's captures set it
    previous = torch._C._cuda_hostTraceSetHarvesting(True)
    try:
        yield
    finally:
        torch._C._cuda_hostTraceSetHarvesting(previous)


# test ops whose functors hold an address or an operand's size
_FUNCTOR_OPS = r"""
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <torch/library.h>

// fills the stack below the caller with a CUDA address; a Python function, not
// an op, so that the dispatcher's frames don't move the buffer below the
// next op's launch site
__attribute__((noinline)) void smear(int64_t address, int64_t nbytes) {
  uint64_t buf[1 << 15];
  for (int64_t i = 0; i < std::min<int64_t>(nbytes / 8, 1 << 15); i++) {
    buf[i] = static_cast<uint64_t>(address);
  }
  asm volatile("" : : "r"(buf) : "memory");
}

namespace {

template <typename F>
at::Tensor map(const at::Tensor& x, const F& f) {
  auto out = at::empty_like(x);
  auto iter = at::TensorIteratorConfig().add_output(out).add_const_input(x).build();
  at::native::gpu_kernel(iter, f);
  return out;
}

// x + *q
struct AddAt {
  const float* q;
  __device__ float operator()(float a) const {
    return a + *q;
  }
  auto host_trace_fields() const {
    return std::tie(q);
  }
};

// x + p[0]
at::Tensor add_first(const at::Tensor& x, const at::Tensor& p) {
  return map(x, AddAt{p.const_data_ptr<float>()});
}

// x + p.numel(), a member eager computes from an operand's size
struct AddSize {
  float n;
  __device__ float operator()(float a) const {
    return a + n;
  }
  static constexpr bool host_trace_sizes = true;
};

at::Tensor add_numel(const at::Tensor& x, const at::Tensor& p) {
  return map(x, AddSize{static_cast<float>(p.numel())});
}

// x + the float at address
at::Tensor add_at(const at::Tensor& x, int64_t address) {
  return map(x, AddAt{reinterpret_cast<const float*>(address)});
}

at::Tensor like(const at::Tensor& x, const at::Tensor&) {
  return at::empty_like(x);
}

} // namespace

TORCH_LIBRARY(ht_functor, m) {
  m.def("add_first(Tensor x, Tensor p) -> Tensor");
  m.def("add_numel(Tensor x, Tensor p) -> Tensor");
  m.def("add_at(Tensor x, int address) -> Tensor");
}

TORCH_LIBRARY_IMPL(ht_functor, CUDA, m) {
  m.impl("add_first", add_first);
  m.impl("add_numel", add_numel);
  m.impl("add_at", add_at);
}

TORCH_LIBRARY_IMPL(ht_functor, Meta, m) {
  m.impl("add_first", like);
  m.impl("add_numel", like);
  m.impl("add_at", [](const at::Tensor& x, int64_t) { return at::empty_like(x); });
}
"""


@functools.cache
def _functor_ops():
    # the extension (its smear) and its ops
    from torch.utils.cpp_extension import load_inline

    ext = load_inline("ht_functor_bytes", cpp_sources="void smear(int64_t address, int64_t nbytes);", cuda_sources=_FUNCTOR_OPS, functions=["smear"], extra_cuda_cflags=["--extended-lambda"])
    return ext, torch.ops.ht_functor


@contextlib.contextmanager
def _traced_as_pointwise(namespace):
    # a test op as an ATen pointwise op: the pointwise host's
    pointwise = _host_trace_tape._pointwise
    with mock.patch.object(_host_trace_tape, "_pointwise", lambda func: func.namespace == namespace or pointwise(func)):
        yield


def silu_mul(x, y):
    return F.silu(x) * y


def _other(dtype):
    return torch.float16 if dtype == torch.float32 else torch.float32


# name: (fn, operand shapes, functor bytes the kernel reads); an operand of
# shape None is a bool condition
CASES = {
    "silu_mul": (silu_mul, ("mh", "mh"), 0),
    "silu_mul_broadcast": (silu_mul, ("mh", "h"), 0),
    "add": (lambda x, y: torch.add(x, y, alpha=2), ("mh", "h"), 4),
    "gelu": (F.gelu, ("mh",), 0),
    "gelu_tanh": (lambda x: F.gelu(x, approximate="tanh"), ("mh",), 0),
    "rsqrt": (torch.rsqrt, ("mh",), 0),
    "where": (torch.where, (None, "mh", "h"), 0),
    "cast": (lambda x: x.to(_other(x.dtype)), ("mh",), 0),
    "cast_f64": (lambda x: x.double(), ("mh",), 0),
    "cast_t": (lambda x: x.t().to(_other(x.dtype), memory_format=torch.contiguous_format), ("mh",), 0),
    "cast_f64_t": (lambda x: x.t().to(torch.float64, memory_format=torch.contiguous_format), ("mh",), 0),
    "contiguous_t": (lambda x: x.t().contiguous(), ("mh",), 0),
}


# name: (fn over an (m, h) input, max traces over the sweep, ops functor bytes,
# whether the accumulator is the input dtype)
REDUCTIONS = {
    f"{op.__name__}_{form}": (lambda x, op=op, f=f: f(op, x), traces, ops, same)
    for op, ops, same in ((torch.sum, 0, False), (torch.mean, 4, False), (torch.amax, 0, True))
    for form, f, traces in (
        ("last", lambda op, x: op(x, -1), 2),
        ("first", lambda op, x: op(x, 0), 2),
        ("keepdim", lambda op, x: op(x, -1, keepdim=True), 2),
        ("all", lambda op, x: op(x), 3),
        ("t", lambda op, x: op(x.t(), -1), 2),
    )
}


# name: (fn of h, operand shapes as CASES', WelfordOps bytes or None for a
# kernel of scalar and pointer parameters)
NORMS = {
    "softmax": (lambda h: lambda x: torch.softmax(x, -1), ("mh",), None),
    "log_softmax": (lambda h: lambda x: torch.log_softmax(x, -1), ("mh",), None),
    "softmax_to_float": (lambda h: lambda x: torch.softmax(x, -1, dtype=torch.float32), ("mh",), None),
    "var": (lambda h: lambda x: torch.var(x, -1), ("mh",), 5),
    "std_keepdim": (lambda h: lambda x: torch.std(x, -1, keepdim=True, correction=0), ("mh",), 5),
    "var_first": (lambda h: lambda x: torch.var(x, 0, correction=0), ("mh",), 5),
    "var_all": (lambda h: torch.var, ("mh",), 5),
    "layer_norm": (lambda h: lambda x, w, b: F.layer_norm(x, x.shape[-1:], w, b), ("mh", "h", "h"), None),
    "layer_norm_no_affine": (lambda h: lambda x: F.layer_norm(x, x.shape[-1:]), ("mh",), None),
    "rms_norm": (lambda h: lambda x, w: F.rms_norm(x, x.shape[-1:], w), ("mh", "h"), None),
    "rms_norm_no_weight": (lambda h: lambda x: F.rms_norm(x, x.shape[-1:]), ("mh",), None),
    # expect_contiguous's copy, then the kernel
    "layer_norm_t": (lambda h: lambda x, w, b: F.layer_norm(x.t(), x.shape[:1], w, b), ("hm", "h", "h"), None),
    "rms_norm_t": (lambda h: lambda x, w: F.rms_norm(x.t(), x.shape[:1], w), ("hm", "h"), None),
}


@contextlib.contextmanager
def _no_native_rms_norm():
    # torch._native routes _fused_rms_norm to a CuTe DSL kernel where
    # nvidia-cutlass-dsl is installed; without it eager runs the ATen kernel
    from torch._native import registry

    registry.deregister_op_overrides(disable_op_symbols="_fused_rms_norm")
    try:
        yield
    finally:
        registry.reenable_op_overrides(enable_op_symbols="_fused_rms_norm")


def _norm_inputs(case, m, h, dtype):
    shapes = {"mh": (m, h), "hm": (h, m), "h": (h,)}
    return [torch.randn(shapes[s], device="cuda", dtype=dtype) for s in NORMS[case][1]]


# bytes of ReduceConfig, OffsetCalculator<1, uint32_t> and OffsetCalculator<2, uint32_t>
REDUCE_CONFIG_BYTES, OFFSET_CALC_1_BYTES, OFFSET_CALC_2_BYTES = 64, 404, 504
# OffsetCalculator<N>: dims, then sizes_ (MAX_DIMS IntDividers), then strides_
# (MAX_DIMS x N uint32)
OFFSET_CALC_STRIDES_AT, OFFSET_CALC_STRIDES_BYTES_PER_OPERAND = 304, 100


def _shared_root_copy(x):
    t = x * x
    return t[:32].copy_(t[32:])


# name: (fn, args from (x f16 [64, 4096], y f32 [4096]), the tape's step types)
DECLINES = {
    "integer_sum": (lambda x: x.sum(-1), lambda x, y: (x.long(),), [EagerCall]),
    "sum_with_dtype": (lambda x: x.sum(-1, dtype=torch.float32), lambda x, y: (x,), [EagerCall]),
    "double_mean": (lambda x: x.mean(-1), lambda x, y: (x.double(),), [EagerCall]),
    "shared_root_copy": (_shared_root_copy, lambda x, y: (x,), [KernelLaunch, EagerCall]),
    "softmax_inner": (lambda x: torch.softmax(x, 0), lambda x, y: (x,), [EagerCall]),
    "softmax_strided": (lambda x: torch.softmax(x.t(), -1), lambda x, y: (x,), [EagerCall]),
    "var_no_dof": (lambda x: torch.var(x[:1], 0), lambda x, y: (x,), [EagerCall]),
    "index_select_gather_strided": (lambda x, i: x.t().index_select(0, i), lambda x, y: (x, torch.arange(17, device="cuda")), [EagerCall]),
    "cat_legacy_empty": (lambda x: torch.cat([x, x.new_empty(0)]), lambda x, y: (x,), [EagerCall]),
    "cat_mixed_dtypes": (lambda x, y: torch.cat([x[0], y]), lambda x, y: (x, y), [EagerCall]),
}


def _randn(*shape, dtype=torch.float16):
    return torch.randn(shape, device="cuda", dtype=dtype)


_jit_unary = torch.cuda.jiterator._create_jit_fn("template <typename T> T unary(T x) { return x * x + x; }")
_jit_binary = torch.cuda.jiterator._create_jit_fn("template <typename T> T binary(T x, T y, T alpha) { return alpha * x + y; }", alpha=1.0)
_jit_four = torch.cuda.jiterator._create_jit_fn("template <typename T> T four(T a, T b, T c, T d, T alpha, T beta) { return alpha * a + beta * b * c - d; }", alpha=0.5, beta=2)
_jit_two = torch.cuda.jiterator._create_multi_output_jit_fn("template <typename T> void two(T x, T y, T& out0, T& out1) { out0 = x + y; out1 = x - y; }", num_outputs=2)

# name: (fn, args from (m, h, dtype)): ops of the generic pointwise host
POINTWISE = {
    "mul_python_float": (lambda x: x * 0.5, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "add_python_int": (lambda x: x + 2, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "pow_3": (lambda x: torch.pow(x, 3.0), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "pow_2_5": (lambda x: torch.pow(x, 2.5), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "tanh": (torch.tanh, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "neg": (torch.neg, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "cos": (torch.cos, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "sin": (torch.sin, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "mul_Scalar": (lambda x: torch.ops.aten.mul.Scalar(x, 0.5), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "tanh_backward": (torch.ops.aten.tanh_backward, lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=d))),
    "tanh_backward_broadcast": (torch.ops.aten.tanh_backward, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d))),
    "mixed_dtypes": (lambda x, y: x * y, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=_other(d)))),
    "mixed_dtypes_contiguous": (lambda x, y: x + y, lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=_other(d)))),
    "integer_rsqrt": (torch.rsqrt, lambda m, h, d: (torch.randint(1, 100, (m, h), device="cuda"),)),
    "cos_strided": (lambda x: x[:, ::2].cos(), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "neg_t": (lambda x: x.t().neg(), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "mul_strided_cast": (lambda x, y: x[:, ::2] * y[:, ::2], lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=_other(d)))),
    "neg_misaligned": (lambda x: x.view(-1)[1:].neg(), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "add_inplace_mixed": (lambda x, y: (x * 1).add_(y), lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=_other(d)))),
    "mul_inplace_python_float": (lambda x: x.neg().mul_(0.5), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "tanh_out": (lambda x: torch.tanh(x, out=x.neg()), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "add_out_broadcast": (lambda x, y: torch.add(x, y, alpha=2, out=x.neg()), lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d))),
    "add_Scalar": (lambda x: torch.ops.aten.add.Scalar(x, 2, alpha=3), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "where_python_scalar": (lambda c, x: torch.where(c, x, 0.5), lambda m, h, d: (_randn(m, h) > 0, _randn(m, h, dtype=d))),
    "addcmul": (lambda x, y, z: torch.addcmul(x, y, z, value=0.5), lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=d), _randn(h, dtype=d))),
    "lerp": (torch.lerp, lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=d), _randn(m, h, dtype=d))),
    "clamp_tensor": (torch.clamp, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d) - 1, _randn(h, dtype=d) + 1)),
    "gt_bool": (torch.gt, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d))),
    "integer_true_divide": (torch.div, lambda m, h, d: (torch.randint(-50, 50, (m, h), device="cuda"), torch.randint(1, 50, (h,), device="cuda"))),
    "integer_times_float": (torch.mul, lambda m, h, d: (torch.randint(-50, 50, (m, h), device="cuda"), _randn(h, dtype=d))),
    "fill_nullary": (lambda x: x.neg().fill_(0.5), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "fill_functional": (lambda x: torch.fill(x, 0.5), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "zero_dim_operand": (torch.mul, lambda m, h, d: (_randn(m, h, dtype=d), _randn(dtype=torch.float32))),
    # functional forms over a CUDA out= kernel; an in-place op of no tag
    "abs": (torch.abs, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "logical_not": (torch.logical_not, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "abs_inplace": (lambda x: x.neg().abs_(), lambda m, h, d: (_randn(m, h, dtype=d),)),
    # no CUDA out= kernel: the witness is the op
    "relu": (torch.relu, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "isnan": (torch.isnan, lambda m, h, d: (_randn(m, h, dtype=d).log(),)),
    "rsub": (lambda x, y: torch.rsub(x, y, alpha=2), lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d))),
    "dropout_backward": (lambda g, mask: torch.ops.aten.native_dropout_backward(g, mask, 2.0), lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h) > 0)),
    # jitted_gpu_kernel's NVRTC kernels: vectorized, strided, casting
    "jiterator": (torch.special.i1e, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "jiterator_strided": (lambda x: torch.special.i1e(x[:, ::2]), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "jiterator_cast": (torch.special.xlog1py, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=_other(d)).abs())),
    "jiterator_scalar": (lambda x: torch.special.xlog1py(x, 2.0), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "jiterator_extra_arg": (lambda x: torch.polygamma(2, x), lambda m, h, d: (_randn(m, h, dtype=d).abs() + 0.5,)),
    # a user jiterator's (torch.cuda.jiterator) dynamic kernels
    "jiterator_user": (_jit_unary, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "jiterator_user_broadcast": (lambda x, y: _jit_binary(x, y, alpha=-1.5), lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d))),
    "jiterator_user_four": (_jit_four, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d), _randn(m, h, dtype=d), _randn(m, 1, dtype=d))),
    "jiterator_user_two_outputs": (_jit_two, lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=d))),
    "jiterator_user_strided": (lambda x: _jit_unary(x[:, ::2]), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "jiterator_user_cast": (_jit_binary, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=_other(d)))),
    "jiterator_user_cast_contiguous": (_jit_binary, lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=_other(d)))),
    "jiterator_user_misaligned": (lambda x: _jit_unary(x.view(-1)[1:]), lambda m, h, d: (_randn(m, h, dtype=d),)),
    # self twice in the iterator
    "threshold": (lambda x: F.threshold(x, 0.1, 20.0), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "threshold_backward": (lambda g, x: torch.ops.aten.threshold_backward(g, x, 0.0), lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=d))),
    # no pointwise tag: admitted by the witness's iterators
    "floor_divide": (torch.floor_divide, lambda m, h, d: (_randn(m, h, dtype=d), _randn(h, dtype=d))),
    "polar": (torch.polar, lambda m, h, d: (_randn(m, h, dtype=torch.float32).abs(), _randn(m, h, dtype=torch.float32))),
    "complex": (torch.complex, lambda m, h, d: (_randn(m, h, dtype=torch.float32), _randn(h, dtype=torch.float32))),
    # and an empty buffer it returns
    "logsigmoid": (F.logsigmoid, lambda m, h, d: (_randn(m, h, dtype=d),)),
    # gpu_kernel_multiple_outputs
    "frexp": (lambda x: torch.mul(*torch.frexp(x)), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "frexp_strided": (lambda x: torch.mul(*torch.frexp(x[:, ::2])), lambda m, h, d: (_randn(m, h, dtype=d),)),
    # a 0-dim temporary of the base, then the kernel
    "pow_scalar_base": (lambda x: torch.pow(2.5, x), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "zero_strided": (lambda x: x.neg()[:, ::2].zero_(), lambda m, h, d: (_randn(m, h, dtype=d),)),
}


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


# name: (fn, args from (m, h, dtype))
CATS = {
    "last_dim": (lambda a, b: torch.cat([a, b], -1), lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, 64, dtype=d))),
    "rotate_half": (_rotate_half, lambda m, h, d: (_randn(1, 4, m, 64, dtype=d),)),
    "kv_cache": (lambda past, new: torch.cat([past, new], -2), lambda m, h, d: (_randn(1, 4, m, 64, dtype=d), _randn(1, 4, 1, 64, dtype=d))),
    "mixed_sizes_dim0": (lambda a, b, c: torch.cat([a, b, c]), lambda m, h, d: (_randn(m, h, dtype=d), _randn(3, h, dtype=d), _randn(1, h, dtype=d))),
    "empty_input": (lambda a, b: torch.cat([a, b[:0], b]), lambda m, h, d: (_randn(m, h, dtype=d), _randn(2, h, dtype=d))),
    "non_contiguous": (lambda a, b: torch.cat([a.t(), b.t()]), lambda m, h, d: (_randn(m, 64, dtype=d), _randn(m, 32, dtype=d))),
    "many_inputs": (lambda *xs: torch.cat(xs, -1), lambda m, h, d: tuple(_randn(m, 8 + i % 3, dtype=d) for i in range(70))),
}


# name: (fn, args from (m, h, dtype)); m indices, as many as a decode step's
INDEXING = {
    "index_select_dim0": (lambda x, i: x.index_select(0, i), lambda m, h, d: (_randn(300, h, dtype=d), torch.randint(0, 300, (m,), device="cuda"))),
    "index_select_dim1": (lambda x, i: x.index_select(1, i), lambda m, h, d: (_randn(64, h, dtype=d), torch.randint(0, h, (m,), device="cuda"))),
    "index_select_int32_strided": (
        lambda x, i: x.index_select(0, i[::2]),
        lambda m, h, d: (_randn(300, h, dtype=d), torch.randint(0, 300, (2 * m,), device="cuda", dtype=torch.int32)),
    ),
    "embedding": (lambda i, w: F.embedding(i, w), lambda m, h, d: (torch.randint(0, 300, (1, m), device="cuda"), _randn(300, h, dtype=d))),
    # more than 16 indices: gather_out's vectorized gather
    "index_select_gather": (lambda x, i: x.index_select(0, i), lambda m, h, d: (_randn(300, h, dtype=d), torch.randint(0, 300, (m + 16,), device="cuda"))),
    "index_select_gather_int32": (
        lambda x, i: x.index_select(0, i),
        lambda m, h, d: (_randn(300, h, dtype=d), torch.randint(0, 300, (m + 16,), device="cuda", dtype=torch.int32)),
    ),
    "embedding_gather": (lambda i, w: F.embedding(i, w), lambda m, h, d: (torch.randint(0, 300, (2, m + 16), device="cuda"), _randn(300, h, dtype=d))),
}


def _batch_norm(x, w, b, dtype, channels_last=False):
    c = x.shape[1]
    stats = (torch.randn(c, device="cuda", dtype=dtype), torch.rand(c, device="cuda", dtype=dtype) + 0.5)
    params = [None if p is None else torch.randn(c, device="cuda", dtype=p) for p in (w, b)]
    return (x.contiguous(memory_format=torch.channels_last) if channels_last else x, *params, *stats)


def _eval_batch_norm(x, w, b, mean, var):
    return torch.native_batch_norm(x, w, b, mean, var, False, 0.1, 1e-5)[0]


def _class_targets(m, c, ignored=False):
    t = torch.randint(0, c, (m,), device="cuda")
    return t.masked_fill(torch.arange(m, device="cuda") % 3 == 1, -100) if ignored else t


def _nans(m, h, d):
    x = _randn(m, h, dtype=d)
    return (x.masked_fill(torch.rand_like(x) < 0.01, float("nan")),)


# name: (fn, args from (m, h, dtype)): ops of their own host
HOSTS = {
    "batch_norm_nchw": (_eval_batch_norm, lambda m, h, d: _batch_norm(_randn(m, 32, h // 256, 4, dtype=d), d, d, d)),
    "batch_norm_channels_last": (_eval_batch_norm, lambda m, h, d: _batch_norm(_randn(m, 32, h // 256, 4, dtype=d), d, d, d, True)),
    "batch_norm_2d": (_eval_batch_norm, lambda m, h, d: _batch_norm(_randn(m, h, dtype=d), d, d, d)),
    "batch_norm_float_params": (_eval_batch_norm, lambda m, h, d: _batch_norm(_randn(m, 32, h // 256, 4, dtype=d), torch.float, torch.float, torch.float)),
    "batch_norm_no_weight": (_eval_batch_norm, lambda m, h, d: _batch_norm(_randn(m, 32, h // 256, 4, dtype=d), None, None, d)),
    # a bf16 BatchNorm2d in eval, which eager keeps off cudnn
    "batch_norm_bf16_module": (
        lambda x, w, b, mean, var: F.batch_norm(x, mean, var, w, b),
        lambda m, h, d: _batch_norm(_randn(m, 32, h // 256, 4, dtype=torch.bfloat16), *[torch.bfloat16] * 3),
    ),
    "arange": (lambda x: torch.arange(x.shape[0], device=x.device), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "arange_start_step": (lambda x: torch.arange(1, 2 * x.shape[0] + 1, 2, device=x.device, dtype=x.dtype), lambda m, h, d: (_randn(m, h, dtype=d),)),
    # T5's cache_position[-1] + 1: an alignment guard at an offset into an allocation
    "arange_last_plus_one": (lambda x: torch.arange(x.shape[0], device=x.device)[-1] + 1, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "arange_negative_step": (lambda x: torch.arange(x.shape[0], -3, -2, device=x.device), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "causal_mask_triu": (lambda x: torch.triu(x.new_full((x.shape[0], x.shape[0]), float("-inf")), 1), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "triu_rectangular": (lambda x: x.triu(-1), lambda m, h, d: (_randn(m, 64, dtype=d),)),
    "zeros": (lambda x: x + torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "zeros_like": (lambda x: torch.zeros_like(x[:, :64]), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "repeat_symbolic": (lambda mask: mask.unsqueeze(1).repeat(1, mask.shape[1] + 1, 1), lambda m, h, d: (torch.rand(2, m, device="cuda") > 0.5,)),
    "repeat_leading": (lambda x: x.repeat(2, 1, 3), lambda m, h, d: (_randn(m, 64, dtype=d),)),
    "channel_shuffle": (lambda x: torch.channel_shuffle(x, 4), lambda m, h, d: (_randn(m, 16, h // 256, 4, dtype=d),)),
    "channel_shuffle_empty": (lambda x: torch.channel_shuffle(x[:, :, :0], 4), lambda m, h, d: (_randn(m, 16, h // 256, 4, dtype=d),)),
    "eye": (lambda x: torch.eye(x.shape[0], device=x.device, dtype=x.dtype), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "eye_m": (lambda x: torch.eye(x.shape[0], 5, device=x.device, dtype=x.dtype), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "eye_out": (lambda x, out: torch.eye(x.shape[0], out=out), lambda m, h, d: (_randn(m, h, dtype=d), torch.empty(m, m, device="cuda", dtype=d))),
    "eye_empty": (lambda x: torch.eye(0, x.shape[0], device=x.device, dtype=x.dtype), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "soft_margin_loss_backward_sum": (lambda g, x, t: torch.ops.aten.soft_margin_loss_backward(g, x, t, 2), lambda m, h, d: (_randn(dtype=d), _randn(m, h, dtype=d), _randn(m, h, dtype=d).sign())),
    "soft_margin_loss_backward_none": (lambda g, x, t: torch.ops.aten.soft_margin_loss_backward(g, x, t, 0), lambda m, h, d: (_randn(m, h, dtype=d), _randn(m, h, dtype=d), _randn(m, h, dtype=d).sign())),
    "nll_loss_mean": (lambda x, t: F.nll_loss(x, t), lambda m, h, d: (_randn(m, h // 8, dtype=d), _class_targets(m, h // 8))),
    "nll_loss_sum_ignored": (lambda x, t: F.nll_loss(x, t, reduction="sum"), lambda m, h, d: (_randn(m, h // 8, dtype=d), _class_targets(m, h // 8, True))),
    "nll_loss_weight": (lambda x, t, w: F.nll_loss(x, t, w), lambda m, h, d: (_randn(m, h // 8, dtype=d), _class_targets(m, h // 8, True), torch.rand(h // 8, device="cuda", dtype=d))),
    "max_pool2d": (lambda x: F.max_pool2d(x, 3, 2, 1), lambda m, h, d: (_randn(m, 8, h // 128, 16, dtype=d),)),
    "max_pool2d_ceil_dilation": (lambda x: F.max_pool2d(x, 2, (2, 1), 0, 2, ceil_mode=True), lambda m, h, d: (_randn(m, 8, h // 128, 16, dtype=d),)),
    "max_pool2d_unbatched": (lambda x: F.max_pool2d(x, 2), lambda m, h, d: (_randn(m, h // 128, 16, dtype=d),)),
    "adaptive_avg_pool2d": (lambda x: F.adaptive_avg_pool2d(x, (6, 6)), lambda m, h, d: (_randn(m, 8, h // 128, 16, dtype=d),)),
    "adaptive_avg_pool2d_unbatched": (lambda x: F.adaptive_avg_pool2d(x, (5, 7)), lambda m, h, d: (_randn(m, h // 128, 16, dtype=d),)),
    "argmax": (lambda x: x.argmax(-1), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "argmax_keepdim_of_ints": (lambda x: (x > 0).int().argmax(1, keepdim=True), lambda m, h, d: (_randn(m, 8, h // 8, dtype=d),)),
    "argmax_all": (torch.argmax, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "argmax_size_one": (lambda x: x[:, :1].argmax(1), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "argmax_nan": (lambda x: x.argmax(-1), _nans),
    "argmin": (lambda x: x.argmin(0), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "argmin_all_keepdim_t": (lambda x: x.t().argmin(keepdim=True), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "argmin_bf16": (lambda x: x.bfloat16().argmin(-1), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "amin": (lambda x: x.amin(-1), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "amin_all_t": (lambda x: x.t().amin(), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "max_all": (torch.max, lambda m, h, d: (_randn(m, h, dtype=d),)),
    "max_all_nan": (torch.max, _nans),
    "min_all_t": (lambda x: x.t().min(), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "max_dim": (lambda x: tuple(x.max(-1)), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "max_dim_t": (lambda x: tuple(x.t().max(-1)), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "max_dim_bf16": (lambda x: tuple(x.bfloat16().max(-1)), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "max_dim_of_ints": (lambda x: tuple((x * 8).int().max(1)), lambda m, h, d: (_randn(m, 8, h // 8, dtype=d),)),
    "max_dim_of_bools": (lambda x: tuple((x > 0).max(0)), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "max_dim_of_0dim": (lambda x: tuple(x[0, 0].max(0)), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "min_dim_keepdim_first": (lambda x: tuple(x.min(0, keepdim=True)), lambda m, h, d: (_randn(m, h, dtype=d),)),
    "min_dim_nan": (lambda x: tuple(x.min(1)), _nans),
    "min_dim_empty": (lambda x: tuple(x[:0].min(-1)), lambda m, h, d: (_randn(m, h, dtype=d),)),
}


def _unread_or_zeros(node: KernelNode, launch: KernelLaunch) -> set[tuple[int, int]]:
    return _unread(node, launch, 0) if "elementwise_kernel" in node.name else _zeros(launch)


def _zeros(launch: KernelLaunch) -> set[tuple[int, int]]:
    # (param, byte) a host leaves zero: entries past its tensor count and dims,
    # which eager leaves stale
    return {(p, b) for p, image in enumerate(launch.images) for b, v in enumerate(image) if v == 0}


def _reduce_unread(functor_bytes: int, arg_bytes: int, arg_align: int | None = None) -> set[tuple[int, int]]:
    # (param, byte) of a ReduceOp that eager copies from stack garbage despite
    # the harvest memset: an empty ops functor's byte and the struct padding
    arg_align = arg_align or arg_bytes
    ident = -(-max(functor_bytes, 1) // arg_align) * arg_align
    config = -(-(ident + arg_bytes) // 4) * 4
    calcs_end = config + REDUCE_CONFIG_BYTES + OFFSET_CALC_1_BYTES + OFFSET_CALC_2_BYTES
    src = -(-calcs_end // 8) * 8
    skipped = [
        *range(functor_bytes, ident),
        *range(ident + arg_bytes, config),
        *range(config + 57, config + 60),
        *range(calcs_end, src),
        src + 58,
        src + 59,
    ]
    return {(0, b) for b in skipped}


def _inputs(case, m, h, dtype):
    shapes = {"mh": (m, h), "h": (h,), None: (m, h)}
    args = [torch.randn(shapes[s], device="cuda", dtype=dtype) for s in CASES[case][1]]
    return [a > 0 if s is None else a for a, s in zip(args, CASES[case][1])]


def _unread(node: KernelNode, launch: KernelLaunch, functor_bytes: int) -> set[tuple[int, int]]:
    # (param, byte) eager leaves uninitialized and the kernel never reads: a
    # functor's tail (an empty functor's byte), an empty offset calculator's,
    # loader's and storer's byte, a cast loader's and storer's padding, and a
    # strided op's padding and OffsetCalculator entries past dims
    # a jitted kernel's (N, data, ic, oc, loader, storer, scalar, extra arguments) have no functor
    jitted = not node.name.startswith("_Z")
    if jitted or "vectorized" in node.name or "unrolled" in node.name:
        tail = set() if jitted else {(1, b) for b in range(functor_bytes, len(node.images[1]))}
        if "vectorized" in node.name:
            return tail
        empty = {(p, 0) for p in range(3, len(node.images)) if len(node.images[p]) == 1}
        # LoadWithCast<n> / StoreWithCast<n>: n dtypes padded to 4, n sizes
        loaders = (4, 5) if jitted else range(5, len(node.images))
        casts = {p: n for p in loaders for n in range(1, 9) if len(node.images[p]) == -(-n // 4) * 4 + 4 * n}
        return tail | empty | {(p, b) for p, n in casts.items() for b in range(n, -(-n // 4) * 4)}
    n = len(launch.pointers)
    cast = "StridedCastOp" in node.name
    at = 8 * n + (-(-n // 4) * 4 if cast else 0)
    image = node.images[1]
    dims = int.from_bytes(image[at : at + 4], "little")
    sizes, strides = at + 4, at + OFFSET_CALC_STRIDES_AT
    end = strides + OFFSET_CALC_STRIDES_BYTES_PER_OPERAND * n
    skipped = [
        *range(9 * n if cast else at, at),
        *range(sizes + 12 * dims, strides),
        *range(strides + 4 * n * dims, end),
        *range(end + functor_bytes, len(image)),
    ]
    return {(1, b) for b in skipped}


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
@requires_cuda_python_bindings
@unittest.skipIf(not hasattr(torch._C, "_cuda_hostTraceMul"), "needs traced hosts")
class TestHostTraceAten(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # a process's first trace allocates state the memory checks would count:
        # the generator's graph state, the plain launch attributes' probe
        HostTraceReplay(torch.neg)(torch.ones(1, device="cuda"))

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("case", list(CASES))
    def test_replays_new_shapes(self, dtype, case):
        fn = CASES[case][0]
        entry = HostTraceReplay(fn)
        for h in (4096, 768):
            for m in (64, 200, 7, 1):
                args = _inputs(case, m, h, dtype)
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                out = entry(*args)
                torch.cuda.synchronize()
                replay_peak = torch.cuda.max_memory_allocated() - base
                ref = fn(*args)
                self.assertEqual(out, ref, atol=0, rtol=0)
                self.assertEqual(out.stride(), ref.stride())
                del out, ref
                torch.cuda.reset_peak_memory_stats()
                fn(*args)
                torch.cuda.synchronize()
                self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)
        # at most one variant for m > 1 and one for m == 1
        self.assertLessEqual(entry.traces, 2)
        self.assertEqual(entry.eager, 0)
        for v in entry.variants:
            self.assertTrue(all(isinstance(s, range) for s in v.captured.lowered.steps))

    def _assert_launch_matches(self, launch, node, declared):
        self.assertEqual(launch.function, node.function)
        self.assertEqual(tuple(int(_hint(g)) for g in launch.grid), node.grid)
        self.assertEqual((launch.block, launch.smem), (node.block, node.smem))
        self.assertEqual(tuple(launch.layout), tuple(node.layout))
        fields = zip(launch.fields, launch.slots)
        for i, ((param, at, width), v) in enumerate(fields):
            declared |= {(param, b) for b in range(at, at + width)}
            if i not in launch.pointers:
                eager = node.images[param][at : at + width]
                eager = int.from_bytes(eager, "little", signed=True)
                self.assertEqual(int(_hint(v)), eager)
        for p, images in enumerate(zip(launch.images, node.images)):
            pairs = enumerate(zip(*images))
            diff = [b for b, (o, e) in pairs if o != e and (p, b) not in declared]
            self.assertEqual(diff, [], msg=f"{launch.name} param {p}")

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(CASES))
    @parametrize("m", [64, 7, 1])
    def test_records_match_eager(self, dtype, case, m):
        fn, _, functor_bytes = CASES[case]
        args = _inputs(case, m, 4096, dtype)
        tape = trace(fn, tuple(args))
        launches = [c for _, c in tape.launches]
        self.assertTrue(all(isinstance(c, KernelLaunch) for c in launches))
        nodes = capture_kernel_nodes(lambda s: fn(*args))
        self.assertEqual(len(launches), len(nodes))
        for launch, node in zip(launches, nodes):
            self._assert_launch_matches(launch, node, _unread(node, launch, functor_bytes))

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("case", list(REDUCTIONS))
    def test_reduction_replays_new_shapes(self, dtype, case):
        fn, most = REDUCTIONS[case][:2]
        entry = HostTraceReplay(fn)
        for h in (4096, 768):
            for m in (64, 200, 7, 1, 150, 5):
                x = torch.randn(m, h, device="cuda", dtype=dtype)
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                out = entry(x)
                torch.cuda.synchronize()
                replay_peak = torch.cuda.max_memory_allocated() - base
                ref = fn(x)
                self.assertEqual(out, ref, atol=0, rtol=0)
                self.assertEqual(out.stride(), ref.stride())
                del out, ref
                torch.cuda.reset_peak_memory_stats()
                fn(x)
                torch.cuda.synchronize()
                self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)
        self.assertLessEqual(entry.traces, most)
        self.assertEqual(entry.eager, 0)

    def _assert_replays_like_eager(self, entry, fn, x):
        # a collection inside the window frees an earlier test's garbage
        gc.collect()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = entry(x)
        torch.cuda.synchronize()
        replay_peak = torch.cuda.max_memory_allocated() - base
        ref = fn(x)
        self.assertEqual(out, ref, atol=0, rtol=0)
        del out, ref
        torch.cuda.reset_peak_memory_stats()
        fn(x)
        torch.cuda.synchronize()
        self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", ["sum_last", "amax_last", "mean_first", "sum_t"])
    def test_reduction_one_trace_across_brackets(self, dtype, case):
        # the launch config's power-of-two brackets of both sizes are patched,
        # not guarded
        fn = REDUCTIONS[case][0]
        entry = HostTraceReplay(fn)
        for m, h in ((64, 768), (2, 768), (37, 96), (129, 4096), (600, 200), (9, 1500), (300, 32)):
            self._assert_replays_like_eager(entry, fn, torch.randn(m, h, device="cuda", dtype=dtype))
        self.assertEqual(entry.traces, 1)
        self.assertEqual(entry.eager, 0)

    def test_global_reduction_replays_new_shapes(self):
        # a reduction across CTAs: its staging buffer, semaphores and CTAs per
        # output are sizes of the call
        fn = REDUCTIONS["sum_last"][0]
        entry = HostTraceReplay(fn)
        for m, h in ((2, 1 << 20), (3, 3 << 19), (5, 1 << 21)):
            self._assert_replays_like_eager(entry, fn, torch.randn(m, h, device="cuda"))
        self.assertEqual(entry.traces, 1)
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(REDUCTIONS))
    @parametrize("m", [4096, 64, 7, 1])
    def test_reduction_records_match_eager(self, dtype, case, m):
        fn, _, functor_bytes, same = REDUCTIONS[case]
        unread = _reduce_unread(functor_bytes, dtype.itemsize if same else 4)
        x = torch.randn(m, 768, device="cuda", dtype=dtype)
        launches = [c for _, c in trace(fn, (x,)).launches]
        with _zero_init():
            nodes = capture_kernel_nodes(lambda s: fn(x))
        kinds = [Memset if isinstance(n, MemsetNode) else KernelLaunch for n in nodes]
        self.assertEqual([type(c) for c in launches], kinds)
        for launch, node in zip(launches, nodes):
            if isinstance(node, MemsetNode):
                self.assertEqual(launch.value, node.value)
                width = int(_hint(launch.width)) * launch.element_size
                self.assertEqual(width, node.width * node.element_size)
            else:
                self._assert_launch_matches(launch, node, set(unread))

    def _assert_norm_replays_like_eager(self, entry, fn, args):
        gc.collect()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        traces = entry.traces
        out = entry(*args)
        torch.cuda.synchronize()
        replay_peak = torch.cuda.max_memory_allocated() - base
        # a trace's witness of an op without an out= overload allocates the op's output
        traced = entry.traces != traces
        ref = fn(*args)
        self.assertEqual(out, ref, atol=0, rtol=0)
        self.assertEqual(out.stride(), ref.stride())
        del out, ref
        torch.cuda.reset_peak_memory_stats()
        fn(*args)
        torch.cuda.synchronize()
        if not traced:
            self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @parametrize("case", list(NORMS))
    def test_norm_replays_new_shapes(self, dtype, case):
        self.enterContext(_no_native_rms_norm())
        # h 4096 is softmax's cunn_SoftMaxForwardReg, h 768 its persistent kernel
        for h in (4096, 768):
            fn = NORMS[case][0](h)
            entry = HostTraceReplay(fn)
            for m in (64, 200, 7, 1):
                self._assert_norm_replays_like_eager(entry, fn, _norm_inputs(case, m, h, dtype))
            # at most one variant for m > 1 and one for m == 1, and for a
            # global var one more where it splits across CTAs
            self.assertLessEqual(entry.traces, 3 if case == "var_all" else 2)
            self.assertEqual(entry.eager, 0)
            args = _norm_inputs(case, 64, h, dtype)
            self.assertTrue(all(isinstance(c, (KernelLaunch, Memset)) for _, c in trace(fn, tuple(args)).launches))

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize(
        "case, sizes",
        [
            ("softmax", ((64, 600), (7, 700), (129, 1000), (300, 513))),
            ("log_softmax", ((64, 2100), (3, 3000), (129, 2049))),
            ("softmax_to_float", ((64, 20), (7, 17), (5, 31))),
            ("layer_norm", ((64, 768), (7, 1024), (129, 4096), (300, 20000))),
            ("layer_norm_no_affine", ((64, 767), (7, 1023), (129, 4095))),
            ("rms_norm", ((64, 768), (7, 1024), (129, 4096), (300, 20000))),
            ("var", ((64, 768), (2, 768), (37, 96), (129, 4096), (600, 200))),
        ],
    )
    def test_norm_one_trace_across_sizes(self, dtype, case, sizes):
        self.enterContext(_no_native_rms_norm())
        # sizes that change only launch dims and size parameters: rows, and
        # a softmax's dim within its kernel's log2 bracket or register count
        fn = NORMS[case][0](None)
        entry = HostTraceReplay(fn)
        for m, h in sizes:
            self._assert_norm_replays_like_eager(entry, fn, _norm_inputs(case, m, h, dtype))
        self.assertEqual((entry.traces, entry.eager), (1, 0))

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(NORMS))
    @parametrize("m", [64, 7, 1])
    @parametrize("h", [4096, 768, 20000, 1022])
    def test_norm_records_match_eager(self, dtype, case, m, h):
        self.enterContext(_no_native_rms_norm())
        make, _, welford_bytes = NORMS[case]
        fn, args = make(h), _norm_inputs(case, m, h, dtype)
        unread = set() if welford_bytes is None else _reduce_unread(welford_bytes, 16, 4)
        launches = [c for _, c in trace(fn, tuple(args)).launches]
        with _zero_init():
            nodes = capture_kernel_nodes(lambda s: fn(*args))
        kinds = [Memset if isinstance(n, MemsetNode) else KernelLaunch for n in nodes]
        self.assertEqual([type(c) for c in launches], kinds)
        for launch, node in zip(launches, nodes):
            if isinstance(node, MemsetNode):
                self.assertEqual(launch.value, node.value)
            elif "elementwise_kernel" in node.name:
                self._assert_launch_matches(launch, node, _unread(node, launch, 0))
            else:
                self._assert_launch_matches(launch, node, set(unread))

    def _assert_replays_eager_calls(self, entry, fn, args, peak_at_most=False):
        gc.collect()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        traces = entry.traces
        out = entry(*args)
        torch.cuda.synchronize()
        replay_peak = torch.cuda.max_memory_allocated() - base
        # a trace's witness of an op without an out= overload allocates the op's output
        traced = entry.traces != traces
        ref = fn(*args)
        self.assertEqual(out, ref, atol=0, rtol=0)
        self.assertEqual([t.stride() for t in pytree.tree_leaves(out)], [t.stride() for t in pytree.tree_leaves(ref)])
        del out, ref
        torch.cuda.reset_peak_memory_stats()
        fn(*args)
        torch.cuda.synchronize()
        if not traced:
            eager_peak = torch.cuda.max_memory_allocated() - base
            if peak_at_most:
                self.assertLessEqual(replay_peak, eager_peak)
            else:
                self.assertEqual(replay_peak, eager_peak)

    def _assert_records_match_eager(self, fn, args, unread):
        launches = [c for _, c in trace(fn, tuple(args)).launches]
        self.assertTrue(all(isinstance(c, KernelLaunch) for c in launches))
        with _zero_init():
            nodes = capture_kernel_nodes(lambda s: fn(*args))
        self.assertEqual(len(launches), len(nodes))
        for launch, node in zip(launches, nodes):
            self._assert_launch_matches(launch, node, unread(node, launch))

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("case", list(POINTWISE))
    def test_pointwise_replays_new_shapes(self, dtype, case):
        fn, args_fn = POINTWISE[case]
        entry = HostTraceReplay(fn)
        for h in (4096, 768):
            for m in (64, 200, 7, 1):
                self._assert_replays_eager_calls(entry, fn, args_fn(m, h, dtype))
        self.assertLessEqual(entry.traces, 2)
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(POINTWISE))
    @parametrize("m", [64, 7, 1])
    def test_pointwise_records_match_eager(self, dtype, case, m):
        # the functor's bytes are the witness's, eager's own; a device lambda's
        # hold nvcc's host-side pointer, which differs per call
        fn, args_fn = POINTWISE[case]
        self._assert_records_match_eager(fn, args_fn(m, 4096, dtype), lambda node, launch: _unread(node, launch, 0))

    @parametrize("case", ["mul_python_float", "tanh_backward_broadcast", "mixed_dtypes", "mul_strided_cast", "pow_2_5", "where_python_scalar", "jiterator", "jiterator_cast", "jiterator_scalar", "jiterator_extra_arg", "jiterator_user", "jiterator_user_four", "jiterator_user_two_outputs", "jiterator_user_strided", "jiterator_user_cast", "jiterator_user_misaligned"])
    def test_pointwise_launch_reports_match_the_captured_nodes(self, case):
        # each launch site's report (LaunchLayout.h): its kernel, configuration
        # and, at each byte of a class, the captured node's byte
        fn, args_fn = POINTWISE[case]
        args = args_fn(64, 768, torch.float32)
        torch._C._cuda_hostTraceRecordLaunches(True)
        try:
            with _zero_init():
                nodes = capture_kernel_nodes(lambda s: fn(*args))
        finally:
            launched = torch._C._cuda_hostTraceRecordLaunches(False)
        self.assertEqual(len(launched), len(nodes))
        for (function, grid, block, classes, data), node in zip(launched, nodes):
            self.assertEqual((function, grid, block), (node.function, node.grid, node.block))
            self.assertEqual([len(c) for c in classes], [len(b) for b in node.images])
            for cs, d, image in zip(classes, data, node.images):
                self.assertEqual(bytes(x for c, x in zip(cs, d) if c != "."), bytes(x for c, x in zip(cs, image) if c != "."))

    def test_jiterator_of_a_tensor_extra_argument_runs_eagerly(self):
        # a tensor extra argument is its value, read on the host
        def fn(x, y):
            return _jit_binary(x, x, alpha=y)

        x, y = _randn(8, 16, dtype=torch.float32), torch.tensor(2.0, device="cuda")
        entry = HostTraceReplay(fn)
        self.assertEqual(entry(x, y), fn(x, y), atol=0, rtol=0)
        self.assertEqual(entry.eager, 1)
        self.assertIs(torch._C._cuda_jiterator_compile_and_launch_kernel, _host_trace_tape._jiterator_launch)

    @parametrize("case", ["mul", "rmul", "add_alpha", "rsub", "maximum", "lt", "eq", "div", "half", "double", "long", "complex", "jitted"])
    def test_cpu_scalar_operand_replays_its_value(self, case):
        # a CPU buffer the host makes (torch.rand's), a CPU scalar operand:
        # each replay reads its value as eager's TensorIterator does
        ops = {
            "mul": lambda x, s: x * s,
            "rmul": lambda x, s: s * x,
            "add_alpha": lambda x, s: torch.add(x, s, alpha=2),
            "rsub": lambda x, s: s - x,
            "maximum": lambda x, s: torch.maximum(x, s),
            "lt": lambda x, s: torch.lt(s, x),
            "eq": lambda x, s: x.round() == (s * 4).round(),
            "div": lambda x, s: x / (s + 0.25),
            "half": lambda x, s: x.half() * s,
            "double": lambda x, s: x * s.double(),
            "long": lambda x, s: x.long() + (s * 100).long(),
            "complex": lambda x, s: x.to(torch.complex64) - s,
            "jitted": lambda x, s: torch.special.hermite_polynomial_h(x, (s * 4).floor()),
        }

        def fn(x):
            return ops[case](x, torch.rand(()))

        entry = HostTraceReplay(fn)
        for seed in range(4):
            x = _randn(64, 33, dtype=torch.float32)
            torch.manual_seed(seed)
            want = fn(x)
            torch.manual_seed(seed)
            self.assertEqual(entry(x), want, atol=0, rtol=0)
        steps = [[c.reason for _, c in v.captured.lowered.tape.launches if type(c) is EagerCall and not c.host] for v in entry.variants]
        self.assertEqual((entry.traces, entry.eager, steps), (1, 0, [[]]))

    @parametrize("case", ["div_floor", "div_trunc", "pow", "complex_div"])
    def test_cpu_scalar_operand_of_a_value_read_runs_eagerly(self, case):
        # eager reads these CPU scalars into other functor members (a
        # reciprocal of a trunc, item()) or routes on them: an eager step, and
        # abs a kernel for the trace to capture
        ops = {
            "div_floor": lambda x, s: torch.div(x.abs(), s + 0.5, rounding_mode="floor"),
            "div_trunc": lambda x, s: torch.div(x.abs(), s + 1, rounding_mode="trunc"),
            "pow": lambda x, s: x.abs() ** s,
            "complex_div": lambda x, s: x.to(torch.complex64) / (s + 1),
        }

        def fn(x):
            return ops[case](x, torch.rand(()))

        entry = HostTraceReplay(fn)
        for seed in range(3):
            x = _randn(64, 33, dtype=torch.float32)
            torch.manual_seed(seed)
            want = fn(x)
            torch.manual_seed(seed)
            self.assertEqual(entry(x), want, atol=0, rtol=0)
        steps = [[c.reason for _, c in v.captured.lowered.tape.launches if type(c) is EagerCall and not c.host] for v in entry.variants]
        self.assertEqual((entry.traces, entry.eager, len(steps), len(steps[0])), (1, 0, 1, 1))
        self.assertIn("pointwise host declines", steps[0][0])

    def test_host_step_writing_a_buffer_a_device_step_read_declines(self):
        # a replay runs the host steps first: s.add_ after x * s would change
        # what the device step reads
        def fn(x):
            s = torch.rand(())
            y = x * s
            s.add_(1)
            return y * s

        entry = HostTraceReplay(fn)
        x = _randn(64, 33, dtype=torch.float32)
        torch.manual_seed(0)
        want = fn(x)
        torch.manual_seed(0)
        self.assertEqual(entry(x), want, atol=0, rtol=0)
        self.assertEqual(len(entry.declines), 1)
        self.assertIn("writes a CPU buffer a device step read earlier", entry.declines[0])

    def test_cpu_scalar_launch_report_marks_the_scalar(self):
        # the functor member eager reads from a CPU scalar is of its class:
        # 'A' + its dtype, or '0' + its dtype for div's reciprocal
        x, s = _randn(64, 33, dtype=torch.float32), torch.tensor(0.75)
        cases = ((lambda: x * s, "G", torch.tensor(0.75)), (lambda: x / s, "6", torch.tensor(1 / 0.75)), (lambda: x.double() - s, "H", torch.tensor(0.75, dtype=torch.float64)))
        for fn, cls, value in cases:
            torch._C._cuda_hostTraceRecordLaunches(True)
            try:
                with _zero_init():
                    capture_kernel_nodes(lambda _: fn())
            finally:
                launched = torch._C._cuda_hostTraceRecordLaunches(False)
            want = torch._C._cuda_hostTraceCpuScalarBytes(s, cls)
            self.assertEqual(want, value.numpy().tobytes())
            (classes, data) = launched[-1][3:]
            (p,) = [p for p, cs in enumerate(classes) if cls in cs]
            at = classes[p].index(cls)
            self.assertEqual((classes[p][at : at + len(want)], data[p][at : at + len(want)]), (cls * len(want), want))
            self.assertEqual(classes[p].count(cls), len(want))

    def test_pointwise_witness_on_a_smeared_stack(self):
        # a parameter's padding (StridedCastOp's tail) holds the stack's bytes,
        # here a live CUDA address: the host neither compares nor scans them
        ext, _ = _functor_ops()
        keep = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")

        def smeared(fn, *args, **kwargs):
            def run(s):
                ext.smear(keep.data_ptr() + 4096, 1 << 18)
                return fn(s)

            return capture_kernel_nodes(run, *args, **kwargs)

        def fn(x, y, z):
            return torch.maximum(x, y), x * z

        entry = HostTraceReplay(fn)
        with mock.patch("torch.cuda._host_trace_capture.capture_kernel_nodes", smeared):
            for m in (4, 7, 4):
                args = (_randn(m, 1, dtype=torch.float32), _randn(1, 5, dtype=torch.float32), _randn(1, 5))
                self.assertEqual([type(c) for _, c in trace(fn, args).launches], [KernelLaunch, KernelLaunch])
                self._assert_replays_eager_calls(entry, fn, args)
        self.assertEqual(entry.eager, 0)

    def test_pointwise_functor_holding_an_operand_address(self):
        # a slot of the operand's address at each replay
        _, ops = _functor_ops()

        def fn(x, p):
            return ops.add_first(x, p)

        entry = HostTraceReplay(fn)
        with _traced_as_pointwise("ht_functor"):
            for m in (64, 7, 64):
                args = (_randn(m, 768, dtype=torch.float32), _randn(3, dtype=torch.float32))
                self.assertEqual([type(c) for _, c in trace(fn, args).launches], [KernelLaunch])
                self.assertEqual(entry(*args), fn(*args), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    def test_pointwise_functor_of_an_operands_size_declines(self):
        # a member marked host_trace_sizes would replay the witness's size
        _, ops = _functor_ops()

        def fn(x, p):
            return ops.add_numel(x, p)

        with _traced_as_pointwise("ht_functor"):
            (call,) = [c for _, c in trace(fn, (_randn(64, 768, dtype=torch.float32), _randn(3, dtype=torch.float32))).launches]
        self.assertIsInstance(call, EagerCall)
        self.assertIn("functor has a member of the operands' sizes", call.reason)

    def test_pointwise_functor_holding_a_cuda_address_declines(self):
        # a pointer member at none of the op's tensors
        _, ops = _functor_ops()
        keep = _randn(1, dtype=torch.float32)

        def fn(x):
            return ops.add_at(x, keep.data_ptr())

        with _traced_as_pointwise("ht_functor"):
            (call,) = [c for _, c in trace(fn, (_randn(64, 768, dtype=torch.float32),)).launches]
        self.assertIsInstance(call, EagerCall)
        self.assertIn("functor points at a tensor that is none of the op's", call.reason)

    @parametrize("dtype", [torch.float16, torch.float32])
    def test_pointwise_inplace_on_an_argument(self, dtype):
        def fn(x, y):
            x.mul_(y).add_(1)
            return x * y

        entry = HostTraceReplay(fn)
        for m in (64, 7, 1):
            x, y = _randn(m, 768, dtype=dtype), _randn(m, 768, dtype=dtype)
            x_ref = x.clone()
            self.assertEqual(entry(x, y), fn(x_ref, y), atol=0, rtol=0)
            self.assertEqual(x, x_ref, atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    def test_zero_of_a_dense_tensor_is_a_memset(self, dtype):
        def fn(x):
            y = x.neg()
            y.t().zero_()
            x[1:].zero_()
            return y

        x = _randn(64, 768, dtype=dtype)
        self.assertEqual([type(c) for _, c in trace(fn, (x,)).launches], [KernelLaunch, Memset, Memset])
        entry = HostTraceReplay(fn)
        for m in (64, 7, 2, 1):
            x = _randn(m, 768, dtype=dtype)
            x_ref = x.clone()
            self.assertEqual(entry(x), fn(x_ref), atol=0, rtol=0)
            self.assertEqual(x, x_ref, atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    def test_contiguous_copy_is_a_memcpy(self, dtype):
        def fn(x, y):
            z = x.clone()
            z.copy_(y)
            return torch.masked_fill(z, y > 0, 1.0)

        args = (_randn(64, 768, dtype=dtype), _randn(64, 768, dtype=dtype))
        launches = [c for _, c in trace(fn, args).launches]
        self.assertEqual([type(c) for c in launches], [Memcpy, Memcpy, KernelLaunch, Memcpy, KernelLaunch])
        nodes = capture_kernel_nodes(lambda s: fn(*args), memcpy=True)
        self.assertEqual([type(n) for n in nodes], [MemcpyNode, MemcpyNode, KernelNode, MemcpyNode, KernelNode])
        self.assertEqual([int(_hint(c.nbytes)) for c in launches[:2]], [n.nbytes for n in nodes[:2]])
        entry = HostTraceReplay(fn)
        for m in (64, 7, 200, 1):
            self._assert_replays_eager_calls(entry, fn, (_randn(m, 768, dtype=dtype), _randn(m, 768, dtype=dtype)))
        self.assertLessEqual(entry.traces, 2)
        self.assertEqual(entry.eager, 0)

    def test_max_dim_of_a_zero_dim_tensor(self):
        # values.copy_(self) is a memcpy, then the indices' fill kernel
        def fn(x):
            v, i = torch.max(x * 2, 0)
            return v + 1, i

        self.assertFalse(any(isinstance(c, EagerCall) for _, c in trace(fn, (_randn(),)).launches))
        entry = HostTraceReplay(fn)
        for _ in range(3):
            x = _randn()
            self.assertEqual(entry(x), fn(x), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.eager), (1, 0))

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_masked_fill_of_a_device_value(self, dtype):
        # eager reads the value on the host (item()), which no capture takes
        def fn(x, mask, v):
            return x.neg().masked_fill_(mask, v)

        args = (_randn(64, 768, dtype=dtype), _randn(768) > 0, _randn(dtype=torch.float32))
        self.assertTrue(all(isinstance(c, KernelLaunch) for _, c in trace(fn, args).launches))
        entry = HostTraceReplay(fn)
        for m in (64, 7, 1):
            args = (_randn(m, 768, dtype=dtype), _randn(768) > 0, _randn(dtype=torch.float32))
            self.assertEqual(entry(*args), fn(*args), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    def test_random_ops_replay_through_the_rng_provider(self):
        def fn(x):
            y = F.dropout(x, 0.2, training=True) + torch.rand_like(x)
            return y.bernoulli_(0.3) * x.neg().normal_()

        entry = HostTraceReplay(fn, opaque=(HarvestProvider(("rng",)),))
        for m in (64, 64, 7):
            x = _randn(m, 768)
            torch.cuda.manual_seed(0)
            got = entry(x)
            torch.cuda.manual_seed(0)
            self.assertEqual(got, fn(x), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    def test_aliasing_op_that_copies_traces_its_decomposition(self):
        # under inference mode reshape reaches the trace, and copies a transposed input
        def fn(x):
            return x.t().reshape(-1) * 2

        with torch.inference_mode():
            x = _randn(64, 768)
            self.assertEqual([type(c) for _, c in trace(fn, (x,)).launches], [KernelLaunch, KernelLaunch])
            entry = HostTraceReplay(fn)
            for m in (64, 7, 64):
                x = _randn(m, 768)
                self.assertEqual(entry(x), fn(x), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    def test_factory_parts_and_constant_queries_trace(self):
        # full of a symbolic size is its fill kernel into its allocation; can_cast is a constant
        def fn(x):
            y = torch.full((x.shape[0], 1), 2.0, device=x.device, dtype=x.dtype)
            z = torch.ones_like(x, dtype=torch.bool) & torch.zeros(768, device=x.device, dtype=torch.bool)
            return x * y if torch.can_cast(x.dtype, torch.float) else z

        self.assertFalse(any(isinstance(c, EagerCall) for _, c in trace(fn, (_randn(64, 768),)).launches))
        entry = HostTraceReplay(fn)
        for m in (64, 7, 64):
            x = _randn(m, 768)
            self.assertEqual(entry(x), fn(x), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    @parametrize("m", [64, 0])
    def test_an_empty_buffer_out_is_returned(self, m):
        # log_sigmoid_forward's CUDA kernel leaves its buffer empty
        def fn(x):
            return torch.ops.aten.log_sigmoid_forward.output(x, output=torch.empty_like(x), buffer=x.new_empty(0))

        self.assertFalse(any(isinstance(c, EagerCall) for _, c in trace(fn, (_randn(m, 768),)).launches))
        entry = HostTraceReplay(fn)
        for n in (m, 7, m):
            x = _randn(n, 768)
            self.assertEqual(entry(x), fn(x), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    def test_empty_argument_launches_nothing(self):
        def fn(x, y):
            return x * 2, y.neg()

        tape = trace(fn, (_randn(0, 64), _randn(8, 64)))
        self.assertEqual([type(c) for _, c in tape.launches], [KernelLaunch])
        entry = HostTraceReplay(fn)
        for m in (0, 0, 5, 0):
            x, y = _randn(m, 64), _randn(8, 64)
            self.assertEqual(entry(x, y), fn(x, y), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.eager), (2, 0))

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(CATS) + list(INDEXING))
    def test_cat_and_indexing_replay_new_shapes(self, dtype, case):
        fn, args_fn = {**CATS, **INDEXING}[case]
        entry = HostTraceReplay(fn)
        for h in (4096, 768):
            for m in (16, 7, 1, 5):
                self._assert_replays_eager_calls(entry, fn, args_fn(m, h, dtype))
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(CATS) + list(INDEXING))
    @parametrize("m", [16, 1])
    def test_cat_and_indexing_records_match_eager(self, dtype, case, m):
        fn, args_fn = {**CATS, **INDEXING}[case]
        self._assert_records_match_eager(fn, args_fn(m, 768, dtype), _unread_or_zeros)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", list(HOSTS))
    def test_host_replays_new_shapes(self, dtype, case):
        fn, args_fn = HOSTS[case]
        entry = HostTraceReplay(fn)
        # Loss.cpp holds -target and -target * input to the end of z's full
        # expression; a replay frees each temporary after its last launch
        peak_at_most = case.startswith("soft_margin_loss_backward")
        for h in (4096, 768):
            for m in (16, 7, 1, 5):
                self._assert_replays_eager_calls(entry, fn, args_fn(m, h, dtype), peak_at_most)
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", [c for c in HOSTS if not c.startswith(("batch_norm", "zeros", "causal", "eye")) and not c.endswith(("_0dim", "_empty"))])
    @parametrize("m", [16, 1])
    def test_host_records_match_eager(self, dtype, case, m):
        fn, args_fn = HOSTS[case]
        self._assert_records_match_eager(fn, args_fn(m, 768, dtype), _unread_or_zeros)

    @parametrize("case", ["max_dim", "min_dim_keepdim_first", "argmax_all", "argmin", "max_all", "amin"])
    def test_minmax_replays_new_sizes_in_one_trace(self, case):
        # a new size re-dispatches the traced reduction with patched launch
        # parameters; a full reduction's switch to a global one (129 x 4096) is
        # a second trace, as for sum
        fn, args_fn = HOSTS[case]
        entry = HostTraceReplay(fn)
        for m, h in ((64, 768), (2, 768), (37, 96), (129, 4096), (9, 1500), (300, 32)):
            self._assert_replays_eager_calls(entry, fn, args_fn(m, h, torch.float32))
        self.assertEqual((entry.traces, entry.eager), (2 if case.endswith("_all") else 1, 0))

    def test_symbolic_repeat_replays_new_sizes_in_one_trace(self):
        fn, args_fn = HOSTS["repeat_symbolic"]
        entry = HostTraceReplay(fn)
        for m in (16, 7, 5):
            self._assert_replays_eager_calls(entry, fn, args_fn(m, 768, torch.float32))
        self.assertEqual((entry.traces, entry.eager), (1, 0))

    @parametrize("static", [True, False])
    def test_remainder_of_a_size(self, static):
        # GPT-2's sequence_lengths % input_ids.shape[-1]: remainder.Tensor's arg
        # parser takes no Python number, its Scalar overload does
        def fn(x, ids):
            return (x.argmax(-1) - 1) % (9 if static else ids.shape[-1])

        entry = HostTraceReplay(fn)
        for n in (9, 5, 9):
            args = (_randn(4, 64), torch.zeros(4, n, device="cuda"))
            self.assertEqual(entry(*args), fn(*args), atol=0, rtol=0)
        kinds = [type(c) for _, c in trace(fn, args).launches]
        self.assertEqual(kinds.count(EagerCall), 0)

    @parametrize("case", list(DECLINES))
    def test_decline_is_an_eager_call(self, case):
        x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
        y = torch.randn(4096, device="cuda", dtype=torch.float32)
        fn, args_fn, kinds = DECLINES[case]
        args = args_fn(x, y)
        tape = trace(fn, args)
        self.assertEqual([type(c) for _, c in tape.launches], kinds)
        entry = HostTraceReplay(fn)
        entry(*args)
        self.assertEqual(entry(*args), fn(*args), atol=0, rtol=0)

    @parametrize("h", [768, 8192])
    def test_rms_norm_under_a_native_override_traces_its_kernel(self, h):
        # the torch._native override runs under the trace: its CuTe DSL launch is on the tape
        from torch._native.registry import _aten_override_libs

        if ("_fused_rms_norm", "CUDA") not in _aten_override_libs:
            self.skipTest("no torch._native _fused_rms_norm override")
        fn = NORMS["rms_norm"][0](h)
        args = _norm_inputs("rms_norm", 64, h, torch.float16)
        self.assertEqual([type(c) for _, c in trace(fn, tuple(args)).launches], [KernelLaunch])

    def test_rms_norm_of_a_new_hidden_size_under_a_native_override_compiles_under_the_trace(self):
        # the override reads the hidden size as an int (a guard), so quack's
        # CuTe DSL compile of a new one at a later trace (its warm-up) succeeds
        import torch._vendor.quack.cache as quack_cache
        from torch._native.registry import _aten_override_libs

        if ("_fused_rms_norm", "CUDA") not in _aten_override_libs:
            self.skipTest("no torch._native _fused_rms_norm override")

        def fn(x, w):
            return F.rms_norm(x, x.shape[-1:], w, 1e-6)

        entry = HostTraceReplay(fn)
        with mock.patch.object(quack_cache, "CACHE_ENABLED", False):
            for m, h in ((64, 768), (64, 1280), (64, 1280), (7, 1280)):
                x, w = torch.randn(m, h, device="cuda", dtype=torch.float16), torch.randn(h, device="cuda", dtype=torch.float16)
                self.assertEqual(entry(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.replays, entry.eager), (2, 2, 0))

    @parametrize("native", [False, True])
    def test_rms_norm_of_a_symbolic_shape_replays(self, native):
        # rms_norm hands _fused_rms_norm (SymInt[] normalized_shape) the traced
        # symbolic size, which an eager call replays with
        from torch._native.registry import _aten_override_libs

        if native and ("_fused_rms_norm", "CUDA") not in _aten_override_libs:
            self.skipTest("no torch._native _fused_rms_norm override")

        def fn(x, w):
            return F.rms_norm(x, x.shape[-1:], w, 1e-6) * 2

        w = torch.randn(768, device="cuda", dtype=torch.float16)
        with contextlib.nullcontext() if native else _no_native_rms_norm():
            entry = HostTraceReplay(fn)
            for m in (64, 7, 300):
                x = torch.randn(m, 768, device="cuda", dtype=torch.float16)
                self.assertEqual(entry(x, w), fn(x, w), atol=0, rtol=0)
            self.assertEqual((entry.traces, entry.eager), (1, 0))

    def test_bmm_outer_product_under_a_native_override_traces_its_triton_kernel(self):
        # the override passes its read-only inputs to the launch in ConstTensorWrapper
        from torch._native.registry import _aten_override_libs

        if ("bmm", "CUDA") not in _aten_override_libs:
            self.skipTest("no torch._native bmm override")

        def make(m, n):
            return torch.randn(8, m, 1, device="cuda", dtype=torch.bfloat16), torch.randn(8, 1, n, device="cuda", dtype=torch.bfloat16)

        self.assertEqual([type(c) for _, c in trace(torch.bmm, make(64, 128)).launches], [KernelLaunch])
        entry = HostTraceReplay(torch.bmm)
        for m, n in ((64, 128), (64, 128), (32, 256)):
            a, b = make(m, n)
            self.assertEqual(entry(a, b), torch.bmm(a, b), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)

    def test_copy_between_arguments_checks_overlap_at_replay(self):
        # arguments disjoint at the trace share storage at the third call: the
        # variant's argument pair overlaps, so the call runs eagerly
        def pair(shared):
            buf = torch.arange(64 * 64 + 4, device="cuda", dtype=torch.float32)
            dst = buf[:4096] if shared else torch.zeros(4096, device="cuda")
            return dst.view(64, 64).t(), buf[4:].view(64, 64)

        def fn(dst, src):
            return dst.copy_(src)

        self.assertTrue(all(isinstance(c, KernelLaunch) for _, c in trace(fn, pair(False)).launches))
        entry = HostTraceReplay(fn)
        for _ in range(2):
            self.assertEqual(entry(*pair(False)), fn(*pair(False)), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)
        overlap = "refer to a single memory location"
        with self.assertRaisesRegex(RuntimeError, overlap):
            fn(*pair(True))
        with self.assertRaisesRegex(RuntimeError, overlap):
            entry(*pair(True))
        self.assertEqual(entry(*pair(False)), fn(*pair(False)), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.eager, len(entry.variants)), (1, 1, 1))

    def test_a_trusted_trace_records_no_argument_pairs(self):
        # the caller vouches for trusted inputs' aliasing: no replay overlap check
        def fn(dst, src):
            return dst.copy_(src)

        args = (torch.zeros(64, 64, device="cuda").t(), torch.randn(64, 64, device="cuda"))
        self.assertEqual(trace(fn, args).argument_pairs, ((0, 1),))
        trusted = TrustedInputs(layouts=tuple((tuple(a.shape), a.stride()) for a in args))
        self.assertEqual(trace(fn, args, trusted=trusted).argument_pairs, ())

    def test_inplace_between_arguments_partially_overlapping_at_replay(self):
        def fn(a, b):
            return a.add_(b) * b

        def pair(shift):
            buf = torch.randn(8192, device="cuda")
            return buf[:4096], (buf[shift : shift + 4096] if shift else torch.randn(4096, device="cuda"))

        entry = HostTraceReplay(fn)
        for _ in range(2):
            a, b = pair(0)
            self.assertEqual(entry(a.clone(), b), fn(a.clone(), b), atol=0, rtol=0)
        self.assertEqual(entry.eager, 0)
        overlap = "refer to a single memory location"
        with self.assertRaisesRegex(RuntimeError, overlap):
            fn(*pair(1024))
        with self.assertRaisesRegex(RuntimeError, overlap):
            entry(*pair(1024))
        # disjoint views of one storage hold the disjoint variant
        buf = torch.randn(8192, device="cuda")
        ref = buf.clone()
        self.assertEqual(entry(buf[:4096], buf[4096:]), fn(ref[:4096], ref[4096:]), atol=0, rtol=0)

    @parametrize("traced_alias", [True, False])
    def test_self_alias_at_trace_or_replay(self, traced_alias):
        # a variant traced at an overlap runs that step eagerly at every call;
        # one traced disjoint runs a call whose arguments overlap eagerly
        def fn(a, b):
            return a.copy_(b) * b

        def call(f, alias):
            a = torch.arange(8192, device="cuda", dtype=torch.float32)[::2]
            return f(a, a) if alias else f(a, torch.ones(8192, device="cuda")[::2])

        entry = HostTraceReplay(fn)
        for alias in (traced_alias, traced_alias, not traced_alias, not traced_alias, traced_alias):
            self.assertEqual(call(entry, alias), call(fn, alias), atol=0, rtol=0)
        steps = [[c.reason for _, c in v.captured.lowered.tape.launches if type(c) is EagerCall] for v in entry.variants]
        aliased = ["aten.copy_.default writes a storage another operand is of"]
        self.assertEqual((entry.traces, entry.eager, steps), (1, 0, [aliased]) if traced_alias else (1, 2, [[]]))

    def test_self_alias_decline_is_its_own_class(self):
        # a tape of only an eager step declines its call's class, which
        # includes which arguments overlap
        def fn(a, b):
            return a.copy_(b)

        def call(f, alias):
            a = torch.arange(8192, device="cuda", dtype=torch.float32)[::2]
            return f(a, a) if alias else f(a, torch.ones(8192, device="cuda")[::2])

        entry = HostTraceReplay(fn)
        for alias, eager in ((True, 1), (False, 1), (True, 2), (False, 2)):
            self.assertEqual(call(entry, alias), call(fn, alias), atol=0, rtol=0)
            self.assertEqual(entry.eager, eager)
        self.assertEqual((entry.traces, len(entry.variants)), (2, 1))

    def test_size_one_dim_strides_follow_eager(self):
        # x [1, h] has strides (1, 1): eager's output keeps stride 1 on the
        # size-1 dim, where the fake's is h
        def fn(x, r):
            return x.float() * r

        entry = HostTraceReplay(fn)
        for h in (64, 4096, 768):
            x = torch.randn(h, 1, device="cuda", dtype=torch.bfloat16).t()
            r = torch.ones(1, 1, device="cuda")
            out, ref = entry(x, r), fn(x, r)
            self.assertEqual(out, ref, atol=0, rtol=0)
            self.assertEqual(out.stride(), ref.stride())
        self.assertEqual(entry.eager, 0)
        self.assertTrue(all(isinstance(c, KernelLaunch) for _, c in trace(fn, (x, r)).launches))

    def test_softmax_brackets_redispatch_into_one_variant(self):
        # each softmax's dim bracket picks its kernel under its own guards:
        # another bracket dispatches that softmax again as an entry, with no
        # trace, and a call mixing brackets replays both entries
        def fn(x, y):
            return torch.softmax(x, -1), torch.softmax(y, -1)

        entry = HostTraceReplay(fn)
        for hx, hy in ((600, 600), (100, 600), (600, 100), (100, 100), (90, 700)):
            x = torch.randn(64, hx, device="cuda")
            y = torch.randn(7, hy, device="cuda")
            out, ref = entry(x, y), fn(x, y)
            self.assertEqual(out, ref, atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.redispatches, len(entry.variants), entry.eager), (1, 2, 1, 0))
        self.assertEqual((entry.fold_refusals, entry.redispatch_refusals), ({}, {}))

    def test_redispatched_entry_fails_at_no_other_call(self):
        # the transposed input's launches divide by a value that is 0 at the
        # first call's shape: its entry's rows fail at no other call
        def fn(x):
            return torch.var(x.float(), -1, keepdim=True)

        entry = HostTraceReplay(fn)
        for x in (torch.randn(2, 33), torch.randn(1024, 64).t(), torch.randn(2, 33)):
            x = x.to("cuda", torch.bfloat16)
            self.assertEqual(entry(x), fn(x), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.redispatches, len(entry.variants), entry.eager), (1, 1, 1, 0))

    def test_output_metadata_guards_are_the_graphs(self):
        # where's output size is its first operand's, or another's where the
        # first broadcasts: the op's metadata changed, so neither its dispatch
        # again nor a fold takes it, and it is another variant
        entry = HostTraceReplay(torch.where)
        for mc in (64, 1, 64, 1):
            c = torch.randn(mc, 256, device="cuda") > 0
            x, y = torch.randn(64, 256, device="cuda"), torch.randn(64, 256, device="cuda")
            self.assertEqual(entry(c, x, y), torch.where(c, x, y), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.folds, len(entry.variants), entry.eager), (2, 0, 2, 0))
        self.assertEqual(list(entry.redispatch_refusals), ["aten.where.self allocates otherwise"])
        self.assertEqual(list(entry.fold_refusals), ["aten.where.self returned other metadata"])

    def test_a_new_key_is_harvested_without_a_relower(self):
        # once a variant binds the matmul's keys, a key it does not bind is
        # harvested on the spot and added as the site's row: no trace, no relower
        # (the second trace's call is its warm-up, so its key is learned too)
        def fn(x, w):
            return F.silu(x @ w)

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        entry = HostTraceReplay(fn, opaque=(HarvestProvider(),))
        for m in (1, 2, 3, 4, 5, 3, 7, 9, 7):
            x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(entry(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.relowers, entry.learned, entry.eager), (2, 1, 3, 0))

    def test_fold_refuses_another_launch_chain(self):
        # layer_norm is one vectorized kernel at h % 4 == 0, else two
        fn = NORMS["layer_norm_no_affine"][0](None)
        entry = HostTraceReplay(fn)
        for h in (768, 767, 768, 767):
            self._assert_norm_replays_like_eager(entry, fn, _norm_inputs("layer_norm_no_affine", 64, h, torch.float32))
        self.assertEqual((entry.traces, entry.folds, len(entry.variants), entry.eager), (2, 0, 2, 0))
        self.assertEqual(list(entry.fold_refusals), ["another op sequence"])

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "requires flash attention")
    def test_default_sdpa_scale(self):
        # flash takes 1 / sqrt(head_dim) as a double guarded under the trace:
        # one trace per head_dim, replayed across sequence lengths
        entry = HostTraceReplay(F.scaled_dot_product_attention, opaque=(HarvestProvider(("attention",)),))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for d, s, traces in [(64, 128, 1), (64, 96, 1), (128, 200, 2), (128, 64, 2)]:
                q, k, v = (torch.randn(2, 8, s, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                out = entry(q, k, v)
                self.assertEqual(out, F.scaled_dot_product_attention(q, k, v), atol=0, rtol=0)
                self.assertEqual((entry.traces, entry.eager), (traces, 0))


instantiate_parametrized_tests(TestHostTraceAten)

if __name__ == "__main__":
    run_tests()
