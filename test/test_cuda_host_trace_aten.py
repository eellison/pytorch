# Owner(s): ["module: cuda graphs"]

import contextlib
import gc
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch.cuda._host_trace_capture import (
    capture_kernel_nodes,
    KernelNode,
    MemcpyNode,
    MemsetNode,
)
from torch.cuda._host_trace_harvest import HarvestProvider
from torch.cuda._host_trace_launch import KernelLaunch
from torch.cuda._host_trace_replay import HostTraceReplay
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


@contextlib.contextmanager
def _zero_init():
    # the harvest flag, as a harvest's captures set it
    previous = torch._C._cuda_hostTraceSetHarvesting(True)
    try:
        yield
    finally:
        torch._C._cuda_hostTraceSetHarvesting(previous)


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
    "empty_sum": (lambda x: x[:0].sum(-1), lambda x, y: (x,), [EagerCall]),
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
    if not node.name.startswith("_Z"):
        return set()
    if "vectorized" in node.name or "unrolled" in node.name:
        tail = {(1, b) for b in range(functor_bytes, len(node.images[1]))}
        if "vectorized" in node.name:
            return tail
        empty = {(p, 0) for p in range(3, len(node.images)) if len(node.images[p]) == 1}
        # LoadWithCast<n> / StoreWithCast<n>: n dtypes padded to 4, n sizes
        casts = {p: n for p in range(5, len(node.images)) for n in range(1, 9) if len(node.images[p]) == -(-n // 4) * 4 + 4 * n}
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

    def _assert_replays_eager_calls(self, entry, fn, args):
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
            self.assertEqual(replay_peak, torch.cuda.max_memory_allocated() - base)

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
        for h in (4096, 768):
            for m in (16, 7, 1, 5):
                self._assert_replays_eager_calls(entry, fn, args_fn(m, h, dtype))
        self.assertEqual(entry.eager, 0)

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("case", [c for c in HOSTS if not c.startswith(("batch_norm", "zeros", "causal")) and not c.endswith(("_0dim", "_empty"))])
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
        # CuTe DSL compile of a new one at a later trace succeeds
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
        self.assertEqual((entry.traces, entry.replays, entry.eager), (2, 3, 0))

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
        def fn(x, w):
            return F.silu(x @ w)

        w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
        entry = HostTraceReplay(fn, opaque=(HarvestProvider(),))
        for m in (1, 2, 3, 4, 5, 3, 7, 9, 7):
            x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(entry(x, w), fn(x, w), atol=0, rtol=0)
        self.assertEqual((entry.traces, entry.relowers, entry.learned, entry.eager), (2, 1, 2, 0))

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
