"""torch._native's CuTe overrides under a host trace: closed regions.

Eager serves some ops through torch._native's Python overrides (an eager
router at the op's CUDA key runs the first active override whose condition
holds: torch/_native/registry.py), and the CuTe ones launch DSL programs of
their own (the vendored QuACK RMSNorm behind _fused_rms_norm, topk's radix
and register kernels, scatter_add's TMA and vector kernels). Under a trace
such a call is a closed region, the treatment cuBLAS gets (E22: the
empirical slot classification is opt-in per library; this module is the
list's second entry):

  - the override's condition, evaluated on the traced tensors as the router
    would (every comparison a guard; a pointer's alignment through the
    address symbol; the copy-on-write state from the root), decides the
    route (E40: the conditions are symbolic-clean; scatter_add's runs
    TensorIterator's reordering and coalescing in Python);
  - the region records the op's operands and the outputs the override
    allocates, in its order, and issues nothing;
  - the harvest (torch/cuda/_host_trace.py _harvest) runs the op through the
    router on stand-ins at the region's key (sizes, strides, dtypes, address
    alignment classes), so the template holds eager's own route at that key,
    the override's kernel or the ATen fallback, byte for byte; the runtime's
    template rows lower it, a new key harvesting on first sight.

Per op the entry says how the trace describes the call (what the override
hands to its kernel as is and what it allocates, in order) and how the
harvest makes the call. What the override would copy first (a non-contiguous
or misaligned operand) declines by name: that copy is an allocation and a
kernel of the call's own ahead of its outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


aten = torch.ops.aten


def _host_trace() -> Any:
    from torch.cuda import _host_trace

    return _host_trace


@dataclass(frozen=True)
class NativeRegion:
    """One override on the closed-region list."""

    # the region's op tag (the harvest's key with the scalars)
    op: str
    # (trace, func, args, kwargs) -> (inputs, scalars, outputs, result): the
    # (name, traced tensor) operands, the call's non-tensor arguments, the
    # traced allocations the override makes in order, and what the op returns
    describe: Callable[..., tuple]
    # (op, scalars, tensors) -> the outputs, through the router
    run: Callable[..., Any]
    # how many outputs the override allocates and returns (an int, or a
    # function of the scalars)
    outputs: Any
    # modules of the CuTe DSL programs the override launches (the warm-up's
    # observations of them are the region's)
    programs: tuple = ()


def _arguments(func: Any, args: tuple, kwargs: dict) -> list:
    # the call's arguments in schema order
    values = list(args)
    for a in func._schema.arguments[len(args) :]:
        values.append(kwargs.get(a.name, a.default_value))
    return values


def _traced(func: Any, name: str, t: Any) -> None:
    ht = _host_trace()
    if not isinstance(t, ht._TracedTensor):
        raise ht.Declined(
            f"host_trace: {func}: {name} is a tensor the trace did not create; a closed "
            "region's operands must be inputs, allocations or views of them (declined)"
        )


def _as_is(func: Any, name: str, t: Any, align: int) -> None:
    # what the override hands to its kernel unchanged (torch/_native/ops/norm/
    # norms.py _reshape_2d and _aligned_weight: contiguous, the base aligned
    # to the vector width); anything else it copies first, a launch of its
    # own ahead of the region. Each test is a guard
    ht = _host_trace()
    if not t.is_contiguous():
        raise ht.Declined(
            f"host_trace: {func}: torch._native's override copies the non-contiguous {name} "
            "before its kernel, a launch of its own ahead of the region (declined)"
        )
    if not bool(t.data_ptr() % align == 0):
        raise ht.Declined(
            f"host_trace: {func}: torch._native's override copies {name}, whose base is not "
            f"{align}-byte aligned, before its kernel, a launch of its own ahead of the "
            "region (declined)"
        )


def _vector_align(t: Any, n: int) -> int:
    # norms.py _required_align_bytes: gcd(N, 128 // dtype_bits) elements
    itemsize = t.element_size()
    return math.gcd(n, 128 // (itemsize * 8)) * itemsize


def _rms_norm_describe(tr: Any, func: Any, args: tuple, kwargs: dict) -> tuple:
    # norms.py quack_rmsnorm_fwd: x as is (M, N), the weight as is (N,), then
    # out = empty_like(x) and rstd = empty(M, float32), the kernel, and the
    # returns as views of the input's shape and the statistics' shape
    input, normalized_shape, weight, eps = _arguments(func, args, kwargs)
    _traced(func, "input", input)
    shape = tuple(int(v) for v in normalized_shape)
    n = math.prod(shape)
    _as_is(func, "input", input, _vector_align(input, n))
    if weight is not None:
        _traced(func, "weight", weight)
        _as_is(func, "weight", weight, _vector_align(weight, n))
    lead = list(input.shape[: input.dim() - len(shape)])
    m = math.prod(lead) if lead else 1
    out = tr.allocate(
        aten.empty.memory_format, ([m, n],), {"dtype": input.dtype, "device": tr.device}
    )
    rstd = tr.allocate(
        aten.empty.memory_format, ([m],), {"dtype": torch.float32, "device": tr.device}
    )
    inputs = [("input", input)] + ([("weight", weight)] if weight is not None else [])
    result = (
        tr.reshape_view(out, list(input.shape)),
        tr.reshape_view(rstd, lead + [1] * len(shape)),
    )
    return inputs, (shape, eps), [("out", out), ("rstd", rstd)], result


def _rms_norm_run(op: str, scalars: tuple, tensors: list) -> Any:
    shape, eps = scalars
    weight = tensors[1] if len(tensors) > 1 else None
    return aten._fused_rms_norm(tensors[0], list(shape), weight, eps)


def _rms_norm_backward_describe(tr: Any, func: Any, args: tuple, kwargs: dict) -> tuple:
    # norms.py quack_rmsnorm_bwd: x and dout as is (M, N), rstd as is (M,),
    # the weight as is (N,); dx = empty_like(x), dw_partial = empty(sm_count,
    # N, float32) (scratch), the kernel, then dw = dw_partial.sum(0)
    # .to(weight.dtype): ATen's reduction into a float32 (N,), the returned
    # dw for a float32 weight, else a cast into a (N,) of the weight's dtype
    # (the reduction's is scratch); the returns are the region's outputs
    grad_out, input, normalized_shape, rstd, weight, mask = _arguments(
        func, args, kwargs
    )
    for name, t in (("grad_out", grad_out), ("input", input), ("rstd", rstd)):
        _traced(func, name, t)
    shape = tuple(int(v) for v in normalized_shape)
    n = math.prod(shape)
    mask = tuple(bool(v) for v in mask)
    ht = _host_trace()
    if not mask[0]:
        raise ht.Declined(
            f"host_trace: {func} without grad_input: torch._native's override computes a "
            "grad_input it does not return, an allocation the tape cannot place (declined)"
        )
    _as_is(func, "grad_out", grad_out, _vector_align(grad_out, n))
    _as_is(func, "input", input, _vector_align(input, n))
    if not rstd.is_contiguous():
        raise ht.Declined(
            f"host_trace: {func}: torch._native's override copies the non-contiguous rstd "
            "before its kernel, a launch of its own ahead of the region (declined)"
        )
    inputs = [("grad_out", grad_out), ("input", input), ("rstd", rstd)]
    if weight is not None:
        _traced(func, "weight", weight)
        _as_is(func, "weight", weight, _vector_align(weight, n))
        inputs.append(("weight", weight))
    lead = list(input.shape[: input.dim() - len(shape)])
    m = math.prod(lead) if lead else 1
    dx = tr.allocate(
        aten.empty.memory_format, ([m, n],), {"dtype": input.dtype, "device": tr.device}
    )
    outputs = [("grad_input", dx)]
    grad_weight = None
    if weight is not None and mask[1]:
        sizes = [n] if weight.dtype == torch.float32 else list(shape)
        dw = tr.allocate(
            aten.empty.memory_format,
            (sizes,),
            {"dtype": weight.dtype, "device": tr.device},
        )
        outputs.append(("grad_weight", dw))
        grad_weight = tr.reshape_view(dw, list(shape))
    scalars = (shape, mask, weight is not None)
    return (
        inputs,
        scalars,
        outputs,
        (tr.reshape_view(dx, list(input.shape)), grad_weight),
    )


def _rms_norm_backward_run(op: str, scalars: tuple, tensors: list) -> Any:
    shape, mask, has_weight = scalars
    grad_out, input, rstd = tensors[:3]
    weight = tensors[3] if has_weight else None
    result = aten._fused_rms_norm_backward(
        grad_out, input, list(shape), rstd, weight, list(mask)
    )
    return tuple(t for t in result if t is not None)


def _rms_norm_backward_outputs(scalars: tuple) -> int:
    _shape, mask, has_weight = scalars
    return 1 + (has_weight and mask[1])


def _topk_describe(tr: Any, func: Any, args: tuple, kwargs: dict) -> tuple:
    # torch/_native/ops/topk/cutedsl_impl.py _run: self viewed as (M, N) (the
    # condition took a contiguous input reduced over its last axis), then
    # values = empty(M, k, self.dtype) and indices = empty(M, k, int64), the
    # kernel, and the returns viewed as self.shape[:-1] + (k,)
    self, k, dim, largest, sorted_ = _arguments(func, args, kwargs)
    _traced(func, "self", self)
    lead = list(self.shape[:-1])
    m = math.prod(lead) if lead else 1
    values = tr.allocate(
        aten.empty.memory_format, ([m, k],), {"dtype": self.dtype, "device": tr.device}
    )
    indices = tr.allocate(
        aten.empty.memory_format, ([m, k],), {"dtype": torch.int64, "device": tr.device}
    )
    result = (
        tr.reshape_view(values, lead + [k]),
        tr.reshape_view(indices, lead + [k]),
    )
    scalars = (int(k), int(dim), bool(largest), bool(sorted_))
    return [("self", self)], scalars, [("values", values), ("indices", indices)], result


def _topk_run(op: str, scalars: tuple, tensors: list) -> Any:
    k, dim, largest, sorted_ = scalars
    return aten.topk(tensors[0], k, dim, largest, sorted_)


def _scatter_add_describe(tr: Any, func: Any, args: tuple, kwargs: dict) -> tuple:
    # torch/_native/ops/scatter_add/cutedsl_impl.py: the in-place override
    # scatters src into self through the views the condition's TensorIterator
    # analysis fixed (one kernel); the functional one runs the same on
    # self.clone(), a device memcpy the tape records as its own copy ahead of
    # the region. The written operand is the region's output, handed in at
    # the harvest (the call allocates nothing it returns)
    self, dim, index, src = _arguments(func, args, kwargs)
    for name, t in (("self", self), ("index", index), ("src", src)):
        _traced(func, name, t)
    if func is aten.scatter_add_.default:
        out = self
    else:
        with tr.mode:
            out = self.clone()
    return [("index", index), ("src", src)], (int(dim),), [("self", out)], out


def _scatter_add_run(op: str, scalars: tuple, tensors: list) -> Any:
    (dim,) = scalars
    index, src, self = tensors
    return aten.scatter_add_(self, dim, index, src)


REGIONS: dict[Any, NativeRegion] = {
    aten._fused_rms_norm.default: NativeRegion(
        "_fused_rms_norm",
        _rms_norm_describe,
        _rms_norm_run,
        2,
        programs=("torch._vendor.quack.rmsnorm",),
    ),
    aten._fused_rms_norm_backward.default: NativeRegion(
        "_fused_rms_norm_backward",
        _rms_norm_backward_describe,
        _rms_norm_backward_run,
        _rms_norm_backward_outputs,
        programs=("torch._vendor.quack.rmsnorm",),
    ),
    aten.topk.default: NativeRegion(
        "topk", _topk_describe, _topk_run, 2, programs=("torch._native.ops.topk.",)
    ),
    aten.scatter_add.default: NativeRegion(
        "scatter_add_",
        _scatter_add_describe,
        _scatter_add_run,
        0,
        programs=("torch._native.ops.scatter_add.",),
    ),
    aten.scatter_add_.default: NativeRegion(
        "scatter_add_",
        _scatter_add_describe,
        _scatter_add_run,
        0,
        programs=("torch._native.ops.scatter_add.",),
    ),
}


def closed_calls() -> dict:
    """The harvest's calls for the overrides on the list, by op tag."""
    ht = _host_trace()
    return {
        r.op: ht._ClosedCall(r.run, r.run, r.outputs, padding=True)
        for r in REGIONS.values()
    }
