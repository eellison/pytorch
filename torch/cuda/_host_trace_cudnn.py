"""cuDNN attention under a host trace: closed regions.

On sm90 and up eager's SDPA selector gives a masked bf16 / fp16 call at head
dims 64 and 128 to aten._scaled_dot_product_cudnn_attention (sdp_utils.cpp
check_prefer_cudnn_attention), whose host runs a cuDNN frontend graph
(aten/src/ATen/native/cudnn/MHA.cpp): the plan is built by heuristics (HeurMode
A, no autotuning) from the dims, strides and dtypes of q, k, v, the bias and the
output, the causal flag, the dropout probability and whether the softmax stats
are asked for, cached per thread under that key (MHAParams), and executed with
the pointers of the call, a pass-by-value scale and a workspace the host
allocates per call through the caching allocator. Its kernels are cuDNN's own
(cudnn_generated_fort_native_sdpa_*, and the backward's dot_do_o and dq
convert). Under a trace such a call is a closed region, the treatment cuBLAS
gets (E22: the empirical slot classification is opt-in per library; this
module is the list's entry for cuDNN attention):

  - the region records the call's operands (query, key, value, the attention
    bias as the host expands it) and the outputs the host allocates in order
    (the attention, the logsumexp when asked; the backward's dq, dk, dv) as
    values, and issues nothing; the seed and offset words the forward returns
    are allocations of the tape nobody writes without dropout;
  - the harvest (torch/cuda/_host_trace.py _harvest) runs the op on stand-ins
    at the region's key: every operand's dtype, sizes, strides and address
    alignment class (MHAParams keys on the dims and strides; the plan sees no
    pointer, so a class only selects a template), the scalars (the softmax
    stats flag, the causal flag, the scale when the caller gives one, a
    pass-by-value float of the kernel image; None keeps 1 / sqrt(d) a function
    of the head dim in the key) and the device identity. The template's slots
    are the operands, its returns and its workspace (the call's own
    allocations, the replay's scratch); the harvest measured no per-call host
    state and no stream-dependent word in any image.

What eager would not run as one closed call declines by name: dropout (the
generator advance and the seed / offset unpack kernel ahead of the graph),
the ragged dense path (TORCH_CUDNN_SDPA_AVOID_RECOMPILE: seqlen kernels ahead
of the graph), the varlen / nested path, a grad_out the backward copies into
the output's layout first. cuDNN faults (CUDA misaligned address, a sticky
error) on a q / k / v base not 16-byte aligned and on a bias not 4-byte
aligned: guards on the address, declined by name where eager would fault.
"""

from __future__ import annotations

import contextlib
import functools
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


aten = torch.ops.aten

# the alignment cuDNN's kernels need of each operand's base (measured on this
# box, cuDNN 9.16: q / k / v at 8 bytes and a bias at 2 bytes fault)
_QKV_ALIGN = 16
_BIAS_ALIGN = 4
_DTYPES = (torch.float16, torch.bfloat16)
_BIAS_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _host_trace() -> Any:
    from torch.cuda import _host_trace

    return _host_trace


@dataclass(frozen=True)
class LibraryRegion:
    """One closed library op on the region list."""

    # the region's op tag (the harvest's key with the scalars)
    op: str
    # (trace, func, args, kwargs) -> (inputs, scalars, outputs, result): the
    # (name, traced tensor) operands, the call's non-tensor arguments, the
    # traced allocations the host returns in order, and what the op returns
    describe: Callable[..., tuple]
    # (op, scalars, tensors) -> the returned outputs, the call on stand-ins
    run: Callable[..., Any]
    # how many outputs the call allocates and returns (an int, or a function
    # of the scalars)
    outputs: Any


def _arguments(func: Any, args: tuple, kwargs: dict) -> list:
    # the call's arguments in schema order
    values = list(args)
    for a in func._schema.arguments[len(args) :]:
        values.append(kwargs.get(a.name, a.default_value))
    return values


def _address(t: Any) -> Any:
    # the data address as a value: the root's address symbol plus the view's
    # offset, as the C++ hosts read it through sym_const_data_ptr
    return t._root.sym + t._sym_offset * t.element_size()


def _traced(func: Any, name: str, t: Any) -> None:
    ht = _host_trace()
    if not isinstance(t, ht._TracedTensor):
        raise ht.Declined(
            f"host_trace: {func}: {name} is a tensor the trace did not create; a closed "
            "region's operands must be inputs, allocations or views of them (declined)"
        )
    if t.device.type != "cuda":
        raise ht.Declined(
            f"host_trace: {func}: {name} on {t.device} inside a trace (declined)"
        )


def _kernel_choice() -> Any:
    # the alignment guards decide a template of the same call (plan item 38
    # stage 0 marks such guards where the binding exists)
    choice = getattr(torch._C, "_HostTraceKernelChoice", None)
    return choice() if choice is not None else contextlib.nullcontext()


def _aligned(func: Any, name: str, t: Any, n: int) -> None:
    ht = _host_trace()
    with _kernel_choice():
        ok = bool(_address(t) % n == 0)
    if not ok:
        raise ht.Declined(
            f"host_trace: {func}: {name}'s base is not {n}-byte aligned, on which cuDNN "
            "attention faults (CUDA misaligned address); the ordinary host would too (declined)"
        )


def _ragged_dense() -> bool:
    # MHA.cpp use_ragged_in_dense: a static read of the environment at first use
    return os.environ.get("TORCH_CUDNN_SDPA_AVOID_RECOMPILE", "").strip().lower() in (
        "1",
        "y",
        "yes",
        "true",
        "on",
    )


def _scale(scale: Any) -> Any:
    # None: the host computes 1 / sqrt(d) itself, at the harvest as at the
    # call, so the key's head dim carries it; a value is a float of the kernel
    # image, a symbolic one pinned as eager's float
    return None if scale is None else float(scale)


def _expanded_bias(tr: Any, func: Any, bias: Any, b: Any, s_q: Any, s_kv: Any) -> Any:
    # _cudnn_attention_forward_impl / _cudnn_attention_backward: a 2-D or 3-D
    # bias expands to (B, 1, S_q, S_kv), a 4-D one keeps its head dim; the
    # expanded view is what MHAParams keys on (its dims and strides) and what
    # the kernel reads
    _traced(func, "attn_bias", bias)
    if bias.dtype not in _BIAS_DTYPES:
        raise RuntimeError(
            f"cuDNN SDPA got attn_bias of unsupported dtype {bias.dtype}, expected one of "
            "float, half, bfloat16."
        )
    if bias.dim() in (2, 3):
        shape = [b, 1, s_q, s_kv]
    elif bias.dim() == 4:
        shape = [b, bias.shape[1], s_q, s_kv]
    else:
        raise RuntimeError(
            f"cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got {bias.dim()}D"
        )
    expanded = tr.view(aten.expand.default, (bias, shape), {})
    _aligned(func, "attn_bias", expanded, _BIAS_ALIGN)
    return expanded


def _check_qkv(func: Any, query: Any, key: Any, value: Any) -> None:
    for name, t in (("query", query), ("key", key), ("value", value)):
        _traced(func, name, t)
    if query.dtype not in _DTYPES:
        raise RuntimeError(
            f"cuDNN attention only supports float16 and bfloat16, got {query.dtype}"
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError(
            "cuDNN attention expects query, key and value to have the same dtype, got "
            f"{query.dtype}, {key.dtype} and {value.dtype}"
        )
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise RuntimeError(
            "Q tensor has unexpected number of dims, please report a bug to PyTorch."
        )
    for name, t in (("query", query), ("key", key), ("value", value)):
        _aligned(func, name, t, _QKV_ALIGN)


def _closed_call_only(func: Any, dropout_p: Any) -> None:
    ht = _host_trace()
    if not isinstance(dropout_p, (int, float)) or dropout_p != 0.0:
        raise ht.Declined(
            f"host_trace: {func} with dropout advances the generator and unpacks the seed / "
            "offset with a kernel ahead of cuDNN's graph: not one closed call (declined)"
        )
    if _ragged_dense():
        raise ht.Declined(
            f"host_trace: {func} under TORCH_CUDNN_SDPA_AVOID_RECOMPILE runs seqlen kernels "
            "ahead of cuDNN's graph (the ragged dense path): not one closed call (declined)"
        )


def _matching_layout(tr: Any, q: Any, shape: list) -> Any:
    # sdp_utils.h alloc_with_matching_layout: the requested shape equal to q's
    # allocates empty_like(q); else the strides follow q's stride order (a
    # zero stride sorts last), each comparison a guard
    ht = _host_trace()
    if ht._guard_each([a == b for a, b in zip(q.shape, shape)]):
        return tr.allocate(aten.empty_like.default, (q,), {})
    strides = list(q._sym_strides)
    zero = [bool(s == 0) for s in strides]

    def less(i: int, j: int) -> int:
        if zero[i]:
            return 0
        if zero[j]:
            return -1
        return -1 if bool(strides[i] < strides[j]) else 0

    order = sorted(range(len(shape)), key=functools.cmp_to_key(less))
    out_strides: list = [None] * len(shape)
    current: Any = 1
    for d in order:
        out_strides[d] = current
        current = current * shape[d]
    return tr.allocate(
        aten.empty_strided.default,
        (shape, out_strides),
        {"dtype": q.dtype, "device": tr.device},
    )


def _scalar_word(tr: Any) -> Any:
    # at::empty({}, kLong) for the philox seed / offset the forward returns
    return tr.allocate(
        aten.empty.memory_format, ([],), {"dtype": torch.int64, "device": tr.device}
    )


def _sdpa_describe(tr: Any, func: Any, args: tuple, kwargs: dict) -> tuple:
    # attention.cu _cudnn_attention_forward_impl and MHA.cpp run_cudnn_SDP_fprop
    (
        query,
        key,
        value,
        attn_bias,
        compute_log_sumexp,
        dropout_p,
        is_causal,
        _return_debug_mask,
        scale,
    ) = _arguments(func, args, kwargs)
    _check_qkv(func, query, key, value)
    _closed_call_only(func, dropout_p)
    b, h, s_q, _d_qk = query.shape
    s_kv = key.shape[2]
    d_v = value.shape[3]
    inputs = [("query", query), ("key", key), ("value", value)]
    if attn_bias is not None:
        inputs.append(("attn_bias", _expanded_bias(tr, func, attn_bias, b, s_q, s_kv)))
    # eager's allocation order: the seed and offset words, the attention in
    # q's layout, the softmax stats (B, H, S_q, 1) in float32 when asked
    seed, offset = _scalar_word(tr), _scalar_word(tr)
    out = _matching_layout(tr, query, [b, h, s_q, d_v])
    outputs = [("output", out)]
    lse = None
    if compute_log_sumexp:
        lse = tr.allocate(
            aten.empty.memory_format,
            ([b, h, s_q, 1],),
            {"dtype": torch.float32, "device": tr.device},
        )
        outputs.append(("logsumexp", lse))
    scalars = (
        bool(compute_log_sumexp),
        bool(is_causal),
        _scale(scale),
        attn_bias is not None,
    )
    result = (out, lse, None, None, s_q, s_kv, seed, offset, None)
    return inputs, scalars, outputs, result


def _sdpa_run(op: str, scalars: tuple, tensors: list) -> Any:
    compute_log_sumexp, is_causal, scale, has_bias = scalars
    q, k, v = tensors[:3]
    bias = tensors[3] if has_bias else None
    res = aten._scaled_dot_product_cudnn_attention(
        q, k, v, bias, compute_log_sumexp, 0.0, is_causal, False, scale=scale
    )
    return (res[0], res[1]) if compute_log_sumexp else (res[0],)


def _sdpa_outputs(scalars: tuple) -> int:
    return 1 + bool(scalars[0])


@functools.cache
def _dead_words(device: torch.device) -> tuple:
    # the seed / offset arguments of a backward without dropout: never read
    return tuple(torch.empty((), dtype=torch.int64, device=device) for _ in range(2))


def _sdpa_backward_describe(tr: Any, func: Any, args: tuple, kwargs: dict) -> tuple:
    # attention_backward.cu _cudnn_attention_backward and MHA.cpp run_cudnn_SDP_bprop
    (
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        _philox_seed,
        _philox_offset,
        attn_bias,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        scale,
    ) = _arguments(func, args, kwargs)
    ht = _host_trace()
    if cum_seq_q is not None or cum_seq_k is not None:
        raise ht.Declined(
            f"host_trace: {func} on the varlen (nested) path is not recorded as a closed "
            "region (declined)"
        )
    _check_qkv(func, query, key, value)
    _closed_call_only(func, dropout_p)
    for name, t in (("grad_out", grad_out), ("out", out), ("logsumexp", logsumexp)):
        _traced(func, name, t)
        _aligned(func, name, t, _QKV_ALIGN)
    b, _h, s_q, _d_qk = query.shape
    s_kv = key.shape[2]
    # the host takes the lengths the caller passes as the plan's S_q / S_kv:
    # the dense autograd passes the sizes
    if not (bool(max_q == s_q) and bool(max_k == s_kv)):
        raise ht.Declined(
            f"host_trace: {func} with max_q / max_k other than the query and key lengths is "
            "not recorded as a closed region (declined)"
        )
    # run_cudnn_SDP_bprop copies a grad_out whose innermost stride is not 1
    # into the output's layout first (permute_to_matching_layout): a launch
    # of the host's own ahead of the region
    if not bool(grad_out._sym_strides[-1] == 1):
        raise ht.Declined(
            f"host_trace: {func}: the host copies a grad_out whose innermost stride is not 1 "
            "into the output's layout ahead of cuDNN's graph: not one closed call (declined)"
        )
    inputs = [
        ("grad_out", grad_out),
        ("query", query),
        ("key", key),
        ("value", value),
        ("out", out),
        ("logsumexp", logsumexp),
    ]
    if attn_bias is not None:
        inputs.append(("attn_bias", _expanded_bias(tr, func, attn_bias, b, s_q, s_kv)))
    # empty_like(query), empty_like(key), empty_like(value), in this order
    dq = tr.allocate(aten.empty_like.default, (query,), {})
    dk = tr.allocate(aten.empty_like.default, (key,), {})
    dv = tr.allocate(aten.empty_like.default, (value,), {})
    outputs = [("grad_query", dq), ("grad_key", dk), ("grad_value", dv)]
    scalars = (bool(is_causal), _scale(scale), attn_bias is not None)
    return inputs, scalars, outputs, (dq, dk, dv)


def _sdpa_backward_run(op: str, scalars: tuple, tensors: list) -> Any:
    is_causal, scale, has_bias = scalars
    grad_out, q, k, v, out, lse = tensors[:6]
    bias = tensors[6] if has_bias else None
    seed, offset = _dead_words(q.device)
    return aten._scaled_dot_product_cudnn_attention_backward(
        grad_out,
        q,
        k,
        v,
        out,
        lse,
        seed,
        offset,
        bias,
        None,
        None,
        q.shape[2],
        k.shape[2],
        0.0,
        is_causal,
        scale=scale,
    )


REGIONS: dict[Any, LibraryRegion] = {
    aten._scaled_dot_product_cudnn_attention.default: LibraryRegion(
        "cudnn_sdpa", _sdpa_describe, _sdpa_run, _sdpa_outputs
    ),
    aten._scaled_dot_product_cudnn_attention_backward.default: LibraryRegion(
        "cudnn_sdpa_backward", _sdpa_backward_describe, _sdpa_backward_run, 3
    ),
}


def closed_calls() -> dict:
    """The harvest's calls for the ops on the list, by op tag."""
    ht = _host_trace()
    return {r.op: ht._ClosedCall(r.run, r.run, r.outputs) for r in REGIONS.values()}
