"""Symmetric-memory collectives under host tracing (torch/cuda/_host_trace.py).

The one-shot all-reduce of torch.distributed._symmetric_memory keeps no
per-call host state: the peer pointer table and the signal pads live in the
handle established at rendezvous (process-lifetime, constants of the launch),
the barrier is a self-resetting compare-and-swap toggle on the pads (every slot
is back at 0 when a call returns), and everything else the kernel reads is a
size, stride or offset of an argument. So the collective traces like any other
kernel: the converted host (CUDASymmetricMemoryOps.cu) reads sizes as SymInts
and launches through the typed helper.

What the entries below add is the rendezvous at trace time: a traced tensor
has no storage, and the handle is looked up by the real buffer's storage, so
the entry passes the real buffer behind the traced input. The host records the
handle's fields (peer pointer table, signal pads, rank, world size) as opaque
lookups over the input's base address, evaluated again at replay from the
caller's real buffer, exactly as eager looks the handle up on every call: a
different rendezvoused buffer serves through its own table, a buffer without a
rendezvous raises eager's error before any launch, and a different world size
misses (the kernel is instantiated for it). The symmetric buffer must be an
input of the traced function (a buffer allocated inside the host has no
rendezvous and declines).

Contract across ranks: every rank traces the same program with the same
shapes, so the guards, declines and variants are the same on every rank. A
decline is raised before any collective launch, on every rank that reaches it;
a rank-local decline (one rank passing a different shape or dtype) leaves the
other ranks in the eager collective the way a mismatched eager call would.
Two-shot and multimem all-reduces are not converted in this version and
decline by name.
"""

import torch
import torch.distributed._symmetric_memory as symm_mem
from torch.cuda._host_trace import (
    _active,
    _host_bindings,
    _real_input_of,
    _TracedTensor,
    Declined,
    register_traced_entry,
)


ops = torch.ops.symm_mem
_C = _host_bindings  # torch._C's bindings, the innermost op marked while one runs


def _symm_buffer(op, input):
    if not isinstance(input, torch.Tensor) or not input.is_cuda:
        raise Declined(f"host_trace: {op} on a non-CUDA input is not traced (declined)")
    if getattr(_active, "trace", None) is not None and not isinstance(
        input, _TracedTensor
    ):
        # a buffer captured by a closure or allocated inside the traced
        # function has no traced root: it would be hidden per-trace state
        raise Declined(
            f"host_trace: {op}: the symmetric buffer is not an input of the "
            "traced function (captured by a closure or allocated inside it); "
            "pass it as an input (declined)"
        )
    real = _real_input_of(input)
    if real is None:
        raise Declined(
            f"host_trace: {op}: the symmetric buffer must be an input of the traced "
            "function (a buffer allocated inside the host has no rendezvous) (declined)"
        )
    if not symm_mem.is_symm_mem_tensor(real):
        raise Declined(
            f"host_trace: {op}: input must be allocated with symm_mem.empty (declined)"
        )
    return real


def _check_rank_uniform_guards(op, symm_input, env, before):
    """Serve or miss at replay must be a function of rank-shared state only,
    or ranks could decide differently and one of them would wait in the
    kernel's cross-rank barrier for a peer that never launched. Shapes,
    dtypes, the world size and the rank are shared by the contract; the
    storage offset and the base address of a rank-local tensor are not. The
    symmetric buffer's offsets are rank-identical by construction. So a guard
    the host recorded during the collective that mentions the offset or base
    of a non-symmetric input declines by name, statically, with no exchange."""
    symm_root = getattr(symm_input, "_root", None)
    symm_arg = symm_root.name.replace("p", "arg", 1) + "." if symm_root else None
    for g in env.guards[before:]:
        for sym in g.expr.free_symbols:
            for src in env.var_to_sources.get(sym, ()):
                name = src.name if isinstance(src.name, str) else str(src)
                if not name.endswith((".storage_offset()", ".base")):
                    continue
                if symm_arg is not None and name.startswith(symm_arg):
                    continue
                raise Declined(
                    f"host_trace: {op}: collective tape depends on a rank-local fact: "
                    f"{g.expr} ({name}); pass a value computed in the traced function "
                    "or the symmetric buffer (declined)"
                )


def _with_rank_uniform_guards(op, symm_input, call):
    tr = getattr(_active, "trace", None)
    if tr is None:
        return call()
    before = len(tr.shape_env.guards)
    out = call()
    _check_rank_uniform_guards(op, symm_input, tr.shape_env, before)
    return out


def _one_shot_all_reduce_out(input, reduce_op, group_name, out):
    real = _symm_buffer(ops.one_shot_all_reduce_out.default, input)
    return _with_rank_uniform_guards(
        ops.one_shot_all_reduce_out.default,
        input,
        lambda: _C._host_trace_symm_one_shot_all_reduce_out(
            input, real, None, reduce_op, group_name, out
        ),
    )


def _one_shot_all_reduce(input, reduce_op, group_name):
    out = torch.empty_like(input)
    return _one_shot_all_reduce_out(input, reduce_op, group_name, out)


def _one_shot_all_reduce_copy_out(input, local_input, reduce_op, group_name, out):
    real = _symm_buffer(ops.one_shot_all_reduce_copy_out.default, input)
    if not (isinstance(local_input, torch.Tensor) and local_input.is_cuda):
        raise Declined(
            "host_trace: one_shot_all_reduce_copy with a non-CUDA local input (declined)"
        )
    return _with_rank_uniform_guards(
        ops.one_shot_all_reduce_copy_out.default,
        input,
        lambda: _C._host_trace_symm_one_shot_all_reduce_out(
            input, real, local_input, reduce_op, group_name, out
        ),
    )


def _one_shot_all_reduce_copy(input, local_input, reduce_op, group_name):
    out = torch.empty_like(local_input)
    return _one_shot_all_reduce_copy_out(input, local_input, reduce_op, group_name, out)


register_traced_entry(ops.one_shot_all_reduce.default, _one_shot_all_reduce)
register_traced_entry(ops.one_shot_all_reduce_out.default, _one_shot_all_reduce_out)
register_traced_entry(ops.one_shot_all_reduce_copy.default, _one_shot_all_reduce_copy)
register_traced_entry(
    ops.one_shot_all_reduce_copy_out.default, _one_shot_all_reduce_copy_out
)
