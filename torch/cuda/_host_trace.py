"""Host tracing for CUDA C++ kernels (private).

A traceable kernel host runs once with symbolic sizes and no data pointers, and
the recorder writes down what it did: the allocations it made, the branches it
took, and every kernel launch with each argument as a value over the input
shapes. Replaying that tape at a new shape evaluates those values, patches the
captured CUDA graph and launches it, without running the host again. A call
whose inputs fail a recorded branch is a miss, never a wrong answer.

    tape = trace(torch.layer_norm, (x, (x.shape[-1],), w, b, 1e-5))

A consumer prepares the replay from the tape (the CUDA-graph runtime's
host-trace adapter, torch/_inductor/runtime/_cudagraph/direct_hosttrace.py,
serves calls through Entry below); this module records and never replays.

The symbolic backend. `symbolic = "sympy"` (the default) backs the values
with torch's ShapeEnv; `symbolic = "ir"` backs them with the integer
expression IR of _host_trace_ir (the same guards from the same host reads,
interned nodes in place of sympy expressions, an operation the IR cannot
express declines by name with a census). An IR tape is exported to sympy as
it is built, so the tape's consumers see one form; `Tape.symbolic` records
which backend traced it.

The symbolic values are ordinary torch.SymInt / SymFloat / SymBool over a
ShapeEnv created for the trace; the C++ host sees them as c10::SymInt and every
branch it takes lands in the ShapeEnv's guards. A traced tensor is a wrapper
subclass on the CUDA device with symbolic sizes and no storage: its raw pointer
raises, its address is read in C++ through sym_const_data_ptr (inputs; in
ordinary mode const_data_ptr, so a copy-on-write input stays lazy) or
sym_mutable_data_ptr (outputs and in-place operands). Views made inside the
host are stride arithmetic over the same root (below), FakeTensor twins over
the same ShapeEnv where that does not cover them.

Only the CUDA hosts that were converted to the recorder's types can be traced
(_TRACEABLE); everything else declines by name. Traced and replayed outputs
carry no requires_grad or grad_fn. As with any CUDA graph capture, no thread
may call torch.cuda.synchronize() while a trace is in progress.

The host contract. A traceable host performs no synchronous memory API call
(cudaMemcpy, cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaMemset), no device
or stream synchronize and no cudaMalloc. Asynchronous copies and memsets on
the current stream are fine: they become nodes the completeness rule sees.
The trace is complete with respect to everything the capture sees; the
synchronous memory calls are invisible to a stream capture in every mode on
CUDA 13.0 (measured), so they are a contract violation, caught by the
HOSTTRACE_SYNC_API lint on any translation unit that includes the host_trace
headers, and a host that makes one is already wrong under a plain CUDA graph
capture. The other forbidden calls fail inside the thread-local capture and
the trace declines by the CUDA error's name.

trace() runs the function once on the real inputs before the symbolic run
(warm_up=True): one-time initializations (a lazily loaded module, cuBLAS's
constant upload, an autotune or kernel-image cache) complete there and never
inside the trace, the same way cudagraph trees warm up before capturing.

Views made inside the host or on its outputs are computed as stride arithmetic
over the same root (view, transpose, permute, unsqueeze, squeeze, expand,
select, slice, narrow); a view whose output shape depends on a size being 1
(squeeze, a broadcasting expand) guards on the traced value, the others
record nothing; view and _unsafe_view take at::detail::computeStride's
strides, each of its decisions a guard. split and split_with_sizes (and
chunk, a composite over split) are the sequence of narrows eager makes, the
piece count a guard. What is not covered that way takes the FakeTensor twins.
unbind and diagonal decline in this version: use select / as_strided. Indexing
a traced tensor with a SymInt (x[M - 1], h[M // 2 :]) is routed to select /
slice on the symbolic value; a plain-int index, None, an ellipsis or a mask
takes the stock path (a SymInt mixed with a mask is converted through
guard_int there, a sound pin).

The warm-up call runs the function's side effects for real, once, before the
symbolic run and even when the trace then declines: an in-place op on an
input stays applied, an RNG draw advances the generator, a global the
function writes holds the warm-up's value. What it must leave as it found
it is every tensor argument's metadata (sizes, strides, storage offset, dtype,
storage), which the symbolic run reads after it. An out= of another shape is
resized in place by eager (resize_output, with a deprecation warning), and a
resize_ or set_ on an argument does the same, so a symbolic run after such a
warm-up would describe the resized call and not the one made: the trace
compares the metadata around the warm-up and declines by name, the argument
left as eager left it. torch.cuda.caching_allocator_alloc inside a trace is
invisible to the recorder: such a pointer can only reach a kernel as a
constant.

A caller serving calls through this surface (a trace, then hits, a miss traced
again) executes the user's function exactly once per call, as eager would: the
ordinary call whose outputs it returns is that call's execution and its
warm-up, and it then traces with warm_up=False, which executes nothing (the
symbolic run does not run the kernels), and prepares its replay from the tape,
which runs nothing of the function either.

Every branch on a size is guarded on the value the trace saw, including the
size-1 branches of the view code: a squeeze of a symbolic dim traced at size 1
pins the tape to size 1, and traced at size 8 misses at size 1. Contiguity and
density are the exception by construction: each is asked as a shape-generic
predicate per dim (a size-1 dim has any stride), so a plain layer norm traced
at batch 1 serves every batch.

The argument contract is part of the tape: the arity, which positions hold
tensors, and every other argument by value and type. A replay whose
arguments differ in any of these is a miss before it touches the GPU.
Inputs carrying a math bit (a negative or conjugate view) decline at the
trace and miss at replay: ordinary dispatch resolves the bit, a replayed graph
would not. A host may launch only on the trace's capturing stream (its current
stream, or a stream forked from it with an event); a launch whose current
stream is any other declines instead of executing. A verbatim launch that
names another stream explicitly is outside the contract: it cannot be seen
before it runs, the completeness rule declines it afterwards, and traced
addresses are placeholders (non-canonical addresses that keep the real
address's low 52 bits, so alignment and offset arithmetic are exact) so that
such a launch faults instead of reading or writing real memory. A host that
must launch on another stream uses the typed launch, which checks the stream
it is given. A host's comparison of two addresses (copy_'s `src != dst`) is
decided by root identity when either is an allocation: two live roots never
share memory, so the answer is fixed whatever the hints and is kept on the
tape as a root fact, not a guard; two inputs may be one storage passed twice,
so their comparison stays an address guard.

A trace that declines carries what it had established on the exception
(Declined.partial): the argument contract, the input symbols and the guards
recorded up to the decline, in order, with the reason and the op that raised.
The calls that bind the same way and hold every one of those guards take the
same path through the host and reach the same decline, so an entry remembers
a declined class by them instead of tracing each new shape again (Entry,
below). A decline raised before the inputs are bound (the checks trace() makes
on the real arguments: no tensor or no CUDA argument, a pageable or empty
input, a math bit, a device mismatch, a trace already open) carries None.

Surface. The stable contract is the tape (aten/src/ATen/cuda/host_trace/
Tape.h and the records this module derives from it), the order its guards are
evaluated in, and the entry policy (Entry); the replay is the consumer's.
Private, underscored, and expected to change.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import gc
import json
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import sympy
from sympy.core.relational import Relational

import torch
from torch._guards import GuardSource, ShapeGuard, SLoc, Source
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, is_fake
from torch.cuda import _host_trace_ir as _ir
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import (
    canonicalize_bool_expr,
    DimDynamic,
    ShapeEnv,
)
from torch.utils._python_dispatch import _disable_current_modes, TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils._sympy.functions import (
    CeilToInt,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    Max,
    Min,
    Mod,
    OpaqueUnaryFn_sqrt,
    PythonMod,
    ToFloat,
    TruncToInt,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import PythonPrinter
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


__all__ = [
    "Declined",
    "Miss",
    "TopologyMiss",
    "Tape",
    "trace",
    "Entry",
]

aten = torch.ops.aten

# The backend behind a trace's symbolic values: "sympy" (torch's ShapeEnv,
# _TraceShapeEnv below) or "ir" (the integer expression IR of _host_trace_ir).
# Read once when a trace starts; a tape records which backend traced it. An
# IR tape is exported to sympy as it is built (_SympyExport), so what a tape's
# consumers read is the same either way at this stage.
symbolic: str = "sympy"


def _binding(name: str) -> Any:
    # A torch without CUDA has no recorder bindings; the module still imports
    # (test_public_bindings and test_testing import every torch module) and
    # trace() declines on its first non-CUDA input.
    found = getattr(torch._C, name, None)
    if found is not None:
        return found
    return type(name.removeprefix("_HostTrace"), (RuntimeError,), {})


TapeMismatch = _binding("_HostTraceTapeMismatch")


class _Declined(RuntimeError):
    """The static shape of Declined, which C++ registers: the host did
    something this tracer does not describe."""

    # what the trace had established when it declined (a _PartialTrace);
    # None when it declined before its inputs were bound. Set by trace().
    partial: _PartialTrace | None = None
    # the IR backend's census when the decline is an operation the IR does
    # not express: (operation, site) pairs, in order; empty otherwise
    census: tuple = ()


# The C++ recorder raises the same type (registered as _HostTraceDeclined),
# so one except clause catches a decline from either side.
Declined: type[_Declined] = _binding("_HostTraceDeclined")
Declined.__doc__ = _Declined.__doc__
Declined.partial = None
Declined.census = ()


class Miss(RuntimeError):
    """This call cannot use the tape: run the ordinary host."""


class TopologyMiss(Miss):
    """The tape's guards held, but the call selects a class of the variant
    that its exec does not hold and that only the call decides (a closed
    region's cuBLAS node chain, known from the template cache). The tape
    describes the call all the same: prepared again at these inputs it is a
    variant of that class (no re-trace). `tape` is that tape; an entry keeps
    the new variant beside the first."""

    def __init__(self, msg: str, tape: Tape) -> None:
        super().__init__(msg)
        self.tape = tape


# The CUDA hosts converted to the recorder's types. An op outside this set
# never reaches a kernel under a trace.
_TRACEABLE = {
    aten.native_layer_norm.default,
    aten.native_layer_norm_backward.default,
}

# the recorder's message when a host reads a raw pointer of a traced tensor
# (Recorder.cpp kNoDataPtr); the trace turns it into a decline naming the op
_NO_DATA_PTR = "host_trace: data_ptr() / storage() on a traced tensor"

_ALLOC_OPS = {
    aten.empty.memory_format,
    aten.empty_strided.default,
    aten.empty_like.default,
    aten.new_empty.default,
    aten.new_empty_strided.default,
}
_VIEW_OPS = {
    aten.view.default,
    aten._unsafe_view.default,
    aten.as_strided.default,
    aten._reshape_alias.default,
    aten.alias.default,
    aten.detach.default,
    aten.permute.default,
    aten.t.default,
    aten.transpose.int,
    aten.slice.Tensor,
    aten.select.int,
    aten.expand.default,
    aten.unsqueeze.default,
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.narrow.default,
    aten.split.Tensor,
    aten.split_with_sizes.default,
    aten.unfold.default,
    aten.view_as_real.default,
    aten.view_as_complex.default,
}
# metadata queries the wrapper routes to Python (dispatch_sizes_strides_policy
# "strides"): the shape-generic contiguity below is what keeps a trace taken at
# batch 1 from pinning itself to batch 1. The wrapper's own TensorImpl computes
# the same predicates (SymbolicShapeMeta), but on hinted symbols c10 answers
# them through the eager templates (Contiguity.h's guard_or shortcuts, then
# _compute_contiguous<SymInt> under all_hinted, the dense test's sort): eager's
# branch outcomes as guards, Ne(s, 1) at every dim and Eq(s, 1) at a size-1
# dim, a pin. Measured (hosttrace_review/small_rows/SMALL_ROWS.md): a layer
# norm traced at batch 1 then misses batch 8; the GPT-2 decode tape gains 77
# pins of its size-1 dims. The mirrors record per dim the disjunction the
# answer depends on, `Eq(s, 1) | Eq(stride, expected)`, and stay shape-generic.
_ROUTED = {
    aten.sym_stride.default,
    aten.stride.default,
    aten.sym_is_contiguous.default,
    aten.is_contiguous.default,
    aten.is_contiguous.memory_format,
    aten.is_non_overlapping_and_dense.default,
    aten.is_strides_like_format.default,
}


@dataclass(frozen=True)
class _Src(Source):
    nm: str

    @property
    def name(self) -> str:
        return self.nm

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return GuardSource.LOCAL


_PLACEHOLDER_TAG = 0x4A5 << 52
_PLACEHOLDER_LOW = (1 << 52) - 1
# an allocation's base hint: a distinct 64 GiB-aligned value under another
# non-canonical top, so code that evaluates on hints sees distinct addresses;
# what decides a comparison of two roots' addresses is their identity
# (_TraceShapeEnv), not these values
_ALLOC_TAG = 0x4A6 << 52
_ALLOC_SHIFT = 36


def _placeholder_address(base: int) -> int:
    """The hint of a traced input's base address.

    A traced tensor has no storage, and its base address exists only as the
    hint of a symbol: the launch under the trace's capture is recorded, never
    executed, and a replay binds the symbol to the real address. The hint
    keeps the real address's low 52 bits (every alignment, modulo and offset
    computation a host performs is exact) under a non-canonical top: a launch
    that escaped the capture (a verbatim launch naming a never-forked stream)
    faults on the placeholder instead of touching the real tensor.
    """
    return _PLACEHOLDER_TAG | (base & _PLACEHOLDER_LOW)


@dataclass
class _Root:
    name: str
    sym: Any  # the base address as a value: an input's symbol, 256*q for an allocation
    itemsize: int
    # made inside the traced call (a<k>), never an input's storage
    allocation: bool = False


@dataclass
class _InputRec:
    position: int
    name: str
    dtype: torch.dtype
    sizes: list
    strides: list
    offset: Any
    root: _Root


@dataclass
class _AllocRec:
    seq: int
    name: str
    sizes: list
    strides: list
    dtype: torch.dtype
    root: _Root
    q: Any  # the symbol the allocation's address / 256 binds to


@dataclass
class _OutputRec:
    name: str
    root: _Root
    sizes: list
    strides: list
    offset: Any
    dtype: torch.dtype
    # ("argument", i) or ("output", k) when this output is that very object,
    # as eager returns it; None for a view or a fresh allocation
    identity: tuple[str, int] | None = None


_SYM_TYPES = (torch.SymInt, torch.SymFloat, torch.SymBool)


def _hint(v: Any) -> Any:
    return v.node.hint if isinstance(v, _SYM_TYPES) else v


def _same(a: Any, b: Any) -> bool:
    # the same value by construction (the same expression, or equal ints):
    # decided without a guard
    if isinstance(a, _SYM_TYPES) and isinstance(b, _SYM_TYPES):
        return bool(a.node.expr == b.node.expr)
    if isinstance(a, _SYM_TYPES) or isinstance(b, _SYM_TYPES):
        return False
    return a == b


def _norm_dim(d: Any, nd: int) -> int | None:
    if not isinstance(d, int):
        return None
    d = d + nd if d < 0 else d
    return d if 0 <= d < nd else None


def _symbol_name(v: Any) -> str | None:
    # the symbol a value was created as (neither backend replaces one, so
    # this is also the symbol every later expression names)
    if not isinstance(v, _SYM_TYPES):
        return None
    node = v.node
    if isinstance(node, _ir.IRSymNode):
        return node.symbol_name()
    if isinstance(node._expr, sympy.Symbol):
        return str(node._expr)
    return None


def _contiguous_strides(sizes: list) -> list:
    # c10::contiguous_strides: a zero size counts as 1 in the products behind
    # it (torch.empty((16, 0)) has strides (1, 1)); the zero is guarded on the
    # traced value so no other size's expression changes
    strides: list = [1] * len(sizes)
    for d in range(len(sizes) - 2, -1, -1):
        size = sizes[d + 1]
        if _hint(size) == 0 and bool(size == 0):
            size = 1
        strides[d] = strides[d + 1] * size
    return strides


def _guard_each(terms: list) -> bool:
    # One guard per term rather than one conjunction: a True answer records
    # every term, a False answer records only the first failing term's
    # negation, which alone implies the answer. Guarding the conjunction sends
    # its negation through the ShapeEnv's implication pass, whose to_cnf is
    # exponential in the number of terms (a non-contiguous rank-4 input never
    # finished tracing).
    for term in terms:
        if not bool(term):
            return False
    return True


def _contiguous_terms(sizes: list, strides: list) -> list:
    # per dim: a size-1 dim has any stride, otherwise the stride is the
    # product of the sizes behind it; each disjunction stays symbolic so the
    # guard is shape-generic. An empty tensor is contiguous whatever its
    # strides (c10 _compute_contiguous's numel == 0 rule): the zero dim is
    # the one term
    for size in sizes:
        if _hint(size) == 0:
            return [size == 0]
    terms, expected = [], 1
    for size, stride in zip(reversed(sizes), reversed(strides)):
        terms.append((size == 1) | (stride == expected))
        expected = expected * size
    return terms


def _dense_terms(sizes: list, strides: list) -> list:
    # dense in the permutation the strides had at the traced call; a size-1
    # dim's stride is irrelevant
    order = sorted(
        range(len(sizes)), key=lambda d: (_hint(sizes[d]) < 2, _hint(strides[d]))
    )
    terms, require = [], 1
    for d in order:
        terms.append((sizes[d] == 1) | (strides[d] == require))
        require = require * sizes[d]
    return terms


def _infer_dense_strides(sizes: list, strides: list) -> list:
    # at::infer_dense_strides (ExpandUtils.cpp), what empty_like gives a
    # strided source that is not dense: the dims sorted by stride with
    # TensorIterator's insertion sort (a zero stride is an ambiguous
    # comparison and does not move; equal strides put the smaller size
    # first), then dense strides in that order (a size of 1 or 0 does not
    # advance). Every comparison is a guard on the traced value
    ndim = len(sizes)
    if ndim == 0:
        return []
    if ndim == 1:
        return [1]
    perm = list(range(ndim - 1, -1, -1))

    def should_swap(dim0: int, dim1: int) -> int:
        s0, s1 = strides[dim0], strides[dim1]
        if bool(s0 == 0) or bool(s1 == 0):
            return 0
        if bool(s0 < s1):
            return -1
        if bool(s0 > s1):
            return 1
        if bool(sizes[dim0] > sizes[dim1]):
            return 1
        return 0

    for i in range(1, ndim):
        dim1 = i
        for j in range(1, i + 1):
            dim0 = i - j
            comparison = should_swap(perm[dim0], perm[dim1])
            if comparison > 0:
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                dim1 = dim0
            elif comparison < 0:
                break
    out: list = [None] * ndim
    current: Any = 1
    for idx in perm:
        out[idx] = current
        if bool(sizes[idx] > 1):
            current = current * sizes[idx]
    return out


def _or(a: Any, b: Any) -> Any:
    # a disjunction over bools and SymBools, decided (guarded) as one term
    if a is True or b is True:
        return True
    if a is False:
        return b
    if b is False:
        return a
    return a | b


def _compute_stride(oldshape: list, oldstride: list, newshape: list) -> list | None:
    # at::detail::computeStride (TensorUtils.cpp), for the strides a view
    # takes: the source is walked from the back in chunks of dims that are
    # contiguous with each other, and the new dims are laid over each chunk.
    # Every decision is a guard in the order the algorithm makes it; the
    # chunk test is guarded as the disjunction it negates (a size-1 dim, or
    # the stride equal to the chunk's extent), so a trace at batch 1 records
    # that disjunction rather than a pin. None: not viewable (eager
    # raises for view, copies for reshape).
    if not oldshape:
        return [1] * len(newshape)
    numel = 1
    for sz in oldshape:
        numel = numel * sz
    if bool(numel == 0):
        # an empty tensor keeps its strides when the shape is unchanged,
        # else takes the strides a resize would give
        same = len(oldshape) == len(newshape) and all(
            bool(a == b) for a, b in zip(oldshape, newshape)
        )
        if same:
            return list(oldstride)
        strides: list = [1] * len(newshape)
        for d in range(len(newshape) - 2, -1, -1):
            strides[d] = torch.sym_max(newshape[d + 1], 1) * strides[d + 1]
        return strides
    newstride: list = [None] * len(newshape)
    view_d = len(newshape) - 1
    chunk_base_stride = oldstride[-1]
    tensor_numel: Any = 1
    view_numel: Any = 1
    for tensor_d in range(len(oldshape) - 1, -1, -1):
        tensor_numel = tensor_numel * oldshape[tensor_d]
        if tensor_d == 0 or not bool(
            _or(
                oldshape[tensor_d - 1] == 1,
                oldstride[tensor_d - 1] == tensor_numel * chunk_base_stride,
            )
        ):
            while view_d >= 0 and bool(
                _or(view_numel < tensor_numel, newshape[view_d] == 1)
            ):
                newstride[view_d] = view_numel * chunk_base_stride
                view_numel = view_numel * newshape[view_d]
                view_d -= 1
            if bool(view_numel != tensor_numel):
                return None
            if tensor_d > 0:
                chunk_base_stride = oldstride[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return newstride


def _is_symint_index(it: Any) -> bool:
    return isinstance(it, torch.SymInt) or (
        isinstance(it, slice)
        and any(isinstance(v, torch.SymInt) for v in (it.start, it.stop, it.step))
    )


def _has_symint(items: tuple) -> bool:
    return any(_is_symint_index(it) for it in items)


def _sym_getitem(t: torch.Tensor, items: tuple) -> Any:
    # x[i, a:b, None, ...] as the view ops it stands for, dim by dim, with the
    # SymInt values intact; NotImplemented for an index form (a mask, a tensor,
    # a list) that only the stock path handles
    consuming = 0
    for it in items:
        if it is Ellipsis or it is None:
            continue
        if isinstance(it, bool) or not isinstance(it, (int, torch.SymInt, slice)):
            return NotImplemented
        if (
            isinstance(it, slice)
            and it.step is not None
            and not isinstance(it.step, int)
        ):
            return NotImplemented
        consuming += 1
    if sum(1 for it in items if it is Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    cur, d, consumed = t, 0, 0
    for it in items:
        if it is None:
            cur = aten.unsqueeze.default(cur, d)
            d += 1
        elif it is Ellipsis:
            d += cur.dim() - d - (consuming - consumed)
        elif isinstance(it, slice):
            cur = aten.slice.Tensor(
                cur, d, it.start, it.stop, 1 if it.step is None else it.step
            )
            d += 1
            consumed += 1
        else:
            cur = aten.select.int(cur, d, it)
            consumed += 1
    return cur


def _routed(func: Any, args: tuple, kwargs: dict) -> Any:
    self = args[0]
    sizes, strides = list(self.shape), self._sym_strides
    if func is aten.sym_stride.default:
        return tuple(strides)
    if func is aten.stride.default:
        # an int read specializes, as a guard
        return tuple(int(s) for s in strides)
    mf = (
        args[1]
        if len(args) > 1
        else kwargs.get("memory_format", torch.contiguous_format)
    )
    plain = mf in (None, torch.contiguous_format)
    if func in (
        aten.sym_is_contiguous.default,
        aten.is_contiguous.default,
        aten.is_contiguous.memory_format,
    ):
        return _guard_each(_contiguous_terms(sizes, strides)) if plain else False
    if func is aten.is_non_overlapping_and_dense.default:
        return _guard_each(_dense_terms(sizes, strides))
    if func is aten.is_strides_like_format.default:
        return False
    raise AssertionError(f"host_trace: unexpected routed query {func}")


class _TracedTensor(torch.Tensor):
    """A tensor the host sees during a trace: CUDA device, symbolic sizes,
    strides and storage offset, no storage. Every one belongs to a root (an
    input's storage or a host allocation); views share their source's root."""

    _root: _Root
    _sym_strides: list
    _sym_offset: Any
    _fake: FakeTensor

    # pyrefly: ignore [bad-override]
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        tr: _Trace,
        root: _Root,
        sizes: list,
        strides: list,
        offset: Any,
        dtype: torch.dtype,
    ):
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            sizes,
            strides,
            storage_offset=offset,
            dtype=dtype,
            device=tr.device,
            dispatch_sizes_strides_policy="strides",
        )
        torch._C._host_trace_drop_storage(t)
        t._root = root
        t._sym_strides = list(strides)
        t._sym_offset = offset
        elem = torch.empty_strided(sizes, strides, dtype=dtype, device="meta")
        if not (isinstance(offset, int) and offset == 0):
            elem = elem.as_strided(sizes, strides, offset)
        t._fake = FakeTensor(tr.fake_mode, elem, tr.device)
        # the offset is in this view's element units (view_as_real halves
        # them), the root's address is shared: register this tensor's itemsize
        tr.rec.register_root(t, root.sym, t.element_size(), root.allocation, root.name)
        tr.tensors.append(t)
        return t

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"_TracedTensor({self._root.name}, {tuple(self.shape)}, {tuple(self._sym_strides)})"

    # A read of the tensor's value on the host (a data-dependent branch, a
    # copy into Python) has no place on the tape. item() and the number
    # conversions decline by name at _local_scalar_dense; these reach the
    # value without dispatching (bool() and is_nonzero() call numel() in
    # C++, tolist() and numpy() refuse subclasses) and would raise PyTorch's
    # own errors, so they decline here.
    def _host_read(self, what: str) -> Any:
        raise Declined(
            f"host_trace: {what} of a traced tensor reads its value on the host; not traced (declined)"
        )

    def __bool__(self) -> bool:
        return self._host_read("bool()")

    def is_nonzero(self) -> bool:
        return self._host_read("is_nonzero()")

    def tolist(self) -> Any:
        return self._host_read("tolist()")

    def numpy(self, *, force: bool = False) -> Any:
        return self._host_read("numpy()")

    def __format__(self, format_spec: str) -> str:
        # Tensor.__format__ formats item() for a 0-dim tensor
        if self.dim() == 0:
            return self._host_read("format()")
        return super().__format__(format_spec)

    def __getitem__(self, index: Any) -> Any:
        # Tensor.__getitem__ turns an integer index into a plain int through
        # SymInt.__index__ (a guard_int), so x[M - 1] would pin the tape to the
        # traced batch. Inside the trace an index built from a SymInt is routed
        # to select / slice / unsqueeze on the symbolic value; everything else
        # (plain ints, None, Ellipsis, masks) takes the stock path.
        items = index if isinstance(index, tuple) else (index,)
        if getattr(_active, "trace", None) is not None:
            # a Python sequence or a CPU tensor among the indices: the stock
            # path builds a device index tensor from it on the host (a copy
            # the capture refuses with its own error); advanced indexing is
            # traced over a CUDA index tensor only (the index entries)
            for it in items:
                if isinstance(it, (list, tuple, range)) or (
                    isinstance(it, torch.Tensor) and not it.is_cuda
                ):
                    raise Declined(
                        "host_trace: a sequence or CPU-tensor index on a traced tensor "
                        "(x[:, [-1, 0]]) builds its index tensor on the host; index with "
                        "a CUDA tensor; not traced (declined)"
                    )
            if _has_symint(items):
                out = _sym_getitem(self, items)
                if out is not NotImplemented:
                    return out
        return super().__getitem__(index)

    def __len__(self) -> int:
        # Tensor.__len__ returns the first dim and CPython converts a SymInt
        # through __index__ (a guard_int): a pin of the traced value, which
        # eager code writes to test emptiness (transformers' DynamicCache).
        # The pin is recorded like any guard; the frame that called len() is
        # noted beside it so a Miss on that guard names it. One stack walk
        # per len() at the trace, nothing at replay.
        n = super().__len__()
        tr = getattr(_active, "trace", None)
        if isinstance(n, torch.SymInt) and tr is not None:
            tr.shape_env.note_pin(n, f"len() of {self._root.name} at {_user_frame()}")
            return int(n)
        return n

    @classmethod
    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _ROUTED:
            return _routed(func, args, kwargs)
        raise Declined(f"host_trace: {func} on a traced tensor outside its trace")


_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__)) + os.sep


def _user_frame() -> str:
    # the innermost frame outside torch: where the user's code stands
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename.startswith(_TORCH_DIR):
        f = f.f_back
    if f is None:
        return "?"
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"


_BOOL_ATOMS = (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)
_NO_SLOC = SLoc(None, None)
_SYM_WRAP = {int: torch.SymInt, float: torch.SymFloat, bool: torch.SymBool}
# the binary SymNode operations defined on a subset of their operands, by the
# method name sym_node.py memoizes them under (a right shift is a FloorDiv by
# a power of two there; the sign of an integer pow's exponent is a guard
# SymInt.__pow__ evaluates)
_DIVISIONS = frozenset({"int_floordiv", "mod", "int_truediv", "float_truediv"})


class _SymOpMemo(dict):
    """The ShapeEnv's memo of binary SymNode operations (sym_node.py, Note
    [symbolic op memo]): every new operation over the trace's symbols is
    stored here once, keyed by method and operand expressions, so this is
    where the trace learns of a partial operation the moment it is created,
    before any value or guard is built on it."""

    __slots__ = ("note",)

    def __init__(self, note: Callable[[Any, Any, Any], None]) -> None:
        super().__init__()
        self.note = note

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        method, lhs, rhs, _version = key
        if method in _DIVISIONS:
            self.note(lhs, rhs, value[0])


class _TraceShapeEnv(ShapeEnv):
    """The trace's ShapeEnv: a guard is the expression the host evaluated,
    decided by its hint.

    The ordinary evaluate_expr proves what it can statically, turns an
    equality into a symbol replacement and refines value ranges so that later
    expressions simplify; that is half of a trace's time and none of it is
    needed for a tape, whose guards are re-evaluated at every replay. Here
    every expression the host branches on that is not a constant is recorded
    once, as written, with its value under the hints. Symbols are never
    replaced: a specialization stays an explicit guard (`Eq(s, 1)`), and the
    same fact may recur in more forms (over-guarding), but a fact the host
    depended on is never left unrecorded. A partial operation (a division, a
    modulo) records its domain as a guard when it is created (`domain`), so
    the ordered guard list meets `Ne(divisor, 0)` before anything built on the
    operation; the host's own evaluation could only raise there.

    One rule of its own besides: a relation between the addresses of two
    roots is decided by root identity when either root is an allocation. A
    host's `dst != src` over two live tensors of different roots (an
    allocation against another allocation or against an input) is true by
    construction, so it is answered without a guard and kept as a root fact
    on the tape; two inputs may be one storage passed twice, so a relation
    between input addresses stays an address guard, evaluated on the real
    addresses at replay. An ordering of two roots' addresses has no answer
    by identity and declines.
    """

    def __init__(self) -> None:
        # duck sizing off, 0/1 specialization off: one symbol per traced value
        super().__init__(duck_shape=False, specialize_zero_one=False)
        # keep every float operation in program order (sym_node.py)
        self.exact_float_arithmetic = True
        self._evaluated: set = set()
        self._recorded: set = set()
        # a root's base symbol -> (root name, is an allocation)
        self.roots: dict[sympy.Symbol, tuple[str, bool]] = {}
        self.root_facts: list[tuple[str, str]] = []  # distinct pairs, by name
        # a guard's origin when a Python construct pinned a symbol (a len())
        self.guard_notes: dict[sympy.Basic, str] = {}
        # per raw guard (op index, kernel-choice depth, origin): the trace's
        # attribution at the record (`attribute`, set by the trace; _NO_ROW
        # for an env used outside one)
        self.guard_rows: list[tuple[int, int, str]] = []
        self.attribute: Any = None
        # every binary SymNode operation on this ShapeEnv is stored here once
        self._symop_cache = _SymOpMemo(self.domain)
        # an env an IR trace was exported to answers tape_guards() with the
        # pass output it was exported with (_SympyExport)
        self._exported_guards: tuple[list, dict, dict, list] | None = None

    # the raw record with its rows; an exported IR trace's is converted on
    # first read
    _lazy_guards: Callable[[], list] | None = None

    @property
    # pyrefly: ignore [bad-override]
    def guards(self) -> list:
        fill, self._lazy_guards = self._lazy_guards, None
        if fill is not None:
            for g, row in fill():
                self._record(g, row=row)
        return self._guards

    @guards.setter
    def guards(self, value: list) -> None:
        self._guards = value

    def note_root(self, sym: torch.SymInt, name: str, alloc: bool) -> None:
        self.roots[sym.node.expr] = (name, alloc)

    def note_pin(self, v: torch.SymInt, note: str) -> None:
        # the guard the int conversion of `v` records (evaluate_expr's
        # Eq(expr, hint)), with where it came from
        expr = v.node.expr
        if not expr.is_number:
            self.guard_notes.setdefault(
                sympy.Eq(expr, sympy.sympify(v.node.hint)), note
            )

    def evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: int | bool | float | None = None,
        fx_node: Any = None,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
        *,
        forcing_spec: bool = False,
    ) -> sympy.Basic:
        # a constant (a size compared with itself) is not a guard
        if isinstance(orig_expr, _BOOL_ATOMS) or orig_expr.is_number:
            return orig_expr
        if self.roots and isinstance(orig_expr, sympy.Rel):
            fact = self._root_identity(orig_expr)
            if fact is not None:
                return fact
        if hint is None:
            hint = self.guarding_hint_or_throw(orig_expr)
        concrete = sympy.sympify(hint)
        key = (orig_expr, concrete)
        if key in self._evaluated:
            return concrete
        self._evaluated.add(key)
        if concrete is sympy.true:
            g = orig_expr
        elif concrete is sympy.false:
            g = sympy.Not(orig_expr)
        else:
            g = sympy.Eq(orig_expr, concrete)
        # sympy decides a relation between symbols' declared properties (an
        # integer size and a fraction) at construction; a true one is no
        # guard, a false one contradicts the hint
        if g is sympy.false:
            raise AssertionError(f"host_trace: {orig_expr} is not {hint}")
        self._record(g, size_oblivious)
        return concrete

    def _record(
        self, g: sympy.Basic, size_oblivious: bool = False, row: tuple | None = None
    ) -> None:
        # the same relation from another evaluation (`not a < b` and `a >= b`,
        # a division's domain the host then tests itself) is one guard
        if g is not sympy.true and g not in self._recorded:
            self._recorded.add(g)
            self.guards.append(ShapeGuard(g, _NO_SLOC, size_oblivious))
            if row is None:
                row = _ir._NO_ROW if self.attribute is None else self.attribute(self, g)
            self.guard_rows.append(row)

    def domain(self, lhs: Any, rhs: Any, out: Any) -> None:
        """A partial operation's domain, recorded as a guard at the point the
        operation is created (the memo's store): the divisor of a floor
        division, a modulo or a true division is nonzero, and torch's Mod,
        which sym_node.py builds only when it knows both operands nonnegative
        and whose value is defined only there, has both operands nonnegative.
        Built through sympy as a host's own test would be: a relation sympy
        decides from the symbols' declared properties (a size is positive, a
        literal is nonzero) is no guard, as in evaluate_expr, nor is one the
        declared ranges decide by interval arithmetic (a block dimension
        `min(512, t, 512 // min(32, s))` over positive opaque results); the
        rest hold at the hints, where the operation's hint arithmetic already
        ran."""
        self._domain(sympy.Ne(rhs, 0), rhs, lambda r: 0 not in r)
        if isinstance(out, Mod):
            self._domain(sympy.Ge(lhs, 0), lhs, lambda r: r.lower >= 0)
            self._domain(sympy.Ge(rhs, 0), rhs, lambda r: r.lower >= 0)

    def _domain(self, g: Any, e: Any, decided: Callable[[Any], bool]) -> None:
        if g is not sympy.true and not decided(self._range(e)):
            self._record(g)

    def _range(self, e: Any) -> ValueRanges:
        # the value range over the declared domains alone (no guard refines a
        # range on this ShapeEnv): a size or a positive opaque result is >= 1,
        # every other symbol unbounded
        try:
            return bound_sympy(e, self.var_to_range)
        except (KeyError, NotImplementedError):  # no rule for a function in e
            return ValueRanges.unknown()

    def _root_identity(self, rel: sympy.Rel) -> sympy.Basic | None:
        roots = [x for x in rel.free_symbols if x in self.roots]
        if len(roots) != 2 or not any(self.roots[x][1] for x in roots):
            return None
        # a relation between the two addresses: one root's address less the
        # other's (an input's address is its symbol, an allocation's is 256
        # times its symbol) plus terms over no root
        d = sympy.expand(rel.lhs - rel.rhs)
        units = [d.coeff(x) / (256 if self.roots[x][1] else 1) for x in roots]
        if sorted(units) != [-1, 1]:
            return None
        a, b = sorted(self.roots[x][0] for x in roots)
        if not isinstance(rel, (sympy.Eq, sympy.Ne)):
            raise Declined(
                f"host_trace: the host ordered the addresses of two roots ({a}, {b}); "
                "only their identity is decided under a trace (declined)"
            )
        if (a, b) not in self.root_facts:
            self.root_facts.append((a, b))
        return sympy.true if isinstance(rel, sympy.Ne) else sympy.false

    def _set_replacement(self, a: sympy.Symbol, tgt: sympy.Expr, msg: str) -> None:
        raise AssertionError(f"host_trace: {a} is never replaced ({msg})")

    def tape_guards(self) -> tuple[list, dict, dict, list]:
        """The recorded guards in order, without those the guards before them
        already imply: a guard that is structurally true once the earlier
        guards' pins (`Eq(s, 1024)`), unifications (`Eq(a, b)`) and relations
        (a kept `a < b` inside a later `Or`) are substituted into it. Nothing
        is simplified and nothing solved beyond one linear symbol; a kept
        guard is the expression as evaluated, with what the kept guards
        before it pin integer symbols to (a number, or the earlier symbols
        the equality stands for) read into it; float symbols are never
        pinned. Returns the guards, the pins (symbol to its value,
        `Eq(s, 1024)`: 1024, `Eq(b, a)`: a), which the tape reads into every
        use of the symbols, the notes of the kept guards (`note_pin`), and
        the raw index of each kept guard (its row in `guard_rows`)."""
        if self._exported_guards is not None:
            return self._exported_guards
        order = {s: k for k, s in enumerate(self.backed_var_to_val)}
        parent: dict = {}
        facts: dict = {}
        lower_of: dict = {}  # a size's bounds from the kept guards
        upper_of: dict = {}

        def find(x: Any) -> Any:
            while x in parent:
                x = parent[x]
            return x

        canonical: dict = {}

        def canon(e: Any) -> Any:
            # one orientation per relation (`1 < a` is `a > 1`); no CNF conversion
            r = canonical.get(e)
            if r is None:
                if isinstance(e, (sympy.And, sympy.Or)):
                    r = type(e)(*(canon(a) for a in e.args))
                elif isinstance(e, Relational):
                    r = canonicalize_bool_expr(e)
                else:
                    r = e
                canonical[e] = r
            return r

        def substitute(e: Any) -> Any:
            # the earlier pins and unifications applied side by side; sympy
            # decides the relation from the symbols' declared properties
            # (`8*s > 1` for a positive s) only when it is small: that walk
            # is quadratic on a wide cat's sums
            if isinstance(e, (sympy.And, sympy.Or)):
                return type(e)(*(substitute(a) for a in e.args))
            if not isinstance(e, Relational):
                return e
            lhs, rhs = (roots(x) for x in e.args)
            if lhs is e.lhs and rhs is e.rhs:
                return e
            if lhs == rhs:
                holds = isinstance(e, (sympy.Eq, sympy.Le, sympy.Ge))
                return sympy.true if holds else sympy.false
            if lhs.is_Number and rhs.is_Number:
                return type(e)(lhs, rhs)
            r = type(e)(lhs, rhs, evaluate=False)
            return sympy.true if held_by_bounds(canon(r)) else r

        def bounds(x: Any) -> tuple:
            # (lower, upper) of an expression over sizes (integers >= 1) from
            # the bounds the kept guards gave them; None where unknown
            if x.is_Number:
                return x, x
            if x.is_Symbol:
                lo = 1 if x.is_positive and x.is_integer else None
                return lower_of.get(x, lo), upper_of.get(x)
            if isinstance(x, sympy.Add):
                los, his = zip(*(bounds(a) for a in x.args))
                return (
                    None if None in los else sum(los),
                    None if None in his else sum(his),
                )
            if isinstance(x, sympy.Mul):
                c, rest = x.as_coeff_Mul()
                lo, hi = sympy.S.One, sympy.S.One
                for f in sympy.Mul.make_args(rest):
                    flo, fhi = bounds(f)
                    if flo is None or flo < 0:
                        return None, None
                    lo = lo * flo if lo is not None else None
                    hi = hi * fhi if hi is not None and fhi is not None else None
                lo = None if lo is None else c * lo
                hi = None if hi is None else c * hi
                return (hi, lo) if c < 0 else (lo, hi)
            return None, None

        def held_by_bounds(r: Any) -> bool:
            # a canonical `lhs < rhs`, `lhs <= rhs`, `lhs != rhs` decided by the
            # bounds of `lhs - rhs`: `256*s <= K` once `512*s <= K` is kept,
            # `8*s > 1` for a size s, without sympy's assumption walk
            if not isinstance(r, (sympy.Lt, sympy.Le, sympy.Ne)):
                return False
            lo, hi = bounds(r.lhs - r.rhs)
            negative = hi is not None and bool(hi < 0)
            if isinstance(r, sympy.Lt):
                return negative
            if isinstance(r, sympy.Le):
                return hi is not None and bool(hi <= 0)
            return negative or (lo is not None and bool(lo > 0))

        def raise_bounds(r: Any) -> None:
            # a kept canonical guard affine in one size (`c*s + d <= 0`)
            # tightens that size's bound
            if not isinstance(r, (sympy.Lt, sympy.Le, sympy.Ne)):
                return
            terms = (r.lhs - r.rhs).as_coefficients_dict()
            d = terms.pop(sympy.S.One, sympy.S.Zero)
            if len(terms) != 1:
                return
            sym, c = next(iter(terms.items()))
            if not (sym.is_Symbol and sym.is_integer and c.is_Integer and d.is_Integer):
                return
            lo, hi = bounds(sym)
            if isinstance(r, sympy.Ne):
                # c*s + d != 0 at the bound: the bound moves by one
                if lo is not None and c * lo + d == 0:
                    lower_of[sym] = lo + 1
                elif hi is not None and c * hi + d == 0:
                    upper_of[sym] = hi - 1
                return
            # c*s + d < 0 is c*s + d + 1 <= 0 over integers
            d = d + 1 if isinstance(r, sympy.Lt) else d
            if c > 0:
                bound = (-d) // c  # s <= floor(-d / c)
                upper_of[sym] = bound if hi is None else min(hi, bound)
            else:
                bound = -(d // c)  # s >= ceil(-d / c), c < 0
                lower_of[sym] = bound if lo is None else max(lo, bound)

        def roots(x: Any) -> Any:
            # a replacement may itself name symbols unified since; a few
            # rounds reach the roots
            for _ in range(4):
                sub = {}
                for s in x.free_symbols:
                    r = find(s)
                    if r is not s:
                        sub[s] = r
                if not sub:
                    break
                x = x.xreplace(sub)
            return x

        def strip(term: Any, common: set) -> Any:
            return sympy.Mul(*[f for f in sympy.Mul.make_args(term) if f not in common])

        def pin_or_unify(e: Any) -> None:
            # Eq(s, 1024), Eq(a, b), Eq(s, a*b + 1): the latest symbol with a
            # unit coefficient stands for the rest; s*(a - b) == 0 with s a
            # size (positive) is a == b
            d = e.lhs - e.rhs
            terms = d.as_coefficients_dict()
            if len(terms) == 2:
                (t1, c1), (t2, c2) = terms.items()
                common = set(sympy.Mul.make_args(t1)) & set(sympy.Mul.make_args(t2))
                if common and all(f.is_Symbol and f.is_positive for f in common):
                    d = c1 * strip(t1, common) + c2 * strip(t2, common)
                    terms = d.as_coefficients_dict()
            one = sympy.S.One
            units = [x for x, c in terms.items() if x.is_Symbol and c in (1, -1)]
            if units:
                s = max(units, key=order.__getitem__)
                value = s - d / terms[s]
            elif len(terms) == 2 and one in terms:
                s, c = next((x, c) for x, c in terms.items() if x is not one)
                value = -terms[one] / c
                if not (s.is_Symbol and value.is_Integer):
                    return
            else:
                return
            # integer symbols only: a float symbol keeps its Identity barrier
            if s.is_integer and s not in value.free_symbols:
                parent[s] = value

        out = []
        notes = {}
        kept = []
        for k, g in enumerate(self.guards):
            expr = g.expr
            e = substitute(expr)
            # facts are kept for the relations a later guard can repeat (a
            # size, a numel, a bound): not for a wide cat's sums
            small = isinstance(e, (sympy.And, sympy.Or)) or len(e.free_symbols) <= 4
            if e is sympy.true or (
                facts and small and canon(e).xreplace(facts) is sympy.true
            ):
                continue
            out.append(e)
            kept.append(k)
            note = self.guard_notes.get(expr)
            if note is not None:
                notes[e] = note
            if small and isinstance(e, Relational):
                facts[canon(e)] = sympy.true
                raise_bounds(canon(e))
            elif small:
                facts[canon(e)] = sympy.true
            if isinstance(e, sympy.Eq):
                pin_or_unify(e)
        return out, {s: roots(s) for s in parent}, notes, kept


_REL_CLASSES = {"eq": sympy.Eq, "ne": sympy.Ne, "lt": sympy.Lt, "le": sympy.Le}


class _SympyExport:
    """An IR trace's symbols, guards and values as sympy over a ShapeEnv of
    its own: the one-way conversion at the tape boundary while the tape's
    consumers (the adapter's lowering, the runtime's payload contract, the
    two-hint check) take sympy. Symbols keep their
    names, sources, hints and domains; a relation is emitted with its
    positive terms on the left and unevaluated (the IR decided what the
    declared domains decide); float operations keep their order under
    Identity, as the sympy backend records them."""

    def __init__(self, env: _ir.Env) -> None:
        self.env = env
        self.shape_env = _TraceShapeEnv()
        self.symbols: dict[str, sympy.Symbol] = {}
        self._exprs: dict[int, Any] = {}
        self._values: dict[int, Any] = {}
        se = self.shape_env
        size_range, int_range = ValueRanges(1, int_oo), ValueRanges.unknown_int()
        float_range = ValueRanges(-sympy.oo, sympy.oo)
        for name, node in env.symbols.items():
            src = _Src(env.sources[name])
            if node.is_float:
                sym = sympy.Symbol(name, real=True)
                se.var_to_range[sym] = float_range
                se.backed_var_to_val[sym] = sympy.Float(node.hint)
            else:
                positive = name in env.ctx.positive
                sym = sympy.Symbol(name, integer=True, positive=positive or None)
                se.var_to_range[sym] = size_range if positive else int_range
                se.backed_var_to_val[sym] = sympy.Integer(node.hint)
            self.symbols[name] = sym
            se.name_to_symbol[name] = sym
            se.source_to_var[src.name] = sym
            se.var_to_sources[sym] = [src]
        se.unique_ids.update(env.unique_ids)
        se.roots = {self.expr(n): r for n, r in env.roots.items()}
        se.root_facts = list(env.root_facts)
        se.guard_notes = {self.expr(g): n for g, n in env.guard_notes.items()}
        # the raw record (twice the kept guards on a model tape) is converted
        # on its first read: tests and the two-hint check read it, a replay
        # and the lowering do not
        se._lazy_guards = lambda: [
            (self.expr(g), row) for g, row in zip(env.guards, env.guard_rows)
        ]

    def expr(self, n: _ir.Node) -> Any:
        r = self._exprs.get(n.id)
        if r is None:
            r = self._expr(n)
            self._exprs[n.id] = r
        return r

    def _sides(self, d: _ir.Node) -> tuple:
        # rel(d, 0) as lhs rel rhs: the positive terms against the negated rest
        c, ts = self.env.ctx._as_terms(d)
        lhs = [sympy.Integer(cf) * self.expr(t) for t, cf in ts.items() if cf > 0]
        rhs = [sympy.Integer(-cf) * self.expr(t) for t, cf in ts.items() if cf < 0]
        if c > 0:
            lhs.append(sympy.Integer(c))
        elif c < 0:
            rhs.append(sympy.Integer(-c))
        return sympy.Add(*lhs), sympy.Add(*rhs)

    def _expr(self, n: _ir.Node) -> Any:
        op, e = n.op, self.expr
        if op == "const":
            return sympy.Integer(n.args[0])
        if op in ("sym", "fsym"):
            return self.symbols[n.args[0]]
        if op == "add":
            c, ts = n.args
            return sympy.Add(
                sympy.Integer(c), *(sympy.Integer(cf) * e(t) for t, cf in ts)
            )
        if op == "mul":
            c, fs = n.args
            return sympy.Mul(sympy.Integer(c), *(e(f) ** ex for f, ex in fs))
        if op == "floordiv":
            # unevaluated: the IR applied the sound folds; torch's FloorDiv.eval
            # composes nested floor divisions whatever the divisors' signs
            a, b = e(n.args[0]), e(n.args[1])
            if a.is_Number and b.is_Number:
                return sympy.Integer(int(a) // int(b))
            return FloorDiv(a, b, evaluate=False)
        if op == "ceildiv":
            return CeilToInt(IntTrueDiv(e(n.args[0]), e(n.args[1])))
        if op == "mod":
            a, b = n.args
            ctx = self.env.ctx
            cls = Mod if ctx.is_nonnegative(a) and ctx.is_nonnegative(b) else PythonMod
            return cls(e(a), e(b))
        if op == "min":
            return Min(*(e(a) for a in n.args))
        if op == "max":
            return Max(*(e(a) for a in n.args))
        if op == "nod":
            return IsNonOverlappingAndDenseIndicator(*(e(a) for a in n.args))
        if op == "true":
            return sympy.true
        if op == "false":
            return sympy.false
        if op in _REL_CLASSES:
            lhs, rhs = self._sides(n.args[0])
            return _REL_CLASSES[op](lhs, rhs, evaluate=False)
        if op == "and":
            return sympy.And(*(e(a) for a in n.args))
        if op == "or":
            return sympy.Or(*(e(a) for a in n.args))
        if op == "not":
            return sympy.Not(e(n.args[0]))
        if op == "fcmp":
            rel, a, b = n.args
            return _REL_CLASSES[rel](e(a), e(b), evaluate=False)
        if op == "fconst":
            return sympy.Float(n.hint)
        if op == "ffromint":
            return ToFloat(e(n.args[0]))
        if op == "fadd":
            return Identity(e(n.args[0]) + e(n.args[1]))
        if op == "fsub":
            return Identity(e(n.args[0]) - e(n.args[1]))
        if op == "fmul":
            return Identity(e(n.args[0]) * e(n.args[1]))
        if op == "fdiv":
            return FloatTrueDiv(e(n.args[0]), e(n.args[1]))
        if op == "fneg":
            return Identity(-e(n.args[0]))
        if op == "fpow":
            return FloatPow(e(n.args[0]), e(n.args[1]))
        if op == "fsqrt":
            return OpaqueUnaryFn_sqrt(e(n.args[0]))
        if op == "fround32":
            return Float32(e(n.args[0]))
        if op == "ftrunc":
            return TruncToInt(e(n.args[0]))
        if op == "ffloor":
            return FloorToInt(e(n.args[0]))
        if op == "fceil":
            return CeilToInt(e(n.args[0]))
        raise AssertionError(f"host_trace: no sympy form for the IR node kind {op}")

    def value(self, v: Any) -> Any:
        # a traced value over the IR as the same value over sympy: a number
        # when constant, else a SymInt / SymFloat / SymBool at the same hint
        if not isinstance(v, _SYM_TYPES) or not isinstance(v.node, _ir.IRSymNode):
            return v
        node = v.node
        n = node.node
        r = self._values.get(n.id)
        if r is None:
            if n.op in ("const", "fconst", "true", "false"):
                r = node.pytype(n.hint)
            else:
                hint = node.pytype(n.hint)
                sym = SymNode(self.expr(n), self.shape_env, node.pytype, hint)
                r = _SYM_WRAP[node.pytype](sym)
            self._values[n.id] = r
        return r


def _raw_guard_sources(env: Any, start: int) -> list[tuple[Any, list[str]]]:
    # the raw guards recorded from `start` on, each with the source names of
    # its symbols, from either backend's env (the symm rank-uniformity check)
    if isinstance(env, _ir.Env):
        return [
            (g, [env.sources[n] for n in g.free_symbols]) for g in env.guards[start:]
        ]
    out = []
    for g in env.guards[start:]:
        srcs = [
            s.name for x in g.expr.free_symbols for s in env.var_to_sources.get(x, ())
        ]
        out.append((g.expr, srcs))
    return out


def _raw_guards(env: Any) -> tuple:
    # the raw record as sympy: a sympy trace's guards as recorded, an IR
    # trace's exported (a partial trace's evaluator reads sympy)
    if isinstance(env, _ir.Env):
        ex = _SympyExport(env)
        return tuple(ex.expr(g) for g in env.guards)
    return tuple(g.expr for g in env.guards)


# a guard's origin by the route the mode took for the innermost op: which code
# raised it, without a frame walk (a sibling entry's own Python before its
# C++ binding is "entry-python", the sibling's C++ once the binding runs
# "entry-host"; a converted host reached by redispatch "host-branch")
_ORIGIN_OF_ROUTE = {
    "host": "host-branch",
    "view": "view-meta",
    "alloc": "alloc-meta",
    "region": "region-record",
    "composite": "composite-body",
    None: "recorder",
}


class _OpRec:
    """One op the trace mode dispatched: a row of the tape's op table. The
    route the mode took (alloc, view, region, entry = a traced sibling entry,
    host = a converted host, composite = a decomposition or an E38 body), the
    arguments as the op received them (traced tensors by reference), the
    outputs as returned, the recorder's `seq` at entry and exit (every record
    the op issued, allocations and memsets included, lies in [seq[0], seq[1])),
    and the raw guard indices raised while it ran (nested ops' included; the
    op's own are the rows naming its index)."""

    __slots__ = (
        "index", "func", "route", "parent", "depth", "args", "kwargs", "seq",
        "guard_range", "outputs", "declined", "in_host",
    )  # fmt: skip

    def __init__(
        self,
        index: int,
        func: Any,
        parent: int,
        depth: int,
        args: tuple,
        kwargs: dict,
        seq: int,
        guard: int,
    ) -> None:
        self.index = index
        self.func = func
        self.route: str | None = None
        self.parent = parent
        self.depth = depth
        self.args = args
        self.kwargs = kwargs
        self.seq = [seq, -1]
        self.guard_range = [guard, -1]
        self.outputs: Any = None
        self.declined = False
        self.in_host = False  # a sibling entry's C++ binding is running

    def origin(self) -> str:
        # a sibling entry: its own Python before the binding, its C++ after
        if self.route == "entry":
            return "entry-host" if self.in_host else "entry-python"
        return _ORIGIN_OF_ROUTE[self.route]


class _HostBindings:
    """torch._C's host bindings as the sibling entries call them, each marking
    the innermost op as inside its C++ for the duration: the guards it raises
    then originate from the host's branch, the entry's own Python checks
    before the call from the entry (no frame walk)."""

    def __getattr__(self, name: str) -> Any:
        fn = getattr(torch._C, name)

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            ops = getattr(_active, "ops", None)
            if not ops:
                return fn(*args, **kwargs)
            op = ops[-1]
            prev, op.in_host = op.in_host, True
            try:
                return fn(*args, **kwargs)
            finally:
                op.in_host = prev

        setattr(self, name, call)
        return call


def _guard_attribution(env: Any, g: Any) -> tuple[int, int, str]:
    # per raw guard: the innermost op on this thread's stack (-1 between ops),
    # the kernel-choice depth (Recorder.h KernelChoice: > 0 picks a kernel or a
    # launch configuration, 0 decides metadata), and the origin by the op's
    # route (a len() pin between ops is the model's Python, noted by __len__
    # before the guard is recorded). A plain function on the env, not a bound
    # method of the trace: a _Trace dropped without end() (a test's) must free
    # by refcount, which closes its capture
    depth = torch._C._host_trace_kernel_choice_depth()
    ops = getattr(_active, "ops", None)
    if not ops:
        origin = "python-len" if g in env.guard_notes else "python"
        return (-1, depth, origin)
    op = ops[-1]
    return (op.index, depth, op.origin())


_host_bindings = _HostBindings()


class _Trace:
    """One trace in progress: the ShapeEnv, the recorder, and the records the
    Python side keeps (inputs, allocations, outputs)."""

    def __init__(self, device: int, hints: dict[str, int] | None = None) -> None:
        self.device = torch.device("cuda", device)
        # a hint per symbol source name in place of the value the input or
        # allocation has: a test's second run of one call at other values
        # (test/host_trace_two_hint.py); None traces the values as they are
        self.hints = hints
        self.symbolic = symbolic
        if symbolic == "ir":
            self.shape_env: Any = _ir.Env(Declined)
            # the fake twins carry the trace's own SymInts; no ShapeEnv
            self.fake_mode = FakeTensorMode(shape_env=None)
        elif symbolic == "sympy":
            self.shape_env = _TraceShapeEnv()
            self.fake_mode = FakeTensorMode(shape_env=self.shape_env)
        else:
            raise ValueError(
                f"host_trace: symbolic={symbolic!r} is not 'sympy' or 'ir'"
            )
        # a cached fake output is materialized with set_(), whose contiguity
        # refresh guards `size == 1` per dim on the hint; the uncached meta
        # path asks nothing until a view needs it
        self.fake_mode.cache_enabled = False
        self.rec = torch._C._HostTraceRecorder(device)
        self.tensors: list = []
        self.inputs: list[_InputRec] = []
        self.allocs: list[_AllocRec] = []
        # the op table, in dispatch order (the trace mode's enter_op / exit_op)
        self.ops: list[_OpRec] = []
        self.shape_env.attribute = _guard_attribution
        # the real tensor behind each input root (by root name): a host that
        # needs process-lifetime state keyed by the real storage (a symmetric
        # memory handle) looks it up here
        self.real_inputs: dict[str, torch.Tensor] = {}
        # the op whose dispatch raised a decline (the trace mode notes it)
        self.declined_at: Any = None

    @contextlib.contextmanager
    def on_thread(self) -> Iterator[None]:
        # this trace as the calling thread's, for an op the trace mode routes
        # off the tracing thread (the mode's own thread keeps _active.trace
        # for the whole trace, _trace_once)
        prev = getattr(_active, "trace", None)
        prev_ops = getattr(_active, "ops", None)
        _active.trace = self
        _active.ops = []
        try:
            with self.rec.on_this_thread():
                yield
        finally:
            _active.trace = prev
            _active.ops = prev_ops

    def enter_op(self, func: Any, args: tuple, kwargs: dict) -> _OpRec:
        stack = _active.ops
        op = _OpRec(
            len(self.ops),
            func,
            stack[-1].index if stack else -1,
            len(stack),
            args,
            kwargs,
            self.rec.seq(),
            len(self.shape_env.guards),
        )
        self.ops.append(op)
        stack.append(op)
        return op

    def exit_op(self, op: _OpRec, out: Any) -> None:
        _active.ops.pop()
        op.seq[1] = self.rec.seq()
        op.guard_range[1] = len(self.shape_env.guards)
        op.outputs = out

    def symbol(
        self, value: Any, name: str, *, positive: bool = False, fresh: bool = False
    ) -> Any:
        env = self.shape_env
        if self.hints is not None:
            value = self.hints.get(name, value)
        if self.symbolic == "ir":
            return env.create_symbol(name, value, positive)
        src = _Src(name)
        if positive:
            sym = env.create_symbol(
                value,
                src,
                DimDynamic.DYNAMIC,
                None,
                positive=True,
                do_not_specialize_zero_one=True,
            )
            # the declared domain as a range too (create_symbol's own admits
            # 0): what a division's domain is tested against
            env.var_to_range[sym] = ValueRanges(1, int_oo)
        else:
            sym = env.create_unspecified_symbol(value, src, DimDynamic.DYNAMIC)
        if isinstance(value, float):
            return env.create_symfloatnode(sym, hint=value, source=src)
        return env.create_symintnode(sym, hint=value, source=src)

    def input(self, position: int, t: torch.Tensor) -> _TracedTensor:
        name = f"arg{position}"
        sizes = [
            self.symbol(t.size(d), f"{name}.size({d})", positive=True)
            for d in range(t.dim())
        ]
        strides = [
            self.symbol(t.stride(d), f"{name}.stride({d})") for d in range(t.dim())
        ]
        offset = self.symbol(t.storage_offset(), f"{name}.storage_offset()")
        # the const read of the storage base: a copy-on-write input stays lazy
        base = torch._C._host_trace_storage_address(t)
        sym = self.symbol(_placeholder_address(base), f"{name}.base")
        root = _Root(f"p{position}", sym, t.element_size())
        self.shape_env.note_root(sym, root.name, alloc=False)
        traced = _TracedTensor(self, root, sizes, strides, offset, t.dtype)
        self.real_inputs[root.name] = t
        self.inputs.append(
            _InputRec(position, name, t.dtype, sizes, strides, offset, root)
        )
        return traced

    def allocate(self, func: Any, args: tuple, kwargs: dict) -> _TracedTensor:
        device = kwargs.get("device")
        if device is not None and torch.device(device) != self.device:
            raise Declined(
                f"host_trace: allocation on {device} inside a trace on {self.device}"
            )
        if kwargs.get("pin_memory"):
            raise Declined(
                "host_trace: pinned allocations inside a host are not traced"
            )
        mf = kwargs.get("memory_format")
        if func is aten.empty.memory_format:
            sizes = list(args[0])
            dtype = kwargs.get("dtype") or torch.get_default_dtype()
            strides = _contiguous_strides(sizes)
        elif func is aten.empty_strided.default:
            sizes, strides = list(args[0]), list(args[1])
            dtype = kwargs.get("dtype") or torch.get_default_dtype()
        elif func is aten.new_empty.default:
            sizes = list(args[1])
            dtype = kwargs.get("dtype") or args[0].dtype
            strides = _contiguous_strides(sizes)
        elif func is aten.new_empty_strided.default:
            sizes, strides = list(args[1]), list(args[2])
            dtype = kwargs.get("dtype") or args[0].dtype
        else:  # empty_like: at::native::empty_like under preserve_format
            src = args[0]
            sizes = list(src.shape)
            dtype = kwargs.get("dtype") or src.dtype
            src_strides = getattr(src, "_sym_strides", None) or list(src.stride())
            if mf not in (None, torch.preserve_format):
                strides = _contiguous_strides(sizes)
            elif _guard_each(_dense_terms(sizes, src_strides)):
                # a non-overlapping dense source keeps its strides
                strides = list(src_strides)
            else:
                # a strided source that is not dense (a head slice of a fused
                # qkv buffer) keeps its layout permutation: infer_dense_strides
                strides = _infer_dense_strides(sizes, src_strides)
            mf = None
        if mf not in (None, torch.contiguous_format, torch.preserve_format):
            raise Declined(
                f"host_trace: only contiguous allocations are traced (got {mf})"
            )
        k = len(self.allocs)
        name = f"alloc{k}"
        q = self.symbol(
            (_ALLOC_TAG | ((k + 1) << _ALLOC_SHIFT)) >> 8, f"{name}.base/256"
        )
        itemsize = torch.empty((), dtype=dtype).element_size()
        root = _Root(f"a{k}", 256 * q, itemsize, allocation=True)
        self.shape_env.note_root(q, root.name, alloc=True)
        t = _TracedTensor(self, root, sizes, strides, 0, dtype)
        self.allocs.append(
            _AllocRec(self.rec.next_seq(), name, sizes, strides, dtype, root, q)
        )
        return t

    def reshape_view(self, src: _TracedTensor, shape: list) -> Any:
        # at::native::view: infer_size on the requested shape (one -1 at
        # most, the element count must match; each check a guard, a failure
        # takes the refs' path for its error), then computeStride on the
        # source's sizes and strides. A shape computeStride cannot lay over
        # the source raises the CUDA op's text; reshape never reaches this
        # with such a shape (its composite copies first, as eager does).
        sizes, strides = list(src.shape), list(src._sym_strides)
        numel = 1
        for sz in sizes:
            numel = numel * sz
        new_sizes = list(shape)
        infer = [
            d for d, sz in enumerate(new_sizes) if isinstance(sz, int) and sz == -1
        ]
        if len(infer) > 1:
            return None
        known = 1
        for d, sz in enumerate(new_sizes):
            if d not in infer:
                known = known * sz
        if infer:
            # guards in evaluation order: a zero product is undefined for the
            # division, then the shape must divide, as the refs would raise
            if not bool(known != 0):
                return None
            if not bool(numel % known == 0):
                return None
            new_sizes[infer[0]] = numel // known
        elif not bool(known == numel):
            return None
        new_strides = _compute_stride(sizes, strides, new_sizes)
        if new_strides is None:
            raise RuntimeError(
                "view size is not compatible with input tensor's size and stride (at "
                "least one dimension spans across two contiguous subspaces). Use "
                ".reshape(...) instead."
            )
        return _TracedTensor(
            self, src._root, new_sizes, new_strides, src._sym_offset, src.dtype
        )

    def split_views(
        self, func: Any, src: _TracedTensor, args: tuple, kwargs: dict
    ) -> Any:
        # TensorShape.cpp split / split_with_sizes: the pieces are narrows
        # along dim with sizes that may be symbolic. split's piece count is a
        # guard (the last piece carries the remainder); the sizes of
        # split_with_sizes must sum to the dim, as eager checks.
        sizes, strides = list(src.shape), list(src._sym_strides)
        offset, nd = src._sym_offset, len(src.shape)
        if nd == 0:
            raise RuntimeError("split expects at least a 1-dimensional tensor")
        d = _norm_dim(args[2] if len(args) > 2 else kwargs.get("dim", 0), nd)
        if d is None:
            return None
        size = sizes[d]

        def piece(start: Any, length: Any) -> _TracedTensor:
            return _TracedTensor(
                self,
                src._root,
                sizes[:d] + [length] + sizes[d + 1 :],
                strides,
                offset + start * strides[d],
                src.dtype,
            )

        if func is aten.split_with_sizes.default:
            lengths = list(args[1])
            if not all(isinstance(n, (int, torch.SymInt)) for n in lengths):
                return None
            start: Any = 0
            out = []
            for length in lengths:
                if not bool(length >= 0):
                    raise RuntimeError(
                        "split_with_sizes expects split_sizes have only non-negative "
                        f"entries, but got split_sizes={lengths}"
                    )
                out.append(piece(start, length))
                start = start + length
            if not bool(start == size):
                raise RuntimeError(
                    f"split_with_sizes expects split_sizes to sum exactly to {size} "
                    f"(input tensor's size at dimension {d}), but got split_sizes={lengths}"
                )
            return out
        split_size = args[1]
        if not isinstance(split_size, (int, torch.SymInt)):
            return None
        # get_num_splits' checks, then the count: max(ceil(size / split_size), 1),
        # guarded as the interval of dim sizes that gives this count
        if not bool(split_size >= 0):
            raise RuntimeError(
                f"split expects split_size be non-negative, but got split_size={split_size}"
            )
        if not bool(_or(split_size > 0, size == 0)):
            raise RuntimeError(
                "split_size can only be 0 if dimension size is 0, but got dimension "
                f"size of {size}"
            )
        if bool(split_size == 0):
            return [piece(0, size)]
        count = max(-(-_hint(size) // _hint(split_size)), 1)
        below = count == 1 or bool(size > (count - 1) * split_size)
        if not (below and bool(size <= count * split_size)):
            raise AssertionError("host_trace: split count does not match its guards")
        last = size - (count - 1) * split_size
        return [
            piece(i * split_size, split_size if i < count - 1 else last)
            for i in range(count)
        ]

    def direct_view(
        self, func: Any, src: _TracedTensor, args: tuple, kwargs: dict
    ) -> Any:
        # Stride arithmetic over the source's root. Only the decisions the op's
        # meaning depends on are guards (a squeezed dim is 1, a broadcast dim
        # is 1, a select index is in range); everything else is an expression,
        # so a trace taken at batch 1 does not pin itself there. None: not
        # handled here, the FakeTensor path takes it.
        sizes, strides = list(src.shape), list(src._sym_strides)
        offset, nd = src._sym_offset, len(src.shape)

        def make(sz: list, st: list, off: Any) -> _TracedTensor:
            return _TracedTensor(self, src._root, sz, st, off, src.dtype)

        if func in (aten.view.default, aten._unsafe_view.default):
            return self.reshape_view(src, list(args[1]))
        if func is aten._reshape_alias.default:
            # reshape's view form: the composite computed these strides
            # (computeStride on the symbolic metadata) and copies otherwise
            return make(list(args[1]), list(args[2]), offset)
        if func in (aten.alias.default, aten.detach.default):
            return make(sizes, strides, offset)
        if func in (aten.split.Tensor, aten.split_with_sizes.default):
            return self.split_views(func, src, args, kwargs)
        if func is aten.t.default:
            if nd < 2:
                return make(sizes, strides, offset)
            return make(sizes[::-1], strides[::-1], offset) if nd == 2 else None
        if func is aten.transpose.int:
            d0, d1 = _norm_dim(args[1], nd), _norm_dim(args[2], nd)
            if d0 is None or d1 is None:
                return None
            sizes[d0], sizes[d1] = sizes[d1], sizes[d0]
            strides[d0], strides[d1] = strides[d1], strides[d0]
            return make(sizes, strides, offset)
        if func is aten.permute.default:
            dims = [d for d in (_norm_dim(x, nd) for x in args[1]) if d is not None]
            if len(dims) != len(args[1]) or sorted(dims) != list(range(nd)):
                return None
            return make([sizes[d] for d in dims], [strides[d] for d in dims], offset)
        if func is aten.unsqueeze.default:
            d = _norm_dim(args[1], nd + 1)
            if d is None:
                return None
            st = sizes[d] * strides[d] if d < nd else 1
            return make(
                sizes[:d] + [1] + sizes[d:], strides[:d] + [st] + strides[d:], offset
            )
        if func in (aten.squeeze.default, aten.squeeze.dim, aten.squeeze.dims):
            if func is aten.squeeze.default:
                dims = list(range(nd))
            else:
                raw = [args[1]] if func is aten.squeeze.dim else list(args[1])
                dims = [d for d in (_norm_dim(x, nd) for x in raw) if d is not None]
                if len(dims) != len(raw):
                    return None
            # squeezing depends on the size being 1: a guard on the traced value
            keep = [d for d in range(nd) if d not in dims or not bool(sizes[d] == 1)]
            return make([sizes[d] for d in keep], [strides[d] for d in keep], offset)
        if func is aten.expand.default:
            target = list(args[1])
            if len(target) < nd or nd == 0:
                return None
            lead = len(target) - nd
            # ATen's inferExpandGeometry, right to left: an added leading dim
            # is a size-1 dim whose stride is the following dim's size times
            # stride (kept when its target is 1, zero when it broadcasts)
            new_sizes: list = [None] * len(target)
            new_strides: list = [None] * len(target)
            for i in range(len(target) - 1, -1, -1):
                tgt = target[i]
                if i >= lead:
                    sz, st = sizes[i - lead], strides[i - lead]
                else:
                    if isinstance(tgt, int) and tgt == -1:
                        return None
                    sz, st = 1, new_sizes[i + 1] * new_strides[i + 1]
                if (isinstance(tgt, int) and tgt == -1) or _same(tgt, sz):
                    pass
                elif _hint(sz) == _hint(tgt) and bool(sz == tgt):
                    pass
                elif bool(sz == 1):
                    # a size-1 dim broadcasts: a guard on the traced value
                    sz, st = tgt, 0
                else:
                    return None
                new_sizes[i], new_strides[i] = sz, st
            return make(new_sizes, new_strides, offset)
        if func is aten.select.int:
            d = _norm_dim(args[1], nd)
            idx = args[2]
            if d is None or not isinstance(idx, (int, torch.SymInt)):
                return None
            if isinstance(idx, int) and idx < 0:
                idx = idx + sizes[d]
            elif isinstance(idx, torch.SymInt) and not bool(idx >= 0):
                idx = idx + sizes[d]
            if not bool(idx >= 0) or not bool(idx < sizes[d]):
                return None
            return make(
                sizes[:d] + sizes[d + 1 :],
                strides[:d] + strides[d + 1 :],
                offset + idx * strides[d],
            )
        if func in (aten.slice.Tensor, aten.narrow.default):
            if func is aten.narrow.default:
                d, start, length = _norm_dim(args[1], nd), args[2], args[3]
                if d is None or not isinstance(start, (int, torch.SymInt)):
                    return None
                if isinstance(start, int) and start < 0:
                    start = start + sizes[d]
                if not bool(start >= 0) or not bool(start + length <= sizes[d]):
                    return None
                end, step = start + length, 1
            else:
                d = _norm_dim(args[1] if len(args) > 1 else kwargs.get("dim", 0), nd)
                start = args[2] if len(args) > 2 else kwargs.get("start")
                end = args[3] if len(args) > 3 else kwargs.get("end")
                step = args[4] if len(args) > 4 else kwargs.get("step", 1)
                if d is None or not isinstance(step, int) or step <= 0:
                    return None
            size = sizes[d]

            def clamp(v: Any, default: Any) -> Any:
                # the slice bounds as ATen clamps them; each clamp is a guard
                # (as the meta kernels record it), so the sizes stay plain
                # expressions the view code downstream can reason about
                if v is None:
                    return default
                if not isinstance(v, (int, torch.SymInt)):
                    return NotImplemented
                if not bool(v >= 0):
                    v = v + size
                    if not bool(v >= 0):
                        return 0
                return v if bool(v <= size) else size

            lo, hi = clamp(start, 0), clamp(end, size)
            if lo is NotImplemented or hi is NotImplemented:
                return None
            length = torch.sym_max(hi - lo, 0)
            sizes[d] = (length + (step - 1)) // step
            offset = offset + lo * strides[d]
            strides[d] = strides[d] * step
            return make(sizes, strides, offset)
        return None

    def view(self, func: Any, args: tuple, kwargs: dict) -> Any:
        src = next((a for a in args if isinstance(a, _TracedTensor)), None)
        if src is None:
            # a view of a tensor the trace does not own (captured by the
            # host's closure): with a symbolic size it cannot be taken without
            # pinning, so it declines; with concrete arguments it is metadata
            # only, taken outside the mode
            values = tree_flatten((args, kwargs))[0]
            if any(isinstance(v, torch.SymInt) for v in values):
                raise Declined(
                    f"host_trace: {func} with a symbolic size on a tensor the trace "
                    "does not own (captured by the host's closure) (declined)"
                )
            with _disable_current_modes():
                return func(*args, **kwargs)
        if args[0] is src:
            out = self.direct_view(func, src, args, kwargs)
            if out is not None:
                return out

        def to_fake(x: Any) -> Any:
            return x._fake if isinstance(x, _TracedTensor) else x

        try:
            with self.fake_mode:
                out = func(*tree_map(to_fake, args), **tree_map(to_fake, kwargs))
        except ValueError as e:
            # the refs raise ValueError for a view that cannot be taken; the
            # CUDA kernel raises RuntimeError for the same input
            raise RuntimeError(str(e)) from e

        def wrap(o: Any) -> Any:
            if is_fake(o):
                return _TracedTensor(
                    self,
                    src._root,
                    list(o.shape),
                    list(o.stride()),
                    o.storage_offset(),
                    o.dtype,
                )
            return o

        return tree_map(wrap, out)


_active = threading.local()


def _deterministic_fill() -> bool:
    return (
        torch.are_deterministic_algorithms_enabled()
        and torch._C._get_deterministic_fill_uninitialized_memory()
    )


def _new_int_symbol(hint: int, name: str, positive: bool) -> torch.SymInt:
    # the recorder's opaque results (Recorder.h opaque()) get a symbol here,
    # with the domain the host declared for the result
    tr = getattr(_active, "trace", None)
    if tr is None:
        raise Declined("host_trace: opaque call outside a trace")
    return tr.symbol(hint, f"opaque {name}", positive=positive)


_SYMBOLIC_READ = re.compile(
    r"Cannot call (\w+\(\)) on tensor with symbolic sizes/strides"
)


def _refused_inside_the_trace(func: Any, e: RuntimeError, via: Any = None) -> Declined:
    # An op the mode ran during the symbolic run raised: a copy from CPU that
    # the capture forbids, a host's own check on symbolic inputs. Nothing the
    # tape can describe; the decline names the op and keeps the cause. `via`
    # is the outermost composite whose body ran `func` (the op the caller
    # wrote): a C++ body reading a concrete size, stride or count of a traced
    # tensor is that op having no traceable host here, named as such.
    if _NO_DATA_PTR in str(e):
        return Declined(
            f"host_trace: {func} read a raw data pointer of a traced tensor "
            "inside the trace: its host is not converted (declined)"
        )
    first = str(e).splitlines()[0] if str(e) else type(e).__name__
    m = _SYMBOLIC_READ.search(first)
    if m is not None:
        outer = func if via is None else via
        body = "its body" if outer is func else f"its body ({func})"
        return Declined(
            f"host_trace: {outer} is not a traceable CUDA host: {body} reads "
            f"{m.group(1)} of a traced tensor (declined)"
        )
    return Declined(f"host_trace: {func} raised inside the trace: {first} (declined)")


class Float32(sympy.Function):
    """A double the traced host narrowed to a float (a float member of a
    parameter struct, ht::round_float32): what the kernel receives, and what
    a later read of that member computes with."""

    nargs = 1
    is_real = True


def _round_float32(value: float | torch.SymFloat) -> float | torch.SymFloat:
    if not isinstance(value, torch.SymFloat):
        return ctypes.c_float(value).value
    node = value.node
    if isinstance(node, _ir.IRSymNode):
        # pyrefly: ignore [bad-argument-type]
        return torch.SymFloat(node._flt(node.env.ctx.fun("fround32", node.node)))
    shape_env = node.shape_env
    if shape_env is None:
        raise AssertionError("host_trace: a symbolic float without a ShapeEnv")
    # pyrefly: ignore [bad-argument-type]
    hint = None if node.hint is None else ctypes.c_float(node.hint).value
    fx_node, _ = shape_env._create_fx_call_function(_round_float32, (node.fx_node,))
    return torch.SymFloat(
        SymNode(Float32(node._expr), shape_env, float, hint, fx_node=fx_node)
    )


class _HostTracePythonPrinter(PythonPrinter):
    def _print_Float32(self, expr: sympy.Expr) -> str:
        # pyrefly: ignore [missing-attribute]
        return f"_round_float32({self._print(expr.args[0])})"


def _concrete_ints(x: Any) -> Any:
    # int and int-list arguments of the op under trace (a normalized_shape
    # written from the input's shape): the CUDA kernels are registered on the
    # non-SymInt signatures, and the dispatcher's wrapper asserts on a
    # symbolic element before the host runs. int() guards on the traced value,
    # a sound pin the tape records (the trace serves that value and misses
    # others by name).
    if isinstance(x, torch.SymInt):
        return int(x)
    if isinstance(x, (list, tuple)) and any(isinstance(e, torch.SymInt) for e in x):
        return type(x)(int(e) if isinstance(e, torch.SymInt) else e for e in x)
    return x


class _TraceMode(TorchDispatchMode):
    # TorchDispatchMode wraps __torch_dispatch__ with torch._disable_dynamo
    # unless a subclass opts out, and that wrapper imports torch._dynamo on
    # its first call (seconds). A trace never runs under Dynamo.
    @classmethod
    def _should_skip_dynamo(cls):
        return False

    def __init__(self, tr: _Trace) -> None:
        super().__init__()
        self.trace = tr
        self.depth = 0
        self.decomposing: list = []  # composite ops whose decomposition is running

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            if getattr(_active, "trace", None) is self.trace:
                return self._dispatch(func, types, args, kwargs)
            # a thread the mode reached that is not the tracing one: the
            # autograd engine runs backward nodes on its device worker thread
            # under a copy of the tracing thread's mode stack, and launches
            # them on the forward op's stream, the capturing one. The trace
            # and its recorder are this thread's for the op (Recorder.h
            # ThreadScope).
            with self.trace.on_thread():
                return self._dispatch(func, types, args, kwargs)
        except Declined:
            # the innermost op is the one that declined; an outer one relays it
            if self.trace.declined_at is None:
                self.trace.declined_at = func
            raise

    def _dispatch(self, func: Any, types: Any, args: tuple, kwargs: dict | None) -> Any:
        kwargs = kwargs or {}
        if func in _ROUTED:
            return _routed(func, args, kwargs)
        # every other op is a row of the tape's op table: its route, its
        # record range by seq, the guards raised while it ran, its outputs
        op = self.trace.enter_op(func, args, kwargs)
        out = None
        try:
            out = self._route(func, types, args, kwargs, op)
            return out
        except (Declined, _ir.Unsupported):
            op.declined = True
            raise
        finally:
            self.trace.exit_op(op, out)

    def _route(
        self, func: Any, types: Any, args: tuple, kwargs: dict, op: _OpRec
    ) -> Any:
        if func in _ALLOC_OPS:
            op.route = "alloc"
            if _deterministic_fill():
                # eager's empty*() fills what it allocates under
                # use_deterministic_algorithms(True) with
                # fill_uninitialized_memory (TensorFactories.h
                # fill_empty_deterministic_), a fill_ launched inside the
                # allocation itself, eager's own kernel; the sibling's fill_
                # is another kernel by name, so no launch of the trace's
                # stands in for it: decline
                raise Declined(
                    "host_trace: use_deterministic_algorithms(True) with fill_uninitialized_memory "
                    "fills every allocation inside empty() with eager's own fill kernel, which the "
                    "trace cannot record as the allocation's launch; the ordinary host serves (declined)"
                )
            return self.trace.allocate(func, args, kwargs)
        if func in _VIEW_OPS:
            op.route = "view"
            return self.trace.view(func, args, kwargs)
        if self.depth == 0 and func in _TRACEABLE:
            op.route = "host"
            # the op under trace: let it reach its CUDA host, which is the
            # code being recorded; the mode stays on for what the host does
            self.depth += 1
            try:
                with self:
                    args = tuple(_concrete_ints(a) for a in args)
                    kwargs = {k: _concrete_ints(v) for k, v in kwargs.items()}
                    return func.redispatch(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA),
                        *args,
                        **kwargs,
                    )
            except (Declined, _ir.Unsupported):
                raise  # a decline, or the IR backend's census item (_trace_once)
            except torch.AcceleratorError:
                raise  # trace() classifies these by CUDA error code
            except RuntimeError as e:
                raise _refused_inside_the_trace(func, e) from e
            finally:
                self.depth -= 1
        # a composite op decomposes under the mode into the ones above
        op.route = "composite"  # a decomposition or an E38 body under the mode
        self.decomposing.append(func)
        try:
            with self:
                r = func.decompose(*args, **kwargs)
        except (Declined, _ir.Unsupported):
            raise
        except torch.AcceleratorError:
            raise
        except RuntimeError as e:
            raise _refused_inside_the_trace(func, e, self.decomposing[0]) from e
        finally:
            self.decomposing.pop()
        if r is not NotImplemented:
            return r
        where = "inside the host" if self.depth else "under trace"
        via = f" (reached from {self.decomposing[0]})" if self.decomposing else ""
        raise Declined(
            f"host_trace: {func} {where} is not a traceable CUDA host{via} (declined)"
        )


class Tape:
    """What one traced call did, in terms of the ShapeEnv's symbols."""

    def __init__(
        self, tr: _Trace, records: dict, outputs: list[_OutputRec], args: tuple
    ) -> None:
        self.shape_env = tr.shape_env
        self.device = tr.device
        self.args = args  # the call the tape describes (a replay is prepared at it)
        # the argument contract: arity, which positions are tensors, and every
        # other argument by value and type; a replay with any difference is
        # a miss, before it touches the GPU
        self.nargs = len(args)
        self.positions = _tensor_positions(args)
        self.constants = _constants(args, self.positions)
        # the device class is part of the contract too (checked before any
        # GPU work)
        self.device_identity = _device_identity(self.device.index)
        self.inputs = tr.inputs
        self.allocs = tr.allocs
        self.launches = records["launches"]
        self.opaque = records["opaque"]
        self.rng_increment = records["rng_increment"]
        # every launch on the trace's own capturing stream (no forked side
        # stream): the fact a one-stream consumer needs before lowering
        self.all_on_capture_stream: bool = records["all_on_capture_stream"]
        # the roots the host read through the mutable accessor (an output, an
        # in-place operand), by name: an input among them is a position the
        # replay writes, so its binding materializes a copy-on-write tensor
        # first, as eager's mutable read does (A98)
        self.written_roots: list[str] = list(records["written_roots"])
        self.outputs = outputs
        # the dtype, sizes and strides of the warm-up's outputs (trace() with
        # warm_up): eager's answer for the traced call, what the test seam
        # checks the output records against; None without a warm-up
        self.warm_up_outputs: list | None = None
        # the argument positions among the written roots, in the call's index
        # space: what a binding reads through the mutable accessor first
        # (torch._C._host_trace_materialize)
        written = set(self.written_roots)
        self.written_inputs: tuple[int, ...] = tuple(
            i.position for i in self.inputs if i.root.name in written
        )
        self.symbolic = tr.symbolic
        self.ir_env: _ir.Env | None = None
        self.export_ms = 0.0
        # the op table and, per raw guard of the env's record, (op index,
        # kernel-choice depth, origin); kept_raw maps each kept guard (a
        # position in `guards`) to its raw index
        self.ops = tr.ops
        self.guard_rows = list(tr.shape_env.guard_rows)
        self.guards, pins, self.guard_notes, self.kept_raw = tr.shape_env.tape_guards()
        self._export: _SympyExport | None = None
        self._kept_index: dict | None = None
        if pins:
            self._pin_uses(pins)
        if tr.symbolic == "ir":
            # the IR backend's tape boundary at this stage: every guard and
            # value exported to sympy over a ShapeEnv of the tape's own, so
            # the adapter's lowering and the runtime's payload contract read
            # what a sympy trace gives them
            t0 = time.perf_counter()
            self.ir_env = tr.shape_env
            ex = _SympyExport(tr.shape_env)
            self._export = ex
            self.guards = [ex.expr(g) for g in self.guards]
            self.guard_notes = {ex.expr(g): n for g, n in self.guard_notes.items()}
            self._map_values(ex.value, export=True)
            self.shape_env = ex.shape_env
            self.shape_env._exported_guards = (
                list(self.guards),
                {ex.expr(s): ex.expr(v) for s, v in pins.items()},
                dict(self.guard_notes),
                list(self.kept_raw),
            )
            self.export_ms = 1e3 * (time.perf_counter() - t0)
        # what the tape declares rather than guards: every input size symbol
        # is a positive integer (a replay misses on an empty input before any
        # guard), as ("domain", symbol, 1, None) rows, then the pairs of roots
        # whose addresses a host compared, decided distinct by identity
        self.root_facts: list[tuple] = [
            ("domain", _symbol_name(s), 1, None) for i in self.inputs for s in i.sizes
        ] + list(tr.shape_env.root_facts)
        self._tensors = tr.tensors  # the symbols' owners stay alive with the tape

    def _pin_uses(self, pins: dict) -> None:
        # what the guards pin integer symbols to (a number, or the earlier
        # symbols an equality stands for), read into every use of the
        # symbols; the pin guard stays, and asserts exactly this. A value
        # that becomes a number is a Python number, as the plain guard path's
        # replacements left it, the rest a SymInt over the remaining symbols
        # at the same hint. The definitions keep their symbols, the replay
        # binds them by name: an input's sizes, strides, offset and base, an
        # allocation's base (`q`), an opaque result
        env = self.shape_env
        done: dict = {}

        def sub(v: Any) -> Any:
            if not isinstance(v, _SYM_TYPES):
                return v
            node = v.node
            expr: Any = node._expr
            r = done.get(expr)
            if r is not None:
                return r
            if isinstance(node, _ir.IRSymNode):
                e = env.ctx.subst(expr, pins)
                if e is expr:
                    r = v
                elif e.op in ("const", "fconst", "true", "false"):
                    r = node.pytype(e.hint)
                else:
                    # pyrefly: ignore [bad-argument-type]
                    r = _SYM_WRAP[node.pytype](_ir.IRSymNode(e, env, node.pytype))
            else:
                e = expr.xreplace(pins)
                if e is expr:
                    r = v
                elif e.is_Number or isinstance(e, _BOOL_ATOMS):
                    r = node.pytype(e)
                else:
                    r = _SYM_WRAP[node.pytype](SymNode(e, env, node.pytype, node._hint))
            done[expr] = r
            return r

        self._map_values(sub)

    def _map_values(self, f: Callable[[Any], Any], *, export: bool = False) -> None:
        # f over every use-site value of the tape (launch fields, grid, block
        # expressions, smem, allocation and output sizes / strides / offset,
        # opaque arguments, the rng increment); with `export` (the IR
        # backend's tape boundary) also over the definitions the replay binds
        # by name (an input's sizes, strides, offset and base, an allocation's
        # q and base, an opaque result), which the pin pass leaves as recorded
        def fs(values: list) -> list:
            return [f(v) for v in values]

        if export:
            roots: dict[int, _Root] = {}
            for i in self.inputs:
                i.sizes, i.strides, i.offset = fs(i.sizes), fs(i.strides), f(i.offset)
                roots[id(i.root)] = i.root
            for a in self.allocs:
                a.q = f(a.q)
                roots[id(a.root)] = a.root
            for o in self.opaque:
                o["sym"] = f(o["sym"])
            for out in self.outputs:
                roots[id(out.root)] = out.root
            for root in roots.values():
                root.sym = f(root.sym)
        for a in self.allocs:
            a.sizes, a.strides = fs(a.sizes), fs(a.strides)
        for L in self.launches:
            for p in L["params"]:
                p["value"] = f(p["value"])
            L["grid"], L["block_expr"] = fs(L["grid"]), fs(L["block_expr"])
            L["smem"] = f(L["smem"])
        for o in self.opaque:
            o["args"] = fs(o["args"])
        for out in self.outputs:
            out.sizes, out.strides = fs(out.sizes), fs(out.strides)
            out.offset = f(out.offset)
        self.rng_increment = f(self.rng_increment)

    def sym_expr(self, v: Any) -> Any:
        """A traced value of either backend (a SymInt / SymFloat / SymBool of
        the op table's arguments and outputs) as a sympy expression over the
        tape's shape_env; a number as itself."""
        if not isinstance(v, _SYM_TYPES):
            return v
        if self._export is not None:
            v = self._export.value(v)
        return v.node.expr if isinstance(v, _SYM_TYPES) else v

    def raw_guards(self) -> tuple:
        """The raw guard record as sympy, aligned with `guard_rows`."""
        return _raw_guards(self.ir_env if self.ir_env is not None else self.shape_env)

    def guard_attribution(self, k: int) -> dict:
        """Kept guard `k`'s attribution: its raw index, the op (index, name,
        route; None between ops), the phase ("kernel" at a kernel-choice
        depth > 0, else "meta") and the origin."""
        raw = self.kept_raw[k]
        op_index, depth, origin = self.guard_rows[raw]
        op = self.ops[op_index] if op_index >= 0 else None
        return {
            "raw": raw,
            "op": op_index if op is not None else None,
            "func": str(op.func) if op is not None else None,
            "route": op.route if op is not None else None,
            "phase": "kernel" if depth > 0 else "meta",
            "depth": depth,
            "origin": origin,
        }

    def guard_site(self, g: Any) -> str:
        """Where kept guard `g` came from, for a Miss text: the op, the phase
        and the origin ("op 3 aten.native_layer_norm.default: kernel choice,
        host-branch"), or the Python between ops."""
        if self._kept_index is None:
            index: dict = {}
            for k, e in enumerate(self.guards):
                index.setdefault(e, k)
            self._kept_index = index
        k = self._kept_index.get(g)
        if k is None:
            return ""
        a = self.guard_attribution(k)
        phase = "kernel choice" if a["phase"] == "kernel" else "metadata"
        if a["op"] is None:
            return f"{a['origin']} between ops, {phase}"
        return f"op {a['op']} {a['func']}: {phase}, {a['origin']}"

    def op_table(self) -> list[dict]:
        """The op table as dicts: per op its index, func, route, parent,
        depth, the record range [seq0, seq1), the raw guards it raised itself
        (rows naming its index) and every raw guard raised while it ran
        (nested ops' included), the arguments and outputs with traced tensors
        as descriptors (root, sizes, strides, offset, dtype, device; the
        values symbolic as recorded)."""
        own: list[list[int]] = [[] for _ in self.ops]
        for raw, (op_index, _depth, _origin) in enumerate(self.guard_rows):
            if op_index >= 0:
                own[op_index].append(raw)
        return [
            {
                "index": o.index,
                "func": str(o.func),
                "route": o.route,
                "parent": o.parent,
                "depth": o.depth,
                "seq": list(o.seq),
                "guards": own[o.index],
                "guard_range": list(o.guard_range),
                "args": _describe_args(o.args),
                "kwargs": {k: _describe_args(v) for k, v in o.kwargs.items()},
                "outputs": _describe_args(o.outputs),
                "declined": o.declined,
            }
            for o in self.ops
        ]

    @property
    def num_launches(self) -> int:
        return len(self.launches)

    @property
    def num_allocations(self) -> int:
        return len(self.allocs)

    @property
    def num_guards(self) -> int:
        return len(self.guards)

    def to_json(self) -> str:
        """A deterministic rendering for tests and debugging: two traces of the
        same call serialize identically; the `hints` section holds the values
        at the traced call, including addresses."""
        p = PythonPrinter()

        def e(v: Any) -> Any:
            if isinstance(v, _SYM_TYPES):
                return p.doprint(v.node.expr)
            if isinstance(v, sympy.Basic):
                return p.doprint(v)
            return v

        return json.dumps(self._structure(e), indent=1)

    def _structure(self, e: Any) -> dict:
        # the tape as nested lists and dicts with every value through `e`: the
        # printer for to_json, the identity for a test comparing two tapes
        hints: dict[str, int | float] = {}
        for name, v in self.shape_env.backed_var_to_val.items():
            hints[str(name)] = int(v) if v.is_integer else float(v)
        d = {
            "contract": {
                "nargs": self.nargs,
                "positions": list(self.positions),
                "constants": [repr(c) for c in self.constants],
                "device": [[k, str(v)] for k, v in self.device_identity],
                "symbolic": self.symbolic,
            },
            "inputs": [
                {
                    "position": i.position,
                    "dtype": str(i.dtype),
                    "sizes": [e(s) for s in i.sizes],
                    "strides": [e(s) for s in i.strides],
                    "offset": e(i.offset),
                    "root": e(i.root.sym),
                }
                for i in self.inputs
            ],
            "allocations": [
                {
                    "seq": a.seq,
                    "name": a.name,
                    "dtype": str(a.dtype),
                    "sizes": [e(s) for s in a.sizes],
                    "strides": [e(s) for s in a.strides],
                    "root": e(a.root.sym),
                }
                for a in self.allocs
            ],
            "launches": [
                {
                    "seq": L["seq"],
                    "kernel": L["kernel"],
                    "grid": [e(g) for g in L["grid"]],
                    "block": list(L["block"]),
                    "block_expr": [e(b) for b in L["block_expr"]],
                    "smem": e(L["smem"]),
                    "params": [
                        {
                            "offset": q["offset"],
                            "size": q["size"],
                            "kind": q["kind"],
                            "name": q["name"],
                            "access": q["access"],
                            "expr": e(q["value"]),
                            "const": not isinstance(q["value"], _SYM_TYPES),
                        }
                        for q in L["params"]
                    ],
                }
                for L in self.launches
            ],
            "opaque": [
                {
                    "seq": o["seq"],
                    "fn": o["fn"],
                    "args": [e(a) for a in o["args"]],
                    "expected": o["expected"],
                    "sym": e(o["sym"]),
                    "kind": o["kind"],
                    "domain": o["domain"],
                }
                for o in self.opaque
            ],
            "guards": [e(g) for g in self.guards],
            "root_facts": [list(f) for f in self.root_facts],
            "written_roots": list(self.written_roots),
            "written_inputs": list(self.written_inputs),
            "outputs": [
                {
                    "name": o.name,
                    "identity": o.identity,
                    "root": e(o.root.sym),
                    "sizes": [e(s) for s in o.sizes],
                    "strides": [e(s) for s in o.strides],
                    "offset": e(o.offset),
                    "dtype": str(o.dtype),
                }
                for o in self.outputs
            ],
            "rng_increment": e(self.rng_increment)
            if self.rng_increment is not None
            else None,
            "all_on_capture_stream": self.all_on_capture_stream,
            "hints": hints,
        }
        return d


def _describe_args(a: Any) -> Any:
    # an op-table argument: a traced tensor as its metadata (symbolic values
    # as recorded), a real tensor met inside the trace as its concrete
    # metadata, a list / tuple elementwise, anything else as is
    if isinstance(a, _TracedTensor):
        return {
            "root": a._root.name,
            "sizes": list(a.shape),
            "strides": list(a._sym_strides),
            "offset": a._sym_offset,
            "dtype": a.dtype,
            "device": str(a.device),
        }
    if isinstance(a, torch.Tensor):
        return {
            "root": None,
            "sizes": list(a.shape),
            "strides": list(a.stride()),
            "offset": a.storage_offset(),
            "dtype": a.dtype,
            "device": str(a.device),
        }
    if isinstance(a, (list, tuple)):
        return type(a)(_describe_args(x) for x in a)
    return a


def _tensor_positions(args: tuple) -> list[int]:
    return [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]


def _constant(a: Any) -> Any:
    # a non-tensor argument is compared by value and type (2, 2.0 and True are
    # different constants); a list, a tuple and a torch.Size of the same
    # elements are the same constant. A float (a complex) is compared by its
    # bits, not by ==: a scalar is baked into the launch as written, and 0.0
    # and -0.0 are different launches (x / -0.0 is -inf) while two nans with
    # the same bits are the same one (a fresh float("nan") serves)
    if isinstance(a, (list, tuple)):
        return tuple(_constant(x) for x in a)
    if a is None:
        return None
    if isinstance(a, float):
        return (type(a), struct.pack("<d", a))
    if isinstance(a, complex):
        return (type(a), struct.pack("<dd", a.real, a.imag))
    return (type(a), a)


def _constants(args: tuple, positions: list[int]) -> tuple:
    skip = set(positions)
    return tuple(_constant(a) for i, a in enumerate(args) if i not in skip)


_DEVICE_PROPERTIES = (
    "name",
    "major",
    "minor",
    "multi_processor_count",
    "max_threads_per_multi_processor",
    "shared_memory_per_block",
    "shared_memory_per_block_optin",
    "regs_per_multiprocessor",
    "warp_size",
)


def _device_identity(device: int) -> tuple:
    # hosts fold device properties into launch configs, grid caps and
    # increments (SM count, threads per SM, shared memory per block, warp
    # size) and select kernels by capability; a tape is bound to a device
    # class, not just to an SM count
    p = torch.cuda.get_device_properties(device)
    return tuple((k, getattr(p, k, None)) for k in _DEVICE_PROPERTIES)


def _math_bits(t: torch.Tensor) -> str | None:
    # a negative or conjugate view carries different logical data behind the
    # same dtype, layout and storage; ordinary dispatch resolves the bit, a
    # replayed graph would not
    if t.is_neg():
        return "a negative view (torch._neg_view)"
    if t.is_conj():
        return "a conjugate view"
    return None


_METADATA = ("sizes", "strides", "storage offset", "dtype", "storage")


def _output_metadata(out: Any) -> list | None:
    # the dtype, sizes and strides of what a call returned (a tensor or a
    # tuple / list of tensors; None for anything else): the warm-up's answer,
    # what the test seam compares a tape's output records against
    outs = (out,) if isinstance(out, torch.Tensor) else out
    if not isinstance(outs, (list, tuple)) or not all(
        isinstance(t, torch.Tensor) for t in outs
    ):
        return None
    return [(t.dtype, tuple(t.shape), tuple(t.stride())) for t in outs]


def _metadata(t: torch.Tensor) -> tuple:
    # what the symbolic run binds to symbols, and the warm-up must leave as it
    # found it; the storage by its address (set_ and a growing resize_ swap it)
    return (
        tuple(t.shape),
        t.stride(),
        t.storage_offset(),
        t.dtype,
        torch._C._host_trace_storage_address(t),
    )


def _real_input_of(t: torch.Tensor) -> torch.Tensor | None:
    """The real tensor an argument stands for: under a trace, the input tensor
    behind a traced tensor's root (None for a host allocation or a tensor of
    another trace); outside a trace, `t` itself."""
    if not isinstance(t, _TracedTensor):
        return t
    tr = getattr(_active, "trace", None)
    if tr is None:
        return None
    return tr.real_inputs.get(t._root.name)


def _trace_once(
    fn: Callable[..., Any],
    args: tuple,
    positions: list[int],
    device: int,
    hints: dict | None,
) -> tuple[_Trace, dict, list[_OutputRec]]:
    # one symbolic run under its own recorder scope
    tr = _Trace(device, hints)
    _active.trace = tr
    _active.ops = []  # this thread's stack of ops being dispatched
    try:
        traced = list(args)
        for i in positions:
            traced[i] = tr.input(i, args[i])
        try:
            with _TraceMode(tr):
                out = fn(*traced)
            tr.rec.finish()
        except torch.AcceleratorError as e:
            # classified by CUDA error code (Recorder.cpp capture_error_name),
            # never by message text: the operations the thread-local capture
            # rejects (a sync, a query, a cudaMalloc) would run at the trace
            # and never in the replayed graph
            code = getattr(e, "error_code", None)
            name = (
                torch._C._host_trace_capture_error_name(int(code))
                if code is not None
                else None
            )
            if name is None:
                raise
            raise Declined(
                "host_trace: the host performed an operation that is not permitted "
                f"inside a stream capture ({name}: {str(e).splitlines()[0]}); it would "
                "run at the trace and never in the replayed graph (declined)"
            ) from e
        if isinstance(out, torch.Tensor):
            outs: tuple = (out,)
        elif isinstance(out, (list, tuple)) and all(
            isinstance(t, torch.Tensor) for t in out
        ):
            outs = tuple(out)
        else:
            raise Declined(
                f"host_trace: the traced call returned {type(out).__name__}, not a tensor or a tuple of tensors"
            )
        outputs = []
        # an output that is an argument or an earlier output is that object at
        # the replay too, as eager returns it; read from what the host
        # returned, never inferred from a layout
        identities = {id(traced[i]): ("argument", i) for i in positions}
        for k, t in enumerate(outs):
            if not isinstance(t, _TracedTensor):
                raise Declined(
                    f"host_trace: output {k} is not a traced tensor ({type(t).__name__}); the host produced it outside the trace"
                )
            outputs.append(
                _OutputRec(
                    f"out{k}",
                    t._root,
                    list(t.shape),
                    list(t._sym_strides),
                    t._sym_offset,
                    t.dtype,
                    identities.get(id(t)),
                )
            )
            identities[id(t)] = ("output", k)
        records = tr.rec.records()
        return tr, records, outputs
    except _ir.Unsupported as e:
        # the IR backend met an operation it does not express: the trace
        # declines by name with the trace's census; nothing falls back to
        # sympy mid-trace
        declined = Declined(
            f"host_trace: the ir backend cannot express {e.op} ({e.site}); the sympy "
            "backend is not used mid-trace (declined)"
        )
        census = list(getattr(tr.shape_env, "census", ()))
        if (e.op, e.site) not in census:  # raised off a bare node, not the env
            census.append((e.op, e.site))
        declined.census = tuple(census)
        if len(tr.inputs) == len(positions):
            declined.partial = _PartialTrace.of(tr, args, str(declined))
        raise declined from e
    except Declined as e:
        # the guards so far describe the class of calls that reach this
        # decline; an entry remembers the class by them (Declined.partial)
        if len(tr.inputs) == len(positions):
            e.partial = _PartialTrace.of(tr, args, str(e))
        raise
    finally:
        # an exception above would otherwise leave the capture open until the
        # traceback releases the recorder; end() closes it, and is a no-op
        # after a completed trace
        tr.rec.end()
        _active.trace = None
        _active.ops = None


class _GcHold:
    """The cyclic collector, held off while any thread traces. One process-wide
    count: a trace ending on one thread must not re-enable the collector under
    a trace still running on another."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._enabled = False

    def __enter__(self) -> None:
        with self._lock:
            if self._depth == 0:
                self._enabled = gc.isenabled()
                gc.disable()
            self._depth += 1

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._depth -= 1
            if self._depth == 0 and self._enabled:
                gc.enable()


_gc_hold = _GcHold()


def trace(
    fn: Callable[..., Any],
    args: tuple,
    device: int | None = None,
    *,
    warm_up: bool = True,
) -> Tape:
    """Trace one call of `fn`. Tensor arguments become symbolic; every other
    argument is a constant of the resulting tape. With `warm_up` (the default)
    `fn` first runs once as written on the real inputs, so anything it
    initializes on first use happens outside the trace; a warm-up that changes
    a tensor argument's metadata (eager's out= resize, a resize_ or set_ inside
    the call) declines, since the symbolic run would describe the resized
    call."""
    positions = _tensor_positions(args)
    if not positions:
        raise Declined("host_trace: the call has no tensor arguments")
    for i in positions:
        if isinstance(args[i], _TracedTensor):
            raise Declined(
                f"host_trace: arg{i} is a traced tensor of another trace; only real tensors are traced"
            )
        if not args[i].is_cuda:
            raise Declined(
                f"host_trace: arg{i} is on {args[i].device}; only CUDA tensors are traced"
            )
        if args[i].numel() == 0:
            raise Declined(
                f"host_trace: arg{i} is empty; empty inputs run the ordinary host"
            )
        bits = _math_bits(args[i])
        if bits is not None:
            raise Declined(
                f"host_trace: arg{i} is {bits}; its semantics are not represented on a tape"
            )
    if device is None:
        device = args[positions[0]].device.index
    for i in positions:
        if args[i].device.index != device:
            raise Declined(
                f"host_trace: arg{i} is on {args[i].device}, the trace is on cuda:{device}"
            )
    if getattr(_active, "trace", None) is not None:
        raise Declined("host_trace: a trace is already in progress on this thread")
    warm = None
    if warm_up:
        # on this thread's current stream, synchronized on that stream only:
        # a device-wide synchronize would invalidate a capture on another thread
        before = [_metadata(args[i]) for i in positions]
        lazy = [torch._C._is_cow_tensor(args[i]) for i in positions]
        with torch.cuda.device(device):
            warm = _output_metadata(fn(*args))
            torch.cuda.current_stream(device).synchronize()
        # the warm-up is the call's execution, so values may change; the
        # metadata the symbolic run binds must not, or the tape would describe
        # the call eager made of the resized argument, not this one. A
        # copy-on-write input the call materialized (its storage address moves
        # and nothing else) is the call as made: eager's first call
        # materializes it too, and the tape describes the materialized tensor
        for i, was, was_lazy in zip(positions, before, lazy):
            now = _metadata(args[i])
            if now != was:
                if (
                    was_lazy
                    and now[:-1] == was[:-1]
                    and not torch._C._is_cow_tensor(args[i])
                ):
                    continue
                changes = ", ".join(
                    "storage replaced" if name == "storage" else f"{name} {a} -> {b}"
                    for name, a, b in zip(_METADATA, was, now)
                    if a != b
                )
                raise Declined(
                    f"host_trace: the warm-up changed the metadata of arg{i} ({changes}); eager resized it "
                    "in place (an out= of another shape, a resize_ or a set_ inside the call), and a trace "
                    "after it would describe the resized call, not the one made"
                )
    # The trace capture is thread-local. A CUDAGraph finalized while it is open
    # (an earlier variant's exec and pool, freed by a cyclic collection on this
    # thread) invalidates it, so hold collections until the trace is over. No
    # collection before the capture: a full one costs more than the trace.
    with _gc_hold:
        tr, records, outputs = _trace_once(fn, args, positions, device, None)
        tape = Tape(tr, records, outputs, args)
    tape.warm_up_outputs = warm
    return tape


_KIND_FMT = {
    "ptr": "<Q",
    "i64": "<Q",
    "u64": "<Q",
    "i32": "<I",
    "u32": "<I",
    "i16": "<H",
    "u8": "<B",
    "f32": "<f",
    "f64": "<d",
}


def _free_symbols(v: Any) -> set:
    if isinstance(v, _SYM_TYPES):
        return {str(s) for s in v.node.expr.free_symbols}
    if hasattr(v, "free_symbols"):
        return {str(s) for s in v.free_symbols}
    return set()


_KIND_MASK = {
    "ptr": (1 << 64) - 1,
    "i64": (1 << 64) - 1,
    "u64": (1 << 64) - 1,
    "i32": (1 << 32) - 1,
    "u32": (1 << 32) - 1,
    "i16": (1 << 16) - 1,
    "u8": 255,
}


def _pack(kind: str, val: Any) -> bytes:
    if kind in ("f32", "f64"):
        return struct.pack(_KIND_FMT[kind], float(val))
    return struct.pack(_KIND_FMT[kind], int(val) & _KIND_MASK[kind])


class _Evaluator:
    """Symbolic values as compiled Python, bound by symbol name: each value is
    compiled once per evaluator and evaluated per call in an env of the
    symbols' values."""

    def __init__(self) -> None:
        self.printer = _HostTracePythonPrinter()
        self.ns: dict[str, Any] = {
            "math": math,
            "torch": torch,
            "min": min,
            "max": max,
            "_round_float32": _round_float32,
        }
        self._code: dict = {}

    def _compile(self, v: Any) -> Any:
        if isinstance(v, _SYM_TYPES):
            v = v.node.expr
        key = id(v)
        c = self._code.get(key)
        if c is None:
            c = compile(self.printer.doprint(v), "<host_trace>", "eval")
            self._code[key] = (c, v)  # keep v alive so its id stays unique
            return c
        return c[0]

    def ev(self, v: Any, env: dict) -> Any:
        if isinstance(v, (int, float, bool)):
            return v
        return eval(self._compile(v), self.ns, env)

    def guard_text(self, g: Any) -> str:
        return self.printer.doprint(g)


def _input_names(inputs: list[_InputRec]) -> list:
    # the input symbols' names, once: binding runs per call
    return [
        (
            rec,
            [_symbol_name(sz) for sz in rec.sizes],
            [_symbol_name(st) for st in rec.strides],
            _symbol_name(rec.offset),
            _symbol_name(rec.root.sym),
        )
        for rec in inputs
    ]


def _bind_inputs(contract: Any, names: list, args: tuple, device: int) -> dict:
    """The env of a call's input symbols, or a Miss. `contract` is a Tape or a
    _PartialTrace: the argument contract first (arity, tensor positions, and
    every other argument by value and type), then each input's dtype, device,
    rank and math bits, and its sizes, strides, offset and base address bound
    to their symbols."""
    if len(args) != contract.nargs:
        raise Miss(f"{len(args)} arguments, the trace had {contract.nargs}")
    positions = _tensor_positions(args)
    if positions != contract.positions:
        raise Miss(
            f"tensor arguments at {positions}, the trace had them at {contract.positions}"
        )
    constants = _constants(args, positions)
    if constants != contract.constants:
        raise Miss(
            f"non-tensor arguments {constants} differ from the traced {contract.constants}"
        )
    env: dict = {}
    for rec, size_names, stride_names, offset_name, root_name in names:
        t = args[rec.position]
        if t.dtype != rec.dtype:
            raise Miss(f"{rec.name} is {t.dtype}, the tape traced {rec.dtype}")
        if not t.is_cuda or t.device.index != device:
            raise Miss(f"{rec.name} is not on the variant's device")
        if t.dim() != len(rec.sizes):
            raise Miss(
                f"{rec.name} has rank {t.dim()}, the tape traced rank {len(rec.sizes)}"
            )
        if t.numel() == 0:
            raise Miss(f"{rec.name} is empty; empty inputs run the ordinary host")
        bits = _math_bits(t)
        if bits is not None:
            raise Miss(f"{rec.name} is {bits}; not represented on the tape")
        for d, (sn, tn) in enumerate(zip(size_names, stride_names)):
            if sn is not None:
                env[sn] = t.size(d)
            if tn is not None:
                env[tn] = t.stride(d)
        if offset_name is not None:
            env[offset_name] = t.storage_offset()
        if root_name is not None:
            env[root_name] = torch._C._host_trace_storage_address(t)
    return env


def _exact_class(args: tuple) -> tuple:
    # the exact class of a call: every tensor by dtype, device, pinning,
    # sizes, strides, storage offset and math bits, every other argument as
    # a constant. What an entry remembers a decline by when the trace
    # declined before binding its inputs (Declined.partial is None)
    return tuple(
        (
            a.dtype,
            a.device,
            a.is_cpu and a.is_pinned(),
            tuple(a.shape),
            a.stride(),
            a.storage_offset(),
            _math_bits(a),
        )
        if isinstance(a, torch.Tensor)
        else _constant(a)
        for a in args
    )


@dataclass(frozen=True, eq=False)
class _PartialTrace:
    """What a trace had established when it declined: the argument contract,
    the inputs it bound to symbols, and the guards it had recorded, in order.

    The trace is deterministic given its branch outcomes, and every branch
    the host takes on a traced value is a guard, so the calls that bind the
    same way and hold every guard recorded up to the decline take the same
    path and reach the same decline: that is the declined class, and an
    entry that remembers it runs the ordinary host for its calls without
    tracing them again, while a call outside it is traced (and may serve).
    A guard over a symbol the inputs do not bind (an allocation's address, an
    opaque result) cannot be decided without running the host; a partial
    trace with one matches nothing."""

    nargs: int
    positions: list[int]
    constants: tuple
    inputs: list[_InputRec]
    guards: tuple  # sympy expressions over the trace's symbols, as recorded
    device: int
    reason: str
    op: Any  # the op whose dispatch raised the decline; None outside any op

    @classmethod
    def of(cls, tr: _Trace, args: tuple, reason: str) -> _PartialTrace:
        positions = _tensor_positions(args)
        return cls(
            len(args),
            positions,
            _constants(args, positions),
            tr.inputs,
            _raw_guards(tr.shape_env),
            tr.device.index,
            reason,
            tr.declined_at,
        )

    @functools.cached_property
    def _names(self) -> list:
        return _input_names(self.inputs)

    @functools.cached_property
    def _ev(self) -> _Evaluator:
        return _Evaluator()

    def matches(self, args: tuple) -> bool:
        """Whether a call at `args` is in the declined class: its inputs bind
        as a replay binds them and every guard so far holds."""
        try:
            env = _bind_inputs(self, self._names, args, self.device)
        except Miss:
            return False
        for g in self.guards:
            try:
                if not self._ev.ev(g, env):
                    return False
            except (NameError, ZeroDivisionError):
                # a symbol the inputs do not bind; undefined at these inputs
                return False
        return True


class _VariantLike(Protocol):
    """What an entry asks of a variant, whichever backend built it: whether a
    call's inputs are its own, as far as the inputs decide, then the call
    served (Miss when only the call itself could decide against it)."""

    def matches(self, args: tuple) -> bool: ...

    def __call__(self, args: tuple) -> list: ...


class Entry:
    """The entry for one function: its calls served by variants, one per class
    of inputs, the outputs as a list. The policy is the entry's and the
    variants are `build_variant`'s (a consumer's replay prepared from the tape
    at the call's inputs): a call is served by the first variant it matches; a
    call that misses every variant is traced at its own inputs and served by
    the variant built from that tape, never by the ordinary host (the cap
    raises). A TopologyMiss from a variant says the tape's guards held and only
    the call's class differs: the same tape is built again at the call's inputs
    (no trace) and the variant kept beside the first, unless a later variant of
    that tape serves the call first. The ordinary host serves a call only when
    its trace declines, warned once per declined class, and the class
    (Declined.partial, or the exact inputs when the trace declined before
    binding them) is remembered so no call of it is traced again. Explicit
    tensor arguments only. Exactly once is the builder's: with a builder that
    runs nothing of `fn`, a call executes it once (its warm-up here is that
    execution, `warm_up`); a missed call's trace with warm_up=True runs it
    again."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        build_variant: Callable[[Tape, tuple], _VariantLike],
        warm_up: bool = True,
        max_variants: int = 16,
    ) -> None:
        self.fn = fn
        self.build_variant = build_variant
        self.warm_up = warm_up
        self.max_variants = max_variants
        self.variants: list[_VariantLike] = []
        # the classes whose trace declined: by their guards so far, or exactly
        self.declined: list[_PartialTrace] = []
        self.declined_exact: set[tuple] = set()
        self.traces = 0  # traces taken: one per variant or declined class
        self.ordinary = 0  # calls the ordinary host served

    def _ordinary(self, args: tuple) -> list:
        self.ordinary += 1
        out = self.fn(*args)
        return [out] if isinstance(out, torch.Tensor) else list(out)

    def __call__(self, *args: Any) -> list:
        # the served call: most calls are the first variant's, so its step of the
        # loop runs here and the rest (`_miss`) only when it did not serve
        variants = self.variants
        if variants and variants[0].matches(args):
            try:
                return variants[0](args)
            except TopologyMiss as e:
                return self._miss(args, e.tape)
            except Miss:
                pass  # decided against by the call itself
        return self._miss(args)

    def _miss(self, args: tuple, tape: Tape | None = None) -> list:
        """A call the first variant did not serve (`tape` when it raised a
        TopologyMiss): the later variants in order, then the declined
        classes, then a build of `tape` or a trace and a build (or the cap)."""
        for v in self.variants[1:]:
            if not v.matches(args):
                continue
            try:
                return v(args)
            except TopologyMiss as e:
                # the tape's guards held: the same tape, built at these
                # inputs, serves the call unless a later variant of it does
                tape = e.tape
            except Miss:
                continue  # decided against by the call itself
        exact = _exact_class(args)
        if tape is None and (
            exact in self.declined_exact or any(p.matches(args) for p in self.declined)
        ):
            return self._ordinary(args)
        if len(self.variants) == self.max_variants:
            raise RuntimeError(
                f"host_trace: the call misses all {self.max_variants} variants of this entry (max_variants)"
            )
        try:
            if tape is None:
                self.traces += 1
                tape = trace(self.fn, args, warm_up=self.warm_up)
            variant = self.build_variant(tape, args)
        except Declined as e:
            # the exact inputs as made (a warm-up may have resized them)
            # always; the class by its guards when they can be decided from
            # the inputs (it then contains these); a build's decline (the
            # allocator backend) has no partial and is remembered exactly
            self.declined_exact.add(exact)
            if e.partial is not None and e.partial.matches(args):
                self.declined.append(e.partial)
            warnings.warn(
                f"host_trace: the trace at these inputs declined, so the ordinary host serves such calls: {e}",
                RuntimeWarning,
                stacklevel=3,  # the entry's caller, above __call__ and _miss
            )
            return self._ordinary(args)
        self.variants.append(variant)
        return variant(args)
