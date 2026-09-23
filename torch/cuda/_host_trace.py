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

Host-to-device data (a pointer table for a grouped kernel, ids arriving in
pinned memory) goes through HostTable and copy_h2d (HostTable.h): the tape
describes the table element by element and the copy as {src, dst, bytes},
the build pairs the copy with a memcpy node of its capture, and each replay
renders the table into a ring of pinned staging slots and updates the node
when an operand moved. A pinned CPU tensor may be an input of the traced
call as the source of such a copy; a pageable one declines at the trace and
misses at replay. The replay's copy reads a pinned input asynchronously: a
caller that rewrites the same pinned buffer in place between calls must wait
for the replay's pending copies (its wait_for_h2d) before each rewrite, or a
CPU running ahead of the GPU hands the copy the next call's bytes. Nothing detects a missing wait (a
contract, like the synchronous-copy rule); a fresh pinned tensor per call
needs no wait, the replay holds it until its copy has run. Passing a small
table by value in the kernel's parameter image instead of copying it is the
tape's future choice when it fits; not done here.

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

Randomness. A host that draws declares each random launch's philox increment
(rng_increment) before it; a function may contain several random launches.
The tape carries every slot and their sum: a replay sets the sum as the
graph's generator increment and hands each random kernel the prefix sum
before it as its intragraph offset, so the streams and the generator state
after the call equal eager's at every shape. The trace consumes randomness
of its own, deterministically: with warm_up=True it advances the generator by
one call's increment (the warm-up; the symbolic run under capture draws
nothing), with warm_up=False by nothing; a replay prepared from the tape draws
nothing. A program that traces mid-stream with the warm-up therefore continues
from a different generator state than the untraced program; reseed after the
trace when the stream matters, or serve the call ordinarily first and trace
without warm-up, which leaves the generator where eager leaves it (the
exactly-once contract). A tape is bound to its device class: hosts fold the SM
count into values (dropout's grid cap and increment), so a replay on a device
with another SM count misses.

Closed regions (cuBLAS). aten.mm and aten.addmm inside a traced function are
not traced through the library: the trace records a closed region (the
operands as values, the output as a traced allocation, the op and its
scalars) and issues nothing. A replay harvests a template for the region's
concrete shapes (raw captures of the same library call on scratch buffers at
different addresses, _harvest: the kernel nodes, and the image bytes that
moved, which must each equal one operand's address, are the pointer slots)
and holds the template's nodes in its graph. Per call a region whose concrete
shape key is the one loaded rebinds its pointer slots; a different key whose
template has the same node chain (kernel count and order of kinds: cuBLAS
adds a reduce kernel and a semaphore memset when it splits K) applies that
key's template in place (the driver lets an exec node change its kernel),
harvesting on first sight. The node chain of every region is part of a
variant's class: a replay holds exactly the nodes of the templates the shapes
it was prepared at selected, never a disabled node, and a key whose template
has another chain is a TopologyMiss, a Miss that carries the tape: the same
tape prepared at that call's inputs (no re-trace: the tape and the templates
are cached) is a variant with that chain, kept beside the first (Entry). The
template cache is process-wide per device and keyed
by everything that decides how the library runs the GEMM: the op, the
scalars, every operand's dtype, sizes and strides, the SM count and compute
capability, and a snapshot of the process-global BLAS settings (preferred
library, TF32 / fp32 precision, reduced-precision reductions, fp16
accumulation, deterministic algorithms, the workspace configuration).
gemm_templates() lists the cache.

Closed regions (cuDNN attention). aten._scaled_dot_product_cudnn_attention and
its backward, eager's default SDPA route for a masked bf16 / fp16 call at head
dims 64 and 128 on sm90 and up, are closed regions the same way
(torch/cuda/_host_trace_cudnn.py): the region records the call's operands and
the outputs the host allocates, the harvest runs the op on stand-ins at the
region's key (every operand's dtype, sizes, strides and alignment class, the
scalars, the device identity), and its templates share the cache above under
their own op tags.

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
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import (
    canonicalize_bool_expr,
    DimDynamic,
    ShapeEnv,
)
from torch.utils._python_dispatch import _disable_current_modes, TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
from torch.utils._sympy.functions import Mod
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
    "gemm_harvests",
    "gemm_templates",
]

aten = torch.ops.aten


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
    # whether trace(warm_up=True) had run the warm-up call when it declined,
    # and that call's return value: the warm-up was the call (E24), so a
    # consumer returns its result with the decline rather than losing it
    warm_up_ran: bool = False
    warm_up_outputs: Any = None


# The C++ recorder raises the same type (registered as _HostTraceDeclined),
# so one except clause catches a decline from either side.
Declined: type[_Declined] = _binding("_HostTraceDeclined")
Declined.__doc__ = _Declined.__doc__
Declined.partial = None
Declined.warm_up_ran = False
Declined.warm_up_outputs = None


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
    aten._fused_rms_norm.default,
    aten._flash_attention_forward.default,
    aten._flash_attention_backward.default,
    aten._scaled_dot_product_flash_attention.default,
    aten._scaled_dot_product_flash_attention_backward.default,
}

# traceable ops whose CUDA kernel takes its SymInt arguments as c10::SymInt
# (a _symint kernel): no pin is needed before redispatch, the host receives
# the symbols (flash's max_q / max_k, which the dense path never reads); and
# the traced entries whose sibling takes a SymInt argument as a value (the
# triu / tril diagonal, a field of the launch)
_SYMINT_KERNELS = {
    aten._flash_attention_forward.default,
    aten._flash_attention_backward.default,
    aten._scaled_dot_product_flash_attention_backward.default,
    aten.triu.default,
    aten.tril.default,
    aten.triu_.default,
    aten.tril_.default,
}

# the recorder's message when a host reads a raw pointer of a traced tensor
# (Recorder.cpp kNoDataPtr); the trace turns it into a decline naming the op
_NO_DATA_PTR = "host_trace: data_ptr() / storage() on a traced tensor"

# CUDA ops with a traced sibling host (torch/cuda/_host_trace_ti.py): the trace
# mode calls the entry in place of the op, so the tape describes the sibling's
# launch, eager's own kernel (E36)
_TRACED_ENTRIES: dict[Any, Any] = {}
# the entries that take the call's SymInt arguments as they are (a factory's
# sizes, arange's bounds); the mode pins them for every other op
_SYMINT_ENTRIES: set[Any] = set()


@functools.cache
def _decomposes_only_off_cuda(func: Any) -> bool:
    # an op with a CUDA kernel (or a CompositeExplicit one) beside a
    # CompositeImplicit kernel: the dispatcher takes the CompositeImplicit
    # kernel only on a backend without the other, so eager on CUDA never runs
    # the decomposition the mode's fallback would run (_fused_rms_norm's
    # rms_norm_composite)
    has = torch._C._dispatch_has_kernel_for_dispatch_key
    name = func.name()
    return has(name, "CompositeImplicitAutograd") and any(
        has(name, key)
        for key in (
            "CUDA",
            "CompositeExplicitAutograd",
            "CompositeExplicitAutogradNonFunctional",
        )
    )


_EXPLICIT_KERNELS = (
    "CompositeExplicitAutogradNonFunctional",
    "CompositeExplicitAutograd",
)


@functools.cache
def _explicit_body_key(func: Any, key: str) -> Any | None:
    # the key set to run the op's own eager body at: an op with no kernel of
    # its own whose computed dispatch entry on the tensors' backend `key`
    # (Undefined without a tensor) is a CompositeExplicit kernel, the C++
    # body eager runs one key below the mode (slice_backward: zeros, then a
    # copy into a view of it). None where eager finds a kernel of the op's
    # own or a BackendSelect kernel there, or nothing but the
    # CompositeImplicit kernel decompose() runs (OperatorEntry.cpp
    # computeDispatchTableEntryWithDebug, steps 1 to 2.2)
    has = torch._C._dispatch_has_kernel_for_dispatch_key
    name = func.name()
    keys = ["BackendSelect", key] if key != "Undefined" else ["BackendSelect"]
    below = next((k for k in keys + list(_EXPLICIT_KERNELS) if has(name, k)), None)
    if below not in _EXPLICIT_KERNELS:
        return None
    return torch._C.DispatchKeySet(getattr(torch._C.DispatchKey, key))


def _key_below(args: tuple, kwargs: dict) -> str:
    # the backend key eager dispatches the call to below the mode's: the
    # tensor arguments' highest (CUDA over CPU), Undefined without one
    key = "Undefined"
    for a in (*args, *kwargs.values()):
        for t in a if isinstance(a, (list, tuple)) else (a,):
            if isinstance(t, torch.Tensor):
                if t.is_cuda:
                    return "CUDA"
                key = "CPU"
    return key


def register_traced_entry(
    op: Any,
    entry: Callable[..., Any],
    *,
    replace: bool = False,
    symint: bool = False,
) -> None:
    """Stand `entry` in for `op` under a trace.
    An entry takes the op's arguments, runs its traced sibling host or raises
    Declined; it must not dispatch `op` itself (that re-entry is declined).
    With `symint` the entry receives the call's SymInt arguments as they are
    (a factory's sizes, arange's bounds); otherwise the mode pins them, as
    the CUDA kernels' non-SymInt signatures require."""
    if not replace and op in _TRACED_ENTRIES:
        raise ValueError(
            f"host_trace: {op} already has a traced entry; pass replace=True to substitute it"
        )
    _TRACED_ENTRIES[op] = entry
    if symint:
        _SYMINT_ENTRIES.add(op)


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
# batch 1 from pinning itself to batch 1
_ROUTED = {
    aten.sym_stride.default,
    aten.stride.default,
    aten.sym_is_contiguous.default,
    aten.is_contiguous.default,
    aten.is_contiguous.memory_format,
    aten.is_non_overlapping_and_dense.default,
    aten.is_strides_like_format.default,
}


# closed library calls: recorded as regions, never traced into, unless
# torch._native's eager override of the op takes the call (the K = 1 bmm runs
# the override's Triton kernel, not cuBLAS: _native_override_takes). baddbmm
# is not one: its structured meta copies the expanded bias into the result
# with ATen's copy kernel before the library call, and that kernel's argument
# image carries per-call host bytes no harvest reproduces (measured: a heap
# address at byte 536 of the StridedOp image differs between two captures)
_CLOSED_OPS = {aten.mm.default, aten.addmm.default, aten.bmm.default}
# the out= forms of the closed calls (Inductor's generated wrapper writes its extern GEMMs
# into buffers it allocated: `extern_kernels.mm(a, b, out=buf)`): the same region, its output
# the given tensor instead of a fresh allocation
_CLOSED_OUT_OPS = {
    aten.mm.out: aten.mm.default,
    aten.addmm.out: aten.addmm.default,
    aten.bmm.out: aten.bmm.default,
}


@functools.cache
def _native_override_nodes(func: Any) -> tuple:
    # the torch._native overrides registered on the op's CUDA entry
    # (torch/_native/registry.py: an eager router at the CUDA key runs the
    # first active node whose condition holds, else the ATen kernel), by the
    # registry's key: the op name with its non-default overload
    from torch._native import registry

    return tuple(registry._graphs.get((func.name().split("::", 1)[1], "CUDA"), ()))


@contextlib.contextmanager
def _cow_from_roots() -> Any:
    """torch._C._is_cow_tensor answered for a traced tensor from its root (an
    input's copy-on-write state at the trace; an allocation is never lazy),
    while an override's condition runs on the traced tensors."""
    original = torch._C._is_cow_tensor

    def is_cow(t: Any) -> bool:
        return t._root.cow if isinstance(t, _TracedTensor) else original(t)

    torch._C._is_cow_tensor = is_cow
    try:
        yield
    finally:
        torch._C._is_cow_tensor = original


def _native_override_takes(func: Any, args: tuple, kwargs: dict) -> bool:
    """Whether eager serves this call through a torch._native override: the op
    is one of the overrides' own `_native::<id>` ops (the router's call to the
    override's implementation), or an active override's condition holds for
    the call, the first-match rule of the router. Under the trace the
    condition runs on the traced tensors, so each comparison it makes is a
    guard, as it is when the router evaluates it one key below (E40: the
    conditions are symbolic-clean)."""
    if func.namespace == "_native":
        return True
    # a call an AOT kernel embedded in the ATen implementation serves: the
    # router declines its Python route ahead of the conditions and the ATen
    # fallback runs that kernel, which no converted host stands for
    from torch._native import aot_manifest

    coverage = aot_manifest.get_coverage(func.name().split("::", 1)[1], "CUDA")
    if coverage is not None and coverage.covers(args, kwargs):
        raise Declined(
            f"host_trace: eager serves {func} through torch._native's AOT-embedded kernel (the "
            "router's ATen fallback), which is not a traced host (declined)"
        )
    for node in _native_override_nodes(func):
        if not node.active:
            continue
        try:
            with _cow_from_roots():
                taken = node.cond_fn(*args, **kwargs)
        except Declined:
            raise
        except Exception as e:
            # the override's condition reads what the trace does not give (a raw
            # data pointer's alignment, the CuTe overrides): eager's route here
            # is decided by a value the tape has no symbol for
            first = str(e).splitlines()[0] if str(e) else type(e).__name__
            raise Declined(
                f"host_trace: eager routes {func} through torch._native's {node.dsl_name} "
                f"override, whose condition raised on the traced tensors ({first}); a "
                f"{node.dsl_name} launch is not recorded on the tape (declined)"
            ) from e
        if taken:
            return True
    return False


_GEMM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


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
    # an input whose storage was copy-on-write at the trace (what a Python
    # condition's torch._C._is_cow_tensor reads of a traced tensor)
    cow: bool = False


@dataclass
class _InputRec:
    position: int
    name: str
    dtype: torch.dtype
    sizes: list
    strides: list
    offset: Any
    root: _Root
    # a CUDA tensor, or a pinned CPU tensor (the source of an in-host copy)
    device: torch.device
    pinned: bool


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


@dataclass
class _RegionOperand:
    name: str
    root: _Root
    address: Any  # the data address as a value: root + offset * itemsize
    sizes: list
    strides: list
    dtype: torch.dtype


@dataclass
class _RegionRec:
    """A closed library call the host made (aten.mm / aten.addmm / aten.bmm
    through cuBLAS; an op eager serves through a torch._native override,
    torch/cuda/_host_trace_native.py): its operands and outputs as values,
    the op and its scalars. The trace issues nothing for it; a replay learns
    its kernel nodes from a harvested template (see _harvest)."""

    seq: int
    op: str  # "mm" | "addmm" | "bmm" | an override's op ("_fused_rms_norm")
    inputs: list[_RegionOperand]  # mm / bmm: (mat1, mat2); addmm: (bias, mat1, mat2)
    # a GEMM's out; an override's returns in the order it allocates them
    outputs: list[_RegionOperand]
    scalars: tuple  # addmm: (beta, alpha); an override's non-tensor arguments
    name: str

    @property
    def out(self) -> _RegionOperand:
        return self.outputs[0]


@dataclass
class _GemmTemplate:
    """One cuBLAS variant, harvested once per process for a template key:
    its nodes in order, kernels (function handle, name, launch configuration,
    argument image with the pointer slots zeroed, parameter layout) and
    memsets (value, size, destination role), and per kernel node the slots:
    (byte offset, operand index, delta) of every qword that is an operand's
    address, (byte offset, allocation index, delta) of every qword into an
    allocation the call makes itself, and the qwords that are the stream's
    workspace or host state of the call."""

    key: tuple
    nodes: list[dict]
    harvest_us: float
    hits: int = 0
    # the first harvest capture's CUDAGraph, kept: cuBLAS may tie host state
    # to the graph (its private pool holds the call's own allocations)
    graph: Any = None
    # bytes of every allocation the call makes itself, in order (its
    # workspace, a contiguous copy of an operand): a replay gives the nodes
    # its arena's buffers instead
    scratch: list[int] = None  # type: ignore[assignment]
    # the call's allocations in order as "out" (a return the tape allocates
    # ahead of the region) or "scratch"; a GEMM's are all scratch
    layout: tuple = ()
    kinds: tuple = ()
    uses_ws: bool = False  # a node holds the stream's workspace base
    # a key whose harvest refused (its kernels are not rebindable): the miss
    # is remembered so every later call with the key misses at once
    miss: str = ""


# the process-wide template cache, per device through the key
_gemm_templates: dict[tuple, _GemmTemplate] = {}
_gemm_lock = threading.Lock()
_harvest_streams: dict[int, tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}
# per device the anchor a harvest capture launches ahead of the closed call
_harvest_anchors: dict[int, torch.Tensor] = {}
_gemm_harvests = 0


def _slot_value(role: Any, const: int, addrs: list, ws: int, scratch: list) -> int:
    # the value a classified qword takes for these operands and this arena
    if role is None:
        return const
    what, idx, delta = role
    if what == "op":
        return addrs[idx] + delta
    if what == "scratch":
        return scratch[idx] + delta
    return ws


def _chain_class(nodes: list) -> tuple:
    # a template's chain as the class compares it (E28): per node its kind and
    # whether its incoming edge is programmatic. An exec's edge types are as
    # fixed as its node set: a kernel the library launched without programmatic
    # serialization must not run behind a programmatic edge, whose dependent
    # may start before the primary completes
    return tuple((n["kind"], n["programmatic"]) for n in nodes)


def _chain_text(chain: Any) -> str:
    rows = [k + (" (programmatic)" if p else "") for k, p in chain]
    return "[" + ", ".join(rows) + "]"


def _blas_settings() -> tuple:
    # every process-global switch that changes which kernels cuBLAS runs for
    # a given shape; part of the template key so a flipped setting is a new
    # template, never a reused one
    m = torch.backends.cuda.matmul
    return (
        str(torch.backends.cuda.preferred_blas_library()),
        # fp32_precision covers allow_tf32 (its legacy setter writes it)
        getattr(m, "fp32_precision", None),
        getattr(m, "allow_fp16_reduced_precision_reduction", None),
        getattr(m, "allow_bf16_reduced_precision_reduction", None),
        getattr(m, "allow_fp16_accumulation", None),
        torch.are_deterministic_algorithms_enabled(),
        # TunableOp routes mm/addmm through its own tuned kernels when enabled
        # (Blas.cpp IsTunableOpEnabled), and tuning mode changes what runs
        torch.cuda.tunable.is_enabled(),
        torch.cuda.tunable.tuning_is_enabled(),
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        os.environ.get("CUBLASLT_WORKSPACE_SIZE"),
    )


def gemm_harvests() -> int:
    """How many cuBLAS variants were harvested so far in this process."""
    return _gemm_harvests


def gemm_templates() -> list[dict]:
    """The closed-region template cache: one entry per harvested cuBLAS
    variant with its key, kernel names, node count, harvest time and hits."""
    with _gemm_lock:
        return [
            {
                "key": t.key,
                "kernels": [n["name"] for n in t.nodes if n["kind"] == "kernel"],
                "kinds": list(t.kinds),
                "programmatic": [n["programmatic"] for n in t.nodes],
                "node_count": len(t.nodes),
                "scratch": list(t.scratch or []),
                "harvest_us": t.harvest_us,
                "hits": t.hits,
                "miss": t.miss,
            }
            for t in _gemm_templates.values()
        ]


def _host_mappings() -> tuple:
    # the calling thread's stack mapping and the process's heap mappings
    # ([heap] and anonymous private writable mappings) from /proc/self/maps:
    # cuBLAS leaves host addresses of the call (alpha / beta on the stack,
    # per-call-site heap state) in some kernel images
    probe = torch._C._host_trace_stack_probe()
    stack = (probe, probe + 1)
    heaps = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            lo, hi = (int(v, 16) for v in parts[0].split("-"))
            path = parts[5] if len(parts) > 5 else ""
            if lo <= probe < hi:
                stack = (lo, hi)
            elif path == "[heap]" or (path == "" and parts[1].startswith("rw")):
                heaps.append((lo, hi))
    return stack, heaps


def _host_class(value: int, stack: tuple, heaps: list) -> str | None:
    if stack[0] <= value < stack[1]:
        return "stack"
    if any(lo <= value < hi for lo, hi in heaps):
        return "heap"
    return None


def _stack_low32(value: int, stack: tuple) -> bool:
    # a 32-bit field holding the low half of an address in the stack mapping
    lo, hi = stack
    return ((value - lo) & 0xFFFFFFFF) < hi - lo


_SMEAR = (0xA5, 0xA5, 0xA5, 0x5A)  # _harvest's stack smear pattern per capture


def _host_slot_class(off: int, imgs: tuple, stack: tuple, heaps: list) -> str | None:
    # what a host slot holds, from the four captures' bytes: an address in
    # the harvesting thread's stack (dead once the call returned) or in a
    # heap mapping (per-call-site library state the kept harvest graph pins),
    # the low 32 bits of a stack address in a 32-bit field, or bytes the
    # stack smear left in uninitialized padding; anything else is per-call
    # state the template cannot keep, and the harvest misses by name
    q = int.from_bytes(imgs[0][off : off + 8], "little")
    cls = _host_class(q, stack, heaps)
    if cls is not None:
        return cls
    diff = [i for i in range(8) if len({img[off + i] for img in imgs}) > 1]
    if diff and all(tuple(img[off + i] for img in imgs) == _SMEAR for i in diff):
        return "padding"
    for half in (0, 4):
        if all(half <= i < half + 4 for i in diff) and all(
            _stack_low32(
                int.from_bytes(img[off + half : off + half + 4], "little"), stack
            )
            for img in imgs
        ):
            return f"stack32:{half}"
    return None


def _closed_call(op: str, scalars: tuple, tensors: list) -> None:
    # the same library entry the ordinary host reaches, into a preallocated
    # output (nothing may allocate inside a raw capture)
    if op in ("mm", "bmm"):
        getattr(aten, op).out(tensors[0], tensors[1], out=tensors[-1])
    else:
        beta, alpha = scalars
        aten.addmm.out(
            tensors[0], tensors[1], tensors[2], beta=beta, alpha=alpha, out=tensors[-1]
        )


def _closed_call_alt(op: str, scalars: tuple, tensors: list) -> None:
    # the same library entry through another binding path: what differs
    # from _closed_call's image is host state of the call (see _harvest)
    if op in ("mm", "bmm"):
        getattr(torch, op)(tensors[0], tensors[1], out=tensors[-1])
    else:
        beta, alpha = scalars
        torch.addmm(
            tensors[0], tensors[1], tensors[2], beta=beta, alpha=alpha, out=tensors[-1]
        )


@dataclass(frozen=True)
class _ClosedCall:
    """How a closed region's call runs at the harvest: `run(op, scalars,
    tensors)` makes it on the operands (a GEMM into its preallocated output; a
    torch._native override's op, which allocates its outputs itself and
    returns them), `alt` makes it through another binding path, and `outputs`
    is how many trailing operands of the region are those returns (0: every
    operand is handed in). Both are plain functions called from the capture's
    lambda, one Python frame below the stack smear: the smear's reach past
    the aten.<op>.out binding is marginal, and a partial or wrapper between
    them puts a cutlass Params struct's padding beyond it."""

    run: Any
    alt: Any
    # an int, or a function of the region's scalars
    outputs: Any = 0
    # the call's kernels are DSL programs whose parameter structs carry
    # uninitialized padding (a 32-bit field ahead of a 64-bit one, inside one
    # driver parameter, so no layout marks it and the stack smear does not
    # reach it): a qword that differs between captures of the same operand
    # set (the first, third and fourth) is no value of the call, kept as the
    # template's own; one that differs only with the operands (the second)
    # and is no pointer still misses (a descriptor)
    padding: bool = False


_closed_calls: dict[str, _ClosedCall] = {
    op: _ClosedCall(_closed_call, _closed_call_alt) for op in ("mm", "addmm", "bmm")
}

_WINDOW = 1 << 21  # 2 MiB: the address bits varied between the two harvest sets


def _operand_spans(metas: tuple) -> list[int]:
    # per operand the bytes from its base to one past its last element
    return [
        (1 + sum((n - 1) * abs(st) for n, st in zip(sizes, strides)))
        * torch.empty(0, dtype=dtype).element_size()
        for dtype, sizes, strides in metas
    ]


def _harvest_operands(
    which: int, metas: tuple, aligns: tuple, dev: torch.device
) -> tuple[list, list]:
    # each operand at the alignment class of the real one (1..256), carved
    # from the set's own buffer in 2 MiB windows: the first set at offset
    # `class` in its window, the second at `class` with every bit above the
    # class up to the window flipped, so that the two sets share no address
    # bit between the class and the window (and the two buffers are different
    # allocations above it): an image field derived from address bits finer
    # than the class differs between the two captures instead of surviving as
    # a constant of the template. The buffer lives as long as its set: the
    # harvest's captures are the only readers, and a buffer kept for the
    # process held the largest set ever harvested (round 8, F4)
    spans = _operand_spans(metas)
    windows = [(span + 2 * _WINDOW - 1) // _WINDOW for span in spans]
    need = (sum(windows) + 1) * _WINDOW
    buf = torch.empty(need, dtype=torch.uint8, device=dev)
    out = []
    cursor = (-buf.data_ptr()) % _WINDOW
    for (dtype, sizes, strides), align, span, n in zip(metas, aligns, spans, windows):
        delta = align
        if which == 1:
            delta ^= (_WINDOW - 1) & ~(2 * align - 1)
        flat = buf[cursor + delta : cursor + delta + span].view(dtype)
        out.append(flat.as_strided(sizes, strides))
        cursor += n * _WINDOW
    return out, spans


def _harvest_capture(
    C: Any,
    st: torch.cuda.Stream,
    device: int,
    call: Any,
    anchor: torch.Tensor,
    smear: int,
) -> tuple:
    # one capture of the closed call on `st` with its allocations logged:
    # (nodes, graph or None, [(address, bytes)]). A kernel on `anchor` runs
    # ahead of the call inside the capture: the call's first node gets an
    # incoming edge whose type says whether the library launched it with
    # programmatic stream serialization (a capture records that as edge
    # data, not as a node attribute); harvest_nodes drops the anchor
    # the library leaves uninitialized padding in some parameter structs
    # (stack leftovers): the stack is smeared with one pattern before the
    # first three captures and another before the fourth, so that padding
    # differs between them and is classified as host state
    C._host_trace_stack_smear(smear)
    # a CUDAGraph with its own pool: the allocator serves the call's
    # allocations from it, so the log is read off the pool (Recorder.h
    # alloc_log_begin). The caller holds every graph until the harvest is
    # over: a CUDAGraph's reset clears the cuBLAS workspace cached for its
    # stream (CUDAGraph.cpp), and a capture after that would allocate the
    # workspace inside itself where the earlier captures did not
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    pool = torch.cuda.graph_pool_handle()
    C._host_trace_alloc_log_begin(device, pool)
    done = False
    try:
        with torch.cuda.stream(st):
            graph.capture_begin(pool=pool, capture_error_mode="thread_local")
            try:
                anchor.fill_(1)
                call()
                done = True
            finally:
                try:
                    graph.capture_end()
                except Exception:
                    if done:
                        raise
    finally:
        log = C._host_trace_alloc_log_end()
    nodes = C._host_trace_harvest_nodes(graph.raw_cuda_graph(), anchored=True)
    return nodes, graph, log


def _harvest(key: tuple, spec: tuple, device: int) -> _GemmTemplate:
    """Two raw captures of the closed call on scratch operands at different
    addresses; the qwords that differ between them are the pointer slots,
    each of which must equal exactly one operand's address in both. The
    call's own allocations are logged in every capture: a qword into one of
    them is a scratch slot (the library's per-call workspace, the host's
    contiguous copy of an operand), which the replay points into its arena.
    A third capture on another stream finds the workspace slots: cuBLAS
    gives each stream its own workspace and bakes its base into the image.
    A fourth, through another binding path, finds the host slots: addresses
    of the call's own host state (stack and heap) that cuBLAS leaves in some
    images; they differ per call path and are dead or library-owned at
    replay, so the template keeps its own. Memset nodes (a split-K
    semaphore, a bias path's scratch)
    are kept with their destination classified the same way."""
    global _gemm_harvests
    op, scalars, metas, aligns = spec
    closed = _closed_calls[op]
    # the operands the harvest hands in (all of a GEMM's; an override's
    # inputs) and, behind them, the ones the call allocates and returns
    n_out = closed.outputs(scalars) if callable(closed.outputs) else closed.outputs
    handed = len(metas) - n_out
    spans = _operand_spans(metas)
    C = torch._C
    streams = _harvest_streams.get(device)
    if streams is None:
        streams = _harvest_streams[device] = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
        )
    stream, other = streams
    dev = torch.device("cuda", device)
    # what the call returned in each capture (an override's outputs), kept
    # until the harvest is over so that no later capture's pool hands their
    # addresses out again
    returned: list = []
    run, alt = closed.run, closed.alt
    with torch.cuda.stream(stream):
        first, _ = _harvest_operands(0, metas[:handed], aligns[:handed], dev)
        second, _ = _harvest_operands(1, metas[:handed], aligns[:handed], dev)
        anchor = _harvest_anchors.get(device)
        if anchor is None:
            anchor = _harvest_anchors[device] = torch.empty(1, device=dev)
        # the library's workspace for this stream (an override's first compile)
        run(op, scalars, first)
        stream.synchronize()
        t0 = time.perf_counter()
        try:
            nodes_a, graph, log_a = _harvest_capture(
                C,
                stream,
                device,
                lambda: returned.append(run(op, scalars, first)),
                anchor,
                0xA5,
            )
        except TapeMismatch as e:
            # the library's call is more than kernels and memsets (a memcpy
            # of host scalars for beta / alpha other than 1): a named miss
            raise Miss(f"closed {op} with scalars {scalars}: {e}") from None
        nodes_b, graph_b, log_b = _harvest_capture(
            C,
            stream,
            device,
            lambda: returned.append(run(op, scalars, second)),
            anchor,
            0xA5,
        )
        ws_a = set(C._host_trace_blas_workspaces(stream.cuda_stream))
        harvest_us = (time.perf_counter() - t0) * 1e6 / 2
    with torch.cuda.stream(other):
        run(op, scalars, first)  # this stream's workspace
        other.synchronize()
        nodes_c, graph_c, log_c = _harvest_capture(
            C,
            other,
            device,
            lambda: returned.append(run(op, scalars, first)),
            anchor,
            0xA5,
        )
        ws_c = set(C._host_trace_blas_workspaces(other.cuda_stream))
    with torch.cuda.stream(stream):
        nodes_d, graph_d, log_d = _harvest_capture(
            C,
            stream,
            device,
            lambda: returned.append(alt(op, scalars, first)),
            anchor,
            0x5A,
        )
    # the three throwaway graphs go together, after the last capture
    del graph_b, graph_c, graph_d
    stack, heaps = _host_mappings()
    _gemm_harvests += 1
    # per capture every operand's address: the handed set's (the second set in
    # the second capture, the first in the others), then the call's returns,
    # each an allocation of the call in the order returned (what the tape
    # allocates ahead of the region); the other allocations are its scratch,
    # and the layout of the two among the log is the template's
    logs = [log_a, log_b, log_c, log_d]
    addresses = []
    layout: tuple = ()
    for x, tensors in enumerate((first, second, first, first)):
        addrs = [t.data_ptr() for t in tensors]
        if n_out:
            outs = returned[x]
            outs = tuple(outs) if isinstance(outs, (list, tuple)) else (outs,)
            outs = tuple(t for t in outs if t is not None)
            bases = [a for a, _n in logs[x]]
            positions = [
                bases.index(t.data_ptr()) if t.data_ptr() in bases else -1 for t in outs
            ]
            if len(outs) != n_out or positions != sorted(
                {p for p in positions if p >= 0}
            ):
                raise Miss(
                    f"closed {op}: the call's returns are not distinct allocations of the call in order (returned {[hex(t.data_ptr()) for t in outs]}, allocated {[hex(a) for a in bases]}): not rebindable"
                )
            found = tuple(
                "out" if p in positions else "scratch" for p in range(len(bases))
            )
            if x == 0:
                layout = found
            elif found != layout:
                raise Miss(
                    f"closed {op}: the call's allocations differ between captures ({layout} vs {found})"
                )
            addrs += [bases[p] for p in positions]
            logs[x] = [e for p, e in enumerate(logs[x]) if p not in positions]
        addresses.append(addrs)
    log_a, log_b, log_c, log_d = logs
    kinds = tuple(n["kind"] for n in nodes_a)
    for others in (nodes_b, nodes_c, nodes_d):
        if tuple(n["kind"] for n in others) != kinds:
            raise Miss(
                f"closed {op}: the call's node structure differs between captures at the same shape"
            )
    if any(
        a["kind"] == "kernel"
        and (
            a["func"] != b["func"]
            or a["grid"] != b["grid"]
            or a["block"] != b["block"]
            or a["attrs"] != b["attrs"]
            or a["programmatic"] != b["programmatic"]
        )
        for a, b in zip(nodes_a, nodes_b)
    ):
        raise Miss(
            f"closed {op}: two captures at the same shape chose different kernels"
        )
    for others in (nodes_c, nodes_d):
        if any(
            a["kind"] == "kernel"
            and (
                a["func"] != o["func"]
                or len(a["image"]) != len(o["image"])
                or a["programmatic"] != o["programmatic"]
            )
            for a, o in zip(nodes_a, others)
        ):
            raise Miss(
                f"closed {op}: the call chose or launched another kernel on another stream or path"
            )
    scratch = [n for _addr, n in log_a]
    for log in (log_b, log_c, log_d):
        if [n for _addr, n in log] != scratch:
            raise Miss(
                f"closed {op}: the call's own allocations differ between captures ({scratch} vs {[n for _addr, n in log]} bytes)"
            )

    def classify(qa: int, qb: int, qc: int, qd: int) -> Any:
        # the role of one qword across the four captures: ("op", i, delta),
        # ("scratch", j, delta), ("ws", 0, 0), ("host", 0, 0), None for a
        # constant, or False for a value that fits no role
        for j, (ta, n) in enumerate(log_a):
            if ta <= qa < ta + n:
                delta = qa - ta
                if (
                    qb - log_b[j][0] == delta
                    and qc - log_c[j][0] == delta
                    and qd - log_d[j][0] == delta
                ):
                    return ("scratch", j, delta)
                return False
        if qa != qb:
            # moved with the operands: one operand's address plus a constant
            # (a tile pointer, a descriptor's base) in every capture
            roles = [
                (i, qa - addresses[0][i])
                for i in range(len(spans))
                if 0 <= qa - addresses[0][i] < spans[i]
                and all(
                    q - addresses[x][i] == qa - addresses[0][i]
                    for x, q in ((1, qb), (2, qc), (3, qd))
                )
            ]
            if len(roles) == 1:
                return ("op", *roles[0])
            return False
        if qa != qc:
            # moved with the stream only: the workspace base, which must be
            # the one cuBLAS registered for the harvesting handle on each
            # stream (CublasHandlePool.cpp); another stream-dependent value
            # fits no role
            return ("ws", 0, 0) if qa == qd and qa in ws_a and qc in ws_c else False
        if qa != qd or stack[0] <= qa < stack[1]:
            # host state of the call (a stack or heap address): per call
            # path, dead or library-owned at replay; kept
            return ("host", 0, 0)
        return None

    nodes = []
    for a, b, c, d in zip(nodes_a, nodes_b, nodes_c, nodes_d):
        if a["kind"] == "memset":
            if (a["value"], a["elem"], a["width"]) != (
                b["value"],
                b["elem"],
                b["width"],
            ) or (
                a["value"],
                a["elem"],
                a["width"],
            ) != (c["value"], c["elem"], c["width"]):
                raise Miss(
                    f"closed {op}: a memset of the call differs between captures"
                )
            role = classify(a["dst"], b["dst"], c["dst"], d["dst"])
            if role is False or (role is not None and role[0] == "host"):
                raise Miss(
                    f"closed {op}: a memset of the call targets an address that is not an operand, workspace or scratch of the call: not rebindable"
                )
            nodes.append(
                {
                    "kind": "memset",
                    "name": "memset",
                    "dst_role": role,
                    "dst": a["dst"],
                    "value": a["value"],
                    "elem": a["elem"],
                    "width": a["width"],
                    "bytes": a["elem"] * a["width"],
                    "programmatic": a["programmatic"],
                }
            )
            continue
        img_a, img_b, img_c, img_d = a["image"], b["image"], c["image"], d["image"]
        if len(img_a) != len(img_b):
            raise Miss(
                f"closed {op}: two captures at the same shape differ in image size"
            )
        image = bytearray(img_a)
        n = len(img_a)
        slots: list = []  # (offset, operand, delta): the qword is operand + delta
        scratch_slots: list = []  # (offset, allocation, delta)
        ws_slots: list = []
        host_slots: list = []
        ws_values: set = set()
        covered = bytearray(n)

        # the bytes inside the driver's parameters; the rest is the struct's
        # padding between them, whatever the launch left there
        inside = bytearray(n)
        for start, size in a["layout"]:
            inside[start : start + size] = b"\x01" * size

        def q(img: bytes, off: int) -> int:
            return int.from_bytes(img[off : off + 8], "little")

        def free(off: int) -> bool:
            return not any(covered[off : off + 8])

        # pointers sit at 8-byte offsets in most images and at 4-byte ones in
        # packed parameter structs (cutlass 2.x); a window is classified on
        # the aligned pass first so a half-pointer window never wins
        for start in (0, 4):
            for off in range(start, n - 7, 8):
                if not free(off):
                    continue
                qa, qb, qc, qd = (
                    q(img_a, off),
                    q(img_b, off),
                    q(img_c, off),
                    q(img_d, off),
                )
                role = classify(qa, qb, qc, qd)
                if role is False and qa == qb == qd and qa != qc and not closed.padding:
                    raise Miss(
                        f"closed {op}: {a['name']} holds a stream-dependent pointer at byte {off} ({qa:#x} on the harvest stream, {qc:#x} on the other) that is not the stream's registered cuBLAS workspace ({sorted(map(hex, ws_a))} / {sorted(map(hex, ws_c))}): not rebindable"
                    )
                if (
                    role is False
                    and closed.padding
                    and all(
                        img[off : off + 4] == img_a[off : off + 4]
                        for img in (img_b, img_c, img_d)
                    )
                ):
                    # the low half a constant, the high half moving: the
                    # padding behind a 32-bit field (an address-derived value
                    # cannot look so: the second operand set flips the address
                    # bits below the 2 MiB window and lives in another buffer,
                    # so its low 32 bits differ)
                    covered[off : off + 8] = b"\x01" * 8
                    host_slots.append((off, "dead"))
                    continue
                if role is None and closed.padding:
                    # a constant of the four captures that is the low half of
                    # an address in this thread's stack (an ATen kernel's
                    # uninitialized functor bytes, the same at every capture
                    # of one call depth, another at the build's): host state
                    half = next(
                        (
                            h
                            for h in (0, 4)
                            if _stack_low32(
                                int.from_bytes(img_a[off + h : off + h + 4], "little"),
                                stack,
                            )
                        ),
                        None,
                    )
                    if half is not None:
                        covered[off : off + 8] = b"\x01" * 8
                        host_slots.append((off, f"stack32:{half}"))
                        continue
                if not role:
                    continue
                covered[off : off + 8] = b"\x01" * 8
                if role[0] == "op":
                    slots.append((off, role[1], role[2]))
                    image[off : off + 8] = bytes(8)
                elif role[0] == "scratch":
                    scratch_slots.append((off, role[1], role[2]))
                    image[off : off + 8] = bytes(8)
                elif role[0] == "ws":
                    ws_slots.append(off)
                    ws_values.add((qa, qc))
                    image[off : off + 8] = bytes(8)
                else:
                    cls = _host_slot_class(
                        off, (img_a, img_b, img_c, img_d), stack, heaps
                    )
                    if cls is None and closed.padding and (qa != qc or qa != qd):
                        cls = "dead"
                    if cls is None:
                        seen = " / ".join(
                            img[off : off + 8].hex()
                            for img in (img_a, img_b, img_c, img_d)
                        )
                        raise Miss(
                            f"closed {op}: {a['name']} carries per-call host state at byte {off} that is neither a stack nor a heap address nor stack-smear padding (qword over the captures: {seen}): not rebindable"
                        )
                    host_slots.append((off, cls))
        # ATen's copy kernel (the host's own contiguous copy of an operand
        # before the library call) carries the uninitialized entries of its
        # offset calculator beyond `dims`, stack leftovers the kernel never
        # reads (OffsetCalculator.cuh fills entries below `dims` only): kept
        # like host state. Its live fields are the count, the two data
        # pointers (classified above) and the entries below `dims`.
        dead_ok = "direct_copy_kernel_cuda" in a["name"]
        for off in range(n):
            if covered[off]:
                continue
            if not inside[off]:
                lo = off - off % 8
                host_slots.append((lo, "dead"))
                covered[lo : lo + 8] = b"\x01" * 8
                continue
            if (
                img_a[off] != img_b[off]
                or img_a[off] != img_c[off]
                or img_a[off] != img_d[off]
            ):
                lo = off - off % 8
                if dead_ok or (
                    closed.padding
                    and (img_a[off] != img_c[off] or img_a[off] != img_d[off])
                ):
                    host_slots.append((lo, "dead"))
                    covered[lo : lo + 8] = b"\x01" * 8
                    continue
                seen = " / ".join(
                    img[lo : lo + 8].hex() for img in (img_a, img_b, img_c, img_d)
                )
                raise Miss(
                    f"closed {op}: {a['name']} carries per-call state at byte {off} that is not an operand, workspace or host address (a descriptor): not rebindable (qword {lo} over the captures: {seen}; the call's allocations: {log_a})"
                )
        if len(ws_values) > 1:
            raise Miss(
                f"closed {op}: {a['name']} holds more than one stream-dependent pointer; only one workspace base is rebased"
            )
        nodes.append(
            {
                "kind": "kernel",
                "func": a["func"],
                "name": a["name"],
                "grid": tuple(a["grid"]),
                "block": tuple(a["block"]),
                "smem": a["smem"],
                "image": bytes(image),
                "layout": a["layout"],
                "slots": slots,
                "scratch_slots": scratch_slots,
                "ws_slots": ws_slots,
                "host_slots": host_slots,
                "attrs": tuple(a["attrs"]),
                "programmatic": a["programmatic"],
            }
        )
    uses_ws = any(
        n["ws_slots"] if n["kind"] == "kernel" else (n["dst_role"] or ("",))[0] == "ws"
        for n in nodes
    )
    return _GemmTemplate(
        key,
        nodes,
        harvest_us,
        # an override's kernel ties nothing to the graph, whose pool would
        # hold the call's outputs for the template's lifetime
        graph=None if n_out else graph,
        scratch=scratch,
        layout=layout if n_out else ("scratch",) * len(scratch),
        kinds=kinds,
        uses_ws=uses_ws,
    )


def _template(key: tuple, spec: tuple, device: int) -> _GemmTemplate:
    with _gemm_lock:
        tpl = _gemm_templates.get(key)
        if tpl is None:
            try:
                tpl = _harvest(key, spec, device)
            except Miss as e:
                tpl = _GemmTemplate(key, [], 0.0, scratch=[], miss=str(e))
            _gemm_templates[key] = tpl
        tpl.hits += 1
        if tpl.miss:
            raise Miss(tpl.miss)
        return tpl


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
    # the symbol a value was created as (the trace's ShapeEnv never replaces
    # one, so this is also the symbol every later expression names)
    if isinstance(v, _SYM_TYPES) and isinstance(v.node._expr, sympy.Symbol):
        return str(v.node._expr)
    return None


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


def _at_least_one(v: Any) -> Any:
    # std::max<int64_t>(1, v) on a size: the size itself when its hint is at
    # least one (a size symbol is positive on the tape), the constant 1 otherwise
    return v if _hint(v) >= 1 else 1


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
    """A tensor the host sees during a trace: the trace's CUDA device (or
    the CPU, for a pinned input a host copies from), symbolic sizes, strides
    and storage offset, no storage. Every one belongs to a root (an input's
    storage or a host allocation); views share their source's root."""

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
        device: torch.device | None = None,
    ):
        device = tr.device if device is None else device
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            sizes,
            strides,
            storage_offset=offset,
            dtype=dtype,
            device=device,
            dispatch_sizes_strides_policy="strides",
        )
        torch._C._host_trace_drop_storage(t)
        t._root = root
        t._sym_strides = list(strides)
        t._sym_offset = offset
        elem = torch.empty_strided(sizes, strides, dtype=dtype, device="meta")
        if not (isinstance(offset, int) and offset == 0):
            elem = elem.as_strided(sizes, strides, offset)
        t._fake = FakeTensor(tr.fake_mode, elem, device)
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

    # a Python condition's read of the address (torch._native's overrides test
    # a pointer's alignment): the root's address symbol plus the view's offset,
    # as the C++ hosts read it through sym_const_data_ptr; a comparison on it
    # lands in the ShapeEnv's guards
    def data_ptr(self) -> Any:
        return self._root.sym + self._sym_offset * self.element_size()

    const_data_ptr = data_ptr
    mutable_data_ptr = data_ptr

    # a DLPack export hands the storage to another library: a launch the trace
    # cannot see. The CuTe DSL's from_dlpack is hooked under the trace
    # (torch/cuda/_host_trace_cute_dsl.py); any other export declines
    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        raise Declined(
            "host_trace: a DLPack export of a traced tensor to a library the recorder does not hook "
            "(the CuTe DSL's from_dlpack and compiled programs are hooked; a program loaded from a "
            "compiled module is not): its launch is not recorded on the tape (declined)"
        )

    def __dlpack_device__(self) -> Any:
        return self.__dlpack__()

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
        if getattr(_active, "trace", None) is not None and _has_symint(items):
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
        # a guard evaluated again by another op or the Python between ops (the
        # record keeps one row per relation): raw index -> the other raisers'
        # rows, so a consumer deciding by the raiser sees every one of them
        self.guard_also: dict[int, list[tuple[int, int, str]]] = {}
        self._raw_index: dict[sympy.Basic, int] = {}
        self._key_raw: dict[tuple, int] = {}
        # every binary SymNode operation on this ShapeEnv is stored here once
        self._symop_cache = _SymOpMemo(self.domain)

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
            self._also(key, orig_expr)
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
        raw = self._raw_index.get(g)
        if raw is not None:
            self._key_raw[key] = raw
        return concrete

    def _also(self, key: tuple, orig_expr: sympy.Basic) -> None:
        raw = self._key_raw.get(key)
        if raw is None or self.attribute is None:
            return
        row = self.attribute(self, orig_expr)
        rows = self.guard_also.get(raw)
        if row == self.guard_rows[raw] or (rows is not None and row in rows):
            return
        if rows is None:
            self.guard_also[raw] = [row]
        else:
            rows.append(row)

    def _record(
        self, g: sympy.Basic, size_oblivious: bool = False, row: tuple | None = None
    ) -> None:
        # the same relation from another evaluation (`not a < b` and `a >= b`,
        # a division's domain the host then tests itself) is one guard
        if g is not sympy.true and g not in self._recorded:
            self._recorded.add(g)
            self.guards.append(ShapeGuard(g, _NO_SLOC, size_oblivious))
            self._raw_index[g] = len(self.guards) - 1
            if row is None:
                row = _NO_ROW if self.attribute is None else self.attribute(self, g)
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
        use of the symbols, and the notes of the kept guards (`note_pin`)."""
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


# a guard's origin by the route the mode took for the innermost op: which code
# raised it, without a frame walk (a sibling entry's own Python before its
# C++ binding is "entry-python", the sibling's C++ once the binding runs
# "entry-host"; a converted host reached by redispatch "host-branch"; a
# torch._native override's condition "override-cond")
_ORIGIN_OF_ROUTE = {
    "host": "host-branch",
    "view": "view-meta",
    "alloc": "alloc-meta",
    "region": "region-record",
    "composite": "composite-body",
    "override": "override-cond",
    None: "recorder",
}
# the row of a guard no trace attributes (an env used outside one)
_NO_ROW = (-1, 0, "python")


class _OpRec:
    """One op the trace mode dispatched: a row of the tape's op table. The
    route the mode took (alloc, view, region, entry = a traced sibling entry,
    host = a converted host, override = a torch._native override, composite =
    a decomposition or an E38 body), the arguments as the op received them
    (traced tensors by reference), the outputs as returned, the recorder's
    `seq` at entry and exit (every record the op issued, allocations and
    memsets included, lies in [seq[0], seq[1])), and the raw guard indices
    raised while it ran (nested ops' included; the op's own are the rows
    naming its index)."""

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
        self.shape_env = _TraceShapeEnv()
        self.fake_mode = FakeTensorMode(shape_env=self.shape_env)
        # a cached fake output is materialized with set_(), whose contiguity
        # refresh guards `size == 1` per dim on the hint; the uncached meta
        # path asks nothing until a view needs it
        self.fake_mode.cache_enabled = False
        self.rec = torch._C._HostTraceRecorder(device)
        # the recorder's capture stream is the current one from here (its
        # scope holds a stream guard): what a launch under the trace goes to
        self.stream = torch.cuda.current_stream(device)
        # the Python-launched Triton kernels of this trace
        # (torch/cuda/_host_trace_triton.py), set by _trace_once
        self.triton: Any = None
        # the CuTe DSL invocations of this trace (torch/cuda/_host_trace_cute.py),
        # set by its tracing scope in _trace_once
        self.cute: Any = None
        # the mode of this trace (_TraceMode sets it): what a closed region's
        # description traces of its own runs under it
        self.mode: Any = None
        self.tensors: list = []
        self.inputs: list[_InputRec] = []
        self.allocs: list[_AllocRec] = []
        self.regions: list[_RegionRec] = []
        # the op table, in dispatch order (the trace mode's enter_op / exit_op)
        self.ops: list[_OpRec] = []
        self.shape_env.attribute = _guard_attribution
        # the real tensor behind each input root (by root name): a host that
        # needs process-lifetime state keyed by the real storage (a symmetric
        # memory handle, torch/cuda/_host_trace_symm.py) looks it up here
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
        src = _Src(name)
        if self.hints is not None:
            value = self.hints.get(name, value)
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
        # a pinned host input is read by copy_h2d at trace time, so its hint
        # must stay the real address; device inputs get a placeholder hint
        pinned = t.is_cpu and t.is_pinned()
        hint = base if pinned else _placeholder_address(base)
        sym = self.symbol(hint, f"{name}.base")
        root = _Root(
            f"p{position}", sym, t.element_size(), cow=bool(torch._C._is_cow_tensor(t))
        )
        self.shape_env.note_root(sym, root.name, alloc=False)
        traced = _TracedTensor(
            self, root, sizes, strides, offset, t.dtype, device=t.device
        )
        self.real_inputs[root.name] = t
        self.inputs.append(
            _InputRec(
                position,
                name,
                t.dtype,
                sizes,
                strides,
                offset,
                root,
                t.device,
                pinned,
            )
        )
        return traced

    def allocate(self, func: Any, args: tuple, kwargs: dict) -> _TracedTensor:
        device = kwargs.get("device")
        if device is not None:
            # `device(at::kCUDA)` without an index is the current device
            dev = torch.device(device)
            if dev.type != self.device.type or dev.index not in (
                None,
                self.device.index,
            ):
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

    def closed_region(self, func: Any, args: tuple, kwargs: dict) -> _TracedTensor:
        """aten.mm / aten.addmm / aten.bmm through cuBLAS: record the call
        as a closed region (operands and output as values), allocate its
        output, issue nothing. The output is what the ordinary op would
        allocate: a contiguous [M, N] ([B, M, N] for bmm) of the operands'
        dtype; for the out= forms, the given tensor."""
        base = _CLOSED_OUT_OPS.get(func, func)
        batched = base is aten.bmm.default
        if base in (aten.mm.default, aten.bmm.default):
            op = "bmm" if batched else "mm"
            mats, bias, scalars = (args[0], args[1]), None, ()
        else:
            op = "addmm"
            mats, bias = (args[1], args[2]), args[0]
            scalars = (kwargs.get("beta", 1), kwargs.get("alpha", 1))
            if any(s != 1 for s in scalars):
                # the host then adds the bias itself (a copy kernel of its
                # own before the library call): not one closed call
                raise Declined(
                    f"host_trace: {func} with beta / alpha other than 1 is not recorded as a closed region (declined)"
                )
            if not all(isinstance(s, (int, float)) for s in scalars):
                raise Declined(
                    f"host_trace: {func} with a symbolic beta / alpha is not recorded as a closed region (declined)"
                )
        operands = ([bias] if bias is not None else []) + list(mats)
        for t in operands:
            if not isinstance(t, _TracedTensor):
                raise Declined(
                    f"host_trace: {func} on a tensor the trace did not create; a closed region's operands must be inputs, allocations or views of them (declined)"
                )
            if t.device.type != "cuda":
                raise Declined(
                    f"host_trace: {func} on {t.device} inside a trace (declined)"
                )
            if t.dtype not in _GEMM_DTYPES:
                raise Declined(
                    f"host_trace: {func} in {t.dtype} is not recorded as a closed region (declined)"
                )
        a, b = mats
        nd = 3 if batched else 2
        if a.dim() != nd or b.dim() != nd:
            raise Declined(
                f"host_trace: {func} with {a.dim()}-D and {b.dim()}-D operands; only {nd}-D closed GEMMs are recorded here (declined)"
            )
        if a.dtype != b.dtype or (bias is not None and bias.dtype != a.dtype):
            raise Declined(f"host_trace: {func} with mixed dtypes (declined)")
        # what the host hands to cuBLAS as is (cuBlasCommonArgs.h
        # prepare_matrix_for_cublas; Blas.cpp prepare_batch_matrix_for_cublas
        # for the batched ops): a unit stride along one dimension and a
        # leading dimension of at least the other extent, or a dense matrix.
        # Anything else it clones into a contiguous temporary of its own
        # first: eager's copy kernel, recorded as the tape's launch ahead of
        # the region, which then reads the clone (_operand_copy). Decided on
        # the traced strides without a guard: the strides are in the template
        # key, and a later layout is another key or another copy
        given = kwargs.get("out")
        if given is not None:
            # the out= form (Inductor's extern GEMMs write into buffers of
            # its own): the region writes the given tensor. What eager's host
            # takes as is (Blas.cpp baddbmm_out_cuda_impl for the batched
            # ops, cuBlasCommonArgs.h prepare_matrix_for_cublas for a 2-D
            # result): a unit stride along one dimension with the other
            # stride at least that extent (a row-major buffer padded beyond
            # N: the host passes the leading dimension), or a dense layout; a
            # result it would compute into a copy of and copy back declines.
            # Decided on the traced strides, as the operands are: the strides
            # are in the template key. The tensor may be a view of an input
            # (a buffer the caller reuses: Inductor's donated buffers), which
            # the region then writes
            if not isinstance(given, _TracedTensor):
                raise Declined(
                    f"host_trace: {func} with out= that is a plain {type(given).__name__} made outside the trace (declined)"
                )
            if given.dtype != a.dtype or given.dim() != nd:
                raise Declined(
                    f"host_trace: {func} with out= of {given.dtype} {tuple(given.shape)} for a {a.dtype} {nd}-D result (declined)"
                )
        if batched:
            # the batched host reads the result as column-major when its
            # strides say so (Blas.cpp baddbmm_out_cuda_impl: a unit stride
            # along dim 1 and dim 2's stride at least M), as C^T = B^T A^T
            # otherwise, and the roles of the two batches follow that choice;
            # the functional op's contiguous result is column-major exactly
            # when N == 1
            m_s, k_s, n_s = a.shape[1], a.shape[2], b.shape[2]
            n_ = _hint(n_s)
            if given is None:
                transpose_result = n_ != 1
            else:
                gm, gn = (_hint(v) for v in given.shape[1:])
                gst = [_hint(v) for v in given._sym_strides]
                if gst[1] == 1 and (gn == 1 or gst[2] >= max(1, gm)):
                    transpose_result = False
                elif gst[2] == 1 and (gm == 1 or gst[1] >= max(1, gn)):
                    transpose_result = True
                else:
                    raise Declined(
                        f"host_trace: {func} with out= of shape {tuple(given.shape)} with strides {tuple(given._sym_strides)} is not a cuBLAS result as is; the host would compute into a copy and copy it back (declined)"
                    )
            checks = (
                (("mat1", a, k_s, m_s), ("mat2", b, n_s, k_s))
                if transpose_result
                else (("mat1", a, m_s, k_s), ("mat2", b, k_s, n_s))
            )
            mats = list(mats)
            for index, (name, t, rows, cols) in enumerate(checks):
                fast, lead = (2, 1) if transpose_result else (1, 2)
                sst = t._sym_strides
                disjuncts = (
                    (sst[fast] == 1, sst[lead] >= _at_least_one(rows)),
                    (sst[lead] == 1, sst[fast] >= _at_least_one(cols)),
                    (
                        sst[1] != 0,
                        sst[2] != 0,
                        *_contiguous_terms(list(t.shape), list(t._sym_strides)),
                    ),
                )
                if not self._operand_ready(disjuncts):
                    mats[index] = self._operand_copy(t)
            a, b = mats
        else:
            mats = list(mats)
            for index, (name, t) in enumerate((("mat1", a), ("mat2", b))):
                rows, cols = t.shape
                sst = t._sym_strides
                disjuncts = (
                    (sst[0] == 1, sst[1] >= _at_least_one(rows)),
                    (sst[1] == 1, sst[0] >= _at_least_one(cols)),
                    tuple(_dense_terms(list(t.shape), list(t._sym_strides))),
                )
                if not self._operand_ready(disjuncts):
                    mats[index] = self._operand_copy(t)
            a, b = mats
            if given is not None:
                rows, cols = (_hint(n) for n in given.shape)
                s0, s1 = (_hint(st) for st in given._sym_strides)
                ready = (
                    (s0 == 1 and s1 >= max(1, rows))
                    or (s1 == 1 and s0 >= max(1, cols))
                    or all(
                        bool(_hint(term))
                        for term in _dense_terms(
                            list(given.shape), list(given._sym_strides)
                        )
                    )
                )
                if not ready:
                    raise Declined(
                        f"host_trace: {func} with out= of shape {tuple(given.shape)} with strides {tuple(given._sym_strides)} is not a cuBLAS result as is; the host would compute into a copy and copy it back (declined)"
                    )
                if bias is not None and not all(
                    bool(_hint(term))
                    for term in _contiguous_terms(
                        list(given.shape), list(given._sym_strides)
                    )
                ):
                    # Blas.cpp addmm_out_cuda_impl fuses a 1-D bias into the
                    # library call (the Lt epilogue) for a contiguous result
                    # only; into any other layout the host copies the bias in
                    # with a kernel of its own before the GEMM: not one closed
                    # call
                    raise Declined(
                        f"host_trace: {func} with a bias into an out= with strides {tuple(given._sym_strides)} that is not contiguous: the host copies the bias into the result before the GEMM (declined)"
                    )
        operands = ([bias] if bias is not None else []) + list(mats)
        if bias is not None and bias.dim() == 1 and _hint(bias._sym_strides[0]) != 1:
            raise Declined(
                f"host_trace: {func}: a bias with stride {bias._sym_strides[0]} is not a cuBLAS operand as is; the host would copy it first, which is not recorded as a closed region (declined)"
            )
        # the library's own shape checks, as guards
        if batched and bool(a.shape[0] != b.shape[0]):
            raise Declined(
                f"host_trace: {func}: batch1 and batch2 must have the same number of batches ({a.shape} x {b.shape})"
            )
        if bool(a.shape[-1] != b.shape[-2]):
            raise Declined(
                f"host_trace: {func}: mat1 and mat2 shapes cannot be multiplied ({a.shape} x {b.shape})"
            )
        m, n = a.shape[-2], b.shape[-1]
        out_shape = [a.shape[0], m, n] if batched else [m, n]
        if bias is not None:
            if bias.dim() == 1:
                # a length-1 bias broadcasts to N under another template of
                # the same output: the region key's decision (a kernel choice)
                with torch._C._HostTraceKernelChoice():
                    broadcasts = bool(bias.shape[0] != n)
                if broadcasts:
                    raise Declined(
                        f"host_trace: {func}: bias of {bias.shape} does not broadcast to [M, {n}]"
                    )
            elif bias.dim() == 2:
                # the host copies a 2-D bias into the output before the GEMM
                # (a device memcpy the library call does not own)
                raise Declined(
                    f"host_trace: {func}: a 2-D bias is not recorded as a closed region (declined)"
                )
            else:
                raise Declined(
                    f"host_trace: {func}: a {bias.dim()}-D bias is not recorded (declined)"
                )
        if given is None:
            out = self.allocate(
                aten.empty.memory_format,
                (out_shape,),
                {"dtype": a.dtype, "device": self.device},
            )
        else:
            # the library's shape check on the given result, as guards
            for d, sz in enumerate(out_shape):
                if bool(given.shape[d] != sz):
                    raise Declined(
                        f"host_trace: {func} with out= of shape {tuple(given.shape)} for a {tuple(out_shape)} result (declined)"
                    )
            out = given
        names = ["bias", "mat1", "mat2"] if bias is not None else ["mat1", "mat2"]
        self.region(op, list(zip(names, operands)), [("out", out)], scalars)
        return out

    @staticmethod
    def _operand_ready(disjuncts: tuple) -> bool:
        # the host's "as is" test on an operand, a disjunction of conjunctions
        # over its strides. Decided on the hints when it holds (no guard: the
        # strides are in the template key, and a later layout the host copies
        # misses by name at its harvest). When it fails the host copies, and
        # the tape then holds a copy eager makes for this layout only: one
        # failing term per disjunct is guarded, so a later layout the host
        # takes as is misses by a guard (E24) instead of replaying the copy
        for terms in disjuncts:
            if all(bool(_hint(term)) for term in terms):
                return True
        for terms in disjuncts:
            for term in terms:
                if not bool(_hint(term)):
                    if not bool(term):
                        break
                    raise Declined(
                        "host_trace: a closed GEMM operand's layout test changed under the guard (declined)"
                    )
        return False

    def _operand_copy(self, t: _TracedTensor) -> _TracedTensor:
        # the host's own copy of an operand it cannot hand to cuBLAS as is
        # (Blas.cpp: tensor.clone(at::MemoryFormat::Contiguous) before the
        # library call): the traced clone under the mode, eager's copy kernel
        # as a launch of the tape and its allocation, ahead of the region
        with self.mode:
            return t.clone(memory_format=torch.contiguous_format)

    def region(
        self, op: str, inputs: list, outputs: list, scalars: tuple
    ) -> _RegionRec:
        # one closed region at this point of the call, over (name, traced
        # tensor) operands
        def desc(name: str, t: _TracedTensor) -> _RegionOperand:
            address = t._root.sym + t._sym_offset * t.element_size()
            return _RegionOperand(
                name, t._root, address, list(t.shape), list(t._sym_strides), t.dtype
            )

        k = len(self.regions)
        rec = _RegionRec(
            self.rec.next_seq(),
            op,
            [desc(nm, t) for nm, t in inputs],
            [desc(nm, t) for nm, t in outputs],
            scalars,
            f"region{k}",
        )
        self.regions.append(rec)
        return rec

    def native_region(self, func: Any, args: tuple, kwargs: dict, entry: Any) -> Any:
        """An op eager serves through a torch._native override on the closed-
        region list (torch/cuda/_host_trace_native.py): the call's operands
        and the outputs the override allocates, as values; nothing issued.
        The harvest runs the op through the router at the region's key."""
        inputs, scalars, outputs, result = entry.describe(self, func, args, kwargs)
        self.region(entry.op, inputs, outputs, scalars)
        _host_trace_cute_dsl.claim(entry.programs)
        return result

    def library_region(self, func: Any, args: tuple, kwargs: dict, entry: Any) -> Any:
        """A closed library op on the region list of torch/cuda/_host_trace_cudnn.py
        (cuDNN attention): the call's operands and the outputs the host
        allocates, as values; nothing issued. The harvest runs the op on
        stand-ins at the region's key."""
        inputs, scalars, outputs, result = entry.describe(self, func, args, kwargs)
        self.region(entry.op, inputs, outputs, scalars)
        return result

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


def _refused_inside_the_trace(func: Any, e: RuntimeError) -> Declined:
    # An op the mode ran during the symbolic run raised: a copy from CPU that
    # the capture forbids, a host's own check on symbolic inputs. Nothing the
    # tape can describe; the decline names the op and keeps the cause.
    if _NO_DATA_PTR in str(e):
        return Declined(
            f"host_trace: {func} read a raw data pointer of a traced tensor "
            "inside the trace: its host is not converted (declined)"
        )
    first = str(e).splitlines()[0] if str(e) else type(e).__name__
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


# c10::SymInt::MAX_UNREPRESENTABLE_INT: an int at or below it is the pointer
# bits of a heap-allocated SymInt (bit 63 set, bit 62 clear)
_SYMINT_POINTER_BITS = -(1 << 62) - 1


def _not_pointer_bits(v: Any) -> None:
    # an ATen composite that reinterprets a SymIntArrayRef as ints
    # (layer_norm.cpp rms_norm_symint hands torch.rms_norm's normalized_shape
    # to _fused_rms_norm so) delivers a traced size here as its node's pointer
    # bits: nothing the trace can pin
    if type(v) is int and v <= _SYMINT_POINTER_BITS:
        raise Declined(
            "host_trace: an int argument of the call is a traced size's pointer bits (an "
            "ATen composite passed a SymInt unchecked to an int[] argument: torch.rms_norm's "
            "normalized_shape written from a traced shape reaches _fused_rms_norm so); write "
            "it as ints (declined)"
        )


def _concrete_ints(x: Any) -> Any:
    # int and int-list arguments of the op under trace (a normalized_shape
    # written from the input's shape): the CUDA kernels are registered on the
    # non-SymInt signatures, and the dispatcher's wrapper asserts on a
    # symbolic element before the host runs. int() guards on the traced value,
    # a sound pin the tape records (the trace serves that value and misses
    # others by name).
    if isinstance(x, torch.SymInt):
        return int(x)
    if isinstance(x, (list, tuple)):
        if any(isinstance(e, torch.SymInt) for e in x):
            x = type(x)(int(e) if isinstance(e, torch.SymInt) else e for e in x)
        for e in x:
            _not_pointer_bits(e)
        return x
    _not_pointer_bits(x)
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
        tr.mode = self
        self.depth = 0
        self.decomposing: list = []  # composite ops whose decomposition or body runs
        self.entering: list = []  # ops whose traced entry is running

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
        except Declined:
            op.declined = True
            raise
        finally:
            self.trace.exit_op(op, out)

    def _route(
        self, func: Any, types: Any, args: tuple, kwargs: dict, op: _OpRec
    ) -> Any:
        if func is aten.resize_.default and isinstance(args[0], _TracedTensor):
            # eager's out= variants resize their out tensor to the result's size
            # (Inductor's `aten.randint.low_out(..., out=buf)` for its seeds): a
            # resize to the tensor's own size is a no-op, each comparison a guard
            t, size = args[0], list(args[1])
            if len(size) == t.dim() and all(
                bool(a == b) for a, b in zip(size, t.shape)
            ):
                return t
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
        # a call eager serves through a torch._native override (the K = 1 bmm:
        # a Triton kernel, not cuBLAS) takes eager's route: the override's
        # Python runs under the mode, its allocations traced, its Triton
        # launch recorded by the hook (torch/cuda/_host_trace_triton.py). An
        # override on the closed-region list (the CuTe ones) is a region
        # instead: its non-tensor arguments are pinned ahead of its condition
        # (the region's scalars)
        entry = _host_trace_native.REGIONS.get(func)
        if entry is not None:
            args = tuple(_concrete_ints(a) for a in args)
            kwargs = {k: _concrete_ints(v) for k, v in kwargs.items()}
        # the override's condition is a kernel choice (E40, LOCAL_MISS 3.1):
        # its comparisons are kernel-tagged and originate from the condition
        op.route = "override"
        with self, torch._C._HostTraceKernelChoice():
            native = _native_override_takes(func, args, kwargs)
        op.route = None
        # the override's program has a launch descriptor (what eager's warm-up
        # launched here, read off eager's own call): eager's route under the
        # trace, the program's call recorded from the descriptor
        # (torch/cuda/_host_trace_cute_desc.py); else the closed region
        if (
            entry is not None
            and native
            and not _host_trace_cute_dsl.descriptor_ahead(entry.programs)
        ):
            op.route = "region"
            return self.trace.native_region(func, args, kwargs, entry)
        # a closed library op on the cuDNN attention list: recorded as a
        # region, never traced into (torch/cuda/_host_trace_cudnn.py)
        entry = _host_trace_cudnn.REGIONS.get(func)
        if entry is not None:
            op.route = "region"
            return self.trace.library_region(func, args, kwargs, entry)
        # a closed library call (cuBLAS): recorded as a region, never traced into
        # (the out= form the same region over the given allocation)
        if (func in _CLOSED_OPS or func in _CLOSED_OUT_OPS) and not native:
            op.route = "region"
            return self.trace.closed_region(func, args, kwargs)
        # an op with a traced sibling host is traceable at any depth: under
        # trace, or inside another host (a layer norm copying a non-contiguous
        # input calls copy_)
        entry = None if native else _TRACED_ENTRIES.get(func)
        # a converted host is traceable under trace and inside another
        # converted host (the SDPA flash entry calls _flash_attention_forward)
        if entry is not None or native or func in _TRACEABLE:
            op.route = (
                "entry" if entry is not None else ("override" if native else "host")
            )
            if entry is not None and func in self.entering:
                raise Declined(
                    f"host_trace: the traced entry for {func} dispatched {func} itself; "
                    "an entry runs its sibling host or declines (declined)"
                )
            # the op under trace: let it reach its CUDA host (or the traced
            # sibling host that stands in for it), which is the code being
            # recorded; the mode stays on for what the host does
            self.depth += 1
            if entry is not None:
                self.entering.append(func)
            try:
                with self, _cow_from_roots() if native else contextlib.nullcontext():
                    # a torch._native override is a Python kernel on the op's
                    # int schema: its integers are pinned to the traced values
                    # whatever the op's own SymInt registration
                    if native or (
                        func not in _SYMINT_KERNELS and func not in _SYMINT_ENTRIES
                    ):
                        args = tuple(_concrete_ints(a) for a in args)
                        kwargs = {k: _concrete_ints(v) for k, v in kwargs.items()}
                    if entry is not None:
                        return entry(*args, **kwargs)
                    return func.redispatch(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA),
                        *args,
                        **kwargs,
                    )
            except Declined:
                raise
            except torch.AcceleratorError:
                raise  # trace() classifies these by CUDA error code
            except RuntimeError as e:
                raise _refused_inside_the_trace(func, e) from e
            except Exception as e:
                if not native:
                    raise
                # the override's Python on traced tensors (a size used where
                # only an int serves): nothing the tape describes
                raise Declined(
                    f"host_trace: torch._native's override of {func} raised inside the trace: "
                    f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''} (declined)"
                ) from e
            finally:
                if entry is not None:
                    self.entering.pop()
                self.depth -= 1
        # a composite op decomposes under the mode into the ones above only
        # where eager's dispatcher runs the same CompositeImplicit kernel; one
        # eager serves with a kernel of its own is traced by a converted host
        # or declined, never decomposed (DECISIONS A190, E34)
        if _decomposes_only_off_cuda(func):
            raise Declined(
                f"host_trace: {func} runs its own CUDA kernel in eager, not its "
                f"CompositeImplicit decomposition; a converted host for {func.name()} "
                "is the way to trace it; the ordinary host serves (declined)"
            )
        # an op with no kernel of its own whose entry on the tensors' backend
        # is a CompositeExplicit body runs that body, eager's own, under the
        # mode: its pieces reach the mode and are traced as eager launches
        # them (slice_backward: the zeros' memset and one copy); a host read
        # inside the body declines where it occurs (DECISIONS E38)
        body = _explicit_body_key(func, _key_below(args, kwargs))
        op.route = "composite"  # a decomposition or an E38 body under the mode
        self.decomposing.append(func)
        try:
            with self:
                if body is not None:
                    return func.redispatch(body, *args, **kwargs)
                r = func.decompose(*args, **kwargs)
        except Declined:
            raise
        except torch.AcceleratorError:
            raise
        except RuntimeError as e:
            raise _refused_inside_the_trace(func, e) from e
        finally:
            self.decomposing.pop()
        if r is not NotImplemented:
            return r
        where = "inside the host" if self.depth else "under trace"
        via = f" (reached from {self.decomposing[0]})" if self.decomposing else ""
        via += _host_trace_cute_dsl.unmet_hint()
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
        # the warm-up call's return value (trace(warm_up=True)); a consumer that
        # treats the warm-up as the call takes it and clears it, like `args`
        self.warm_up_outputs: Any = None
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
        # one slot per random launch (Tape.h RngSlotRec): the launch index, the
        # u32 param over its philox state's intragraph offset, and its own
        # increment; rng_increment is their sum
        self.rng_slots = records["rng_slots"]
        # cudaMemsetAsync calls the host issued (a split reduction's semaphore
        # reset): memset nodes of a replay's graph, in this order
        self.memsets = records["memsets"]
        # host tables as each copy_h2d read them (HostTable.h, one image per
        # copy) and the copies issued from them, from a pinned CPU input or
        # between device addresses (copy_d2d), each record declaring its kind
        # ("h2d" / "d2d"): memcpy nodes of a replay's graph, in this order,
        # the host-sourced ones re-issued from the replay's own staging
        # buffers
        self.host_buffers = records["host_buffers"]
        self.memcpys = records["memcpys"]
        # closed library calls (cuBLAS mm / addmm / bmm): kernel nodes of a
        # replay's graph the tape describes only as a shape key, a harvested
        # template per concrete key
        self.regions = tr.regions
        # a region's outputs are written by the call: their roots join the
        # written roots (an allocation, or the input an out= that is a view
        # of an input writes through the region)
        for r in self.regions:
            for o in r.outputs:
                if o.root.name not in self.written_roots:
                    self.written_roots.append(o.root.name)
        self.outputs = outputs
        # the argument positions among the written roots, in the call's index
        # space: what a binding reads through the mutable accessor first
        # (torch._C._host_trace_materialize)
        written = set(self.written_roots)
        self.written_inputs: tuple[int, ...] = tuple(
            i.position for i in self.inputs if i.root.name in written
        )
        # the op table and, per raw guard of the env's record, (op index,
        # kernel-choice depth, origin); kept_raw maps each kept guard (a
        # position in `guards`) to its raw index
        self.ops = tr.ops
        self.guard_rows = list(tr.shape_env.guard_rows)
        self.guard_also = dict(tr.shape_env.guard_also)
        self.guards, pins, self.guard_notes, self.kept_raw = tr.shape_env.tape_guards()
        self._kept_index: dict | None = None
        if pins:
            self._pin_uses(pins)
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
        # allocation's base (`q`), an opaque result, a host table's root
        env = self.shape_env
        done: dict = {}

        def sub(v: Any) -> Any:
            if not isinstance(v, _SYM_TYPES):
                return v
            node = v.node
            expr: Any = node._expr
            r = done.get(expr)
            if r is None:
                e = expr.xreplace(pins)
                if e is expr:
                    r = v
                elif e.is_Number or isinstance(e, _BOOL_ATOMS):
                    r = node.pytype(e)
                else:
                    wrap = _SYM_WRAP[node.pytype]
                    r = wrap(SymNode(e, env, node.pytype, node._hint))
                done[expr] = r
            return r

        def subs(values: list) -> list:
            return [sub(v) for v in values]

        for a in self.allocs:
            a.sizes, a.strides = subs(a.sizes), subs(a.strides)
        for L in self.launches:
            for p in L["params"]:
                p["value"] = sub(p["value"])
            L["grid"], L["block_expr"] = subs(L["grid"]), subs(L["block_expr"])
            L["smem"] = sub(L["smem"])
        for o in self.opaque:
            o["args"] = subs(o["args"])
        for out in self.outputs:
            out.sizes, out.strides = subs(out.sizes), subs(out.strides)
            out.offset = sub(out.offset)
        for hb in self.host_buffers:
            for q in hb["elements"]:
                q["value"] = sub(q["value"])
        for m in self.memcpys:
            m["src"], m["dst"] = sub(m["src"]), sub(m["dst"])
            m["bytes"] = sub(m["bytes"])
        for r in self.regions:
            for op in (*r.inputs, *r.outputs):
                op.address = sub(op.address)
                op.sizes, op.strides = subs(op.sizes), subs(op.strides)
        self.rng_increment = sub(self.rng_increment)
        for slot in self.rng_slots:
            slot["increment"] = sub(slot["increment"])

    def sym_expr(self, v: Any) -> Any:
        """A traced value (a SymInt / SymFloat / SymBool of the op table's
        arguments and outputs) as a sympy expression over the tape's
        shape_env; a number as itself."""
        return v.node.expr if isinstance(v, _SYM_TYPES) else v

    def raw_guards(self) -> tuple:
        """The raw guard record as sympy, aligned with `guard_rows`."""
        return tuple(g.expr for g in self.shape_env.guards)

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
            # the other raisers of the same relation (op index, depth, origin)
            "also": list(self.guard_also.get(raw, ())),
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
    def num_regions(self) -> int:
        return len(self.regions)

    @property
    def num_allocations(self) -> int:
        return len(self.allocs)

    @property
    def num_guards(self) -> int:
        return len(self.guards)

    @property
    def num_host_buffers(self) -> int:
        return len(self.host_buffers)

    @property
    def num_memcpys(self) -> int:
        return len(self.memcpys)

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
            },
            "written_inputs": list(self.written_inputs),
            "inputs": [
                {
                    "position": i.position,
                    "dtype": str(i.dtype),
                    "sizes": [e(s) for s in i.sizes],
                    "strides": [e(s) for s in i.strides],
                    "offset": e(i.offset),
                    "root": e(i.root.sym),
                    "device": i.device.type,
                    "pinned": i.pinned,
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
            "memsets": [
                {
                    "seq": m["seq"],
                    "dst": e(m["dst"]),
                    "value": m["value"],
                    "bytes": e(m["bytes"]),
                }
                for m in self.memsets
            ],
            "host_buffers": [
                {
                    "seq": hb["seq"],
                    "name": hb["name"],
                    "root": e(hb["root"]),
                    "nbytes": hb["nbytes"],
                    "elements": [
                        {
                            "offset": q["offset"],
                            "size": q["size"],
                            "kind": q["kind"],
                            "expr": e(q["value"]),
                            "const": not isinstance(q["value"], _SYM_TYPES),
                        }
                        for q in hb["elements"]
                    ],
                }
                for hb in self.host_buffers
            ],
            "memcpys": [
                {
                    "seq": m["seq"],
                    "src": e(m["src"]),
                    "dst": e(m["dst"]),
                    "bytes": e(m["bytes"]),
                    "kind": m["kind"],
                }
                for m in self.memcpys
            ],
            "regions": [
                {
                    "seq": r.seq,
                    "name": r.name,
                    "op": r.op,
                    "scalars": list(r.scalars),
                    "inputs": [
                        {
                            "name": o.name,
                            "root": e(o.root.sym),
                            "address": e(o.address),
                            "sizes": [e(s) for s in o.sizes],
                            "strides": [e(s) for s in o.strides],
                            "dtype": str(o.dtype),
                        }
                        for o in r.inputs
                    ],
                    "outputs": [
                        {
                            "name": o.name,
                            "root": e(o.root.sym),
                            "address": e(o.address),
                            "sizes": [e(s) for s in o.sizes],
                            "strides": [e(s) for s in o.strides],
                            "dtype": str(o.dtype),
                        }
                        for o in r.outputs
                    ],
                }
                for r in self.regions
            ],
            "rng_increment": e(self.rng_increment)
            if self.rng_increment is not None
            else None,
            "all_on_capture_stream": self.all_on_capture_stream,
            "rng_slots": [
                {
                    "launch": r["launch"],
                    "offset": r["offset"],
                    "size": r["size"],
                    "increment": e(r["increment"]),
                }
                for r in self.rng_slots
            ],
            "hints": hints,
        }
        return d


def _check_host_buffers(tr: _Trace, records: dict) -> None:
    # the typed guarantee (HostTable.h): a pointer element of a host table is
    # an address the trace created, i.e. a value over an input's base, an
    # allocation's base or another table's root; anything else is an address
    # the host obtained some other way
    roots = {
        name
        for name in (
            [_symbol_name(i.root.sym) for i in tr.inputs]
            + [_symbol_name(a.q) for a in tr.allocs]
            + [_symbol_name(hb["root"]) for hb in records["host_buffers"]]
        )
        if name is not None
    }
    for hb in records["host_buffers"]:
        for q in hb["elements"]:
            if q["kind"] != "ptr":
                continue
            v = q["value"]
            if not isinstance(v, _SYM_TYPES) or not (
                {str(x) for x in v.node.expr.free_symbols} & roots
            ):
                raise Declined(
                    f"host_trace: host table '{hb['name']}': the pointer element at byte "
                    f"{q['offset']} is not an address the trace created (declined)"
                )
    # a copy's destination is device memory: an address over a CUDA input's
    # base or an allocation's base, never a pinned input or a host table; its
    # source is an address the trace created (a host table image, a pinned
    # input, or for a device-to-device copy the same roots as a destination)
    host_roots = {
        name
        for name in (
            [_symbol_name(i.root.sym) for i in tr.inputs if i.device.type == "cpu"]
            + [_symbol_name(hb["root"]) for hb in records["host_buffers"]]
        )
        if name is not None
    }

    for j, m in enumerate(records["memcpys"]):
        syms = _free_symbols(m["dst"])
        if not (syms & roots) or (syms & host_roots):
            raise Declined(
                f"host_trace: memcpy {j}: the destination must be device memory "
                "(a CUDA input or a host allocation), not host memory (declined)"
            )
        # the record declares its kind (the recording site knows which copy it
        # made); the source is checked against the declaration, not classified
        src = _free_symbols(m["src"])
        if m["kind"] == "h2d":
            ok = bool(src & host_roots)
        elif m["kind"] == "d2d":
            ok = bool(src & roots) and not (src & host_roots)
        else:
            raise Declined(
                f"host_trace: memcpy {j}: kind {m['kind']!r} is neither h2d nor d2d (declined)"
            )
        if not ok:
            raise Declined(
                f"host_trace: memcpy {j} is declared {m['kind']} but its source is not "
                + (
                    "a host table image or a pinned input"
                    if m["kind"] == "h2d"
                    else "device memory the trace created"
                )
                + " (declined)"
            )


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


def _merge_launches(records: dict, extra: list) -> None:
    launches = [*records["launches"], *extra]
    order = sorted(range(len(launches)), key=lambda i: launches[i]["seq"])
    positions = {old: new for new, old in enumerate(order)}
    # RNG slots name launch-vector positions, not the stable event sequence.
    for slot in records["rng_slots"]:
        slot["launch"] = positions[slot["launch"]]
    records["launches"] = [launches[i] for i in order]


def _trace_once(
    fn: Callable[..., Any],
    args: tuple,
    positions: list[int],
    device: int,
    hints: dict | None,
    *,
    triton: list | None = None,
    cute: list | None = None,
) -> tuple[_Trace, dict, list[_OutputRec]]:
    # one symbolic run under its own recorder scope; `triton` is the warm-up's
    # observations of Python-launched Triton kernels (None: a run without a
    # warm-up, where such a launch declines by name), `cute` the names of the
    # CuTe DSL programs the warm-up called
    tr = _Trace(device, hints)
    tr.triton = _host_trace_triton.TritonTrace(triton)
    _active.trace = tr
    _active.ops = []  # this thread's stack of ops being dispatched
    try:
        traced = list(args)
        for i in positions:
            traced[i] = tr.input(i, args[i])
        try:
            with (
                _host_trace_triton.hooked(),
                _host_trace_triton.tracing(),
                _host_trace_cute.tracing(tr),
                _host_trace_cute_dsl.tracing(cute) as cute_met,
                _TraceMode(tr),
            ):
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
        _host_trace_triton.merge(tr, records)
        _host_trace_cute.merge(tr, records)
        _host_trace_cute_dsl.check_met(cute, cute_met)
        _check_host_buffers(tr, records)
        return tr, records, outputs
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
        try:
            tr.rec.end()
        finally:
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
        if not (args[i].is_cuda or (args[i].is_cpu and args[i].is_pinned())):
            where = "pageable CPU memory" if args[i].is_cpu else str(args[i].device)
            raise Declined(
                f"host_trace: arg{i} is in {where}; only CUDA tensors and pinned CPU tensors are traced"
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
    cuda_positions = [i for i in positions if args[i].is_cuda]
    if device is None:
        if not cuda_positions:
            raise Declined("host_trace: the call has no CUDA tensor argument")
        device = args[cuda_positions[0]].device.index
    for i in cuda_positions:
        if args[i].device.index != device:
            raise Declined(
                f"host_trace: arg{i} is on {args[i].device}, the trace is on cuda:{device}"
            )
    if getattr(_active, "trace", None) is not None:
        raise Declined("host_trace: a trace is already in progress on this thread")
    result = None
    observations = None
    cute_programs = None
    # Python-launched Triton kernels: observed at the warm-up (the
    # compilation eager selects for these inputs), intercepted and recorded
    # under the trace (torch/cuda/_host_trace_triton.py)
    with _host_trace_triton.hooked(), _host_trace_cute_dsl.hooked():
        if warm_up:
            # on this thread's current stream, synchronized on that stream only:
            # a device-wide synchronize would invalidate a capture on another thread
            before = [_metadata(args[i]) for i in positions]
            lazy = [torch._C._is_cow_tensor(args[i]) for i in positions]
            with (
                torch.cuda.device(device),
                _host_trace_triton.observing() as observations,
                _host_trace_cute_dsl.observing() as cute_programs,
            ):
                result = fn(*args)
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
                        "storage replaced"
                        if name == "storage"
                        else f"{name} {a} -> {b}"
                        for name, a, b in zip(_METADATA, was, now)
                        if a != b
                    )
                    declined = Declined(
                        f"host_trace: the warm-up changed the metadata of arg{i} ({changes}); eager resized it "
                        "in place (an out= of another shape, a resize_ or a set_ inside the call), and a trace "
                        "after it would describe the resized call, not the one made"
                    )
                    declined.warm_up_ran, declined.warm_up_outputs = True, result
                    raise declined
        # The trace capture is thread-local. A CUDAGraph finalized while it is open
        # (an earlier variant's exec and pool, freed by a cyclic collection on this
        # thread) invalidates it, so hold collections until the trace is over. No
        # collection before the capture: a full one costs more than the trace.
        try:
            with _gc_hold:
                tr, records, outputs = _trace_once(
                    fn,
                    args,
                    positions,
                    device,
                    None,
                    triton=observations,
                    cute=cute_programs,
                )
        except Declined as e:
            e.warm_up_ran, e.warm_up_outputs = warm_up, result
            raise
    tape = Tape(tr, records, outputs, args)
    tape.warm_up_outputs = result
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

    def compile_seq(self, values: list) -> Any:
        # one code object evaluating every value to an int, for the per-call
        # paths (a closed region's key and addresses)
        parts = []
        for v in values:
            if isinstance(v, _SYM_TYPES):
                v = v.node.expr
            parts.append(
                repr(int(v))
                if isinstance(v, (int, bool))
                else f"int({self.printer.doprint(v)})"
            )
        return compile("(" + ",".join(parts) + ",)", "<host_trace>", "eval")

    def ev_seq(self, code: Any, env: dict) -> tuple:
        return eval(code, self.ns, env)

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
        if rec.device.type == "cpu":
            if not t.is_cpu:
                raise Miss(
                    f"{rec.name} is on {t.device}, the tape traced a pinned CPU tensor"
                )
            if not t.is_pinned():
                raise Miss(
                    f"{rec.name} is in pageable CPU memory, the tape traced a pinned tensor"
                )
        elif not t.is_cuda or t.device.index != device:
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
            tuple(g.expr for g in tr.shape_env.guards),
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
    at the call's inputs): a call is served by the first
    variant it matches; a call that misses every variant is traced at its
    own inputs and served by the variant built from that tape, never by the
    ordinary host (the cap raises). A TopologyMiss from a variant says the
    tape's guards held and only the call's class differs: the same tape is
    built again at the call's inputs (no trace) and the variant kept beside
    the first, unless a later variant of that tape serves the call first.
    The ordinary host serves a call only when its trace declines, warned once
    per declined class, and the class (Declined.partial, or the exact inputs
    when the trace declined before binding them) is remembered so no call of
    it is traced again. Explicit tensor arguments only. Exactly once is the
    builder's: with a builder that runs nothing of `fn`, a call executes it
    once (its warm-up here is that execution, `warm_up`); a missed call's
    trace with warm_up=True runs it again."""

    # the frame the declined-class warning names: this class's caller, above
    # __call__ and _miss (a subclass that wraps __call__ raises it by one)
    warn_stacklevel = 4

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
        self.declined_reasons: list[str] = []  # why, once per distinct reason
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
        return self._serve_unmatched(args, tape)

    def _serve_unmatched(self, args: tuple, tape: Tape | None) -> list:
        exact = _exact_class(args)
        if tape is None and (
            exact in self.declined_exact or any(p.matches(args) for p in self.declined)
        ):
            return self._ordinary(args)
        if len(self.variants) == self.max_variants:
            raise RuntimeError(
                f"host_trace: the call misses all {self.max_variants} variants of this entry (max_variants={self.max_variants})"
            )
        try:
            if tape is None:
                self.traces += 1
                tape = trace(self.fn, args, warm_up=self.warm_up)
            # a backend's builder declines like a trace does (Declined, with the
            # class on `partial` when the builder can name it; a build's decline,
            # the allocator backend, has no partial and is remembered exactly):
            # the same memo
            variant = self.build_variant(tape, args)
        except Declined as e:
            self._decline(e, args, exact, self.warn_stacklevel + 1)
            return self._ordinary(args)
        self.variants.append(variant)
        return variant(args)

    def _decline(self, e: Declined, args: tuple, exact: tuple, stacklevel: int) -> None:
        """Remember the declined class: the exact inputs as made (a warm-up may
        have resized them) always; the class by its guards when they can be
        decided from the inputs (it then contains these). Warned at the frame
        `stacklevel` names, once per decline."""
        self.declined_exact.add(exact)
        if e.partial is not None and e.partial.matches(args):
            self.declined.append(e.partial)
        if str(e) not in self.declined_reasons:
            self.declined_reasons.append(str(e))
        warnings.warn(
            f"host_trace: the trace at these inputs declined, so the ordinary host serves such calls: {e}",
            RuntimeWarning,
            stacklevel=stacklevel,
        )


# registers the TensorIterator entries (add, mul, silu, gelu, copy_); the Python-launched
# Triton kernels and the CuTe DSL invocations under a trace (the hooks _trace_once enters)
from torch.cuda import (  # noqa: F401
    _host_trace_cudnn,
    _host_trace_cute,
    _host_trace_cute_dsl,
    _host_trace_native,
    _host_trace_ti,
    _host_trace_triton,
)


# the calls of torch._native's overrides and of the cuDNN attention ops on the
# closed-region list
_closed_calls.update(_host_trace_native.closed_calls())
_closed_calls.update(_host_trace_cudnn.closed_calls())


if torch.distributed.is_available():
    from torch.cuda import _host_trace_symm  # noqa: F401
