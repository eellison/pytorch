"""Host tracing (private): the symbolic run of one call and its Tape.

trace() runs a call's host code once over traced tensors, whose sizes,
strides, storage offsets and addresses are symbols of the trace's ShapeEnv,
and int arguments that are symbols too. The host allocates (every allocation's
address is 256*q for a fresh symbol q) and launches kernels (recorded by the
launch interceptors through _Trace.record_launch); a view of a traced tensor is
another traced tensor over the same root, its metadata computed by the view's
fake kernel over the trace's symbols; any other operator is an EagerCall that a
replay runs eagerly, its outputs' metadata computed by its fake kernel and each
fresh output storage a new root. The run is under a thread-local stream
capture on a side stream, so a synchronizing call raises and any work enqueued
without a record shows up as a node of the discarded graph: both decline.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import functools
import gc
import heapq
import itertools
import math
import operator
import struct
import sys
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TYPE_CHECKING

import sympy

import torch
from torch._C import _dispatch_has_kernel_for_dispatch_key as _has_kernel, DispatchKey
from torch._dispatch.python import enable_python_dispatcher
from torch._ops import OpOverload
from torch import _meta_registrations, _prims_common as utils
from torch._prims.rng_prims import _impl_graphsafe_rng, graphsafe_run_with_rng_state
from torch._subclasses import fake_impls
from torch._subclasses.fake_impls import _compute_stride
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily
from torch.cuda import _host_trace_hint_audit as _hint_audit, _host_trace_ir as _ir
from torch.cuda._host_trace import _drop_tracebacks, _TraceShapeEnv, Declined, declined, ProcessHold
from torch.cuda._host_trace_opaque import bind_at_trace, library_state, library_state_as, real_key, record_binding, takes_generator, trace_key
from torch.cuda._utils import _check_cuda_bindings
from torch.fx.experimental import _config as fx_config
from torch.fx.experimental.symbolic_shapes import free_symbols, free_unbacked_symbols
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes, TorchDispatchMode
from torch.utils._sympy.value_ranges import ValueRanges


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping, Sequence

    from torch.cuda._host_trace_opaque import KeyedSite, OpaqueProvider


aten = torch.ops.aten

# A traced input's base address is the hint of a symbol: the real address's
# low 52 bits (every alignment and modulo a host computes is exact) under a
# non-canonical top, so a launch that escaped the trace faults on it instead of
# touching the real tensor
_PLACEHOLDER_TAG = 0x4A5 << 52
_PLACEHOLDER_LOW = (1 << 52) - 1
# an allocation's hint: a distinct 64 GiB-aligned value under another top
_ALLOC_TAG = 0x4A6 << 52
_ALLOC_SHIFT = 36
_ALLOC_ALIGNMENT = 256
# an eager op's output address, unknown until the op runs at a replay: a unique hint under
# a non-canonical top so an escaped launch faults; base/256 keeps it 256-aligned like the allocator's
_EAGER_TAG = 0x4A7 << 52

_SYM_TYPES = (torch.SymInt, torch.SymFloat, torch.SymBool)


@dataclass(frozen=True)
class Unconverted:
    """An op whose own CUDA host code (_traced_redispatch) went past what is
    converted to symbolic: what to convert next."""

    op: str
    # the C++ refusal (at::detail::SymbolicRefusal): "accessor", "output",
    # "generator"; "int read" for host code that read a traced tensor's sizes,
    # strides or offset as ints (a guard of each value: it works, specialized);
    # None for another error (a data pointer read)
    refusal: str | None
    error: str  # the exception's type
    message: str  # its first line
    site: str | None  # the C++ frames below the refusing accessor
    symbolic: bool  # whether a symbolic TensorIterator was built before it


class UnconvertedHost(BaseException):
    """In strict mode, an op's CUDA host code that built a symbolic
    TensorIterator and then stopped at code not converted to symbolic, or that
    read a traced tensor's sizes as ints. A BaseException: the trace's own
    handlers do not turn it into a decline."""


# what C++ host code dispatches when it reads a traced tensor's sizes(),
# strides() or storage_offset() as ints (the Python sizes hook): each value
# becomes a guard of its hint. Python code reads them as SymInts (sym_size)
_INT_READS = {aten.size.default: "sizes()", aten.stride.default: "strides()", aten.storage_offset.default: "storage_offset()"}


# every Unconverted of the process, by (op, refusal, site or message):
# TORCH_HOST_TRACE_UNCONVERTED_LOG=<path> appends them as JSON lines at exit
UNCONVERTED: collections.Counter = collections.Counter()


def _log_unconverted() -> None:
    import json
    import os

    if (path := os.environ.get("TORCH_HOST_TRACE_UNCONVERTED_LOG")) and UNCONVERTED:
        with open(path, "a") as f:
            for (op, refusal, message), n in UNCONVERTED.most_common():
                f.write(json.dumps({"op": op, "refusal": refusal, "message": message, "count": n}) + "\n")


import atexit as _atexit  # noqa: E402

_atexit.register(_log_unconverted)


def _has_cuda_kernel(func: Any) -> bool:
    # a C++ CUDA kernel: an ATen op's, a C++ extension's (a Python one routes as _python_kernel / _library_kernel)
    return isinstance(func, OpOverload) and _has_kernel(func.name(), "CUDA") and _python_kernel(func) is None and _library_kernel(func) is None


@dataclass
class _Root:
    name: str  # p<i> for argument i's storage, a<k> for allocation k, e<k> for eager output k
    sym: Any  # the base address: an input's symbol, 256*q otherwise
    kind: Literal["argument", "allocation", "eager"] = "argument"
    index: int = -1  # k of allocation k or eager output k: its placeholder (_placeholder)
    # a pinned CPU argument's storage, which only a memcpy node reads or writes
    host: bool = False
    # an unpinned CPU argument's storage, which nothing the trace reads (the unread kind, 10-22): its sizes are the call's
    # and nothing of it is copied; an op that reads its bytes declines the trace (Tape._eager_call), which then keys or
    # pins the tensor as today
    unread: bool = False


def _placeholder(root: _Root) -> int:
    # an allocation's or eager output's base at the trace (q's value is a 256th of it): a
    # non-canonical address an escaped launch faults on, which the first replay patches
    return (_ALLOC_TAG if root.kind == "allocation" else _EAGER_TAG) | ((root.index + 1) << _ALLOC_SHIFT)


@dataclass
class _InputRec:
    position: int
    name: str
    dtype: torch.dtype
    sizes: list
    strides: list
    offset: Any
    root: _Root
    extent: tuple[int, int]  # the first and last byte of its elements at the trace
    cow: bool  # a lazy copy-on-write storage at the trace; not guarded


@dataclass(frozen=True)
class TrustedInputs:
    """What the caller vouches for at every call (under Inductor, Dynamo's
    guards and Inductor's own input handling): the trace records no guard a
    fact below decides, and a replay checks none of it.

    `layouts[i]` is tensor argument i's (sizes, strides), each an int or a
    sympy expression over the caller's symbols, or int argument i's int or
    expression; `ranges` bounds the caller's symbols. Only the caller's symbols are symbols of the trace; every other size and
    stride is a constant. An
    address is never vouched for: each call's is read. A guard the trace
    still records comes from a decision of the host (a size-based dispatch),
    not from an input's validity."""

    layouts: tuple[Any, ...]
    ranges: Mapping[sympy.Symbol, ValueRanges] = field(default_factory=dict)


@dataclass
class _IntInputRec:
    position: int
    name: str
    sym: torch.SymInt


@dataclass
class _AllocRec:
    seq: int
    name: str
    sizes: list
    strides: list
    dtype: torch.dtype
    root: _Root
    q: torch.SymInt  # the address is _ALLOC_ALIGNMENT * q


@dataclass
class _OutputRec:
    name: str
    root: _Root
    sizes: list
    strides: list
    offset: Any
    dtype: torch.dtype
    # ("argument", i) or ("output", k) when this output is that very object,
    # as the host returned it; None for a fresh allocation
    identity: tuple[str, int] | None = None


@dataclass
class _IntOutputRec:
    name: str
    value: Any  # an int or a SymInt of the trace


@dataclass(frozen=True)
class EagerCall:
    """An operator the trace does not put in a graph; a replay runs it
    eagerly, in program order."""

    # an OpOverload, or ("triton", JITFunction, grid, options) for a launch
    # whose compilation the trace does not describe, or ("cute", compiled
    # function, its stream arguments' (position, type), its CuTe tensor
    # arguments' (position, _DLPackArg)), or ("host",
    # function) for a host step (_Trace.host_step)
    target: Any
    args: tuple  # as the host passed them: traced tensors, SymInts, constants
    kwargs: dict
    # the returned tensors: a fresh output over an eager root, or for an
    # in-place or out= return the argument object itself
    outputs: tuple[_TracedTensor, ...]
    # why a Triton launch runs eagerly, or why each opaque provider declined
    reason: str | None = None
    # the generator the call draws from, a graph input under graphsafe RNG
    generator: torch.Generator | None = None
    # library_state() at a call an opaque provider accepted, which its eager
    # runs are under; () for the state at the replay
    state: tuple = ()

    @property
    def host(self) -> bool:
        return isinstance(self.target, tuple) and self.target[0] == "host"

    @property
    def name(self) -> str:
        if isinstance(self.target, tuple) and self.target[0] == "cute":
            return f"CuTe function {self.target[1].function_name}"
        if self.host:
            return f"host step {self.target[2]}"
        if isinstance(self.target, tuple):
            return f"Triton kernel {self.target[1].fn.__name__}"
        return str(self.target)


# OpRec.redo of an operator call and of a jiterator launch: objects, not closures, so a variant's ops hold no
# function or cell (_host_trace_replay._gc_untrack); eq=False: a redo template keys a non-operator op by its redo
@dataclass(frozen=True, eq=False)
class _EagerRedo:
    func: Any

    def __call__(self, args: tuple, kwargs: dict) -> Any:
        return current_trace().eager_call(self.func, args, kwargs)


@dataclass(frozen=True, eq=False)
class _JiteratorRedo:
    def __call__(self, args: tuple, kwargs: dict) -> Any:
        return current_trace().jiterator(*args, **kwargs)


@dataclass
class OpRec:
    """A top-level op (an operator, a Triton or CuTe DSL launch): the records
    its fake kernel and host made."""

    func: Any
    # its arguments, traced tensors and ints among them: the op alone runs
    # again from these, at any call, from their roles and metadata
    call: tuple[tuple, dict]
    launches: range  # in Tape.launches
    allocs: range  # in Tape.allocs
    # its returned tensors and ints
    outputs: tuple[Any, ...]
    # "traced": every launch recorded (an ATen or Python host's, a Triton or
    # CuTe DSL launch); "eager": one EagerCall; "other" (an opaque call, a
    # host step, a nested eager call), whose guards are graph-level
    kind: str
    guards: tuple[int, ...] = ()  # the Tape.guards it owns
    # the op's dispatch alone, redo(args, kwargs), in current_trace(): its
    # host again at other metadata (_host_trace_redispatch)
    redo: Callable[[tuple, dict], Any] | None = None
    # an op "other" only for its opaque calls: the guards it recorded, graph-level until every one of those calls
    # binds as a keyed site (bind_opaque), then owned as a traced op's (shared_op_guards' owner sets); else None
    held: tuple[int, ...] | None = None


@dataclass(frozen=True)
class OpaqueCall(EagerCall):
    """An eager call `provider` accepted: a replay binds its key
    (_host_trace_opaque)."""

    provider: Any = None
    # its key bound at the trace, which did not record the binding (a fresh
    # output at a storage offset): a replay at a key that binds is no reason
    # to relower
    bound: bool = False


# the records a trace writes while it runs and nothing writes once its tape is built (bind_opaque and a twin's
# tape replace them): final for _host_trace_replay._gc_untrack
_GC_RECORDS = (_Root, _InputRec, _IntInputRec, _AllocRec, _OutputRec, _IntOutputRec, OpRec)


@dataclass(frozen=True)
class Memset:
    """A memset node: `height` rows `pitch` bytes apart of `width` elements
    of `element_size` bytes, each `value`, from the address slots[0]."""

    name: str
    slots: tuple[Any]
    roots: tuple[_Root, ...]  # the root slots[0] may be of
    value: int
    element_size: int  # 1, 2 or 4
    width: Any
    height: Any
    pitch: Any


@dataclass(frozen=True)
class Memcpy:
    """A 1D memcpy node of `nbytes` from slots[1] to slots[0]: device to device,
    or between the device and a pinned host argument ("H2D", "D2H")."""

    name: str
    slots: tuple[Any, Any]  # dst, src
    roots: tuple[_Root, ...]  # the roots the slots may be of
    nbytes: Any
    kind: Literal["D2D", "H2D", "D2H"] = "D2D"


# each Memcpy kind's cudaMemcpyKind: a node's parameters can only be set with its capture's
MEMCPY_KINDS = {"H2D": 1, "D2H": 2, "D2D": 3}


_active = _ir.ACTIVE


def current_trace() -> _Trace | None:
    return getattr(_active, "trace", None)


def _declined(msg: str) -> Declined:
    tr = current_trace()
    if tr is None:
        return declined(msg)
    return tr.decline(msg)


def _run_host_op(func: OpOverload, held: tuple, *args: Any, **kwargs: Any) -> None:
    out = func(*args, **kwargs)
    outs = [out] if isinstance(out, torch.Tensor) else list(out)
    fresh = [o for r, o in zip(func._schema.returns, outs) if r.alias_info is None]
    for h, o in zip(held, fresh):
        h.copy_(o)


class _Trace:
    """One trace in progress: its ShapeEnv and the records of the host's
    inputs, allocations and launches, in program order (`seq`)."""

    def __init__(
        self,
        device: torch.device,
        trusted: TrustedInputs | None = None,
        opaque: Sequence[OpaqueProvider] = (),
        static_shapes: Collection[int] = (),
        check_escapes: bool = False,
        eager_ops: Mapping[int, str] | None = None,
        fake_mode: FakeTensorMode | None = None,
        declined_compiles: set[Any] | None = None,
        shape_env: Any = None,
        markers: _Markers | None = None,
    ) -> None:
        self.device = device
        self.trusted = trusted
        self.opaque = opaque
        self.static_shapes = frozenset(static_shapes)
        # the top-level ops (by index) that run eagerly, each with why
        self.eager_ops = eager_ops or {}
        # the functions whose CuTe compile declined in an earlier trace of the caller's
        self.declined_compiles = declined_compiles
        # the trace's symbol for each of the caller's, under trusted inputs
        self.given: dict[sympy.Symbol, torch.SymInt] = {}
        # the trace's capturing stream; a launch may also be recorded on a side
        # stream forked from it in the capture (`markers`)
        cuda = device.type == "cuda"
        self.stream = torch.cuda.current_stream(device) if cuda else None
        # the capture's markers (trace's _capture); None in a redo's or a
        # binding's trace, where a launch is recorded on `stream` only
        self.markers = markers
        # the side streams launches were recorded on: logical stream k + 1 is side_streams[k]
        self.side_streams: list[torch.cuda.Stream] = []
        # per record on a side stream its logical stream (Tape.streams); per
        # record, the records on other streams it directly follows (Tape.after),
        # read off the capture's edges once it ends
        self.stream_of: dict[int, int] = markers.streams if markers is not None else {}
        self.after: dict[int, tuple[int, ...]] = markers.after if markers is not None else {}
        # computes a view's and an eager call's metadata; without its cache
        # every guard a kernel evaluates is recorded, and without fallback
        # kernels an op with no meta raises instead of running on zeros
        if trusted is None and torch.cuda._host_trace.symbolic == "ir":
            self.shape_env: Any = _ir.Env() if shape_env is None else shape_env
            # the fake twins carry the trace's own SymInts; no ShapeEnv
            self.fake_mode = fake_mode or FakeTensorMode(allow_fallback_kernels=False)
        else:
            self.shape_env = _TraceShapeEnv(trusted=trusted is not None) if shape_env is None else shape_env
            self.fake_mode = FakeTensorMode(
                shape_env=self.shape_env, allow_fallback_kernels=False
            )
        self.fake_mode.cache_enabled = False
        if torch.cuda._host_trace.shared_op_guards:
            self.shape_env.owner_sets = []
        self._seq = itertools.count()
        self.inputs: list[_InputRec] = []
        self.arguments: dict[int, _InputRec] = {}  # by id of its root
        # argument positions i < j, where a step not run eagerly writes one
        # and reads the other (Tape.argument_pairs)
        self.argument_pairs: set[tuple[int, int]] = set()
        # the keys the warm-up learned at their calls: their calls stay
        # OpaqueCalls, bound after the trace as a learned key's are (bind_opaque)
        self.warm_learned: set[Any] = set()
        self.int_inputs: list[_IntInputRec] = []
        self.allocs: list[_AllocRec] = []
        # allocation k's symbol is alloc<first_alloc + k>: bind_opaque's follow its tape's
        self.first_alloc = 0
        self.launches: list[tuple[int, Any]] = []  # (seq, launch or EagerCall)
        if markers is not None:
            markers.launches = self.launches
        self.eager_outputs: list[_TracedTensor] = []
        # a redispatch's tensors spanning the variant's allocations and eager outputs, by the fresh root's id
        self.outer: dict[int, _TracedTensor] = {}
        # every tensor of the trace, to find one fn kept past it (trace's check_escapes)
        self.tensors: weakref.WeakSet[_TracedTensor] | None = weakref.WeakSet() if check_escapes else None
        # the traced call's values (_symbolic_run, a redispatch's _redo)
        self.values: CallValues | None = None
        # eager's checks of the call's values that its calls must meet (check)
        self.checks: list[torch.SymBool] = []
        # the traced hosts' calls with their outputs, for _check_witness
        self.aten_calls: list[tuple[int, EagerCall]] = []
        # inside _traced_redispatch: the traced tensors whose storages are
        # sealed (a data pointer read raises) until it returns
        self.sealed: list[_TracedTensor] | None = None
        # the ops whose own CUDA host code went past what is converted to
        # symbolic (_traced_redispatch): the list of what to convert
        self.unconverted: list[Unconverted] = []
        # fn's own operator calls in order, as the warm-up's (_order_key)
        self.order: list[tuple] = []
        self.depth = 0  # of _TraceMode dispatches
        # a redo's resolution of a plain tensor to an argument's stand-in (redo_resolves_arguments), or None
        self.resolve: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.sites: list[KeyedSite] = []  # the calls bound at trace time
        # the CPU tensors the host's steps write, by storage: the trace's own,
        # each step writing them again at a replay
        self.host: dict[int, torch.Tensor] = {}
        # the host buffers a device step or eager call reads, by storage: a
        # replay runs every host step before them
        self.host_reads: set[int] = set()
        # inside graphsafe_run_with_rng_state, the generator its op draws from
        self.generator: torch.Generator | None = None
        # the first decline; kept so that host code that catches it (a
        # try/except around an op) cannot make the trace succeed
        self.declined: Declined | None = None
        # inside a traced ATen host (_traced_aten), whose own ops are not routed again
        self.in_aten = False
        # inside a top-level op, whose calls and launches are its own
        self.in_op = False
        # C++ host code's int reads of traced tensors' sizes (int_read), reported per op by op()
        self.int_reads: list[str] = []
        self.ops: list[OpRec] = []
        # the Library.impl or custom_op kernel being traced and its kind, and
        # why its body declines (a host-trace entry it called, raised in the
        # body as reentered())
        self.library_body: tuple[OpOverload, str] | None = None
        self.reentry: str | None = None
        # the kind of the body the trace runs inside (library_impls_reentry,
        # custom_op_reentry)
        self.under_library_body = _library_body_on_stack()

    def reentered(self) -> Declined:
        op, kind = self.library_body  # type: ignore[misc]
        self.reentry = f"{op}'s {kind} body calls a host-trace entry"
        return declined(self.reentry)

    def decline(self, msg: str) -> Declined:
        e = declined(msg)
        if self.declined is None:
            self.declined = e
        return e

    def check_stream(self, what: str) -> None:
        """Declines work on a stream other than the trace's, but for a side
        stream forked from it in the trace's capture."""
        if self.stream is None:
            return
        current = torch.cuda.current_stream(self.device)
        if current == self.stream or (self.markers is not None and _capture_info(current)[:2] == _capture_info(self.stream)[:2]):
            return
        joined = "" if self.markers is None else " and never forked from it (a wait_stream or an event wait)"
        raise self.side_decline(f"{what} on a stream other than the trace's{joined}")

    def _join_side_streams(self) -> None:
        """The trace's stream waits on each side stream in its capture: a body
        that declined after it launched on a side stream left the fork open
        (its records are rolled back and the op is one eager step on the
        trace's stream, which forks and joins as eager does)."""
        here = _capture_info(self.stream)[:2]
        for side in self.side_streams:
            if _capture_info(side)[:2] == here:
                self.stream.wait_stream(side)

    def side_decline(self, msg: str) -> Declined:
        """A decline of side-stream work: warned once per entry and cached per
        argument contract (no retrace per shape)."""
        e = self.decline(msg)
        e.side_stream = True
        return e

    def record_launch(self, record: Any) -> None:
        if isinstance(record, EagerCall) and not record.host:
            leaves = pytree.tree_leaves((record.args, record.kwargs))
            self.host_reads.update(a.untyped_storage()._cdata for a in leaves if isinstance(a, torch.Tensor) and self._is_host(a))
        seq = next(self._seq)
        if self.markers is not None and not (isinstance(record, EagerCall) and record.host):
            self._mark(seq, record)
        self.launches.append((seq, record))

    def _mark(self, seq: int, record: Any) -> None:
        """A marker node for the record on its stream in the capture
        (_Markers): the capture's edges between markers order the records."""
        from cuda.bindings import runtime

        handle = torch._C._cuda_getCurrentRawStream(self.stream.device_index)
        if handle != self.stream.cuda_stream:
            current = torch.cuda.current_stream(self.device)
            self.check_stream(record.name)
            if isinstance(record, EagerCall) and not isinstance(record, OpaqueCall):
                # its replay would need the stream and the events around it; an opaque call binds
                # after the trace (bind_opaque), or the lowering declines it
                raise self.side_decline(f"{record.name} on a side stream; an eager step there is not supported")
            if current not in self.side_streams:
                self.side_streams.append(current)
            self.stream_of[seq] = self.side_streams.index(current) + 1
        markers = self.markers
        if markers.event is None:
            markers.event = _check_cuda_bindings(runtime.cudaEventCreateWithFlags(runtime.cudaEventDisableTiming))
        _check_cuda_bindings(runtime.cudaEventRecordWithFlags(markers.event, handle, runtime.cudaEventRecordExternal))
        node = [int(n) for n in _check_cuda_bindings(runtime.cudaStreamGetCaptureInfo(handle))[3]]
        if len(node) != 1 or node[0] in markers.seqs:
            raise AssertionError(f"host_trace: a marker of record {seq} added {node} to the capture")
        markers.seqs[node[0]] = seq

    def input(self, position: int, t: torch.Tensor) -> _TracedTensor:
        env = self.shape_env
        name = f"arg{position}"
        sizes: list[Any] = []
        strides: list[Any] = []
        trusted = self.trusted
        if trusted is None:
            for d in range(t.dim()):
                sizes.append(env.symbol(t.size(d), f"{name}.size({d})", positive=t.size(d) > 0))
                strides.append(env.symbol(t.stride(d), f"{name}.stride({d})"))
            offset = env.symbol(int(t.storage_offset()), f"{name}.storage_offset()")
        else:
            given_sizes, given_strides = trusted.layouts[position]
            if len(given_sizes) != t.dim():
                raise self.decline(f"{name} has {t.dim()} dims; the caller described {len(given_sizes)}")
            for d in range(t.dim()):
                sizes.append(self._given(given_sizes[d], t.size(d), f"{name}.size({d})"))
                if given_sizes[d] == 1:
                    # addresses nothing; the producer may have set any stride
                    strides.append(t.stride(d))
                else:
                    strides.append(self._given(given_strides[d], t.stride(d), f"{name}.stride({d})"))
            offset = env.symbol(int(t.storage_offset()), f"{name}.storage_offset()")
        # the input record's symbols read the argument; a static input's are
        # guarded to their values (int() records Eq) and constants in the trace
        layout = sizes, strides, offset
        if trusted is None and position in self.static_shapes:
            sizes, strides, offset = [int(v) for v in sizes], [int(v) for v in strides], int(offset)
        # const_data_ptr leaves a copy-on-write input lazy
        base = t.const_data_ptr() - t.storage_offset() * t.element_size()  # type: ignore[attr-defined]
        sym = env.symbol(_PLACEHOLDER_TAG | (base & _PLACEHOLDER_LOW), f"{name}.base")
        root = _Root(f"p{position}", sym, host=t.device != self.device, unread=t.device.type == "cpu" and not t.is_pinned())
        first = t.const_data_ptr()  # type: ignore[attr-defined]
        last = first + (sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1) * t.element_size() - 1 if t.numel() else first - 1
        rec = _InputRec(position, name, t.dtype, *layout, root, (first, last), torch._C._is_cow_tensor(t))
        self.inputs.append(rec)
        self.arguments[id(root)] = rec
        return _TracedTensor(root, sizes, strides, offset, t.dtype, t.device)

    def int_input(self, position: int, value: int) -> torch.SymInt:
        name = f"arg{position}"
        if not -(1 << 63) <= value < 1 << 63:
            raise self.decline(f"{name} is {value}, outside int64")
        if self.trusted is None:
            sym = self.shape_env.symbol(value, name)
        else:
            sym = self._given(self.trusted.layouts[position], value, name)
            if not isinstance(sym, torch.SymInt):
                return sym
        self.int_inputs.append(_IntInputRec(position, name, sym))
        return sym

    def bind_given(self, args: tuple) -> None:
        """A symbol of the trace for each of the caller's symbols, at its value
        where it first appears as an int argument or a plain size or stride."""
        trusted = self.trusted
        if trusted is None:
            raise AssertionError("host_trace: bind_given without trusted inputs")
        values: dict[sympy.Symbol, int] = {}
        for i, layout in enumerate(trusted.layouts):
            if isinstance(layout, sympy.Symbol):
                values.setdefault(layout, args[i])
            elif isinstance(layout, tuple):
                for d, (size, stride) in enumerate(zip(*layout)):
                    if isinstance(size, sympy.Symbol):
                        values.setdefault(size, args[i].size(d))
                    if isinstance(stride, sympy.Symbol):
                        values.setdefault(stride, args[i].stride(d))
        env = self.shape_env
        for s, v in values.items():
            r = trusted.ranges.get(s, ValueRanges.unknown())
            if v not in r:
                raise self.decline(f"{s} is {v}, outside the caller's range {r}")
            sym = env.symbol(v, str(s), positive=r.lower >= 1)
            env.var_to_range[sym.node.expr] = r
            self.given[s] = sym

    def _given(self, e: Any, value: int, what: str) -> Any:
        # the caller's int or expression, checked against the traced call
        if isinstance(e, int):
            if e != value:
                raise self.decline(f"{what} is {value}; the caller described {e}")
            return e
        missing = e.free_symbols - self.given.keys()
        if missing:
            raise self.decline(
                f"{what} is {e}; {', '.join(map(str, missing))} is no int argument, size or stride"
            )
        expr = e.xreplace({s: v.node.expr for s, v in self.given.items()})
        hint = expr.xreplace(self.shape_env.backed_var_to_val)
        if hint != value:
            raise self.decline(f"{what} is {value}; the caller described {e} = {hint}")
        if isinstance(expr, sympy.Symbol):
            return next(v for v in self.given.values() if v.node.expr == expr)
        return self.shape_env.create_symintnode(expr, hint=value)

    def allocate(self, func: Any, args: tuple, kwargs: dict) -> _TracedTensor:
        # no fill here: under use_deterministic_algorithms(True) with fill_uninitialized_memory, eager fills what its
        # empty / empty_strided return, and the mode records that fill for a factory op (_Mode._dispatch); its other
        # callers' allocations eager makes without one (a structured kernel's output, Inductor's empty_strided_cuda) or
        # overwrites whole (a resized fresh output, a pinned copy's destination)
        device = kwargs.get("device")
        if device is None:
            # the factories' default is the CPU, the *_like / new_* ops' their source's device
            factory = func in (aten.empty.memory_format, aten.empty_strided.default)
            device = torch.device("cpu") if factory else args[0].device
        dev = torch.device(device)
        if dev.type != self.device.type or dev.index not in (None, self.device.index):
            raise self.decline(f"an allocation on {dev} in a trace on {self.device}")
        if kwargs.get("pin_memory"):
            raise self.decline("a pinned allocation")
        if kwargs.get("layout") not in (None, torch.strided):
            raise self.decline(f"an allocation with layout {kwargs['layout']}")
        mf = kwargs.get("memory_format")
        if func is aten.empty.memory_format:
            sizes = list(args[0])
            dtype = kwargs.get("dtype") or torch.get_default_dtype()
            strides = _format_strides(sizes, mf)
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
        else:  # empty_like: at::native::empty_like
            src = args[0]
            sizes = list(src.shape)
            dtype = kwargs.get("dtype") or src.dtype
            traced = isinstance(src, _TracedTensor)
            src_strides = src._sym_strides if traced else list(src.stride())
            strides = _empty_like_strides(sizes, src_strides, mf, self.values)
            mf = None
        if mf not in _ALLOC_FORMATS:
            raise self.decline(f"an allocation in memory format {mf}")
        for s in sizes:
            # c10's check_size_nonnegative
            message = functools.partial(self._negative_size, s, sizes)
            self.check(s >= 0, message)
        k = self.first_alloc + len(self.allocs)
        name = f"alloc{k}"
        root = _Root(f"a{k}", None, "allocation", k)
        q = self.shape_env.symbol(_placeholder(root) // _ALLOC_ALIGNMENT, f"{name}.base/{_ALLOC_ALIGNMENT}")
        root.sym = _ALLOC_ALIGNMENT * q
        rec = _AllocRec(next(self._seq), name, sizes, strides, dtype, root, q)
        self.allocs.append(rec)
        return _TracedTensor(root, sizes, strides, 0, dtype, self.device)

    def _negative_size(self, s: Any, sizes: list) -> RuntimeError:
        at = [self.values(n) for n in sizes]
        return RuntimeError(f"Trying to create tensor with negative dimension {self.values(s)}: {at}")

    def check(self, cond: Any, error: Callable[[], Exception]) -> None:
        """eager's check `cond` of the call's values, which raises error()
        where it fails: at the traced call, a condition of the trace's calls
        (Tape.checks) with validity_checks on, else a guard."""
        if not isinstance(cond, torch.SymBool) or not torch.cuda._host_trace.validity_checks:
            if not bool(cond):
                raise error()
            return
        if not self.values(cond):
            raise error()
        node = cond.node
        if isinstance(node, _ir.IRSymNode):
            env = self.shape_env
            if node.node.op == "true" or _ir._record_key(node.node, node.written) in env._index:
                return  # the declared domains or a guard decide it
        elif node.expr is sympy.true:
            return
        self.checks.append(cond)

    def zero(self, t: Any) -> bool:
        """Record t.zero_() of a non-overlapping and dense tensor on the
        trace's stream or a side stream (_mark) as a Memset of its bytes
        (nothing if it is empty); otherwise it is an op like another."""
        if not isinstance(t, _TracedTensor) or t._root.host or self.stream is None:
            return False
        if self.markers is None and torch.cuda.current_stream(self.device) != self.stream:
            return False
        if not t.numel():
            return True
        if not _is_non_overlapping_and_dense(t, self.values):
            return False
        nbytes = t.numel() * t.element_size()
        self.record_launch(Memset(f"zero_ of {t._root.name}", (t.data_ptr(),), (t._root,), 0, 1, nbytes, 1, nbytes))
        return True

    def view(self, func: Any, args: tuple, kwargs: dict) -> Any:
        src = args[0]
        rest = pytree.tree_leaves((args[1:], kwargs))
        if any(isinstance(a, torch.Tensor) for a in rest):
            raise self.decline(f"{func} with a tensor argument")
        if not isinstance(src, _TracedTensor):
            if any(isinstance(a, _SYM_TYPES) for a in rest):
                raise self.decline(
                    f"{func} of an untraced tensor with symbolic arguments"
                )
            return func(*args, **kwargs)
        if func in (aten.view.default, aten._unsafe_view.default) and torch.cuda._host_trace.free_size_one_strides:
            shape = list(args[1])
            inferred = [i for i, n in enumerate(shape) if type(n) is int and n == -1]
            if len(inferred) == 1 and torch.cuda._host_trace.free_inferred_views:
                others = functools.reduce(operator.mul, shape[: inferred[0]] + shape[inferred[0] + 1 :], 1)
                if not (type(others) is int and others == 0):
                    shape[inferred[0]] = functools.reduce(operator.mul, src.shape, 1) // others
                    inferred = []
            if not inferred:
                strides = _free_view_strides(src.shape, src._sym_strides, shape)
                if strides is not None:
                    return _TracedTensor(src._root, shape, strides, src._sym_offset, src.dtype, src.device)
        out = None
        if (route := _META_VIEWS.get(func)) is not None:
            twin = torch.empty(0, dtype=src.dtype, device="meta").as_strided(src.shape, src._sym_strides, src._sym_offset)
            try:
                out = route(self.fake_mode, func, twin, *args[1:], **kwargs)
            except Exception:
                out = None  # the fake's verdict, a user error's too
        if out is None:
            with self.fake_mode:
                twin = self._twin(src)
                try:
                    out = func(twin, *args[1:], **kwargs)
                except Exception as e:
                    # the fake kernel's error need not be eager's type; host code
                    # that caught it could take a path eager does not
                    raise self.decline(f"{func} raised {type(e).__name__}: {e}") from e
        if func is aten.as_strided.default and self.trusted is None:
            self._check_as_strided(src._root, out)
        storage = twin.untyped_storage()._cdata
        if any(o.untyped_storage()._cdata != storage for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)):
            # an op that may alias (reshape, to.dtype) and copied
            return None
        view_strides = None
        if func in (aten.view.default, aten._unsafe_view.default):
            # the fake kernel (_reshape_view_helper) strides size-1 dims and
            # empty views unlike eager's computeStride
            view_strides = _compute_stride(src.shape, src._sym_strides, out.shape)

        def wrap(o: torch.Tensor) -> _TracedTensor:
            if o.layout != torch.strided or o.is_conj() or o.is_neg():
                raise self.decline(f"{func} is not a plain strided view")
            strides = list(o.stride()) if view_strides is None else view_strides
            return _TracedTensor(src._root, list(o.shape), strides, o.storage_offset(), o.dtype, src.device)

        return pytree.tree_map_only(torch.Tensor, wrap, out)

    def _twin(self, t: _TracedTensor) -> torch.Tensor:
        # under fake_mode, a twin with t's metadata over an empty storage:
        # building it evaluates nothing, so the only guards are the ones the
        # kernel run on it evaluates
        with unset_fake_temporarily(), torch._C._DisablePythonDispatcher():
            meta = torch.empty(0, dtype=t.dtype, device="meta").as_strided(t.shape, t._sym_strides, t._sym_offset)
        return FakeTensor(self.fake_mode, meta, t.device)

    def host_buffer(self, size: Any, stride: Any, dtype: torch.dtype) -> torch.Tensor:
        if free_symbols((size, stride)):
            raise self.decline("a CPU buffer of symbolic size")
        with _disable_current_modes():  # the wrapper's, under the trace's mode
            t = torch.empty_strided(size, stride, dtype=dtype, device="cpu")
        self.host[t.untyped_storage()._cdata] = t
        return t

    def host_step(self, fn: Callable[..., Any], args: tuple, kwargs: dict, name: str, writes: Sequence[Any] | None = None) -> None:
        """fn(*args, **kwargs) over the trace's host buffers and constants:
        host code that reads no device memory, which a replay runs before its
        first graph, in the host's order (as eager draws from the CPU
        generator). It writes `writes`; None: any of its buffers."""
        for a in pytree.tree_leaves((args, kwargs)):
            if isinstance(a, torch.Tensor) and not self._is_host(a):
                raise self.decline(f"{name} of a tensor other than a CPU buffer of the trace")
            if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                raise self.decline(f"{name} of a {type(a).__name__}")
        for a in pytree.tree_leaves((args, kwargs) if writes is None else writes):
            if isinstance(a, torch.Tensor) and a.untyped_storage()._cdata in self.host_reads:
                raise self.decline(f"{name} writes a CPU buffer a device step read earlier")
        self.record_launch(EagerCall(("host", fn, name), args, kwargs, ()))

    def _is_host(self, t: torch.Tensor) -> bool:
        return not isinstance(t, _TracedTensor) and t.untyped_storage()._cdata in self.host

    def _host_fake(self, t: torch.Tensor) -> torch.Tensor:
        return self.fake_mode.from_tensor(t, static_shapes=True)

    def _host_op(self, func: OpOverload, args: tuple, kwargs: dict) -> Any:
        # an operator of host buffers and constants alone: a host step if its
        # outputs are on the CPU, each fresh one a host buffer the step writes
        schema = func._schema
        written = {}
        for i, a in enumerate(schema.arguments):
            if a.alias_info is not None and a.alias_info.is_write:
                written[frozenset(a.alias_info.before_set)] = args[i] if i < len(args) else kwargs.get(a.name)
        with self.fake_mode:
            fakes = pytree.tree_map_only(torch.Tensor, self._host_fake, (args, kwargs))
            out = func(*fakes[0], **fakes[1])
        rets = [out] if len(schema.returns) == 1 else list(out or ())
        if not all(isinstance(o, torch.Tensor) and o.device.type == "cpu" for o in rets):
            return NotImplemented
        result, held = [], []
        for r, o in zip(schema.returns, rets):
            if r.alias_info is not None:
                result.append(written[frozenset(r.alias_info.before_set)])
            else:
                held.append(self.host_buffer(o.shape, o.stride(), o.dtype))
                result.append(held[-1])
        self.host_step(functools.partial(_run_host_op, func, tuple(held)), args, kwargs, str(func), list(written.values()))
        return result[0] if len(schema.returns) == 1 else type(out)(result)

    def eager_call(self, func: Any, args: tuple, kwargs: dict) -> Any:
        run = functools.partial(self._eager_call, func, args, kwargs)
        return self.op(func, args, kwargs, run, redo=_EagerRedo(func))

    def _host_copy(self, func: OpOverload, args: tuple, kwargs: dict) -> Any:
        """copy_ between a pinned host argument and a tensor on the device as a
        memcpy node, which reads or writes the host buffer when it runs, as
        eager's non_blocking cudaMemcpyAsync does; _to_copy of a pinned host
        argument to the device: its allocation and that copy_."""
        if func is aten._to_copy.default:
            src, (dtype, layout, device, pin, non_blocking, mf) = args[0], (kwargs.get(k) for k in _TO_COPY_KWARGS)
            dev = torch.device(device) if device is not None else None
            if dev is None or dev.type != self.device.type or dev.index not in (None, self.device.index) or dtype not in (None, src.dtype):
                raise self.decline(f"{func} of a pinned host argument other than to the trace's device")
            if layout not in (None, torch.strided) or pin or mf not in (None, torch.preserve_format, torch.contiguous_format):
                raise self.decline(f"{func} of a pinned host argument to another layout")
            out = self.allocate(aten.empty.memory_format, (list(src.shape),), {"dtype": src.dtype, "device": self.device})
            return self._host_copy(aten.copy_.default, (out, src, bool(non_blocking)), {})
        if func is not aten.copy_.default:
            raise self.decline(f"{func} of a pinned host argument; only its copy_ to or from the device is traced")
        dst, src = args[:2]
        non_blocking = args[2] if len(args) > 2 else kwargs.get("non_blocking", False)
        h2d = isinstance(src, _TracedTensor) and src._root.host
        dev = dst if h2d else src
        if not isinstance(dev, _TracedTensor) or dev._root.host or dev.device != self.device:
            raise self.decline(f"{func} between a pinned host argument and a tensor not on the trace's device")
        if not non_blocking:
            # eager synchronizes: the host may reuse or read the buffer once copy_ returns
            raise self.decline(f"a blocking {func} to or from a pinned host argument")
        self.check_stream(str(func))
        if dst.dtype != src.dtype or any(t.is_neg() or t.is_conj() for t in (dst, src)):
            raise self.decline(f"{func} of a pinned host argument across dtypes or neg/conj bits")
        if dst.dim() != src.dim() or not _guard_each([a == b for a, b in zip(dst.shape, src.shape)]):
            raise self.decline(f"{func} of a pinned host argument that broadcasts")
        if bool(dst.numel() == 0):
            return dst
        for t in (dst, src):
            # copy_kernel_cuda's cudaMemcpyAsync; it copies other layouts through temporaries
            terms, require = [], 1
            for n, stride in reversed(list(zip(t.shape, t._sym_strides))):
                terms.append((n == 1) | (stride == require))
                require = require * n
            if not _guard_each(terms):
                raise self.decline(f"{func} of a pinned host argument and a tensor that are not both contiguous")
        nbytes = dst.numel() * dst.element_size()
        self.record_launch(Memcpy(str(func), (dst.data_ptr(), src.data_ptr()), (dst._root, src._root), nbytes, "H2D" if h2d else "D2H"))
        return dst

    def jiterator(self, code: str, name: str, return_by_ref: bool, num_outputs: int, tensors: tuple, kwargs: dict | None = None) -> Any:
        """A user jiterator's launch (torch.cuda.jiterator), which no operator
        dispatches: its CUDA host code over symbolic TensorIterators."""
        label = f"jiterator {name}"
        args = (code, name, return_by_ref, num_outputs, tensors, kwargs or {})

        def run() -> Any:
            if not all(isinstance(t, _TracedTensor) for t in tensors):
                raise self.decline(f"{label} of a tensor the trace does not track")
            if any(isinstance(v, (torch.Tensor, *_SYM_TYPES)) for v in args[5].values()):
                raise self.decline(f"{label} of an extra argument that is not a constant")
            out = self._traced_jiterator(label, lambda: _jiterator_launch(*args), tensors)
            if isinstance(out, str):
                raise self.decline(f"{label}'s CUDA host declines: {out}")
            return out

        return self.op(label, args, {}, run, host=True, redo=_JiteratorRedo())

    def op(
        self, func: Any, args: tuple, kwargs: dict, run: Callable[[], Any], host: bool = False, redo: Callable[[tuple, dict], Any] | None = None
    ) -> Any:
        """run(), the call func(*args, **kwargs); at the top level an op (OpRec).
        What decides a library step's outputs' metadata guards for the graph;
        an eager step's (its fake kernel) for the op; a traced host, which
        picks and launches its kernels and allocates its outputs, for the op
        (env.op), from the start where run() is all host (`host`)."""
        if self.in_op:
            return run()
        env = self.shape_env
        k, first, guards = len(self.ops), (len(self.launches), len(self.allocs), len(self.sites)), len(env.owners)
        self.in_op, env.op, reads = True, k if host else None, len(self.int_reads)
        if isinstance(env, _ir.Env):
            env.library_python = _library_body(func)
        try:
            # guard_sloc memoizes its walks within the op (its body's guards, an aten op's, a CuTe or Triton launcher's,
            # share their frames), holding no frame past the op
            with _ir.sloc_memo():
                out = run()
        finally:
            self.in_op, env.op = False, None
            if isinstance(env, _ir.Env):
                env.library_python = False
        # a later op reading a property this op cached would not guard it again
        for t in pytree.tree_leaves((args, kwargs, out)):
            if isinstance(t, _TracedTensor):
                torch._C._cuda_hostTraceRefreshContiguous(t)
        records = [rec for _, rec in self.launches[first[0] :]]
        kind, held = "other", None
        if not any(isinstance(rec, EagerCall) for rec in records):
            # a keyed site's nodes are launches too: the op's other launches are its selector's
            kind = "traced"
        elif len(self.sites) == first[2] and len(records) == 1 and type(records[0]) is EagerCall and not records[0].host:
            kind = "eager"
        if kind == "other" and env.owner_sets is not None and all(type(rec) is OpaqueCall for rec in records if isinstance(rec, EagerCall)):
            # only its opaque calls make it "other": its guards are graph-level (OpRec.held) until each binds as a keyed site
            env.owners[guards:] = [None if o == k else o for o in env.owners[guards:]]
            held = tuple(i for i in range(guards, len(env.owners)) if k in env.owner_sets[i])
        elif kind == "other":
            _to_graph(env, guards, k)
        elif kind == "eager" and not host:
            env.owners[guards:] = [k] * (len(env.owners) - guards)
            if env.owner_sets is not None:
                env.owner_sets[guards:] = [frozenset((k,)) for _ in env.owner_sets[guards:]]
        if read := self.int_reads[reads:]:
            del self.int_reads[reads:]
            # an op run eagerly took no value of its host's reads
            if kind == "traced":
                self._report_int_reads(func, read)
        outputs = tuple(o for o in pytree.tree_leaves(out) if isinstance(o, (torch.Tensor, int, torch.SymInt)))
        spans = (range(first[0], len(self.launches)), range(first[1], len(self.allocs)))
        self.ops.append(OpRec(func, (args, kwargs), *spans, outputs, kind, redo=redo, held=held))
        return out

    def _eager_call(self, func: Any, args: tuple, kwargs: dict) -> Any:
        if not isinstance(func, OpOverload):
            raise self.decline(f"{func} is not an operator")
        hosted = [a._root for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, _TracedTensor) and a._root.host]
        if (unread := next((r for r in hosted if r.unread), None)) is not None:
            # the unread kind holds only while nothing of the trace reads the tensor's bytes (its sizes are read freely):
            # declined here, the call keys, pins or guards the tensor as a read one is today (coordinator 10-22: no eager step)
            raise self.decline(f"{unread.name} read at op {len(self.ops)}: unread kind declined ({func} reads the storage of an unpinned CPU argument)")
        if func is aten._local_scalar_dense.default:
            raise self.decline(f"{func} reads a traced tensor's value on the host")
        if func is aten.record_stream.default:
            raise self.side_decline(f"{func}: a traced tensor's lifetime across streams in the allocator is not supported")
        if hosted:
            return self._host_copy(func, args, kwargs)
        if func in _SDPA_OPS:
            q, k, v = args[:3]
            # the shapes the kernels assume and their metas do not check (cuDNN
            # faults on others): guards, so a replay never runs them
            same = [q.size(0) == k.size(0), k.size(0) == v.size(0), k.size(1) == v.size(1)]
            if not _guard_each([*same, q.size(-1) == k.size(-1), k.size(-2) == v.size(-2)]):
                raise self.decline(f"{func} of query, key and value shapes its kernels do not take")
        packet = func.overloadpacket
        if torch.Tag.inplace_view in func.tags or packet in _METADATA_OPS:
            raise self.decline(f"{func} changes a traced tensor's metadata")
        if self.stream is not None:
            self.check_stream(str(func))
            if (backend := torch.cuda.get_allocator_backend()) != "native":
                raise self.decline(f"{func} under the {backend} allocator")
        leaves, spec = pytree.tree_flatten((args, kwargs))
        returns = func._schema.returns
        if returns and not any(isinstance(a, (torch.Tensor, *_SYM_TYPES)) for a in leaves) and not any(r.type.isSubtypeOf(torch._C.TensorType.get()) for r in returns):
            # a function of constants (can_cast, promote_types)
            return func(*args, **kwargs)
        if all(self._is_host(a) for a in leaves if isinstance(a, torch.Tensor)):
            if (result := self._host_op(func, args, kwargs)) is not NotImplemented:
                return result
        for a in leaves:
            if isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) and not self._is_host(a):
                raise self.decline(f"{func} of a tensor the trace does not track")
            if isinstance(a, (torch.SymFloat, torch.SymBool)):
                raise self.decline(f"{func} of a {type(a).__name__}")
        # a SymInt for a Tensor operand (x % n), or a number where the arg parser
        # takes none (a composite's wrapped number, remainder.Scalar's): the
        # op's Scalar overload
        numeric = not torch._C._should_allow_numbers_as_tensors(func._schema.name.split("::")[1])
        numbers = tuple(isinstance(s.type, torch.TensorType) and (isinstance(a, torch.SymInt) or (numeric and type(a) in (int, float))) for s, a in zip(func._schema.arguments, args))
        # the op's own CUDA host code takes a constant number for a Tensor
        # operand as eager's arg parser passes it, a wrapped number
        original = func
        if any(numbers):
            if (scalar := _scalar_overload(func, numbers)) is None:
                raise self.decline(f"{func} of a number for a Tensor operand")
            func = scalar
        # a traced host's, unless inside one or the op runs eagerly
        forced = self.eager_ops.get(len(self.ops))
        hosts = not self.in_aten and forced is None
        if hosts and func is aten.masked_fill_.Tensor and isinstance(args[2], _TracedTensor) and not args[0].is_complex():
            # the CUDA kernel reads a device value on the host (item()); where
            # reads it on the device, into self as masked_fill_ writes it
            x, mask, value = args
            with _TraceMode(self):
                if value.dtype != x.dtype:
                    value = aten._to_copy.default(value, dtype=x.dtype)
                return torch.where(mask, value, x, out=x)
        if hosts and func is aten.repeat.default:
            with _TraceMode(self):
                return _repeat(*args)
        if hosts and func in _MVLGAMMA and type(args[1]) is int and args[1] >= 1 and args[0].dtype is not torch.bool:
            x, p = args
            out = x if func is aten.mvlgamma_.default else kwargs.get("out")
            if out is None:
                with _TraceMode(self):
                    return _mvlgamma(x, p)
            if isinstance(out, _TracedTensor) and torch.can_cast(_mvlgamma_dtype(x), out.dtype):
                # mvlgamma_out: the op's result, resize_output and copy_
                with _TraceMode(self):
                    r = _mvlgamma(x, p)
                    if r.dim() != out.dim() or not _guard_each([a == b for a, b in zip(r.shape, out.shape)]):
                        raise self.decline(f"{func} resizes its out= argument")
                    return out.copy_(r)
        if hosts and func is aten.channel_shuffle.default and type(args[1]) is int and args[1] > 0 and args[0].dim() > 2 and _guard_each([args[0].shape[1] % args[1] == 0]):
            with _TraceMode(self):
                return _channel_shuffle(*args)
        if hosts and func is aten.soft_margin_loss_backward.default and all(isinstance(a, _TracedTensor) and a.dtype == args[1].dtype for a in args[:3]) and (args[3] != 1 or type(args[1].numel()) is int):
            with _TraceMode(self):
                return _soft_margin_loss_backward(*args)
        if hosts and func in _EYE and (eye := _eye_args(func, args, kwargs)) is not None:
            n, m, out, dtype, device = eye
            with _TraceMode(self):
                if out is None:
                    # eye's empty({0}) resized by eye_out
                    out = torch.empty((n, m), dtype=dtype, device=device)
                elif out.dim() != 2 or not _guard_each([out.shape[0] == n, out.shape[1] == m]):
                    raise self.decline(f"{func} resizes its out= argument")
                # eye_out_cuda: its as_strided({min(n, m)}, {stride(0) + stride(1)})
                out.zero_()
                out.diagonal().fill_(1)
                return out
        # the C++ composite takes no symbolic size: a symbolic factory is the pointwise host's
        factory = not any(isinstance(a, (torch.Tensor, *_SYM_TYPES)) for a in leaves)
        redispatch = factory and func not in _TRACED_ATEN
        if hosts and (func in _ZEROS or func is aten.embedding.default or (redispatch and _composite(func) and _out_overload(func) is None)):
            # CompositeExplicitAutograd (embedding, a factory (full,
            # scalar_tensor)): its parts, traced, as eager runs them
            with _TraceMode(self):
                return func.redispatch(torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA), *args, **kwargs)
        schema = func._schema
        # the argument each written alias set names
        written: dict[frozenset, Any] = {}
        for i, a in enumerate(schema.arguments):
            if a.alias_info is None or not a.alias_info.is_write:
                continue
            v = args[i] if i < len(args) else kwargs.get(a.name)
            if isinstance(v, torch.Tensor) and self._is_host(v):
                raise self.decline(f"{func} writes a CPU buffer on the device's stream")
            written[frozenset(a.alias_info.before_set)] = v

        reasons, provider, binding, state = [] if forced is None else [forced], None, None, ()
        # a symbolic number for a Scalar argument of a composite that wraps it
        # (remainder.Scalar's body: remainder.Tensor of the wrapped number): the
        # Tensor overload's own host code, the number a wrapped number's stand-in
        number = str(torch.NumberType.get())
        sym_scalars = tuple(str(a.type) == number and isinstance(v, torch.SymInt) for a, v in zip(schema.arguments, args))
        if self.device.type == "cuda" and hosts and not written and any(sym_scalars) and _kernel_less(func):
            wrapping = _tensor_overload(func, sym_scalars)
            if wrapping is not None and _has_cuda_kernel(wrapping):
                traced = self._traced_redispatch(wrapping, args, kwargs)
                if not isinstance(traced, str):
                    return traced
                reasons.append(f"{wrapping}'s CUDA host declines: {traced}")
        # a library kernel may assume its output overlaps no input. Arguments
        # may overlap at one call and not at another: a step not run eagerly
        # holds only while its written arguments overlap no other operand
        tensors = [t for t in leaves if isinstance(t, _TracedTensor)]
        roots = [t._root for t in tensors]
        writes = [self.arguments[id(v._root)] for v in written.values() if isinstance(v, _TracedTensor) and id(v._root) in self.arguments]
        reads = {id(r): self.arguments[id(r)] for r in roots if id(r) in self.arguments}.values()
        # a zero-element argument (its size specialized) addresses nothing
        others = [(w, o) for w in writes for o in reads if o is not w and w.extent[0] <= w.extent[1] and o.extent[0] <= o.extent[1]]
        pairs = {(min(w.position, o.position), max(w.position, o.position)) for w, o in others}
        overlap = any(w.extent[0] <= o.extent[1] and o.extent[0] <= w.extent[1] for w, o in others)

        on_roots = [[t for t in tensors if t._root is v._root] for v in written.values() if isinstance(v, _TracedTensor)]

        def shared(full_overlap: bool) -> bool:
            # TensorIterator allows a full overlap: an op may read the tensor it writes
            return any(len({id(t) for t in on_root} if full_overlap else on_root) > 1 for on_root in on_roots)

        # the written tensor read as itself, a TensorIterator's full overlap:
        # only the op's own CUDA host code over symbolic TensorIterators, whose
        # every launch is a TensorIterator kernel (_traced_redispatch), takes it
        full_only = not overlap and shared(False) and not shared(True)
        redispatched = self.device.type == "cuda" and hosts and _has_cuda_kernel(original)

        # a provider that keys the operands' layout on the storage harvests and checks them as they lie
        aliased = any(func in p.aliased for p in self.opaque)
        # a registered launcher records its kernel's own launch, on the operands as they lie
        impl = torch.cuda._host_trace._TRACED_IMPLS.get(func) if hosts and torch.cuda._host_trace.traced_impls else None
        if full_only and not aliased and impl is None and redispatched:
            # the one route that takes a full overlap
            traced = self._traced_redispatch(original, args, kwargs)
            if not isinstance(traced, str):
                self.argument_pairs |= pairs
                return traced
            reasons.append(f"{original}'s CUDA host declines: {traced}")
        elif overlap or (shared(False) and not aliased and impl is None):
            reasons.append(f"{func} writes a storage another operand is of")
        else:
            if impl is not None:
                traced = self._traced_host(func, args, kwargs, lambda: impl(*args, **kwargs))
                if not isinstance(traced, str):
                    self.argument_pairs |= pairs
                    return traced
                if shared(False) and not aliased:
                    reasons.append(f"{func} writes a storage another operand is of")
            if self.opaque and shared(False) and not aliased:
                reasons.append(f"{func} writes a tensor it reads: no library kernel")
            for p in self.opaque if not reasons else ():
                if (why := p.accepts(func, args, kwargs)) is None:
                    provider = p
                    break
                if why:
                    reasons.append(why)
            if impl is not None and provider is None:
                reasons.insert(0, f"{func}'s traced impl declines: {traced}")
            # a traced host's outputs and guards are eager's own: no fake kernel
            if provider is None and hosts and (kernel := _python_kernel(func)) is not None:
                body = _library_body_kind(func, kernel)
                if func.namespace == "aten":
                    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
                    traced = self._traced_host(func, args, kwargs, lambda: kernel(keyset, *args, _fallback=_Fallback(self, func), **kwargs))
                elif body is not None and self.under_library_body is not None:
                    traced = f"the trace runs inside a {self.under_library_body} body, whose state its body may read"
                else:
                    traced = self._traced_host(func, args, kwargs, lambda: kernel(*args, **kwargs), body=body)
                if not isinstance(traced, str):
                    self.argument_pairs |= pairs
                    return traced
                reasons.append(f"{func}'s kernel declines: {traced}")
            elif provider is None and hosts:
                if func.namespace != "aten" and _library_kernel(func) is not None:
                    reasons.append(f"{func}'s CUDA kernel is a Library.impl Python function (library_impls is off)")
                traced = self._traced_aten(func, args, kwargs) if func in _TRACED_ATEN else _UNTRACED
                if isinstance(traced, str):
                    reasons.append(f"{func}'s traced host declines: {traced}")
                    traced = _UNTRACED
                if traced is _UNTRACED and redispatched:
                    traced = self._traced_redispatch(original, args, kwargs)
                    if isinstance(traced, str):
                        reasons.append(f"{original}'s CUDA host declines: {traced}")
                        traced = _UNTRACED
                if traced is _UNTRACED and _kernel_less(func):
                    # a CompositeExplicitAutograd(NonFunctional) op: its body, as
                    # eager runs it, each of its parts routed as any op
                    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)

                    def redispatch() -> Any:
                        try:
                            return func.redispatch(keyset, *args, **kwargs)
                        except RuntimeError as e:
                            # an int-signature body's wrapper (C10_AS_INTARRAYREF_SLOW)
                            if "expected to contain only concrete integers" not in str(e):
                                raise
                            return _INT_BODY

                    def body() -> Any:
                        if func in _SYM_BODIES and any(isinstance(a, _SYM_TYPES) for a in pytree.tree_leaves((args, kwargs))):
                            return _SYM_BODIES[func](*args, **kwargs)
                        # its own host code: a TensorIterator the body builds
                        # (copy_'s copy_impl) is symbolic, its launches recorded
                        out = self._symbolic_host(func, lambda sink: torch._C._cuda_hostTraceRedispatch(redispatch, sink), tensors, {})
                        if out is not _INT_BODY:
                            return out
                        if func not in _NEW_FACTORIES:
                            raise declined("its body takes no symbolic size")
                        # Tensor.new_full's body: the factory at self's options merged with the call's
                        x, size, *fill = args
                        options = {"dtype": x.dtype, "layout": x.layout, "device": x.device}
                        options.update((k, v) for k, v in kwargs.items() if v is not None)
                        return _NEW_FACTORIES[func](size, *fill, **options)

                    traced = self._traced_host(func, args, kwargs, body, parts=True)
                    if isinstance(traced, str):
                        if traced:
                            reasons.append(f"{func}'s body declines: {traced}")
                        traced = _UNTRACED
                # (an op of no returns traces as None)
                if traced is not _UNTRACED:
                    self.argument_pairs |= pairs
                    return traced
        # an eager or library step's outputs: its meta function's metadata on
        # bare meta twins, else the fake kernel's
        routed = func in _META_OPAQUE and (meta := self._opaque_meta(func, leaves, spec)) is not None
        if routed:
            out, twins = meta
        else:
            twins, out = self._fake_call(func, args, kwargs)
        out = _without_unwritten_lse(func, args, kwargs, out)
        rets = [out] if len(schema.returns) == 1 else list(out or ())
        for o in pytree.tree_leaves(rets):
            if o is None:  # an optional output, a replay's None too
                continue
            if (type(o) is int or isinstance(o, torch.SymInt)) and not free_unbacked_symbols(o):
                continue  # the fake kernel's value, as a view's metadata is
            if not isinstance(o, torch.Tensor):
                raise self.decline(f"{func} returns a {type(o).__name__}")
            if free_unbacked_symbols((o.shape, o.stride(), o.storage_offset())):
                raise self.decline(f"{func} returns a data-dependent shape")
            if o.device != self.device and not routed:
                raise self.decline(f"{func} returns a tensor on {o.device}")
            if o.layout != torch.strided or o.is_conj() or o.is_neg():
                raise self.decline(f"{func} returns a tensor that is not plain strided")
        if provider is not None:
            fakes = [o for r, o in zip(schema.returns, rets) if r.alias_info is None]
            fakes = [o for o in pytree.tree_leaves(fakes) if isinstance(o, torch.Tensor)]
            binding, values, refusal = bind_at_trace(provider, func, args, kwargs, fakes, self.values)
            for v in values:
                int(v)  # a guard: Eq(v, its value at the call)
            # an operand on an eager step's output with a size-1 dim, whose stride there the warm-up may yet set to
            # eager's (_check_witness): the call binds after the trace (bind_opaque), at the key its rows then hold.
            # Either way it is a keyed site: the choice reads the call, no guard
            pending = {id(o._root) for o in self.eager_outputs if any(self.values(n) == 1 for n in o.shape)}
            if binding is not None and any(isinstance(t, _TracedTensor) and id(t._root) in pending for t in leaves):
                binding = None
            # a key the warm-up learned binds after the trace, as one learned then would
            if binding is not None and trace_key(func, args, kwargs, fakes, self.values)[0] in self.warm_learned:
                binding = None
            if refusal is not None:
                provider, reasons, state = None, [refusal], library_state()
        if provider is not None:
            self.argument_pairs |= pairs
        made: list[_TracedTensor] = []  # the binding's fresh outputs

        storages = {t.untyped_storage()._cdata for t in twins if isinstance(t, torch.Tensor)}

        def fresh(o: torch.Tensor) -> _TracedTensor:
            storage = o.untyped_storage()._cdata
            if storage in storages:
                raise self.decline(f"{func} returns an alias of a tensor")
            storages.add(storage)
            if binding is not None:
                sizes, strides = list(o.shape), list(o.stride())
                kw = {"dtype": o.dtype, "device": self.device}
                made.append(self.allocate(aten.empty_strided.default, (sizes, strides), kw))
                return made[-1]
            k = len(self.eager_outputs)
            root = _Root(f"e{k}", None, "eager", k)
            q = self.shape_env.symbol(_placeholder(root) // _ALLOC_ALIGNMENT, f"eager{k}.base/{_ALLOC_ALIGNMENT}")
            root.sym = _ALLOC_ALIGNMENT * q
            sizes, strides, offset = list(o.shape), list(o.stride()), o.storage_offset()
            t = _TracedTensor(root, sizes, strides, offset, o.dtype, self.device)
            self.eager_outputs.append(t)
            return t

        result = []
        for r, o in zip(schema.returns, rets):
            if r.alias_info is None:
                result.append(pytree.tree_map_only(torch.Tensor, fresh, o))
                continue
            v = written.get(frozenset(r.alias_info.before_set))
            if not r.alias_info.is_write or not isinstance(v, _TracedTensor):
                raise self.decline(f"{func} returns an alias of an argument")
            # an out= argument of other sizes or strides is resized, a metadata
            # mutation
            same = [a == b for a, b in zip(o.shape, v.shape)]
            same += [a == b for a, b in zip(o.stride(), v._sym_strides)]
            if o.dim() != v.dim() or not _guard_each(same):
                raise self.decline(f"{func} resizes its out= argument")
            result.append(v)
        outputs = tuple(o for o in pytree.tree_leaves(result) if isinstance(o, torch.Tensor))
        if binding is not None:
            operands = [a for a in leaves if isinstance(a, torch.Tensor)] + made
            scalars = [a for a in leaves if not isinstance(a, torch.Tensor)]
            call = (spec, frozenset(j for j, a in enumerate(leaves) if isinstance(a, torch.Tensor)))
            self.sites.append(record_binding(self, func, provider, binding, operands, scalars, call, trace_key(func, args, kwargs, fakes, self.values)[0]))
        elif provider is not None:
            bound = provider.bind(trace_key(func, args, kwargs, fakes, self.values)[0]) is not None
            self.record_launch(OpaqueCall(func, args, kwargs, outputs, generator=self.generator, provider=provider, state=library_state(), bound=bound))
        else:
            why = "; ".join(reasons) or None
            self.record_launch(EagerCall(func, args, kwargs, outputs, why, self.generator, state))
        if len(schema.returns) == 1:
            return result[0]
        return type(out)(result) if result else None  # a tuple or a structseq

    def _fake_call(self, func: OpOverload, args: tuple, kwargs: dict) -> tuple[Any, Any]:
        # the Python dispatcher, as under Dynamo: a C++ composite (a meta
        # function's expand) takes no symbolic sizes
        with self.fake_mode, enable_python_dispatcher():
            twins = pytree.tree_map_only(_TracedTensor, self._twin, (args, kwargs))
            twins = pytree.tree_map_only(torch.Tensor, lambda t: self._host_fake(t) if self._is_host(t) else t, twins)
            try:
                out = func(*twins[0], **twins[1])
            except Exception as e:
                why = f"{type(e).__name__}: {str(e).splitlines()[0]}"
                raise self.decline(f"{func} has no traced metadata ({why})") from e
        # an op that changes an argument's metadata (an in-place view, a resize), tagged or not: its twin's.
        # One it returns is its written output, which _eager_call compares
        returned = {id(o) for o in pytree.tree_leaves(out)}
        for t, twin in zip(pytree.tree_leaves((args, kwargs)), pytree.tree_leaves(twins)):
            if isinstance(t, _TracedTensor) and id(twin) not in returned and (twin.dim() != t.dim() or not all(map(_same, (*twin.shape, *twin.stride(), twin.storage_offset()), (*t.shape, *t._sym_strides, t._sym_offset)))):
                raise self.decline(f"{func} changes a traced tensor's metadata")
        return twins, out
    def _opaque_meta(self, func: OpOverload, leaves: list, spec: Any) -> tuple[Any, list] | None:
        """func's outputs (meta tensors) and operand twins from its meta
        function on bare meta twins, without FakeTensorMode's dispatch; None
        leaves the call to the fake kernel."""
        if any(isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) for a in leaves):
            return None
        twins = [
            torch.empty(0, dtype=a.dtype, device="meta").as_strided(a.shape, a._sym_strides, a._sym_offset)
            if isinstance(a, _TracedTensor)
            else a
            for a in leaves
        ]
        args, kwargs = pytree.tree_unflatten(twins, spec)
        try:
            return _META_OPAQUE[func](*args, **kwargs), twins
        except Exception:
            return None  # the fake's verdict

    def _traced_host(
        self,
        func: OpOverload,
        args: tuple,
        kwargs: dict,
        kernel: Callable[[], Any],
        parts: bool = False,
        body: str | None = None,
        forget: bool = False,
    ) -> Any:
        """func's output from kernel(), its Python CUDA kernel (_python_kernel)
        or its composite body on the traced tensors, its calls and launches
        traced as any; why where it declines (or with `parts`, a part runs
        eagerly), which leaves the call to EagerCall: '' where that part runs
        eagerly for no reason, as a plain eager call, and so does the op."""
        env = self.shape_env
        marks = len(self.allocs), len(self.launches), len(self.sites), len(self.eager_outputs), len(self.aten_calls)
        guards, op, env.op, prior, host, host_reads = len(env.owners), env.op, len(self.ops), self.declined, dict(self.host), set(self.host_reads)
        plain_part = False
        outer = self.library_body
        if body is not None:
            self.library_body = func, body
        try:
            # the Python and global state a torch._native condition reads
            # (node.active, a config flag, the arguments' copy-on-write state)
            # is fixed for the entry under the trace's contract that Python
            # state is fixed at trace time, so it is not guarded
            with _TraceMode(self):
                out = kernel()
            # the body may catch reentered()
            if self.reentry is not None:
                raise declined(self.reentry)
            # a part run eagerly would split the op's eager step in more
            if parts and (eager := next((r for _, r in self.launches[marks[1] :] if isinstance(r, EagerCall)), None)) is not None:
                plain_part = eager.reason is None
                raise declined(f"its part {eager.name} runs eagerly ({eager.reason})")
            written = {id(a) for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, _TracedTensor)}
            made = {id(a.root) for a in self.allocs[marks[0] :]} | {id(t._root) for t in self.eager_outputs[marks[3] :]}
            returns = func._schema.returns
            outs = (out,) if len(returns) == 1 else tuple(out or ())
            for r, o in zip(returns, outs, strict=True):
                if o is None:
                    continue
                if r.alias_info is not None:
                    if id(o) not in written:
                        raise declined(f"{func}'s kernel returned no argument where its schema aliases one")
                    continue
                for t in o if isinstance(o, list) else (o,):
                    if not isinstance(t, _TracedTensor) or id(t._root) not in made:
                        raise declined(f"{func}'s kernel returned a {type(t).__name__} it did not allocate")
            self._witnessed(func, args, kwargs, out)
            return out
        except Exception as e:
            if isinstance(e, Declined) and e.retry:
                raise
            if not isinstance(e, Declined) and torch.cuda._host_trace.raise_unexpected:
                raise
            if any(seq in self.stream_of for seq, _ in self.launches[marks[1] :]):
                self._join_side_streams()
            del self.allocs[marks[0] :], self.launches[marks[1] :], self.sites[marks[2] :], self.eager_outputs[marks[3] :]
            del self.aten_calls[marks[4] :]
            self.declined, self.host, self.host_reads, self.reentry = prior, host, host_reads, None
            if forget:
                # an op's own host code stepping aside (`forget`) runs eagerly
                # at any sizes: what it read before is no guard of the call
                env.forget(guards)
            else:
                # what the kernel declined on chose the eager call
                _to_graph(env, guards)
            if plain_part:
                return ""
            return str(e).splitlines()[0] if isinstance(e, Declined) else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        finally:
            env.op, self.library_body = op, outer

    def _traced_aten(self, func: OpOverload, args: tuple, kwargs: dict, witnessed: bool = True) -> Any:
        """func's traced ATen host (_TRACED_ATEN): its output (a tuple for more than one),
        allocated through this trace, and its kernels as KernelLaunches; why
        where the host declines, which leaves the call to EagerCall."""
        call = (args, kwargs) if witnessed else None
        rest = func._schema.arguments[len(args) :]
        args = (*args, *(kwargs.get(a.name, a.default_value) for a in rest))
        leaves = pytree.tree_leaves(args)
        if any(isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) for a in leaves):
            return "an operand outside the trace"
        if func not in _SYM_SCALAR_HOSTS and any(isinstance(a, _SYM_TYPES) for a in args):
            return "a symbolic scalar operand"
        # a Python number where the schema takes a Tensor (x + 1)
        if any(isinstance(s.type, torch.TensorType) and not isinstance(a, _TracedTensor) for s, a in zip(func._schema.arguments, args)):
            return "a Python number for a Tensor operand"
        tensors = [a for a in leaves if isinstance(a, _TracedTensor)]
        return self._run_host(func, lambda: _TRACED_ATEN[func](*args), tensors, call)

    def _witnessed(self, func: OpOverload, args: tuple, kwargs: dict, out: Any) -> None:
        # a traced host's outputs, checked against the warm-up's as an eager call's are
        outputs = tuple(o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor))
        self.aten_calls.append((next(self._seq), EagerCall(func, args, kwargs, outputs)))

    def _run_host(self, func: OpOverload | str, host: Callable[[], Any], tensors: list, call: tuple | None, check: Callable[..., bool] | None = None) -> Any:
        """host() under this trace, a traced ATen host's (out, records): out, its
        launches recorded and, with the host's `call` (args, kwargs), its
        outputs witnessed; nothing recorded, and why where the host declines,
        None where check(out, launches) fails. Its other exception declines the
        trace: eager's own error for the call."""
        from torch.cuda._host_trace_launch import KernelLaunch

        marks = len(self.allocs), len(self.launches)
        env = self.shape_env
        guards, op, env.op = len(env.owners), env.op, len(self.ops)
        self.in_aten = True
        try:
            with _TraceMode(self):
                out, records = host()
        except NotImplementedError as e:
            del self.allocs[marks[0] :], self.launches[marks[1] :]
            # what the host declined on chose the eager call
            _to_graph(env, guards)
            env.op = op
            return str(e).splitlines()[0]
        except Exception as e:
            del self.allocs[marks[0] :], self.launches[marks[1] :]
            env.op = op
            if isinstance(e, (Declined, AssertionError)):
                raise  # the trace's own verdict, or its bug
            if torch.cuda._host_trace.raise_unexpected and not isinstance(e, (RuntimeError, IndexError, ValueError)):
                raise  # not a TORCH_CHECK: the route's or the trace's bug
            raise self.decline(f"{func} raised {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}") from e
        finally:
            self.in_aten = False
        outs = out if isinstance(out, tuple) else (out,)
        if not all(isinstance(o, _TracedTensor) for o in outs):
            raise AssertionError(f"{func}'s traced host returned {[type(o) for o in outs]}")
        env.op = op
        made = [a.root for a in self.allocs[marks[0] :]]
        roots = tuple({id(r): r for r in (*(t._root for t in (*tensors, *outs)), *made)}.values())
        launches: list[Any] = []
        for record in records:
            if len(record) == 3:
                dst, src, _ = record
                # copy_device_to_device skips a memcpy onto its source
                apart = _sym_expr(dst) - _sym_expr(src)
                if not any(_sym_expr(r.sym).free_symbols & apart.free_symbols for r in roots) and bool(dst == src):
                    continue
            launches.append(self._launch_of(func, record, roots))
        if check is not None and not check(out, launches):
            del self.allocs[marks[0] :], self.launches[marks[1] :]
            _to_graph(env, guards)
            return None
        for launch in launches:
            if isinstance(launch, KernelLaunch):
                self.host_reads.update(t.untyped_storage()._cdata for *_, t in launch.cpu_scalars)
            self.record_launch(launch)
        if call is not None:
            self._witnessed(func, *call, out)
        return out

    def _traced_redispatch(self, func: OpOverload, args: tuple, kwargs: dict) -> Any:
        """func's own CUDA host code (the generated wrapper, its structured meta
        and impl, the DispatchStub and kernel) run under the trace over
        symbolic TensorIterators (torch._C._cuda_hostTraceRedispatch): each
        launch recorded as eager's code builds it, every decision on a symbol a
        guard; outputs allocated through the trace and nested ops routed as any.
        Its output, or why it declines: code not converted to symbolic stops at
        an int64 read (an accessor, a data pointer, the generator) and is
        reported in Tape.unconverted; in strict mode (the hint audit's) a gap
        after a symbolic TensorIterator was built raises."""
        schema = func._schema
        values = (*args, *(kwargs.get(a.name, a.default_value) for a in schema.arguments[len(args) :]))
        for s, a in zip(schema.arguments, values):
            if isinstance(a, torch.Generator) or (a is not None and "Generator" in str(s.type) and isinstance(a, torch.Generator)):
                return "it draws from a generator"
            if isinstance(s.type, torch.TensorType) and isinstance(a, _SYM_TYPES) and not isinstance(a, torch.SymInt):
                return f"a symbolic {type(a).__name__} for its Tensor operand {s.name}"
        # a host buffer (a CPU tensor the trace's host steps write) reaches the
        # kernel as a stand-in whose storage raises on a read: its value is each
        # replay's, which only a CPU scalar member (KernelLaunch.cpu_scalars) reads
        stand_ins: dict[int, torch.Tensor] = {}

        def number_stand_in(a: torch.SymInt) -> torch.Tensor:
            # a symbolic number for a Tensor operand: a wrapped number's stand-in
            n = _SymNumberStandIn(a)
            stand_ins[id(n)] = _SymNumber(a)
            return n

        def stand_in(a: Any) -> Any:
            if not (isinstance(a, torch.Tensor) and self._is_host(a)):
                return a
            if a.dim() != 0:
                raise declined("a CPU buffer operand that is not 0-dim")
            s = _HostBufferStandIn(a)
            stand_ins[id(s)] = a
            return s

        try:
            a2, k2 = pytree.tree_map_only(torch.Tensor, stand_in, (args, kwargs))
            types = {x.name: x.type for x in schema.arguments}
            a2 = [number_stand_in(a) if isinstance(a, torch.SymInt) and isinstance(x.type, torch.TensorType) else a for x, a in zip(schema.arguments, a2)]
            k2 = {k: number_stand_in(a) if isinstance(a, torch.SymInt) and isinstance(types[k], torch.TensorType) else a for k, a in k2.items()}
        except Declined as e:
            return str(e).splitlines()[0]
        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
        leaves = [t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, _TracedTensor)]

        def kernel() -> Any:
            # a number for a Tensor operand is a wrapped number, as eager's arg parser makes it
            return self._symbolic_host(func, lambda sink: torch._C._cuda_hostTraceRedispatchOp(func._handle, keyset, tuple(a2), dict(k2), sink), leaves, stand_ins)

        return self._traced_host(func, args, kwargs, kernel, parts=True, forget=True)

    def _symbolic_host(self, func: Any, invoke: Callable[[Callable[[tuple], None]], Any], tensors: Sequence[_TracedTensor], stand_ins: Mapping[int, torch.Tensor]) -> Any:
        """invoke(sink), an op's own CUDA host code over `tensors`, with
        symbolic TensorIterators on (torch._C._cuda_hostTraceRedispatch*):
        each launch handed to sink and recorded as it happens, the traced tensors' storages sealed (a data
        pointer read raises). Code not converted to symbolic declines and is
        recorded in self.unconverted; in strict mode a gap after a symbolic
        TensorIterator was built raises UnconvertedHost."""
        marks = len(self.allocs), len(self.eager_outputs)
        arg_roots = {id(t._root): t._root for t in tensors}

        def sink(record: tuple) -> None:
            roots = dict(arg_roots)
            roots.update((id(a.root), a.root) for a in self.allocs[marks[0] :])
            roots.update((id(t._root), t._root) for t in self.eager_outputs[marks[1] :])
            launch = self._launch_of(func, record, tuple(roots.values()), stand_ins)
            if not isinstance(launch, Memcpy):
                self.host_reads.update(t.untyped_storage()._cdata for *_, t in launch.cpu_scalars)
            self.record_launch(launch)

        built = torch._C._cuda_hostTraceSymbolicIteratorsBuilt()
        outer = self.sealed
        self.sealed = []
        for t in tensors:
            torch._C._cuda_hostTraceSeal(t, True)
            self.sealed.append(t)
        torch._C._cuda_hostTraceTakeRefusal()
        env = self.shape_env
        library_python = isinstance(env, _ir.Env) and env.library_python
        if library_python:
            env.library_python = False
        try:
            return invoke(sink)
        except Exception as e:
            if isinstance(e, Declined):
                raise
            refusal = torch._C._cuda_hostTraceTakeRefusal()
            symbolic = torch._C._cuda_hostTraceSymbolicIteratorsBuilt() > built
            lines = str(e).splitlines()
            message, site = (lines[0] if lines else ""), ("\n".join(lines[1:]).strip() or None)
            record = Unconverted(str(func), refusal, type(e).__name__, message, site, symbolic)
            self.unconverted.append(record)
            UNCONVERTED[(record.op, record.refusal, record.message)] += 1
            # strict raises at code a symbolic iterator refused (its accessors, its
            # outputs): converted code's gap. A data pointer or value read, a sizes()
            # of a symbolic tensor, the generator are the op's host stepping aside
            if refusal in ("accessor", "output") and symbolic and _hint_audit.mode == "strict":
                raise UnconvertedHost(f"{func}: {type(e).__name__}: {message}{chr(10) + site if site else ''}") from e
            raise declined(f"its host stops at {refusal or type(e).__name__}: {message}") from e
        finally:
            if library_python:
                env.library_python = True
            for t in self.sealed:
                torch._C._cuda_hostTraceSeal(t, False)
            # a nested op's host (a composite body's part) unsealed what the enclosing one sealed
            for t in outer or ():
                torch._C._cuda_hostTraceSeal(t, True)
            self.sealed = outer

    def _traced_jiterator(self, label: str, call: Callable[[], Any], tensors: Sequence[_TracedTensor]) -> Any:
        """A user jiterator's launch (no operator): its CUDA host code
        (jiterator.cu) over symbolic TensorIterators, as _traced_redispatch's;
        its outputs, or why it declines (nothing recorded)."""
        env = self.shape_env
        marks = len(self.allocs), len(self.launches), len(self.eager_outputs)
        guards, op = len(env.owners), env.op
        env.op = len(self.ops)
        try:
            with _TraceMode(self):
                out = self._symbolic_host(label, lambda sink: torch._C._cuda_hostTraceRedispatch(call, sink), tensors, {})
        except Declined as e:
            if e.retry:
                raise
            del self.allocs[marks[0] :], self.launches[marks[1] :], self.eager_outputs[marks[2] :]
            _to_graph(env, guards)
            return str(e).splitlines()[0]
        finally:
            env.op = op
        made = {id(a.root) for a in self.allocs[marks[0] :]}
        # one output is a tensor, more a tuple (torch._C._cuda_jiterator_compile_and_launch_kernel)
        if not all(isinstance(t, _TracedTensor) and id(t._root) in made for t in (out if isinstance(out, tuple) else (out,))):
            raise AssertionError(f"{label} returned a tensor it did not allocate")
        return out

    def int_read(self, accessor: str, t: Any) -> None:
        """C++ host code read a traced tensor's sizes, strides or offset as
        ints: each symbolic value becomes a guard of its hint (sound,
        specialized), owned by the op. Reported (op()) where the op traces."""
        values = {"sizes()": t.shape, "strides()": t.stride(), "storage_offset()": (t.storage_offset(),)}[accessor]
        if any(isinstance(v, torch.SymInt) and v.node.maybe_as_int() is None for v in values):
            self.int_reads.append(f"a traced tensor's {accessor}")

    def _report_int_reads(self, func: Any, reads: list[str]) -> None:
        # a traced op whose host specialized: in self.unconverted; in strict mode it raises
        op, message = str(func), f"host code read symbolic values as ints (each a guard of its hint): {', '.join(sorted(set(reads)))}"
        self.unconverted.append(Unconverted(op, "int read", "IntRead", message, None, False))
        UNCONVERTED[(op, "int read", message)] += 1
        if _hint_audit.mode == "strict":
            raise UnconvertedHost(f"{op}: {message}")

    def resize_fresh(self, t: torch.Tensor, sizes: list, strides: list) -> bool:
        """resize_output of an op's own fresh empty output (its functional form
        over its out= kernel: empty({0}), then the kernel resizes it to the
        iterator's sizes and strides): a new allocation of those, t rebound
        to it, as eager's resize_ allocates a new storage; False for any
        other tensor (its metadata the trace never changes)."""
        if not isinstance(t, _TracedTensor) or t._root.kind != "allocation":
            return False
        rec = next((a for a in self.allocs if a.root is t._root), None)
        # zero elements of a constant size, and nothing launched since
        if rec is None or not any(type(n) is int and n == 0 for n in rec.sizes) or any(s > rec.seq for s, _ in self.launches):
            return False
        options = {"dtype": t.dtype, "device": t.device}
        if strides:
            new = self.allocate(aten.empty_strided.default, (sizes, strides), options)
        else:
            new = self.allocate(aten.empty.memory_format, (sizes,), options)
        t._root, t._sym_strides, t._sym_offset = new._root, list(new._sym_strides), new._sym_offset
        torch._C._cuda_hostTraceRestride(t, list(new.shape), list(new._sym_strides), new._sym_offset)
        return True

    def _launch_of(self, func: Any, record: tuple, roots: tuple, stand_ins: Mapping[int, torch.Tensor] | None = None) -> Any:
        """A C++ host's launch record (torch/csrc/cuda/host_trace/Aten.cpp
        record()) as a KernelLaunch or Memcpy; a CPU scalar of a stand-in reads
        its host buffer."""
        from torch.cuda._host_trace_launch import KernelLaunch

        if len(record) == 3:
            dst, src, nbytes = record
            return Memcpy(str(func), (dst, src), roots, nbytes)
        function, offsets, params, fields, grid, block, smem, *cpu_scalars = record
        places, values, is_pointer = [], [], []
        for param, offset, width, value, pointer in fields:
            places.append((param, offset, width))
            values.append(value)
            is_pointer.append(pointer)
        scalars = []
        for p, at, cls, src in cpu_scalars[0] if cpu_scalars else ():
            src = (stand_ins or {}).get(id(src), src)
            if isinstance(src, _SymNumber):
                # a symbolic number's member: a field of the number's value at each replay
                place, value = _sym_number_field(p, at, cls, src.value)
                places.append(place)
                values.append(value)
                is_pointer.append(False)
            else:
                scalars.append((p, at, cls, src))
        scalars = tuple(scalars)
        return KernelLaunch(
            str(func),
            function,
            None,
            tuple(zip(offsets, map(len, params))),
            tuple(grid),
            tuple(block),
            smem,
            tuple(values),
            roots,
            fields=tuple(places),
            pointers=frozenset(i for i, p in enumerate(is_pointer) if p),
            images=tuple(params),
            generator=self.generator,
            cpu_scalars=scalars,
        )

    def _check_as_strided(self, root: _Root, v: torch.Tensor) -> None:
        # eager's setStrided checks (checkAsStridedArgs, checkInBoundsForStorage),
        # which the fake kernel skips for symbolic values
        sizes, strides, offset = list(v.shape), list(v.stride()), v.storage_offset()

        def error(what: str) -> Callable[[], RuntimeError]:
            def make() -> RuntimeError:
                at = self.values
                return RuntimeError(what.format(f"sizes {list(map(at, sizes))}, strides {list(map(at, strides))}, storage offset {at(offset)}"))

            return make

        for st in strides:
            self.check(st >= 0, error("as_strided: Negative strides are not supported ({})"))
        self.check(offset >= 0, error("Tensor: invalid storage offset ({})"))
        if root.kind == "allocation" and id(root) not in self.outer:
            a = next(a for a in self.allocs if a.root is root)
            empty = _any_zero(a.sizes)
            extent = (1 + sum((n - 1) * st for n, st in zip(a.sizes, a.strides))) * a.dtype.itemsize
            bound = torch.cuda._host_trace.select(empty, 0, extent) if isinstance(empty, torch.SymBool) else 0 if empty else extent
            needed = (offset + 1 + sum((n - 1) * st for n, st in zip(sizes, strides))) * v.dtype.itemsize
            self.check(_either(_any_zero(sizes), needed <= bound), error("setStorage: {} are out of bounds for the storage"))
            return
        if any(bool(s == 0) for s in sizes):
            return
        needed = _storage_nbytes(sizes, strides, offset, v.dtype.itemsize)
        # neither an input's nor an eager output's storage size is part of the
        # trace: the bound is the bytes up to the input's or output's end
        if root.kind == "argument":
            t = next(i for i in self.inputs if i.root is root)
            sizes, strides, offset, what = t.sizes, t.strides, t.offset, t.name
        else:
            t = self.outer[id(root)] if id(root) in self.outer else next(o for o in self.eager_outputs if o._root is root)
            sizes, strides, offset = list(t.shape), t._sym_strides, t._sym_offset
            what = f"eager output {root.name}" if root.kind == "eager" else root.name
        bound = _storage_nbytes(sizes, strides, offset, t.dtype.itemsize)
        if not bool(needed <= bound):
            raise self.decline(f"as_strided past the end of {what}")


def _python_kernel(func: OpOverload) -> Callable[..., Any] | None:
    """func's CUDA kernel where it is Python: a custom op's, a torch._native
    override's, or the router of the overrides on an ATen op."""
    from torch._library.custom_ops import _maybe_get_opdef
    from torch._native import registry

    ns, name = func._schema.name.split("::")
    if ns == "aten":
        kernels = registry._aten_override_kernels
        return kernels.get((f"{name}.{func._overloadname}", "CUDA"), kernels.get((name, "CUDA")))
    if ns == "_native":
        nodes = (n for (_, key), graph in registry._graphs.items() if key == "CUDA" for n in graph)
        return next((n.impl_fn for n in nodes if n.node_id == name), None)
    if (opdef := _maybe_get_opdef(func)) is None:
        return _library_kernel(func) if torch.cuda._host_trace.library_impls else None
    if (kernel := opdef._backend_fns.get("cuda")) is not None:
        return kernel
    # a CUDA kernel registered outside the custom op runs ahead of its
    # CompositeExplicitAutograd one
    if _has_kernel(func.name(), "CUDA"):
        return None
    return opdef._backend_fns.get(None)


def _library_body(func: Any) -> bool:
    """A non-ATen operator or a dispatch unit (a Python function run as one op): its Python body runs in a redo. ATen
    operators, Triton / CuTe DSL launches and jiterator ops redo from their recorded arguments."""
    if isinstance(func, OpOverload):
        return func.namespace != "aten"
    if isinstance(func, str):  # a jiterator op's label
        return False
    module = type(func).__module__ or ""
    if module.startswith(("triton", "cutlass", "torch.cuda._host_trace")):  # a Triton / CuTe DSL launch
        return False
    return callable(func)


def _library_body_kind(func: OpOverload, kernel: Callable[..., Any]) -> str | None:
    """Where `kernel` (_python_kernel's) is a body whose re-entry switch is on,
    its kind: "Library.impl" or "custom_op"."""
    ht = torch.cuda._host_trace
    if func.namespace in ("aten", "_native"):
        return None
    if kernel is _library_kernel(func):
        return "Library.impl" if ht.library_impls_reentry else None
    return "custom_op" if ht.custom_op_reentry else None


_library_codes: tuple[tuple[Any, ...], dict[Any, str]] = ((), {})


def _library_body_on_stack() -> str | None:
    """The kind of the innermost Library.impl or custom_op body on the stack
    whose re-entry switch is on. A custom op's kernels are Library.impl
    functions too, defined in torch._library.custom_ops."""
    global _library_codes
    ht = torch.cuda._host_trace
    # the kernels themselves are the key (held, so no id is reused)
    if (impls := tuple(torch.library._python_impls.values())) != _library_codes[0]:
        codes = {}
        for k in impls:
            if (code := getattr(k, "__code__", None)) is not None:
                codes[code] = "custom_op" if code.co_filename == torch._library.custom_ops.__file__ else "Library.impl"
        _library_codes = (impls, codes)
    on = {"custom_op": ht.custom_op_reentry, "Library.impl": ht.library_impls_reentry}
    frame = sys._getframe(1)
    while frame is not None:
        if (kind := _library_codes[1].get(frame.f_code)) is not None and on[kind]:
            return kind
        frame = frame.f_back
    return None


def _library_kernel(func: OpOverload) -> Callable[..., Any] | None:
    """func's CUDA kernel where Library.impl registered a Python function."""
    ns, name = func._schema.name.split("::")
    kernels = torch.library._python_impls
    if (kernel := kernels.get(f"{ns}/{name}.{func._overloadname}/CUDA")) is not None:
        return kernel
    return kernels.get(f"{ns}/{name}/CUDA") if func._overloadname == "default" else None


class _Fallback:
    """The ATen kernel a torch._native router falls back to, traced: the traced
    ATen host, or a decline that leaves the call to EagerCall."""

    def __init__(self, tr: _Trace, func: OpOverload) -> None:
        self.tr, self.func = tr, func

    def call_boxed(self, keyset: Any, *args: Any, **kwargs: Any) -> Any:
        # the router's call is witnessed as func's (_traced_host)
        out = self.tr._traced_aten(self.func, args, kwargs, witnessed=False) if self.func in _TRACED_ATEN else None
        if out is not None and not isinstance(out, str):
            return out
        raise declined(f"{self.func} falls back to its ATen kernel")


def _owned(owners: list[int | None], owner_sets: Sequence[Collection[int | None]] | None, n: int) -> list[list[int]]:
    """Per op the guards it owns (OpRec.guards): those only it read, and with
    owner sets (shared_op_guards) those only ops read, each of theirs."""
    owned: list[list[int]] = [[] for _ in range(n)]
    for i, o in enumerate(owners):
        if o is not None:
            owned[o].append(i)
        elif owner_sets is not None and None not in owner_sets[i]:
            for k in sorted(owner_sets[i]):  # type: ignore[type-var]
                owned[k].append(i)
    return owned


def _to_graph(env: Any, n: int, op: int | None = None) -> None:
    """The guards recorded after the first n that op owns, or every one
    without op, become graph-level; their owner sets keep the op."""
    env.owners[n:] = [None if op is None or o == op else o for o in env.owners[n:]]
    if env.owner_sets is not None:
        env.owner_sets[n:] = [o | {None} if op is None or op in o else o for o in env.owner_sets[n:]]


def _hint(v: Any) -> Any:
    return v.node.hint if isinstance(v, _SYM_TYPES) else v


class CallValues:
    """A trace's values at one call: each symbol at the value of the call's
    leaf it stands for (an input's size, stride, storage offset or base
    address, an int argument), any expression over them evaluated from those,
    an IR node by the IR's own arithmetic on constants (Ctx.transfer), a sympy
    one by xreplace. Never a node's hint: an allocation's or eager output's
    base, which no call supplies, raises KeyError."""

    def __init__(self, env: Any, symbols: Mapping[Any, int]) -> None:
        self.env = env
        self.ir = isinstance(env, _ir.Env)
        # an IR symbol's name or a sympy symbol -> its value
        self.symbols: dict[Any, int] = {}
        self._ctx = _ir.Ctx()
        self._at: dict[str, _ir.Node] = {}
        self._memo: dict[int, _ir.Node] = {}
        for s, value in symbols.items():
            self._set(s, value)

    def _set(self, s: Any, value: int) -> None:
        self.symbols[s] = int(value)
        if self.ir:
            self._at[s] = self._ctx.const(value)

    @classmethod
    def of_call(cls, env: Any, args: Sequence[Any], inputs: Sequence[_InputRec], int_inputs: Sequence[_IntInputRec]) -> CallValues:
        """At the call `args` (tensors, or release_args' metadata), through
        the input records, as _TapeLowering's leaves read it."""
        values = cls(env, {})
        for rec in inputs:
            a = args[rec.position]
            if isinstance(a, torch.Tensor):
                shape, stride, offset = a.shape, a.stride(), a.storage_offset()
                base = a.const_data_ptr() - offset * a.element_size()  # type: ignore[attr-defined]
            else:
                shape, stride, offset, _, base = a
            for d, (size, st) in enumerate(zip(rec.sizes, rec.strides)):
                values.bind(size, shape[d])
                values.bind(st, stride[d])
            values.bind(rec.offset, offset)
            values.bind(rec.root.sym, base)
        for rec in int_inputs:
            values.bind(rec.sym, args[rec.position])
        return values

    def bind(self, v: Any, value: int) -> None:
        # v's value where v is a symbol; a constant or an expression of one binds nothing
        k = _sym_key(v) if self.ir else _sym_expr(v)
        if self.ir and isinstance(k, _ir.Node) and k.op == "sym":
            self._set(k.args[0], value)
        elif not self.ir and isinstance(k, sympy.Symbol):
            self._set(k, value)

    def __call__(self, v: Any) -> Any:
        if not isinstance(v, _SYM_TYPES):
            return v
        if not self.ir:
            e = _sym_expr(v) if isinstance(v, torch.SymInt) else v.node.expr
            r = e.xreplace({s: sympy.Integer(self.symbols[s]) for s in e.free_symbols})
            return bool(r) if isinstance(v, torch.SymBool) else float(r) if isinstance(v, torch.SymFloat) else int(r)
        n = v.node.node
        r = self._ctx.transfer(n, self.env.ctx, self._at, self._memo)
        if r.op == "const":
            return r.args[0]
        if r.op in ("true", "false"):
            return r.op == "true"
        if r.op == "fconst":
            return struct.unpack("<d", r.args[0])[0]
        raise AssertionError(f"host_trace: {_ir.render(n)} is not a constant at the call's values")


def _sym_expr(v: Any) -> sympy.Expr:
    if isinstance(v, sympy.Basic):
        return v
    return v.node.expr if isinstance(v, torch.SymInt) else sympy.Integer(v)


def _sym_key(v: Any) -> Any:
    """v as a value to compare or lower with no sympy export: an int, an IR
    node (interned, so equal values are one node), or a sympy expression."""
    if not isinstance(v, torch.SymInt):
        return v
    node = v.node
    if isinstance(node, _ir.IRSymNode):
        n = node.node
        return n.args[0] if n.op == "const" else n
    return node.expr


def _free_view_strides(old_shape: Sequence[Any], old_stride: Sequence[Any], new_shape: list) -> list | None:
    """computeStride's view strides, deciding no symbolic size is 1: a dim
    joins the next one's chunk only where its stride continues it, and a chunk
    takes the view dims whose product is its size identically. Where a
    symbolic size is 1 they differ from eager's only on size-1 dims, so a
    view dim's stride selects eager's where its size is 1 (a GEMM reads it:
    cuBLAS's transpose and leading dimension). None where that does not
    decide the view."""
    old = [(n, s) for n, s in zip(old_shape, old_stride) if not _same(n, 1)]
    numel = functools.reduce(operator.mul, old_shape, 1)
    if not old or numel == 0:
        return None
    strides: list = [None] * len(new_shape)
    view_d, d = len(new_shape) - 1, len(old) - 1
    while d >= 0:
        size, base = old[d]
        while d > 0 and not (old[d - 1][1] != size * base):
            d -= 1
            size = size * old[d][0]
        d -= 1
        product = 1
        while view_d >= 0 and (not _same(product, size) or _same(new_shape[view_d], 1)):
            strides[view_d] = product * base
            product = product * new_shape[view_d]
            view_d -= 1
        if not _same(product, size):
            return None
    if view_d >= 0:
        return None
    # computeStride gives a size-1 view dim the stride of the dim after it times
    # that dim's size, the last one the source's last stride
    for d in reversed(range(len(new_shape))):
        one = old_stride[-1] if d == len(new_shape) - 1 else strides[d + 1] * new_shape[d + 1]
        if _same(new_shape[d], 1):
            strides[d] = one
        elif not isinstance(new_shape[d], int) and not _same(one, strides[d]):
            strides[d] = torch.cuda._host_trace.select(new_shape[d] == 1, one, strides[d])
    return strides


def _same(a: Any, b: Any) -> bool:
    # equal as expressions, at every value of their symbols
    return _sym_key(a) == _sym_key(b)


def _guard_each(terms: list) -> bool:
    # One guard per term rather than one conjunction: a True answer records
    # every term, a False answer only the first failing term's negation.
    for term in terms:
        if not bool(term):
            return False
    return True


def _either(a: Any, b: Any) -> Any:
    # a or b, each a bool or a SymBool, with no guard
    if a is True or b is True:
        return True
    return b if a is False else a if b is False else a | b


def _any_zero(sizes: Sequence[Any]) -> Any:
    # whether a size is 0, with no guard
    return functools.reduce(_either, [s == 0 for s in sizes], False)


def _storage_nbytes(sizes: list, strides: list, offset: Any, itemsize: int) -> Any:
    # at::detail::computeStorageNbytes, each zero-size test a guard
    extent: Any = 1
    for size, stride in zip(sizes, strides):
        if bool(size == 0):
            return 0
        extent = extent + stride * (size - 1)
    return (offset + extent) * itemsize


def _contiguous_strides(sizes: list) -> list:
    # c10::contiguous_strides: a zero size counts as 1 in the products (max(size, 1),
    # a size the declared domains put at >= 1 itself)
    strides: list = [1] * len(sizes)
    for d in range(len(sizes) - 2, -1, -1):
        strides[d] = strides[d + 1] * torch.sym_max(sizes[d + 1], 1)
    return strides


_CHANNELS_LAST_FORMATS = (torch.channels_last, torch.channels_last_3d)
_ALLOC_FORMATS = (
    None,
    torch.contiguous_format,
    torch.preserve_format,
    *_CHANNELS_LAST_FORMATS,
)


def _format_strides(sizes: list, mf: Any) -> list:
    # what empty(sizes, memory_format=mf) lays out: contiguous strides, or the
    # dim order (innermost first) of c10 get_channels_last_strides_2d / _3d
    if mf not in _CHANNELS_LAST_FORMATS:
        return _contiguous_strides(sizes)
    if mf is torch.channels_last:
        order = {4: [1, 3, 2, 0], 3: [0, 2, 1]}.get(len(sizes))
    else:
        order = {5: [1, 4, 3, 2, 0], 4: [0, 3, 2, 1]}.get(len(sizes))
    if order is None:
        rank = 4 if mf is torch.channels_last else 5
        raise RuntimeError(
            f"required rank {rank} tensor to use {str(mf).split('.')[-1]} format"
        )
    strides: list = [1] * len(sizes)
    current: Any = 1
    for d in order:
        strides[d] = current
        current = current * sizes[d]
    return strides


def _infer_dense_strides(sizes: list, strides: list) -> list:
    # at::infer_dense_strides (ExpandUtils.cpp), what empty_like gives a source
    # that is not dense: the dims sorted by stride with TensorIterator's
    # insertion sort (a zero stride does not move; equal strides put the
    # smaller size first), then dense strides in that order
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


def _empty_like_strides(sizes: list, strides: list, mf: Any, values: CallValues) -> list:
    # at::native::empty_like's strides of a source's, each branch a guard
    if mf not in (None, torch.preserve_format):
        return _format_strides(sizes, mf)
    # c10's is_contiguous: an empty tensor is
    if bool(functools.reduce(operator.mul, sizes, 1) == 0) or _guard_each(_dense_terms(sizes, strides, values)):
        return list(strides)
    return _infer_dense_strides(sizes, strides)


class _EagerMeta(TorchDispatchMode):
    """Over meta tensors: an op's Python meta function (meta_table) with
    eager's empty_like (_empty_like_strides, each branch a guard) where the
    meta's own decides no size-1 or density condition; an op without one, its
    meta kernel."""

    @classmethod
    def _should_skip_dynamo(cls):
        return False

    def __init__(self, values: CallValues) -> None:
        super().__init__()
        self.values = values

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        kwargs = kwargs or {}
        if func is aten.empty_like.default:
            src = args[0]
            strides = _empty_like_strides(list(src.shape), list(src.stride()), kwargs.get("memory_format"), self.values)
            return torch.empty_strided(src.shape, strides, dtype=kwargs.get("dtype") or src.dtype, device="meta")
        meta = torch._decomp.meta_table.get(func)
        if meta is None:
            return func(*args, **kwargs)
        with self:
            return meta(*args, **kwargs)


def _dense_terms(sizes: list, strides: list, values: CallValues) -> list:
    # dense in the permutation the strides have at the traced call (`values`): a choice the
    # terms, each a guard, decide (they order the dims of size above 1 as it does)
    order = sorted(
        range(len(sizes)), key=lambda d: (values(sizes[d]) < 2, values(strides[d]))
    )
    terms, require = [], 1
    for d in order:
        terms.append((sizes[d] == 1) | (strides[d] == require))
        require = require * sizes[d]
    return terms


# the in-place metadata ops without the inplace_view tag
_METADATA_OPS = (aten.resize_, aten.resize_as_, aten.set_)
# the ops with a traced ATen host, by the name torch._C._cuda_hostTraceAten takes
def _reduce(host: Callable[..., Any], x: torch.Tensor, dim: list[int] | None, keepdim: bool, dtype: torch.dtype | None = None) -> Any:
    if dtype is not None:
        raise NotImplementedError("a reduction with a dtype")
    return host(x, dim or [], keepdim)


def _std_var(x: torch.Tensor, dim: list[int] | None, correction: Any, keepdim: bool, take_sqrt: bool) -> Any:
    correction = 1 if correction is None else correction
    if type(correction) not in (int, float):
        raise NotImplementedError("a var of a non-number correction")
    return torch._C._cuda_hostTraceStdVar(x, dim or [], float(correction), keepdim, take_sqrt)


_ARANGE_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _arange(start: Any, end: Any, step: Any, dtype: torch.dtype | None, layout: Any, device: torch.device) -> Any:
    integral = all(isinstance(v, (int, torch.SymInt)) for v in (start, end, step))
    dtype = (torch.int64 if integral else torch.get_default_dtype()) if dtype is None else dtype
    concrete = all(type(v) in (int, float) for v in (start, end, step)) and dtype.is_floating_point
    if not (integral or concrete) or layout not in (None, torch.strided) or dtype not in _ARANGE_DTYPES:
        raise NotImplementedError("an arange of a symbolic floating or an integral dtype's floating bound or step, a layout or a dtype it has no kernel of")
    if integral:
        # torch._refs.arange's length of integer arguments
        sgn = bool(step > 0) - bool(step < 0)
        size = (end - start + step - sgn) // step
    else:
        # compute_arange_size's, in double; where arange_check_bounds raises, eager's error
        if not (math.isfinite(start) and math.isfinite(end) and ((step > 0 and end >= start) or (step < 0 and end <= start))):
            raise NotImplementedError("an arange eager raises on")
        size = math.ceil((float(end) - float(start)) / float(step))
    return torch._C._cuda_hostTraceArange(size, start, step, dtype, device)


def _max_pool2d(x: torch.Tensor, kernel_size: Any, stride: Any, padding: Any, dilation: Any, ceil_mode: bool) -> Any:
    k, s, p, d = ([v] if isinstance(v, int) else list(v) for v in (kernel_size, stride, padding, dilation))
    return torch._C._cuda_hostTraceMaxPool2d(x, k, s, p, d, ceil_mode)


def _repeat(x: torch.Tensor, repeats: list) -> torch.Tensor:
    # TensorShape.cpp's repeat, whose IntArrayRef takes no symbolic repeat: the
    # copy_ of x into an empty tensor as [r0, n0, r1, n1, ...], the memory order
    # of eager's unfolded view (unfold's int size and step would pin n)
    padded = [1] * (len(repeats) - x.dim()) + list(x.shape)
    target = [n * r for n, r in zip(padded, repeats)]
    out = torch.empty(target, dtype=x.dtype, device=x.device)
    if any(r == 0 for r in repeats):
        return out
    split = [d for n, r in zip(padded, repeats) for d in (r, n)]
    out.view(split).copy_(x.view([d for n in padded for d in (1, n)]).expand(split))
    return out


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    # ChanelShuffle.cpp's channel_shuffle (int sizes) and math_channel_shuffle
    if x.numel() == 0:
        return aten.alias.default(x)
    b, c = x.shape[0], x.shape[1]
    return x.view(b, groups, c // groups, -1).permute(0, 2, 1, 3).contiguous().reshape(x.shape)


def _soft_margin_loss_backward(grad_output: torch.Tensor, x: torch.Tensor, target: torch.Tensor, reduction: int) -> torch.Tensor:
    # Loss.cpp's; its mul_out resizes an empty({0}) grad_input as mul allocates
    n = x.numel()
    norm = (1.0 / n if n != 0 else math.inf) if reduction == 1 else 1.0
    z = torch.exp(-target * x)
    grad_input = torch.mul(target, z).mul_(-norm)
    z.add_(1)
    return grad_input.div_(z).mul_(grad_output)


_EYE = (aten.eye.default, aten.eye.m, aten.eye.out, aten.eye.m_out)


def _eye_args(func: OpOverload, args: tuple, kwargs: dict) -> tuple | None:
    """eye's (n, m, out, dtype, device): out None for a CUDA tensor's factory;
    None for another."""
    n = args[0]
    m = args[1] if func in (aten.eye.m, aten.eye.m_out) else n
    if not all(isinstance(v, (int, torch.SymInt)) and v >= 0 for v in (n, m)):
        return None
    if func in (aten.eye.out, aten.eye.m_out):
        out = kwargs["out"]
        return (n, m, out, out.dtype, out.device) if isinstance(out, _TracedTensor) else None
    device = kwargs.get("device")
    if device is None or torch.device(device).type != "cuda" or kwargs.get("layout") not in (None, torch.strided) or kwargs.get("pin_memory"):
        return None
    return n, m, None, kwargs.get("dtype") or torch.get_default_dtype(), device


# a CompositeExplicitAutograd mvlgamma (mvlgamma_ and the CUDA mvlgamma.out run it): its parts
_MVLGAMMA = (aten.mvlgamma.default, aten.mvlgamma_.default, aten.mvlgamma.out)


def _mvlgamma_dtype(x: torch.Tensor) -> torch.dtype:
    return x.dtype if x.is_floating_point() or x.is_complex() else torch.get_default_dtype()


def _mvlgamma(x: torch.Tensor, p: int) -> torch.Tensor:
    # UnaryOps.cpp's mvlgamma
    args = torch.arange(-p * 0.5 + 0.5, 0.5, 0.5, dtype=_mvlgamma_dtype(x), device=x.device)
    args = args.add(x.unsqueeze(-1))
    return args.lgamma_().sum(-1).add_(float(p * (p - 1)) * math.log(math.pi) * 0.25)


_TO_COPY_KWARGS = ("dtype", "layout", "device", "pin_memory", "non_blocking", "memory_format")


# the C++ sites (SymInt::guard_int's __FILE__) of an implicit int read: an
# int-API use of a symbolic tensor's sizes (their materialization) or of a
# symbolic Scalar. A guard_int an op's code writes is its own, deliberate
# guard (an instantiation's or a loop count's choice), not reported
_IMPLICIT_INT_READS = ("SymbolicShapeMeta.cpp", "Scalar.h")


def _note_int_read(site: str) -> None:
    # _host_trace_ir's INT_READ: C++ code (SymInt::guard_int) read a symbolic
    # value as an int at site; with the first ATen frame above it
    if site.split(":")[0] not in _IMPLICIT_INT_READS:
        return
    if (tr := current_trace()) is not None and tr.in_op:
        if (bt := getattr(torch._C, "_cuda_hostTraceBacktrace", None)) is not None:
            frames = [f for f in bt(1, 24).splitlines() if " at::" in f and "c10::" not in f.split(" at::")[0]]
            if frames:
                site = f"{site} < {frames[0].split(': ', 1)[-1].split(' + 0x')[0].strip()}"
        tr.int_reads.append(site)


_ir.INT_READ = _note_int_read


def _same_start(a: torch.Tensor, b: torch.Tensor) -> bool:
    # copy_impl's same-data test (torch/csrc/cuda/host_trace/Aten.cpp): one
    # root and one offset; distinct roots are distinct allocations (arguments
    # that overlap are a replay check)
    if not (isinstance(a, _TracedTensor) and isinstance(b, _TracedTensor)):
        return False
    return a._root is b._root and bool(a._sym_offset == b._sym_offset)


def _symbolic_resize(t: torch.Tensor, sizes: list, strides: list) -> bool:
    # at::impl::symbolic_resize's hook (torch/csrc/cuda/host_trace/Aten.cpp)
    tr = current_trace()
    return tr is not None and tr.resize_fresh(t, sizes, strides)


@functools.cache
def _kernel_less(func: OpOverload) -> bool:
    name = func.name()
    keys = ("CompositeExplicitAutograd", "CompositeExplicitAutogradNonFunctional")
    return func.namespace == "aten" and not _has_kernel(name, "CUDA") and any(_has_kernel(name, k) for k in keys)


@functools.cache
def _composite(func: OpOverload) -> bool:
    name = func.name()
    return not _has_kernel(name, "CUDA") and _has_kernel(name, "CompositeExplicitAutograd")


@functools.cache
def _scalar_overload(func: OpOverload, numbers: tuple[bool, ...]) -> OpOverload | None:
    """func's overload that takes a Scalar at each Tensor argument numbers marks."""
    scalar = str(torch.NumberType.get())
    want = [(a.name, scalar if i < len(numbers) and numbers[i] else str(a.type)) for i, a in enumerate(func._schema.arguments)]
    for name in func.overloadpacket.overloads():
        op = getattr(func.overloadpacket, name)
        if [(a.name, str(a.type)) for a in op._schema.arguments] == want:
            return op
    return None


@functools.cache
def _tensor_overload(func: OpOverload, scalars: tuple[bool, ...]) -> OpOverload | None:
    """func's overload that takes a Tensor at each Scalar argument scalars marks
    (remainder.Tensor of remainder.Scalar: the op its composite body calls
    with the Scalar wrapped)."""
    want = [(a.name, "Tensor" if i < len(scalars) and scalars[i] else str(a.type)) for i, a in enumerate(func._schema.arguments)]
    for name in func.overloadpacket.overloads():
        op = getattr(func.overloadpacket, name)
        if [(a.name, str(a.type)) for a in op._schema.arguments] == want:
            return op
    return None


@functools.cache
def _out_overload(func: OpOverload) -> tuple[OpOverload, str] | None:
    """func's out= overload of a CUDA kernel, and its out argument's name."""
    want = [(a.name, str(a.type)) for a in func._schema.arguments]
    for name in func.overloadpacket.overloads():
        op = getattr(func.overloadpacket, name)
        outs = [a.name for a in op._schema.arguments if a.is_out]
        rest = [(a.name, str(a.type)) for a in op._schema.arguments if not a.is_out]
        if len(outs) == 1 and rest == want and _has_kernel(op.name(), "CUDA"):
            return op, outs[0]
    return None


# an op's traced host (torch._C._cuda_hostTrace*) on its schema-ordered args
_TRACED_ATEN: dict[OpOverload, Callable[..., Any]] = {
    aten.max.default: lambda x: torch._C._cuda_hostTraceMaxAll(x, None),
    aten.min.default: lambda x: torch._C._cuda_hostTraceMinAll(x, None),
    aten.max.unary_out: lambda x, out: torch._C._cuda_hostTraceMaxAll(x, out),
    aten.min.unary_out: lambda x, out: torch._C._cuda_hostTraceMinAll(x, out),
    aten.max.dim: lambda x, dim, keepdim: torch._C._cuda_hostTraceMaxDim(x, dim, keepdim),
    aten.min.dim: lambda x, dim, keepdim: torch._C._cuda_hostTraceMinDim(x, dim, keepdim),
    aten.argmax.default: lambda x, dim, keepdim: torch._C._cuda_hostTraceArgmax(x, dim, keepdim),
    aten.argmin.default: lambda x, dim, keepdim: torch._C._cuda_hostTraceArgmin(x, dim, keepdim),
    aten.var.correction: lambda x, dim, correction, keepdim: _std_var(x, dim, correction, keepdim, False),
    aten.std.correction: lambda x, dim, correction, keepdim: _std_var(x, dim, correction, keepdim, True),
    aten._softmax.default: lambda x, dim, half_to_float: torch._C._cuda_hostTraceSoftmax(x, dim, half_to_float),
    aten._log_softmax.default: lambda x, dim, half_to_float: torch._C._cuda_hostTraceLogSoftmax(x, dim, half_to_float),
    aten.native_layer_norm.default: lambda x, shape, w, b, eps: torch._C._cuda_hostTraceLayerNorm(x, len(shape), w, b, eps),
    aten._fused_rms_norm.default: lambda x, shape, w, eps: torch._C._cuda_hostTraceRmsNorm(x, len(shape), w, eps),
    aten.index_select.default: lambda x, dim, index: torch._C._cuda_hostTraceIndexSelect(x, dim, index),
    aten.cat.default: lambda tensors, dim: torch._C._cuda_hostTraceCat(tensors, dim),
    aten.native_batch_norm.default: lambda x, w, b, mean, var, training, momentum, eps: torch._C._cuda_hostTraceBatchNorm(x, w, b, mean, var, training, eps),
    aten.arange.default: lambda end, dtype, layout, device, pin_memory: _arange(0, end, 1, dtype, layout, device),
    aten.arange.start: lambda start, end, dtype, layout, device, pin_memory: _arange(start, end, 1, dtype, layout, device),
    aten.arange.start_step: lambda start, end, step, dtype, layout, device, pin_memory: _arange(start, end, step, dtype, layout, device),
    aten.triu.default: lambda x, diagonal: torch._C._cuda_hostTraceTriu(x, diagonal),
    aten.nll_loss_forward.default: lambda x, target, weight, reduction, ignore_index: torch._C._cuda_hostTraceNllLoss(x, target, weight, reduction, ignore_index),
    aten.max_pool2d_with_indices.default: _max_pool2d,
    aten._adaptive_avg_pool2d.default: lambda x, output_size: torch._C._cuda_hostTraceAdaptiveAvgPool2d(x, list(output_size)),
}
# hosts that take a symbolic scalar
_SYM_SCALAR_HOSTS = {aten.arange.default, aten.arange.start, aten.arange.start_step}
# CompositeExplicitAutograd empty and zero_, of symbolic sizes too
_ZEROS = {aten.zeros.default, aten.zeros_like.default}
# int-signature bodies of symbolic sizes, as the factory they call
# a composite body's int-signature wrapper refused a symbolic size
_INT_BODY = object()
# a route not taken, or declined: a traced op of no returns gives None
_UNTRACED = object()

_NEW_FACTORIES = {aten.new_full.default: torch.full, aten.new_zeros.default: torch.zeros, aten.new_ones.default: torch.ones}


def _full(size: list, fill_value: Any, *, dtype: Any = None, layout: Any = None, device: Any = None, pin_memory: Any = None) -> torch.Tensor:
    # TensorFactories.cpp's full: empty, of the fill value's dtype where none is given (infer_full_options), filled
    dtype = utils.type_to_dtype(type(fill_value)) if dtype is None else dtype
    return torch.empty(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory).fill_(fill_value)


def _constant_pad_nd(x: torch.Tensor, pad: list, value: Any = 0) -> torch.Tensor:
    # PadNd.cpp's constant_pad_nd, each of its tests of a pad a guard
    torch._check(len(pad) % 2 == 0, lambda: f"Length of pad must be even but instead it equals {len(pad)}")
    l_inp, l_pad = x.dim(), len(pad) // 2
    l_diff = l_inp - l_pad
    torch._check(l_inp >= l_pad, lambda: "Length of pad should be no more than twice the number of dimensions of the input.")
    nonpositive, c_input = True, x
    for i in range(l_diff, l_inp):
        left, right = pad[2 * (l_inp - i - 1)], pad[2 * (l_inp - i - 1) + 1]
        if left < 0:
            c_input = c_input.narrow(i, -left, c_input.size(i) + left)
        elif left != 0:
            nonpositive = False
        if right < 0:
            c_input = c_input.narrow(i, 0, c_input.size(i) + right)
        elif right != 0:
            nonpositive = False
    if nonpositive:
        return c_input.clone()
    new_shape = list(x.shape[:l_diff])
    for i in range(l_pad):
        k = len(pad) - (i + 1) * 2
        new_dim = x.size(l_diff + i) + pad[k] + pad[k + 1]
        torch._check(new_dim >= 0, lambda: f"constant_pad_nd: dimension {l_diff + i} of the output would be negative")
        new_shape.append(new_dim)
    output = torch.empty(new_shape, dtype=x.dtype, device=x.device, memory_format=utils.suggest_memory_format(x))
    output.fill_(value)
    c_output = output
    for i in range(l_diff, l_inp):
        left, right = pad[2 * (l_inp - i - 1)], pad[2 * (l_inp - i - 1) + 1]
        if left > 0:
            c_output = c_output.narrow(i, left, c_output.size(i) - left)
        if right > 0:
            c_output = c_output.narrow(i, 0, c_output.size(i) - right)
    c_output.copy_(c_input)
    return output


# int-signature composite bodies (their sizes C10_AS_INTARRAYREF_SLOW) a call of
# a symbolic size runs as their Python transcriptions, each part traced as any op
_SYM_BODIES: dict[OpOverload, Callable[..., Any]] = {
    aten.constant_pad_nd.default: _constant_pad_nd,
    aten.full.default: _full,
    aten.ones.default: lambda size, **options: _full(size, 1.0, **options),
}


def _cpp_view(fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


def _ref_view(ref: Callable[..., Any]) -> Callable[..., Any]:
    return lambda fake_mode, func, *args, **kwargs: ref(*args, **kwargs)


# a view's metadata on a bare meta tensor, without FakeTensorMode's dispatch:
# the C++ kernels whose SymInt handling is exact, else the Python impl the fake
# itself runs for the op (the C++ slice and expand raise on symbolic sizes, and
# permute and view specialize them)
_META_VIEWS: dict[OpOverload, Callable[..., Any]] = {
    aten.view.default: fake_impls._view_meta,
    aten.t.default: _cpp_view,
    aten.transpose.int: _cpp_view,
    aten.unsqueeze.default: _cpp_view,
    aten.select.int: _cpp_view,
    aten.as_strided.default: _cpp_view,
    aten.slice.Tensor: fake_impls.slice_forward,
    aten.expand.default: _ref_view(torch._refs.expand),
    aten.permute.default: _ref_view(torch._refs.permute),
}


def _meta_addmm(self: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta: Any = 1, alpha: Any = 1) -> torch.Tensor:
    # the decomposition's layout and guards: mm's output, self expanded to it
    if not self.dtype == mat1.dtype == mat2.dtype:
        raise NotImplementedError("addmm of mixed dtypes")
    out = _meta_registrations.meta_mm(mat1, mat2)
    if beta != 0:
        torch._refs.expand(self, out.shape)
    return out


# an opaque call's output metadata from its meta function, as the fake kernel
# computes it (addmm's is a decomposition)
_META_OPAQUE: dict[OpOverload, Callable[..., Any]] = {
    aten.mm.default: _meta_registrations.meta_mm,
    aten.bmm.default: _meta_registrations.meta_bmm,
    aten.addmm.default: _meta_addmm,
    aten._scaled_dot_product_cudnn_attention.default: _meta_registrations.meta__scaled_dot_product_cudnn_attention,
    aten._scaled_dot_product_flash_attention.default: _meta_registrations.meta__scaled_dot_product_flash_attention,
    aten._scaled_dot_product_efficient_attention.default: _meta_registrations.meta__scaled_dot_product_efficient_attention,
}

_SDPA_OPS = frozenset(
    {
        aten._scaled_dot_product_cudnn_attention.default,
        aten._scaled_dot_product_flash_attention.default,
        aten._scaled_dot_product_efficient_attention.default,
    }
)

def _without_unwritten_lse(func: Any, args: tuple, kwargs: dict, out: Any) -> Any:
    # the fake kernel returns a log-sum-exp the CUDA kernel does not. A
    # metadata fix, not an attention knob: it holds for an eager step as for a
    # bound call, so it is always on
    if func is aten._scaled_dot_product_cudnn_attention.default:
        if not (args[4] if len(args) > 4 else kwargs["compute_log_sumexp"]):
            return (out[0], None, *out[2:])
    return out


# ops whose dropout seed and offset eager returns on the host outside capture
# and on the device under it (Note [Seed and Offset Device])
HOST_SEED_OFFSET = frozenset({aten._scaled_dot_product_efficient_attention.default})


def seed_offset_on_device(func: Any, outs: list, device: torch.device) -> list:
    """An eager call's outputs with a HOST_SEED_OFFSET op's host seed and
    offset copied to `device`, as the trace predicts them: its bound calls'
    kernels write them there, as under capture, and its backward reads either."""
    if func not in HOST_SEED_OFFSET:
        return outs
    return [o.to(device, non_blocking=True) if isinstance(o, torch.Tensor) and o.device.type == "cpu" else o for o in outs]

def _deterministic_fill(dtype: torch.dtype) -> float | int | bool:
    if dtype.is_floating_point or dtype.is_complex:
        return float("nan")
    return True if dtype == torch.bool else torch.iinfo(dtype).max


_ALLOC_OPS = {
    aten.empty.memory_format,
    aten.empty_strided.default,
    aten.empty_like.default,
    aten.new_empty.default,
    aten.new_empty_strided.default,
}


def _is_non_overlapping_and_dense(t: _TracedTensor, values: CallValues) -> bool:
    # c10 answers every other metadata query of a traced tensor from its
    # symbols, each condition a guard; this one is an operator call
    return _guard_each(_dense_terms(list(t.shape), t._sym_strides, values))


class _HostBufferStandIn(torch.Tensor):
    """A host buffer (a 0-dim CPU tensor the trace's host steps write) as an
    op's own CUDA host code sees it (_traced_redispatch): its dtype and
    device, no value; its storage raises on a read."""

    # pyrefly: ignore [bad-override]
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, buffer: torch.Tensor):
        t = torch.Tensor._make_wrapper_subclass(cls, (), (), dtype=buffer.dtype, device=buffer.device)
        torch._C._cuda_hostTraceSeal(t, True)
        return t

    @classmethod
    def __torch_dispatch__(cls, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        raise declined(f"{func} of a host buffer's stand-in: its value is each replay's")


class _SymNumberStandIn(torch.Tensor):
    """A symbolic number (a SymInt) where an op's schema takes a Tensor, as
    an op's own CUDA host code sees it (_traced_redispatch): a wrapped 0-dim
    int64 CPU tensor, as eager's arg parser passes a number, of no value (its
    storage raises on a read). A functor member the host code marks from it
    (host_scalar_member) is the number at each replay."""

    # pyrefly: ignore [bad-override]
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, value: torch.SymInt):
        t = torch.Tensor._make_wrapper_subclass(cls, (), (), dtype=torch.int64, device="cpu")
        # a wrapped number crossing into Python is this (torch::jit::toPyObject)
        t._wrapped_number = value
        torch._C._cuda_hostTraceSeal(t, True)
        torch._C._cuda_hostTraceWrapNumber(t)
        return t

    @classmethod
    def __torch_dispatch__(cls, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        raise declined(f"{func} of a symbolic number's stand-in: its value is each replay's")


@dataclass(frozen=True)
class _SymNumber:
    value: torch.SymInt


# ScalarTypes (c10 ScalarType order) a symbolic number fills a functor member
# of: its width, and the member's bits from the number
_SYM_NUMBER_MEMBERS: dict[int, tuple[int, Callable[[Any], Any]]] = {
    0: (1, lambda v: v),  # Byte
    1: (1, lambda v: v),  # Char
    3: (4, lambda v: v),  # Int
    4: (8, lambda v: v),  # Long
    6: (4, lambda v: _float_bits(v)),  # Float: static_cast<float>(v)
}


def _float_bits(v: torch.SymInt) -> Any:
    # static_cast<float>(v)'s bits as an int32: f32_div takes a numerator >= 0,
    # and a negative float's bits are its magnitude's with the sign bit set
    ht = torch.cuda._host_trace
    negative = v < 0
    bits = ht.f32_div(ht.select(negative, -v, v), 1)
    return ht.select(negative, bits - (1 << 31), bits)


def _sym_number_field(p: int, at: int, cls: str, value: torch.SymInt) -> tuple[tuple[int, int, int], Any]:
    # a functor member of class cls (LaunchLayout.h: 'A' + its ScalarType;
    # '0' + it for a reciprocal) holding the symbolic number
    member = _SYM_NUMBER_MEMBERS.get(ord(cls) - ord("A"))
    if member is None:
        raise declined(f"a symbolic number in a functor member of class {cls!r}, which no program row fills")
    width, bits = member
    return (p, at, width), bits(value)


def _fill_mask(x: torch.Tensor, index: Any, value: Any) -> torch.Tensor | None:
    """x[index] = value as eager runs it when that is masked_fill_
    (try_dispatch_masked_fill_): a Python number, and a boolean mask on x's
    device over x's dims from the full slices before it, then full slices or
    an Ellipsis. The mask, a size per x dim past it: eager reads the sizes as
    ints to check they match, which here is a guard on each pair, not a value
    pin. None for another index or value (Python's indexing, as before)."""
    if not isinstance(value, (bool, int, float)):
        return None
    parts = index if isinstance(index, tuple) else (index,)
    full = [isinstance(p, slice) and p == slice(None) for p in parts]
    at = next((i for i, p in enumerate(parts) if not full[i]), None)
    if at is None:
        return None
    mask = parts[at]
    rest = parts[at + 1 :]
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device != x.device:
        return None
    if any(p is not Ellipsis and not f for p, f in zip(rest, full[at + 1 :])) or sum(p is Ellipsis for p in rest) > 1:
        return None
    if at + mask.dim() > x.dim() or any(m != n for m, n in zip(mask.shape, x.shape[at:])):
        return None
    return mask.view(*mask.shape, *[1] * (x.dim() - at - mask.dim()))


class _TracedTensor(torch.Tensor):
    """A tensor the host sees during a trace: symbolic sizes, strides and
    storage offset, no storage; its address is its root's symbol plus the
    offset."""

    _root: _Root
    _sym_strides: list
    _sym_offset: Any
    # ("arg", i), or (k, i): output i of fn's k-th operator call
    _origin: tuple | None = None

    # pyrefly: ignore [bad-override]
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        root: _Root,
        sizes: list,
        strides: list,
        offset: Any,
        dtype: torch.dtype,
        device: torch.device,
    ):
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            sizes,
            strides,
            storage_offset=offset,
            dtype=dtype,
            device=device,
        )
        t._root = root
        t._sym_strides = list(strides)
        t._sym_offset = offset
        if (tr := current_trace()) is not None:
            if tr.tensors is not None:
                tr.tensors.add(t)
            if tr.sealed is not None:
                torch._C._cuda_hostTraceSeal(t, True)
                tr.sealed.append(t)
        return t

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"_TracedTensor({self._root.name}, {tuple(self.shape)}, {tuple(self._sym_strides)})"

    # Python indexing (applySlicing) returns self for a slice it statically
    # knows covers its dim, a shortcut that changes only the result's identity
    # (the JIT tracer's disable_slice_optimization skips it): its static reads
    # are no guards, so a symbolic slice is one view at every size
    def __getitem__(self, index: Any) -> Any:
        with _ir.static_reads(unguarded=True):
            return super().__getitem__(index)

    def __setitem__(self, index: Any, value: Any) -> None:
        mask = _fill_mask(self, index, value)
        if mask is not None:
            self.masked_fill_(mask, value)
            return
        with _ir.static_reads(unguarded=True):
            super().__setitem__(index, value)

    # A read of the value on the host has no place on the tape. item() and
    # the number conversions decline by name at _local_scalar_dense; these
    # reach the value without dispatching.
    def _host_read(self, what: str) -> Any:
        raise _declined(f"{what} of a traced tensor reads its value on the host")

    def __bool__(self) -> bool:
        return self._host_read("bool()")

    def is_nonzero(self) -> bool:
        return self._host_read("is_nonzero()")

    def tolist(self) -> Any:
        return self._host_read("tolist()")

    def numpy(self, *, force: bool = False) -> Any:
        return self._host_read("numpy()")

    def untyped_storage(self) -> Any:
        # the wrapper's storage has no address, and its size is not the input's
        raise _declined("the storage of a traced tensor is not traced")

    def __format__(self, format_spec: str) -> str:
        # Tensor.__format__ formats item() for a 0-dim tensor
        if self.dim() == 0:
            return self._host_read("format()")
        return super().__format__(format_spec)

    def data_ptr(self) -> Any:
        # TensorImpl::data() is null for a tensor with no elements
        if bool(self.numel() == 0):
            return 0
        return self._root.sym + self._sym_offset * self.element_size()

    const_data_ptr = data_ptr
    mutable_data_ptr = data_ptr

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        raise _declined(
            "a DLPack export of a traced tensor hands it to a library whose launches are not recorded"
        )

    def __dlpack_device__(self) -> Any:
        return self.__dlpack__()

    @classmethod
    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(cls, func, types, args: tuple = (), kwargs=None):
        if func is aten.is_non_overlapping_and_dense.default and (tr := current_trace()) is not None:
            return _is_non_overlapping_and_dense(args[0], tr.values)
        raise _declined(
            f"{func} on a traced tensor outside its trace (fn kept it past the trace: fn must return the tensors it makes)"
        )


class _TraceMode(TorchDispatchMode):
    # TorchDispatchMode wraps __torch_dispatch__ with torch._disable_dynamo
    # unless a subclass opts out, which imports torch._dynamo on first use
    @classmethod
    def _should_skip_dynamo(cls):
        return False

    def __init__(self, tr: _Trace) -> None:
        super().__init__()
        self.trace = tr

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        # an op's own static reads are guards, under indexing too (_TracedTensor.__getitem__)
        with _ir.static_reads(unguarded=False):
            return self._traced_dispatch(func, args, kwargs or {})

    def _traced_dispatch(self, func: OpOverload, args: tuple, kwargs: dict) -> Any:
        tr = self.trace
        if current_trace() is not tr:
            raise tr.decline(f"{func} on a thread other than the trace's")
        if tr.resolve is not None:
            args, kwargs = pytree.tree_map_only(torch.Tensor, tr.resolve, (args, kwargs))
        if func is aten.is_non_overlapping_and_dense.default:
            return _is_non_overlapping_and_dense(args[0], tr.values)
        if (accessor := _INT_READS.get(func)) is not None:
            tr.int_read(accessor, args[0])
        if tr.depth:
            return self._dispatch(func, args, kwargs)
        tr.depth += 1
        try:
            out = self._dispatch(func, args, kwargs)
        finally:
            tr.depth -= 1
        if func is aten.detach.default:
            # a factory's of a symbolic size (its binding's), which eager's of
            # an int does not call: no call of fn's to witness
            if isinstance(out, _TracedTensor) and isinstance(args[0], _TracedTensor):
                out._origin = args[0]._origin
            return out
        if func in _ALLOC_OPS or func.is_view:
            # no launch of fn's to witness, its uses are: Inductor's wrapper
            # allocates and reinterprets below the dispatcher at the warm-up
            return out
        k = len(tr.order)
        tr.order.append(_order_key(func, args, kwargs, out, tr.values))
        for i, o in enumerate(o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)):
            if isinstance(o, _TracedTensor) and o._origin is None:
                o._origin = (k, i)
        return out

    def _dispatch(self, func: OpOverload, args: tuple, kwargs: dict) -> Any:
        # below autograd (a torch._native override, run at the CUDA key) an op
        # whose kernel there is its CompositeImplicitAutograd one arrives whole;
        # eager runs that kernel's ops: aten.to.dtype is no view of its argument
        composite, key = DispatchKey.CompositeImplicitAutograd, torch._C._dispatch_key_for_device(self.trace.device.type)
        if _has_kernel(func.name(), composite) and not _has_kernel(func.name(), key):
            with self:
                return func._op_dk(composite, *args, **kwargs)
        if func in _ALLOC_OPS:
            out = self.trace.allocate(func, args, kwargs)
            if torch.are_deterministic_algorithms_enabled() and torch._C._get_deterministic_fill_uninitialized_memory():
                # eager's at::native::fill_empty_deterministic_: NaN for a floating or complex dtype, the max otherwise
                with self:
                    out.fill_(_deterministic_fill(out.dtype))
            return out
        if func is aten.zero_.default and self.trace.zero(args[0]):
            return args[0]
        # resize_output of an op's own fresh empty output outside a TensorIterator
        # (resize_reduction_result's): a new allocation, as eager's resize_ makes
        if func is aten.resize_.default and len(args) == 2 and kwargs.get("memory_format") in (None, torch.contiguous_format):
            if self.trace.resize_fresh(args[0], list(args[1]), []):
                return args[0]
        # below autograd, where this mode runs, _unsafe_view is view (matmul
        # folds a batched input with it)
        if func is aten._unsafe_view.default:
            func = aten.view.default
        # the in-place metadata ops (transpose_, as_strided_) are no views
        if func.is_view:
            out = self.trace.view(func, args, kwargs)
            if out is not None:
                return out
            with self:
                out = func.decompose(*args, **kwargs)
            if out is NotImplemented:
                raise self.trace.decline(f"{func} returned a copy, not a view")
            return out
        return self.trace.eager_call(func, args, kwargs)


@graphsafe_run_with_rng_state.py_impl(_TraceMode)
def _graphsafe_rng_traced(mode: _TraceMode, op: Any, *args: Any, rng_state: Any = None, **kwargs: Any) -> Any:
    # AOT's checkpointed RNG: op draws from rng_state, a graph input, as a
    # call outside it draws from the default generator
    tr = mode.trace
    if rng_state.device != tr.device:
        raise tr.decline(f"{op} draws from a generator on {rng_state.device}")
    tr.generator = rng_state
    try:
        with mode:
            return op(*args, **kwargs)
    finally:
        tr.generator = None


# 32800 slots: a 512 KiB chunk, about 250 KiB of it free above the ballast
@torch.cuda._host_trace.on_a_fresh_stack_chunk(32800)
def _symbolic_run(
    tr: _Trace,
    fn: Callable[..., Any],
    args: tuple,
    positions: list[int],
    int_positions: list[int],
) -> tuple[Any, list]:
    traced = list(args)
    if tr.trusted is not None:
        tr.bind_given(args)
    for i in positions:
        traced[i] = tr.input(i, args[i])
    for i in int_positions:
        traced[i] = tr.int_input(i, args[i])
    tr.values = CallValues.of_call(tr.shape_env, args, tr.inputs, tr.int_inputs)
    for i in positions:
        # an empty argument's trace launches nothing for it: a size is zero or not at replay as at the trace
        for size in traced[i].shape:
            bool(size != 0) if tr.values(size) else bool(size == 0)
    for i in positions:
        if traced[i]._origin is None:
            traced[i]._origin = ("arg", i)
    if tr.tensors is not None:
        tr.tensors.update(traced[i] for i in positions)
    _active.trace = tr
    try:
        # every symbol has its value, so no condition is decided size-obliviously
        with _cow_hold, _jiterator_hold, _TraceMode(tr), fx_config.patch(backed_size_oblivious=False):  # type: ignore[attr-defined]
            try:
                out = fn(*traced)
            except Exception as e:
                if _ir_census(tr):
                    raise declined(_ir_census(tr)) from e
                if tr.declined is not None and e is not tr.declined:
                    raise tr.declined from e
                raise
    finally:
        _active.trace = None
        _drop_tracebacks(tr.declined)
    # an operation the IR backend does not express, even one the host caught
    if _ir_census(tr):
        raise declined(_ir_census(tr))
    if tr.declined is not None:
        raise tr.declined
    return out, traced


def _ir_census(tr: _Trace) -> str:
    census = getattr(tr.shape_env, "census", None)
    if not census:
        return ""
    ops = ", ".join(f"{op} at {site}" for op, site in census)
    return f"the IR symbolic backend does not express {ops}"


def _output_records(
    out: Any, traced: list, positions: list[int]
) -> tuple[str, list[_OutputRec | _IntOutputRec]]:
    if out is None:
        return "none", []
    if isinstance(out, torch.Tensor):
        kind, outs = "tensor", (out,)
    elif type(out) in (list, tuple):
        kind, outs = type(out).__name__, tuple(out)
    else:
        raise declined(
            f"the traced call returned {type(out).__name__}, not a tensor, a tuple or list of tensors and ints, or None"
        )
    identities = {id(traced[i]): ("argument", i) for i in positions}
    records: list[_OutputRec | _IntOutputRec] = []
    for k, t in enumerate(outs):
        if isinstance(t, (int, torch.SymInt)) and not isinstance(t, bool):
            records.append(_IntOutputRec(f"out{k}", t))
            continue
        if not isinstance(t, _TracedTensor):
            raise declined(
                f"output {k} is {type(t).__name__}, not a tensor of the trace"
            )
        if t.requires_grad:
            raise declined(f"output {k} requires grad; a replay's outputs do not")
        if t._root.host and identities.get(id(t)) is None:
            raise declined(f"output {k} is a view of a pinned host argument")
        records.append(
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
    return kind, records


# cudaErrorIllegalState, cudaErrorNotPermitted and cudaErrorStreamCapture*:
# what an operation the thread-local capture refuses raises
_CAPTURE_ERRORS = frozenset({401, 800, *range(900, 909)})
_CAPTURE_UNJOINED = 904  # cudaErrorStreamCaptureUnjoined


def new_private_stream(device: torch.device | int | None = None, priority: int = 0) -> torch.cuda.ExternalStream:
    """A new non-blocking stream outside torch.cuda.Stream()'s pool, which
    hands its streams to any caller: another thread's work or allocations on
    one would land in a capture on it."""
    from cuda.bindings import runtime

    index = torch.cuda._utils._get_device_index(device, optional=True)
    with torch.cuda.device(index):
        handle = _check_cuda_bindings(runtime.cudaStreamCreateWithPriority(runtime.cudaStreamNonBlocking, priority))
    return torch.cuda.ExternalStream(int(handle), index)


_PRIVATE_STREAMS: dict[tuple[int, int], list[torch.cuda.ExternalStream]] = {}


@contextlib.contextmanager
def private_stream(device: torch.device | int | None = None, priority: int = 0) -> Iterator[torch.cuda.ExternalStream]:
    """A private stream (new_private_stream) no one else holds until exit,
    reused afterwards: the caching allocator keeps a stream's freed blocks
    for that stream, so streams are not made per call."""
    index = torch.cuda._utils._get_device_index(device, optional=True)
    free = _PRIVATE_STREAMS.setdefault((index, priority), [])
    try:
        stream = free.pop()
    except IndexError:
        stream = new_private_stream(index, priority)
    try:
        yield stream
    finally:
        free.append(stream)


@dataclass
class _Markers:
    """The event-record nodes (cudaEventRecordExternal) a trace adds to its
    capture, one per record on the record's stream (_Trace._mark). The
    capture gives a node the dependencies its stream has, so whatever ordered
    the streams (wait_stream, an event a library or a C++ op records and
    waits on) is an edge between markers: measured, nothing intercepted."""

    event: Any = None  # recorded at each node; the graph is never launched
    seqs: dict[int, int] = field(default_factory=dict)  # node -> its record's seq
    streams: dict[int, int] = field(default_factory=dict)  # seq -> logical stream, for a record on a side stream
    # seq -> the records on other streams it directly follows (an edge between their markers), set once the capture ends
    after: dict[int, tuple[int, ...]] = field(default_factory=dict)

    launches: list = field(default_factory=list)  # the trace's records (_Trace.launches), rolled back ones gone

    def read_edges(self, graph: int) -> None:
        """`after` from the capture's edges. A marker of a record the trace
        rolled back (a body or host that declined) passes its dependencies on."""
        from cuda.bindings import runtime

        if not self.streams:
            return  # one stream: its order is the tape's
        n = _check_cuda_bindings(runtime.cudaGraphGetEdges(graph, numEdges=0))[-1]
        src, dst = _check_cuda_bindings(runtime.cudaGraphGetEdges(graph, numEdges=n))[:2]
        into: dict[int, list[int]] = {}
        for a, b in zip(src, dst):
            into.setdefault(int(b), []).append(int(a))
        kept = {seq for seq, _ in self.launches}
        seq_node = {seq: node for node, seq in self.seqs.items() if seq in kept}
        for q, node in seq_node.items():
            mine, found, todo, seen = self.streams.get(q, 0), set(), list(into.get(node, ())), set()
            while todo:
                d = todo.pop()
                if d in seen:
                    continue
                seen.add(d)
                p = self.seqs.get(d)
                if p in kept:
                    if self.streams.get(p, 0) != mine:
                        found.add(p)
                elif p is not None:
                    todo += into.get(d, ())
            if found:
                self.after[q] = tuple(sorted(found))


def _capture_info(stream: torch.cuda.Stream) -> tuple[int, int, list[int]]:
    """The stream's capture status, capture id and dependency nodes."""
    from cuda.bindings import runtime

    status, capture, _, nodes, *_ = _check_cuda_bindings(runtime.cudaStreamGetCaptureInfo(stream.cuda_stream))
    return int(status), int(capture), [int(n) for n in nodes]


# thread-local, unlike capture_tape's relaxed capture: here the capture is the
# detector, so an operation it refuses must raise on this thread
@contextlib.contextmanager
def _capture(device: torch.device) -> Iterator[_Markers]:
    from cuda.bindings import runtime as _cuda_runtime

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    markers = _Markers()
    try:
        with private_stream(device) as stream, torch.cuda.stream(stream):
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                # a node of ours: capture_end warns of an empty graph, and a
                # filter for that would swap the process's warnings filters
                torch.cuda._sleep(0)
                yield markers
            except BaseException:
                with contextlib.suppress(Exception):
                    graph.capture_end()
                raise
            graph.capture_end()
        raw = graph.raw_cuda_graph()
        _, nodes = _check_cuda_bindings(_cuda_runtime.cudaGraphGetNodes(raw, numNodes=0))
        if nodes > 1 + len(markers.seqs):
            raise declined(f"the host enqueued {nodes - 1 - len(markers.seqs)} operations the trace does not record")
        markers.read_edges(raw)
    except torch.AcceleratorError as e:
        if getattr(e, "error_code", None) not in _CAPTURE_ERRORS:
            raise
        if e.error_code == _CAPTURE_UNJOINED:
            raise declined("work on a side stream is not joined to the trace's stream at its end (a wait_stream or an event wait)", side_stream=True) from e
        raise declined(
            "the host performed an operation a stream capture does not permit "
            f"({str(e).splitlines()[0]}); it would run at the trace and never in a replay"
        ) from e
    finally:
        del graph
        if markers.event is not None:
            _check_cuda_bindings(_cuda_runtime.cudaEventDestroy(markers.event))


# the cyclic collector, held off during a trace's or segment's capture: a
# collection on the capturing thread can finalize a CUDAGraph, which
# invalidates the capture. The hold spans only the captures
_gc_was_enabled = [False]


def _gc_off() -> None:
    _gc_was_enabled[0] = gc.isenabled()
    gc.disable()


def _gc_restore() -> None:
    if _gc_was_enabled[0]:
        gc.enable()


_gc_hold = ProcessHold(_gc_off, _gc_restore)


def _swap_hold(name: str, replacement: Callable) -> ProcessHold:
    original = getattr(torch._C, name)
    return ProcessHold(lambda: setattr(torch._C, name, replacement), lambda: setattr(torch._C, name, original))


_is_cow_tensor = torch._C._is_cow_tensor


def _traced_is_cow_tensor(t: torch.Tensor) -> bool:
    # torch._native's conditions ask of traced tensors, which the C++ check
    # rejects: an argument answers its input's state at the trace, anything
    # the trace allocated is not copy-on-write
    tr = current_trace()
    if tr is None or not isinstance(t, _TracedTensor):
        return _is_cow_tensor(t)
    rec = tr.arguments.get(id(t._root))
    return rec is not None and rec.cow


_cow_hold = _swap_hold("_is_cow_tensor", _traced_is_cow_tensor)

_jiterator_launch = torch._C._cuda_jiterator_compile_and_launch_kernel


def _traced_jiterator_launch(code: str, name: str, return_by_ref: bool, num_outputs: int, tensors: tuple, kwargs: dict | None = None) -> Any:
    tr = current_trace()
    if tr is None:
        return _jiterator_launch(code, name, return_by_ref, num_outputs, tensors, *(() if kwargs is None else (kwargs,)))
    return tr.jiterator(code, name, return_by_ref, num_outputs, tensors, kwargs)


_jiterator_hold = _swap_hold("_cuda_jiterator_compile_and_launch_kernel", _traced_jiterator_launch)


def _real_layout(t: torch.Tensor) -> tuple:
    if t.layout != torch.strided:
        return (t.layout,)
    return tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype, t.device


def _traced_layout(t: torch.Tensor, values: CallValues) -> tuple:
    # t's layout at the traced call
    if not isinstance(t, _TracedTensor):  # a host buffer
        return _real_layout(t)
    return tuple(map(values, t.shape)), tuple(map(values, t._sym_strides)), values(t._sym_offset), t.dtype, t.device


def _addressing(layout: tuple) -> tuple:
    # a size-1 dim's stride addresses nothing: kernels (flash, cuDNN) and
    # their fakes disagree on it freely
    if len(layout) == 1:
        return layout
    sizes, strides, *rest = layout
    return (sizes, tuple(None if n == 1 else s for n, s in zip(sizes, strides)), *rest)


def _layout_key(layout: tuple) -> tuple:
    # nor do an empty tensor's strides and offset
    if len(layout) > 1 and 0 in layout[0]:
        return layout[0], layout[3]
    return _addressing(layout)


def _call_key(func: Any, args: tuple, kwargs: dict, values: CallValues | None) -> tuple:
    # an operator call by its tensor arguments' layouts and int arguments, at
    # the traced call's `values` (None: a warm-up call's, of real tensors)
    layout = _real_layout if values is None else functools.partial(_traced_layout, values=values)
    key: list[Any] = [func]
    for a in pytree.tree_leaves((args, kwargs)):
        if isinstance(a, torch.Tensor):
            key.append(_layout_key(layout(a)))
        elif isinstance(a, (int, torch.SymInt, torch.SymBool)):
            key.append(a if values is None else values(a))
    return tuple(key)


def _order_key(func: Any, args: tuple, kwargs: dict, out: Any, values: CallValues | None) -> tuple:
    # a call of fn's by its key and its tensor outputs' layouts
    layout = _real_layout if values is None else functools.partial(_traced_layout, values=values)
    outs = tuple(_layout_key(layout(o)) for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor))
    return _call_key(func, args, kwargs, values), outs


def _order_repr(key: tuple | None) -> str:
    return "no call" if key is None else f"{key[0][0]}{key[0][1:]} returning {key[1]}"


class _Witness(TorchDispatchMode):
    """The warm-up's operator calls: per call key, in call order, the layouts
    of the tensors each returned."""

    @classmethod
    def _should_skip_dynamo(cls):
        return False

    def __init__(self, opaque: Sequence[OpaqueProvider] = (), learn: Callable[..., Any] | None = None) -> None:
        super().__init__()
        self.learn = learn
        self.learned: set[Any] = set()  # the keys learn ran at
        self.calls: dict[tuple, list[list[tuple]]] = {}
        self.order: list[tuple] = []
        # per call its tensor outputs, weakly; after the warm-up those it
        # kept or returned, a leaked traced tensor's twin (trace())
        self.outputs: list[list[Any]] = []
        self.opaque = opaque
        # inside a Python body the trace descends: its calls are witnessed, not fn's own
        self.depth = 0

    def _body(self, func: Any, args: tuple, kwargs: dict) -> Callable[..., Any] | None:
        # a Library.impl or custom_op body the trace runs on its traced tensors
        # (_traced_host) runs under the witness too, as its calls then see a
        # dispatch mode: first-use work they do under one happens here
        if func.namespace in ("aten", "_native") or torch.cuda._host_trace._TRACED_IMPLS.get(func) is not None:
            return None
        if (kernel := _python_kernel(func)) is None or _library_body_kind(func, kernel) is None:
            return None
        if any(p.accepts(func, args, kwargs) is None for p in self.opaque):
            return None
        return kernel

    def _learn(self, func: Any, args: tuple, kwargs: dict, out: Any) -> None:
        # a call the trace keys, learned at the call on its live inputs and
        # output (as a replay's missed bind learns, _eager_step): the trace then
        # binds it with no stand-ins (HostTraceReplay._learn's)
        provider = next((p for p in self.opaque if p.accepts(func, args, kwargs) is None), None)
        if provider is None:
            return
        rets = [out] if len(func._schema.returns) == 1 else list(out or ())
        fresh = [o for r, ret in zip(func._schema.returns, rets) if r.alias_info is None for o in pytree.tree_leaves(ret) if isinstance(o, torch.Tensor)]
        key = real_key(func, args, kwargs, fresh)
        if provider.bind(key) is None and provider.refusal(key) is None:
            self.learned.add(key)
            self.learn(provider, key, args, kwargs, [*(t for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)), *fresh])

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        kwargs = kwargs or {}
        if (body := self._body(func, args, kwargs)) is not None:
            self.depth += 1
            try:
                with self:
                    out = body(*args, **kwargs)
            finally:
                self.depth -= 1
        else:
            out = func(*args, **kwargs)
        if func in HOST_SEED_OFFSET:
            # the warm-up returns what a replay does
            out = tuple(seed_offset_on_device(func, list(out), args[0].device))
        if func is aten.detach.default:
            return out
        if self.learn is not None and not self.depth and not takes_generator(func):
            self._learn(func, args, kwargs, out)
        key = _call_key(func, args, kwargs, None)
        tensors = [o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)]
        self.calls.setdefault(key, []).append([_real_layout(o) for o in tensors])
        if func in _ALLOC_OPS or func.is_view or self.depth:
            return out
        self.order.append(_order_key(func, args, kwargs, out, None))
        self.outputs.append([weakref.ref(o) for o in tensors])
        return out


@graphsafe_run_with_rng_state.py_impl(_Witness)
def _graphsafe_rng_witnessed(mode: _Witness, op: Any, *args: Any, rng_state: Any = None, **kwargs: Any) -> Any:
    with mode:
        return _impl_graphsafe_rng(op, *args, rng_state=rng_state, **kwargs)


def _check_witness(tr: _Trace, witness: _Witness) -> None:
    """Each eager call's and traced host's fresh outputs, as the trace
    predicted them at the hints (an eager call's by its fake kernel), against
    what the operator returned at the warm-up. The k-th call of a key pairs
    with the warm-up's k-th; a key the warm-up called a different number of
    times (a lazy initialization) is not checked here, only at a replay.

    Then fn's own calls but its allocations and views, in order, by their
    int arguments and tensor layouts, against the warm-up's: fn decided each branch the same way at
    the hints, a symbol's hint-dodging reads (has_hint, a library's shape
    check) included, and dispatched to the same operators (which SDPA
    backend). Under trusted inputs Dynamo's guards stand for this."""
    keyed = []
    values = tr.values
    hosts = {id(call) for _, call in tr.aten_calls}
    for _, call in heapq.merge(tr.launches, tr.aten_calls, key=operator.itemgetter(0)):
        if isinstance(call, EagerCall) and not isinstance(call.target, tuple):
            key = _call_key(call.target, call.args, call.kwargs, values)
            keyed.append((key, call))
    counts = collections.Counter(key for key, _ in keyed)
    seen: collections.Counter = collections.Counter()
    inexact: set[int] = set()  # roots whose size-1 strides are the fake kernel's, not eager's
    for key, call in keyed:
        real = witness.calls.get(key, [])
        k, seen[key] = seen[key], seen[key] + 1
        if len(real) != counts[key]:
            continue
        got = real[k]
        what = "its traced host" if id(call) in hosts else "its fake kernel"
        if len(got) != len(call.outputs):
            raise declined(
                f"{call.name} returned {len(got)} tensors at the warm-up; {what} {len(call.outputs)}"
            )
        arguments = {id(a) for a in pytree.tree_leaves((call.args, call.kwargs))}
        for i, o in enumerate(call.outputs):
            if id(o) in arguments:
                continue
            want = _traced_layout(o, values)
            if got[i] != want and _addressing(got[i]) == _addressing(want) and id(call) not in hosts:
                # predict the op's own size-1 strides, which a replay checks: eager's where the size is 1
                # (a select over a symbolic size), the trace's elsewhere. With no rule they stay the fake's
                if (strides := _size_one_strides(tr, call, i, want, got[i])) is None:
                    inexact.add(id(o._root))
                else:
                    o._sym_strides = strides
            elif got[i] != want and _addressing(got[i]) != _addressing(want):
                e = declined(
                    f"{call.name} output {i} is (sizes, strides, storage offset, dtype, device) "
                    f"{got[i]} at the warm-up; {what} predicted {want}"
                )
                e.meta_op = None if id(call) in hosts else call.target
                raise e
    # an opaque call keyed on such a stride would key another than eager's: it runs eagerly, unbound
    for k, (seq, rec) in enumerate(tr.launches):
        leaves = pytree.tree_leaves((rec.args, rec.kwargs)) if isinstance(rec, OpaqueCall) else ()
        if any(isinstance(t, _TracedTensor) and id(t._root) in inexact for t in leaves):
            why = f"{rec.target} reads an eager output's size-1 stride, eager's at the warm-up and not its fake kernel's"
            tr.launches[k] = seq, EagerCall(rec.target, rec.args, rec.kwargs, rec.outputs, why, rec.generator, rec.state)
    for k, (got, want) in enumerate(itertools.zip_longest(witness.order, tr.order) if tr.trusted is None else ()):
        if got != want:
            e = declined(f"fn's call {k} is {_order_repr(want)} in the trace; at the warm-up {_order_repr(got)}")
            e.witness = True
            raise e


def _size_one_strides(tr: _Trace, call: EagerCall, i: int, want: tuple, got: tuple) -> list | None:
    """Output i of an eager call whose fake kernel predicted another stride
    than eager's on a dim of size 1 at the call: its strides with eager's
    rule there, select(size == 1, rule, the fake's), the rule
    _size_one_rule's, evaluated at the call to be eager's. None where a dim
    has none."""
    o = call.outputs[i]
    strides = list(o._sym_strides)
    for d, (n, z, r, s) in enumerate(zip(want[0], o.shape, got[1], o._sym_strides)):
        if n != 1 or r == want[1][d]:
            continue
        rule = _size_one_rule(tr, call, i, d, z)
        if rule is None or tr.values(rule) != r:
            return None
        strides[d] = rule if type(z) is int else torch.cuda._host_trace.select(z == 1, rule, s)
    return strides


def _size_one_rule(tr: _Trace, call: EagerCall, i: int, d: int, z: Any) -> Any:
    """Output i's stride d at size z = 1: the op's meta at the operands with
    each dim of size z at 1, under eager's empty_like (_EagerMeta). Each
    condition it guards, which held at the call, is guarded where z is 1
    (Or(z != 1, g)), or always where z is the constant 1. None where it
    raises or returns another shape, or for a symbolic z off the IR backend
    (Inductor's trusted traces), which records each guard as the meta runs."""
    env = tr.shape_env
    ir = isinstance(env, _ir.Env)
    if not isinstance(call.target, OpOverload) or not (ir or type(z) is int):
        return None

    def at_one(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, _TracedTensor):
            return t.to("meta")
        sizes = [1 if _same(n, z) else n for n in t.shape]
        return torch.empty(0, dtype=t.dtype, device="meta").as_strided(sizes, t._sym_strides, t._sym_offset)

    mark = len(env.records) if ir else 0
    args = pytree.tree_map_only(torch.SymInt, lambda v: 1 if _same(v, z) else v, (call.args, call.kwargs))
    try:
        with unset_fake_temporarily():
            with torch._C._DisablePythonDispatcher():
                twins = pytree.tree_map_only(torch.Tensor, at_one, args)
            with _EagerMeta(tr.values):
                out = _without_unwritten_lse(call.target, *twins, call.target(*twins[0], **twins[1]))
    except Exception:
        if ir:
            env.forget(mark)
        return None
    conditions = env.records[mark:] if ir else []
    if ir:
        env.forget(mark)
    outs = [t for t in pytree.tree_leaves(out) if isinstance(t, torch.Tensor)]
    if len(outs) != len(call.outputs) or outs[i].dim() <= d or not _same(outs[i].shape[d], 1):
        return None
    if type(z) is int:
        # a constant 1: what the meta decided holds at every call
        for g, _ in conditions:
            env.guard_bool(g)
        return outs[i].stride()[d]
    ctx, at_size = env.ctx, _sym_key(z)
    other = ctx.ne(at_size, ctx.const(1))
    for g, _ in conditions:
        env.guard_bool(ctx.or_(other, g))
    return outs[i].stride()[d]


def _metadata(t: torch.Tensor) -> tuple:
    base = t.const_data_ptr() - t.storage_offset() * t.element_size()  # type: ignore[attr-defined]
    return tuple(t.shape), t.stride(), t.storage_offset(), t.dtype, base


def trace(
    fn: Callable[..., Any],
    args: tuple,
    *,
    warm_up: bool = True,
    trusted: TrustedInputs | None = None,
    opaque: Sequence[OpaqueProvider] = (),
    static_shapes: Collection[int] = (),
    check_escapes: bool = False,
    eager_ops: Mapping[int, str] | None = None,
    declined_compiles: set[Any] | None = None,
    learn: Callable[..., Any] | None = None,
    boxed: tuple[Callable[[list[Any]], Any], list[Any]] | None = None,
) -> Tape:
    """Trace one call fn(*args). Tensor arguments become traced tensors and
    int arguments symbols; every other argument is a constant of the tape.
    With `learn`, learn(provider, key, args, kwargs, operands) runs at the
    warm-up's call of each op a provider accepts whose key neither binds nor
    is refused, on the call's live tensors (its inputs, then its fresh outputs).
    With `warm_up` fn first runs once on the real arguments, so what it
    initializes on first use happens outside the trace; a warm-up that
    changes a tensor argument's metadata declines, an eager call whose fake
    kernel predicted other output metadata than its operator returned at the
    warm-up declines (_check_witness), and an exception the trace raises
    after a warm-up that did not is a decline.

    Host code must be a function of its arguments: as with torch.cuda.graph,
    Python state it reads is fixed at trace time, including torch's global
    state (grad and inference mode, autocast, the default device), argument
    identity (`x is y`), requires_grad and an argument's Python attributes;
    Dynamo guards these for Inductor. Only arguments of type torch.Tensor are
    traced, and the host sees each as a subclass (`type(x) is torch.Tensor`
    is False). With `trusted`, the caller vouches for the arguments
    (TrustedInputs), which may also be nn.Parameters. An eager call one of
    the `opaque` providers accepts is an OpaqueCall. The tensor arguments at
    `static_shapes` (a module's parameters and buffers, as Dynamo's
    force_parameter_static_shapes) have static sizes, strides and storage
    offset: guarded to the trace's values, constants in the trace. With
    `check_escapes` a traced tensor fn keeps past the trace, one in a cycle too, declines. The
    top-level ops at the indices `eager_ops` maps (in Tape.ops) are eager
    calls, for the reason it maps them to. A CuTe compile that raises is a
    retried decline unless `declined_compiles` (the caller's record, which it
    joins) has its function. With `boxed` (fn's boxed form and the call's
    list), the warm-up runs boxed_fn on the list, which it consumes, and
    `args` are the stand-ins of the inputs it drops (their metadata and
    addresses)."""
    positions = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
    int_positions = [i for i, a in enumerate(args) if type(a) is int]
    for i, a in enumerate(args):
        if i not in positions and any(
            isinstance(x, torch.Tensor) for x in pytree.tree_leaves(a)
        ):
            raise declined(
                f"arg{i} holds a tensor inside a {type(a).__name__}; only top-level tensor arguments are traced"
            )
    if not positions:
        raise declined("the call has no tensor arguments")
    cuda = next((args[i].device for i in positions if args[i].is_cuda), None)
    device = cuda or args[positions[0]].device
    exact = (torch.Tensor,) if trusted is None else (torch.Tensor, torch.nn.Parameter)
    for i in positions:
        a = args[i]
        if isinstance(a, _TracedTensor):
            raise declined(f"arg{i} is a traced tensor of another trace")
        if type(a) not in exact:
            raise declined(
                f"arg{i} is a {type(a).__name__}; only torch.Tensor arguments are traced"
            )
        # a CPU tensor: pinned, a memcpy's host side; unpinned (a pageable copy_ is synchronous), its sizes alone
        host = cuda is not None and trusted is None and a.device.type == "cpu"
        if not (a.is_cuda and a.device == device) and not host:
            raise declined(
                f"arg{i} is on {a.device}; only CUDA tensors on one device and CPU tensors beside them are traced"
            )
        if a.is_neg() or a.is_conj():
            raise declined(f"arg{i} is a negative or conjugate view")
        if a.requires_grad and torch.is_grad_enabled():
            raise declined(f"arg{i} requires grad under grad mode; a replay records no autograd graph")
    if current_trace() is not None:
        raise declined("a trace is already in progress on this thread")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise declined("the current stream is capturing")
        from torch.cuda._host_trace_cute import intercepting as cute_intercepting
        from torch.cuda._host_trace_triton_launch import intercepting

        result, witness = None, None
        if warm_up:
            before = [_metadata(args[i]) for i in positions]
            with _Witness(opaque, learn) as witness:
                result = fn(*args) if boxed is None else boxed[0](boxed[1])
            boxed = None
            # not at the mode's exit: a higher-order op's impl enters it again
            witness.outputs = [[r() for r in refs] for refs in witness.outputs]
            for i, was in zip(positions, before):
                if _metadata(args[i]) != was:
                    e = declined(
                        f"the warm-up changed the metadata of arg{i} (a resize_, set_ or out= "
                        "inside the call); a trace would describe the changed call"
                    )
                    e.warm_up_ran, e.warm_up_result = True, result
                    raise e
        state = torch._C._host_trace_global_state()
        try:
            with _gc_hold, _capture(device) as markers, intercepting(), cute_intercepting():
                tr = _Trace(device, trusted, opaque, static_shapes, check_escapes, eager_ops, declined_compiles=declined_compiles, markers=markers)
                if witness is not None:
                    tr.warm_learned = witness.learned
                out, traced = _symbolic_run(tr, fn, args, positions, int_positions)
                result_kind, outputs = _output_records(out, traced, positions)
                if witness is not None:
                    _check_witness(tr, witness)
                if torch._C._host_trace_global_state() != state:
                    raise declined("the trace changed the global state")
                tape = Tape(tr, args, outputs, result_kind, result)
                if check_escapes:
                    # the tape keeps copies, so a tensor of the trace alive past it is one fn kept
                    _copy_tensors(tape)
                    tensors, trusted = tr.tensors, tr.trusted
                    del tr, out, traced
                    if kept := list(tensors):
                        holders = [r for r in gc.get_referrers(*kept) if r is not kept]
                        what = f"a {type(holders[0]).__name__}" if holders else "a C++ owner"
                        del holders
                        stale = _swap_twins(tensors, trusted, kept, witness, args)
                        del kept
                        raise declined(
                            f"fn kept a traced tensor past the trace, in {what}; fn must return the tensors "
                            f"it makes and store none{' (it stays a traced tensor)' if stale else ''}"
                        )
        except Declined as e:
            e.warm_up_ran, e.warm_up_result = warm_up, result
            raise
        except Exception as e:
            # the warm-up ran the call: an error now is the trace's
            if not warm_up or torch.cuda._host_trace.raise_unexpected:
                raise
            d = declined(f"the trace raised {type(e).__name__}: {e}")
            d.warm_up_ran, d.warm_up_result = True, result
            try:
                raise d from e
            finally:
                del d
    return tape


def _cuda_address(q: int) -> bool:
    from cuda.bindings import driver

    attribute = driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    return driver.cuPointerGetAttribute(attribute, q)[0] == driver.CUresult.CUDA_SUCCESS


def _copy_tensors(tape: Tape) -> None:
    # each traced tensor of the tape's records becomes a copy outside the trace
    copies: dict[int, tuple[_TracedTensor, _TracedTensor]] = {}

    def copy_of(t: _TracedTensor) -> _TracedTensor:
        if id(t) not in copies:
            copies[id(t)] = t, _TracedTensor(t._root, list(t.shape), t._sym_strides, t._sym_offset, t.dtype, t.device)
        return copies[id(t)][1]

    def copied(tree: Any) -> Any:
        return pytree.tree_map_only(_TracedTensor, copy_of, tree)

    for k, (seq, rec) in enumerate(tape.launches):
        if isinstance(rec, EagerCall):
            args, kwargs, outputs = copied((rec.args, rec.kwargs, rec.outputs))
            tape.launches[k] = seq, replace(rec, args=args, kwargs=kwargs, outputs=outputs)
    for op in tape.ops:
        op.call, op.outputs = copied((op.call, op.outputs))
    for k, site in enumerate(tape.sites):
        tape.sites[k] = replace(site, operands=copied(site.operands), scratch=copied(site.scratch))
    tape.eager_outputs = copied(tape.eager_outputs)


def _swap_twins(tensors: weakref.WeakSet, trusted: TrustedInputs | None, kept: list, witness: _Witness | None, args: tuple) -> int:
    """Each traced tensor fn kept becomes in place a view of its twin, the
    warm-up's tensor at its place (the argument, or the same call's output),
    which the warm-up left there and the trace replaced: fn's state is as
    eager leaves it. The number of traced tensors without one."""
    # the WeakSet's own reference, dropped by identity: its discard compares with ==
    refs = {id(r()): r for r in tensors.data}
    stale = 0
    for t in kept:
        twin = None
        if t._origin is not None and t._origin[0] == "arg":
            twin = args[t._origin[1]]
        elif t._origin is not None and witness is not None and trusted is None:
            # the warm-up's calls pair with the trace's by position once the order is checked
            k, i = t._origin
            twin = witness.outputs[k][i] if k < len(witness.outputs) and i < len(witness.outputs[k]) else None
        tensors.data.discard(refs.pop(id(t)))
        if twin is None or weakref.getweakrefcount(t) or t._use_count() != 1:
            stale += 1
            continue
        torch.utils.swap_tensors(t, twin.view_as(twin))
    return stale


def _constant(a: Any) -> Any:
    # compared by value and type (2, 2.0 and True differ); a list, a tuple and
    # a torch.Size of the same elements are the same constant; a float by its
    # bits (0.0 and -0.0 differ, two nans with the same bits do not); a tensor,
    # which trace() declines, by its type alone, so a contract holds no tensor
    if isinstance(a, (list, tuple)):
        return tuple(_constant(x) for x in a)
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return (type(a),)
    if isinstance(a, float):
        return (float, struct.pack("<d", a))
    if isinstance(a, complex):
        return (complex, struct.pack("<dd", a.real, a.imag))
    return (type(a), a)


def argument_contract(args: tuple, global_state: bool) -> tuple:
    """What a call must match before any GPU work, as trace() classifies its
    arguments: per tensor its exact type, dtype, device, whether it is
    pinned (a CPU tensor), rank, layout,
    neg/conj and requires_grad bits and whether autograd records it (under
    grad mode, which trace() declines), per int its kind, every other argument
    by value; then with `global_state` the global state a replay does not
    read (grad and inference mode, autocast, the cuBLAS, cuDNN and SDPA
    flags, ...)."""
    kinds: list[Any] = []
    grad = torch.is_grad_enabled()
    for a in args:
        if isinstance(a, torch.Tensor):
            rg = a.requires_grad
            pinned = a.device.type == "cpu" and a.is_pinned()
            kinds.append((type(a), a.dtype, a.device, pinned, a.dim(), a.layout, a.is_neg(), a.is_conj(), rg, rg and grad))
        elif type(a) is int:
            kinds.append(int)
        else:
            kinds.append(_constant(a))
    return (tuple(kinds), torch._C._host_trace_global_state() if global_state else ())


class Tape:
    """What one traced call did, in terms of the trace's symbols."""

    def __init__(
        self,
        tr: _Trace,
        args: tuple,
        outputs: list[_OutputRec | _IntOutputRec],
        result_kind: str,
        warm_up_result: Any,
    ) -> None:
        self.shape_env = tr.shape_env
        self.device = tr.device
        # the traced call's arguments, until release_args
        self.args = args
        # the warm-up call's return value (None without one): the warm-up was
        # the call, so a consumer returns it once and drops it
        self.warm_up_result = warm_up_result
        # the argument contract, as at the end of the trace; trusted inputs'
        # kinds and the global state are the caller's to vouch for
        self.contract = argument_contract(args, True) if tr.trusted is None else ()
        self.inputs = tr.inputs
        self.int_inputs = tr.int_inputs
        self.allocs = tr.allocs
        self.launches = tr.launches
        # per launch on a side stream (forked from the trace's in its capture)
        # its logical stream, 0 the trace's; per launch, the launches on other
        # streams it directly follows, as their seqs (filled as the trace's
        # capture ends); per logical stream past 0, its priority
        kept = {seq for seq, _ in tr.launches}
        self.streams = {seq: k for seq, k in tr.stream_of.items() if seq in kept}
        self.after = tr.after
        self.priorities = tuple(s.priority for s in tr.side_streams)
        self.sites = tr.sites
        self.eager_outputs = tr.eager_outputs
        # eager's checks of the call's values every call must meet (_Trace.check)
        self.checks = tr.checks
        self.outputs = outputs
        self.result_kind = result_kind  # "tensor", "tuple", "list" or "none"
        # how many conditions the host branched on (`guards`)
        env = tr.shape_env
        self.guard_count = len(env.records if isinstance(env, _ir.Env) else env.guards)
        # per guard the op that owns it (OpRec.guards), or None: graph-level
        self.owners = list(tr.shape_env.owners)
        # with shared_op_guards on, per guard every op that branched on it, plus None
        # where it is graph-level; owners[i] is the one op, or None
        sets = tr.shape_env.owner_sets
        self.owner_sets = None if sets is None else [frozenset(o) for o in sets]
        self.ops = tr.ops
        held = {g for op in self.ops for g in op.held or ()}
        self.unconverted = tuple(tr.unconverted)
        for op, guards in zip(self.ops, _owned(self.owners, self.owner_sets, len(self.ops))):
            op.guards = tuple(g for g in guards if g not in held)
        _hint_audit.freeze(env, self.guard_count)
        # (i, j): arguments i < j, disjoint at the trace, where a step not run
        # eagerly writes one and reads the other. A call where they overlap
        # runs eagerly; trusted inputs' aliasing is the caller's (none)
        self.argument_pairs = tuple(sorted(tr.argument_pairs)) if tr.trusted is None else ()
        # per op of a twin (_host_trace_replay._eager_sites), its refused key's
        # guards in the tape's symbols, which select it in place of op.guards
        self.twin_guards: dict[int, tuple[Any, ...]] = {}

    @property
    def graph(self) -> list[int]:
        """The guards no op owns: the variant's validity, not a selector's."""
        owned = {i for op in self.ops for i in op.guards}
        return [i for i in range(self.guard_count) if i not in owned]

    @functools.cached_property
    def guards(self) -> list[sympy.Basic]:
        """Every condition the host branched on, in program order, as sympy
        (an IR trace's exported on first read); under trusted inputs, only its
        size-based dispatch decisions."""
        return [g.expr for g in self.shape_env.guards[: self.guard_count]]

    def release_args(self) -> None:
        """Keep only each tensor argument's metadata (sizes, strides, storage
        offset, dtype, base address), so the tape keeps no tensor alive."""
        self.args = tuple(
            _metadata(a) if isinstance(a, torch.Tensor) else a for a in self.args
        )


def _released_arg(meta: tuple, device: torch.device) -> torch.Tensor:
    # a tensor over no memory of its own, at a released argument's metadata and address
    shape, stride, offset, dtype, base = meta
    nbytes = (offset + sum((n - 1) * s for n, s in zip(shape, stride)) + 1) * dtype.itemsize
    storage = torch._C._construct_storage_from_data_pointer(base, device, nbytes)
    return torch.empty(0, dtype=dtype, device=device).set_(storage, offset, shape, stride)


def _guarded_symbols(tape: Tape) -> set[str]:
    # the names of the symbols the tape's guards read, an IR trace's with no export
    env = tape.shape_env
    if not isinstance(env, _ir.Env):
        return {s.name for s in free_symbols(tape.guards)}
    names: set[str] = set()
    todo, seen = [g for g, _ in env.records[: tape.guard_count]], set()
    while todo:
        n = todo.pop()
        if n.id not in seen:
            seen.add(n.id)
            if n.op in ("sym", "fsym"):
                names.add(n.args[0])
            todo.extend(_ir.children(n))
    return names


def bind_opaque(
    tape: Tape, calls: set[int], learn: Callable[..., object] | None = None, keep_guards: bool = False, outputs: Any = None
) -> tuple[Tape, dict[int, KeyedSite]] | None:
    """The tape trace() records at the tape's call where the OpaqueCalls of
    ids `calls` whose keys bind now, after learn(op, provider, (spec, tensor
    positions), key, real) for each that draws no RNG and does not, had bound then
    (their KeyedSites, their fresh outputs allocations), and each such call's
    site by the call's id;
    the tape's args are the traced call's, or stand-ins at their released
    metadata. None where that trace differs beyond those calls: a key it
    refuses, a guard the tape lacks, a zero_ of a fresh output (a Memset), a
    fresh output at a symbolic storage offset, or an address not its root's
    symbol plus an offset. With `keep_guards` a guard the bindings add (a
    size's != 0, say) is kept as a graph-level guard instead. `real` is the
    live tensors a learn may read for the call's operands, by operand
    position: the tape's arguments and its call's `outputs` (the warm-up's
    result) that an operand is exactly; an output only where every input is
    one and no later step reaches an operand, so it holds the op's result."""
    with _hint_audit.reopened(tape.shape_env):
        bound = _bind_opaque(tape, calls, learn, keep_guards, outputs)
        if bound is not None:
            _hint_audit.freeze(tape.shape_env, bound[0].guard_count)
        return bound


def _bind_opaque(
    tape: Tape, calls: set[int], learn: Callable[..., object] | None, keep_guards: bool, outputs: Any
) -> tuple[Tape, dict[int, KeyedSite]] | None:
    live: list[tuple[Any, torch.Tensor]] = [(r, tape.args[r.position]) for r in tape.inputs if isinstance(tape.args[r.position], torch.Tensor)]
    outs = () if outputs is None else (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    live += [(o, t) for o, t in zip(tape.outputs, outs) if isinstance(o, _OutputRec) and isinstance(t, torch.Tensor)]

    def layout(sizes: Any, strides: Any, offset: Any) -> tuple:
        return (len(sizes), *map(_sym_expr, sizes), *map(_sym_expr, strides), _sym_expr(offset))

    def real(t: _TracedTensor) -> torch.Tensor | None:
        at = layout(t.shape, t._sym_strides, t._sym_offset)
        return next((x for r, x in live if r.root is t._root and layout(r.sizes, r.strides, r.offset) == at), None)

    # each root's last launch
    last: dict[int, int] = {}
    for i, (_, rec) in enumerate(tape.launches):
        if isinstance(rec, EagerCall):
            last.update((id(t._root), i) for t in pytree.tree_leaves((rec.args, rec.kwargs, rec.outputs)) if isinstance(t, _TracedTensor))
        else:
            last.update((id(r), i) for r in rec.roots)
    env = tape.shape_env
    tr = _Trace(tape.device)
    tr.shape_env = env
    tr.values = CallValues.of_call(env, tape.args, tape.inputs, tape.int_inputs)
    # after the tape's allocations
    tr.first_alloc = max((a.root.index + 1 for a in tape.allocs), default=0)
    marks, owners = len(env.owners), list(env.owners)
    owner_sets = None if env.owner_sets is None else list(env.owner_sets)
    # each launch's and allocation's index in the new tape, for the ops' spans
    at_launch = {id(rec): i for i, (_, rec) in enumerate(tape.launches)}
    at_alloc = {id(a): j for j, a in enumerate(tape.allocs)}
    launch_start, alloc_start = [0] * (len(tape.launches) + 1), [0] * (len(tape.allocs) + 1)
    made_by: dict[int, range] = {}  # a converted call's launch index -> its fresh allocations' span
    roots: dict[int, _Root] = {}  # a converted call's fresh output's root: its allocation's
    tensors: dict[int, _TracedTensor] = {}
    records: dict[int, Any] = {}
    gone: set[sympy.Symbol] = set()
    allocs: list[_AllocRec] = []
    launches: list[tuple[int, Any]] = []
    sites: list[KeyedSite] = []
    converted: dict[int, KeyedSite] = {}

    def tensor(t: Any) -> Any:
        if not isinstance(t, _TracedTensor) or id(t._root) not in roots:
            return t
        if id(t) not in tensors:
            sizes, strides = list(t.shape), list(t._sym_strides)
            tensors[id(t)] = _TracedTensor(roots[id(t._root)], sizes, strides, t._sym_offset, t.dtype, t.device)
        return tensors[id(t)]

    def rebase(v: Any, of: tuple[_Root, ...]) -> Any:
        for r in of:
            if id(r) in roots and isinstance(v, torch.SymInt) and _sym_expr(r.sym).free_symbols <= v.node.expr.free_symbols:
                v = v - r.sym + roots[id(r)].sym
        return v

    def remap(rec: Any) -> Any:
        if isinstance(rec, EagerCall):
            leaves = pytree.tree_leaves((rec.args, rec.kwargs, rec.outputs))
            if not any(isinstance(t, _TracedTensor) and id(t._root) in roots for t in leaves):
                return rec
            if rec.target is aten.zero_.default:
                return None
            a, k, o = pytree.tree_map_only(_TracedTensor, tensor, (rec.args, rec.kwargs, rec.outputs))
            return replace(rec, args=a, kwargs=k, outputs=o)
        if not any(id(r) in roots for r in rec.roots):
            return rec
        new = replace(rec, slots=tuple(rebase(v, rec.roots) for v in rec.slots), roots=tuple(roots.get(id(r), r) for r in rec.roots))
        if isinstance(new, Memcpy):
            values = (new.slots, new.nbytes)
        elif isinstance(new, Memset):
            values = (new.slots, new.width, new.height, new.pitch)
        else:
            values = (new.slots, new.grid, new.block, new.smem)
        return None if free_symbols(values) & gone else new

    events = sorted([(a.seq, a) for a in tape.allocs] + tape.launches, key=lambda e: e[0])
    shift = 0
    added = None  # the guards the bindings added, once they stand
    # each record's new seq, a converted call's its last node's (on the trace's stream, the record before it there
    # where it has none); the records' streams and the records they follow on others (Tape.streams, Tape.after)
    moved: dict[int, int | None] = {}
    last_main = None
    streams: dict[int, int] = {}
    after: dict[int, tuple[int, ...]] = {}
    carry: dict[int, tuple[int, ...]] = {}
    try:
        for seq, rec in events:
            if isinstance(rec, _AllocRec):
                alloc_start[at_alloc[id(rec)]] = len(allocs)
                allocs.append(replace(rec, seq=seq + shift) if shift else rec)
                continue
            launch_start[at_launch[id(rec)]] = len(launches)
            if isinstance(rec, OpaqueCall) and id(rec) in calls:
                args, kwargs = pytree.tree_map_only(_TracedTensor, tensor, (rec.args, rec.kwargs))
                leaves, spec = pytree.tree_flatten((args, kwargs))
                fresh = [t for t in rec.outputs if t._root.kind == "eager" and not any(t is a for a in pytree.tree_leaves((rec.args, rec.kwargs)))]
                call = (spec, frozenset(j for j, a in enumerate(leaves) if isinstance(a, torch.Tensor)))
                # the library state the call was traced under, which its key and site record
                with library_state_as(rec.state):
                    key = trace_key(rec.target, args, kwargs, fresh, tr.values)[0]
                    if learn is not None and rec.generator is None and rec.provider.bind(key) is None:
                        inputs = [t for t in pytree.tree_leaves((rec.args, rec.kwargs)) if isinstance(t, _TracedTensor)]
                        lent = {j: x for j, t in enumerate(inputs + fresh) if (x := real(t)) is not None}
                        held = all(j in lent for j in range(len(inputs))) and all(last.get(id(t._root)) == at_launch[id(rec)] for t in inputs + fresh)
                        if not held:
                            lent = {j: x for j, x in lent.items() if j < len(inputs)}
                        learn(rec.target, rec.provider, call, key, lent)
                    binding, values, refusal = bind_at_trace(rec.provider, rec.target, args, kwargs, fresh, tr.values)
                if refusal is not None:
                    return None
                if binding is not None and all(type(t.storage_offset()) is int for t in fresh):
                    for v in values:
                        int(v)  # a guard: Eq(v, its value at the call)
                    tr._seq = itertools.count(seq + shift)
                    tr.generator = rec.generator
                    first, tr.launches = len(tr.allocs), []
                    empty = aten.empty_strided.default
                    made = [tr.allocate(empty, (list(t.shape), list(t.stride())), {"dtype": t.dtype, "device": tr.device}) for t in fresh]
                    for t, m in zip(fresh, made):
                        roots[id(t._root)], tensors[id(t)] = m._root, m
                        gone |= _sym_expr(t._root.sym).free_symbols
                    operands = [a for a in leaves if isinstance(a, torch.Tensor)] + made
                    scalars = [a for a in leaves if not isinstance(a, torch.Tensor)]
                    with library_state_as(rec.state):
                        converted[id(rec)] = record_binding(tr, rec.target, rec.provider, binding, operands, scalars, call, key)
                    sites.append(converted[id(rec)])
                    made_by[at_launch[id(rec)]] = range(len(allocs), len(allocs) + len(tr.allocs) - first)
                    allocs += tr.allocs[first:]
                    launches += tr.launches
                    shift = next(tr._seq) - seq - 1
                    k = tape.streams.get(seq, 0)
                    if k:
                        if not tr.launches:
                            return None  # what follows it on its stream would lose its order
                        streams.update((n, k) for n, _ in tr.launches)
                        moved[seq] = tr.launches[-1][0]
                    else:
                        last_main = moved[seq] = launches[-1][0] if tr.launches else last_main
                    # a binding of no launches passes what it follows to the next record on its stream
                    preds = (*carry.pop(k, ()), *tape.after.get(seq, ()))
                    if preds and tr.launches:
                        after[tr.launches[0][0]] = preds
                    elif preds:
                        carry[k] = preds
                    continue
            new = remap(rec)
            if new is None:
                return None
            records[id(rec)] = new
            launches.append((seq + shift, new))
            moved[seq] = seq + shift
            k = tape.streams.get(seq, 0)
            if k:
                streams[seq + shift] = k
            elif not (isinstance(rec, EagerCall) and rec.host):
                last_main = seq + shift
            if preds := (*(carry.pop(k, ()) if not (isinstance(rec, EagerCall) and rec.host) else ()), *tape.after.get(seq, ())):
                after[seq + shift] = preds
        if (len(env.owners) != marks and not keep_guards) or not converted or _guarded_symbols(tape) & {s.name for s in gone}:
            return None
        added = len(env.owners) - marks
    except Declined:
        return None
    finally:
        # a guard the binding evaluated again outside any op is graph-level,
        # as in the trace; the tape's env keeps its own record
        owners, env.owners[:marks] = env.owners[:marks], owners
        if owner_sets is not None:
            owner_sets, env.owner_sets[:marks] = env.owner_sets[:marks], owner_sets
        if added is None:
            env.forget(marks)
    owners += [None] * added
    env.owners[marks:] = [None] * added
    if owner_sets is not None:
        # a kept guard is graph-level, as its owners entry
        owner_sets += [frozenset((None,)) for _ in range(added)]
        env.owner_sets[marks:] = [frozenset((None,)) for _ in range(added)]
    launch_start[-1], alloc_start[-1] = len(launches), len(allocs)
    # an op whose opaque calls all bound is traced, as a trace at their keys records it, and owns its guards
    traced = {k for k, op in enumerate(tape.ops) if op.held is not None and all(id(rec) in converted for _, rec in tape.launches[op.launches.start : op.launches.stop] if isinstance(rec, EagerCall))}
    held = {g for k, op in enumerate(tape.ops) if k not in traced for g in op.held or ()}
    ops = []
    for k, (op, guards) in enumerate(zip(tape.ops, _owned(owners, owner_sets, len(tape.ops)))):
        call, outputs = pytree.tree_map_only(_TracedTensor, tensor, (op.call, op.outputs))
        # a converted call's fresh allocations are its op's, not the op's before
        own = [range(alloc_start[op.allocs.start], alloc_start[op.allocs.stop - 1] + 1)] if op.allocs else []
        held = [*own, *(made_by[i] for i in op.launches if i in made_by and made_by[i])] or [range(alloc_start[op.allocs.start], alloc_start[op.allocs.start])]
        spans = range(launch_start[op.launches.start], launch_start[op.launches.stop]), range(min(r.start for r in held), max(r.stop for r in held))
        kind, mine = ("traced", None) if k in traced else (op.kind, op.held)
        ops.append(replace(op, call=call, outputs=outputs, launches=spans[0], allocs=spans[1], guards=tuple(g for g in guards if g not in held), kind=kind, held=mine))

    order = {id(a.root): a.seq for a in allocs}
    for s in tape.sites:
        operands = tuple(map(tensor, s.operands))
        sites.append(replace(s, operands=operands, nodes=tuple(records.get(id(n), n) for n in s.nodes)))
    sites.sort(key=lambda s: min(order[id(t._root)] for t, _ in s.scratch.values()))
    out = copy.copy(tape)
    out.allocs, out.launches, out.sites, out.ops, out.owners = allocs, launches, sites, ops, owners
    out.streams = streams
    out.after = {seq: tuple(sorted(ps)) for seq, preds in after.items() if (ps := {moved.get(p) for p in preds} - {None})}
    out.checks = [*tape.checks, *tr.checks]
    if added:
        out.guard_count = marks + added
    out.owner_sets = None if owner_sets is None else [frozenset(o) for o in owner_sets]
    out.outputs = [replace(o, root=roots[id(o.root)]) if isinstance(o, _OutputRec) and id(o.root) in roots else o for o in tape.outputs]
    positions = {rec.position for rec in tape.inputs}
    out.args = tuple(_released_arg(a, tape.device) if i in positions and not isinstance(a, torch.Tensor) else a for i, a in enumerate(tape.args))
    out.warm_up_result = None
    return out, converted


if _hint_audit.requested:
    _hint_audit.enable(_hint_audit.requested)
