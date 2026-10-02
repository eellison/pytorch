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
import threading
import warnings
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TYPE_CHECKING

import sympy

import torch
from torch._C import _dispatch_has_kernel_for_dispatch_key as _has_kernel, DispatchKey
from torch._dispatch.python import enable_python_dispatcher
from torch._ops import OpOverload
from torch import _meta_registrations
from torch._prims.rng_prims import _impl_graphsafe_rng, graphsafe_run_with_rng_state
from torch._subclasses import fake_impls
from torch._subclasses.fake_impls import _compute_stride
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda import _host_trace_ir as _ir
from torch.cuda._host_trace import _TraceShapeEnv, Declined, declined, ProcessHold
from torch.cuda._host_trace_opaque import bind_at_trace, library_state, library_state_as, record_binding, trace_key
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


@dataclass
class _Root:
    name: str  # p<i> for argument i's storage, a<k> for allocation k, e<k> for eager output k
    sym: Any  # the base address: an input's symbol, 256*q otherwise
    kind: Literal["argument", "allocation", "eager"] = "argument"


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
    # function, its stream arguments' (position, type), None), or ("host",
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


@dataclass(frozen=True)
class OpaqueCall(EagerCall):
    """An eager call `provider` accepted: a replay binds its key
    (_host_trace_opaque)."""

    provider: Any = None
    # its key bound at the trace, which did not record the binding (a fresh
    # output at a storage offset): a replay at a key that binds is no reason
    # to relower
    bound: bool = False


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
    """A 1D device-to-device memcpy node of `nbytes` from slots[1] to slots[0]."""

    name: str
    slots: tuple[Any, Any]  # dst, src
    roots: tuple[_Root, ...]  # the roots the slots may be of
    nbytes: Any


_active = threading.local()


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
    ) -> None:
        self.device = device
        self.trusted = trusted
        self.opaque = opaque
        self.static_shapes = frozenset(static_shapes)
        # the top-level ops (by index) that run eagerly, each with why
        self.eager_ops = eager_ops or {}
        # the trace's symbol for each of the caller's, under trusted inputs
        self.given: dict[sympy.Symbol, torch.SymInt] = {}
        # the trace's capturing stream, the only one a launch may be recorded on
        cuda = device.type == "cuda"
        self.stream = torch.cuda.current_stream(device) if cuda else None
        # computes a view's and an eager call's metadata; without its cache
        # every guard a kernel evaluates is recorded, and without fallback
        # kernels an op with no meta raises instead of running on zeros
        if trusted is None and torch.cuda._host_trace.symbolic == "ir":
            self.shape_env: Any = _ir.Env()
            # the fake twins carry the trace's own SymInts; no ShapeEnv
            self.fake_mode = FakeTensorMode(allow_fallback_kernels=False)
        else:
            self.shape_env = _TraceShapeEnv(trusted=trusted is not None)
            self.fake_mode = FakeTensorMode(
                shape_env=self.shape_env, allow_fallback_kernels=False
            )
        self.fake_mode.cache_enabled = False
        self._seq = itertools.count()
        self.inputs: list[_InputRec] = []
        self.arguments: dict[int, _InputRec] = {}  # by id of its root
        # argument positions i < j, where a step not run eagerly writes one
        # and reads the other (Tape.argument_pairs)
        self.argument_pairs: set[tuple[int, int]] = set()
        self.int_inputs: list[_IntInputRec] = []
        self.allocs: list[_AllocRec] = []
        # allocation k's symbol is alloc<first_alloc + k>: bind_opaque's follow its tape's
        self.first_alloc = 0
        self.launches: list[tuple[int, Any]] = []  # (seq, launch or EagerCall)
        self.eager_outputs: list[_TracedTensor] = []
        # every tensor of the trace, to find one fn kept past it (trace's check_escapes)
        self.tensors: weakref.WeakSet[_TracedTensor] | None = weakref.WeakSet() if check_escapes else None
        # the traced hosts' calls with their outputs, for _check_witness
        self.aten_calls: list[tuple[int, EagerCall]] = []
        # fn's own operator calls in order, as the warm-up's (_order_key)
        self.order: list[tuple] = []
        self.depth = 0  # of _TraceMode dispatches
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
        self.ops: list[OpRec] = []

    def decline(self, msg: str) -> Declined:
        e = declined(msg)
        if self.declined is None:
            self.declined = e
        return e

    def record_launch(self, record: Any) -> None:
        if isinstance(record, EagerCall) and not record.host:
            leaves = pytree.tree_leaves((record.args, record.kwargs))
            self.host_reads.update(a.untyped_storage()._cdata for a in leaves if isinstance(a, torch.Tensor) and self._is_host(a))
        self.launches.append((next(self._seq), record))

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
        root = _Root(f"p{position}", sym)
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
        deterministic = torch.are_deterministic_algorithms_enabled()
        if deterministic and torch._C._get_deterministic_fill_uninitialized_memory():
            raise self.decline(
                "use_deterministic_algorithms(True) with fill_uninitialized_memory fills every allocation "
                "with a kernel the trace does not record"
            )
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
            if mf not in (None, torch.preserve_format):
                strides = _format_strides(sizes, mf)
            elif bool(src.numel() == 0) or _guard_each(_dense_terms(sizes, src_strides)):
                # c10's is_contiguous: an empty tensor is
                strides = list(src_strides)
            else:
                strides = _infer_dense_strides(sizes, src_strides)
            mf = None
        if mf not in _ALLOC_FORMATS:
            raise self.decline(f"an allocation in memory format {mf}")
        for s in sizes:
            # c10's check_size_nonnegative, a guard for a size from an int argument
            if not bool(s >= 0):
                raise RuntimeError(
                    f"Trying to create tensor with negative dimension {_hint(s)}: {sizes}"
                )
        k = self.first_alloc + len(self.allocs)
        name = f"alloc{k}"
        hint = (_ALLOC_TAG | ((k + 1) << _ALLOC_SHIFT)) // _ALLOC_ALIGNMENT
        q = self.shape_env.symbol(hint, f"{name}.base/{_ALLOC_ALIGNMENT}")
        root = _Root(f"a{k}", _ALLOC_ALIGNMENT * q, "allocation")
        rec = _AllocRec(next(self._seq), name, sizes, strides, dtype, root, q)
        self.allocs.append(rec)
        return _TracedTensor(root, sizes, strides, 0, dtype, self.device)

    def zero(self, t: Any) -> bool:
        """Record t.zero_() of a non-overlapping and dense tensor on the
        trace's stream as a Memset of its bytes (nothing if it is empty);
        otherwise it is an op like another."""
        if not isinstance(t, _TracedTensor) or self.stream is None:
            return False
        if torch.cuda.current_stream(self.device) != self.stream:
            return False
        if not t.numel():
            return True
        if not _is_non_overlapping_and_dense(t):
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
        twin = torch.empty(0, dtype=t.dtype, device=t.device)
        return twin.as_strided(t.shape, t._sym_strides, t._sym_offset)

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
        return self.op(func, args, kwargs, run, redo=lambda a, k: current_trace().eager_call(func, a, k))

    def jiterator(self, code: str, name: str, return_by_ref: bool, num_outputs: int, tensors: tuple, kwargs: dict | None = None) -> Any:
        """A user jiterator's launch (torch.cuda.jiterator), which no operator
        dispatches: its pointwise host, of the witness launch at stand-ins."""
        label = f"jiterator {name}"
        args = (code, name, return_by_ref, num_outputs, tensors, kwargs or {})

        def run() -> Any:
            if not all(isinstance(t, _TracedTensor) for t in tensors):
                raise self.decline(f"{label} of a tensor the trace does not track")
            if any(isinstance(v, (torch.Tensor, *_SYM_TYPES)) for v in args[5].values()):
                raise self.decline(f"{label} of an extra argument that is not a constant")
            out = self._pointwise_host(label, args, {}, _jiterator_launch, None, [])
            if isinstance(out, str):
                raise self.decline(f"{label}'s pointwise host declines: {out}")
            return out

        return self.op(label, args, {}, run, host=True, redo=lambda a, k: current_trace().jiterator(*a, **k))

    def op(
        self, func: Any, args: tuple, kwargs: dict, run: Callable[[], Any], host: bool = False, redo: Callable[[tuple, dict], Any] | None = None
    ) -> Any:
        """run(), the call func(*args, **kwargs); at the top level an op (OpRec).
        What decides an eager or library step's outputs' metadata (its fake
        kernel) guards for the graph; a traced host, which picks and launches
        its kernels and allocates its outputs, for the op (env.op), from the
        start where run() is all host (`host`)."""
        if self.in_op:
            return run()
        env = self.shape_env
        k, first, guards = len(self.ops), (len(self.launches), len(self.allocs), len(self.sites)), len(env.owners)
        self.in_op, env.op = True, k if host else None
        try:
            out = run()
        finally:
            self.in_op, env.op = False, None
        # a later op reading a property this op cached would not guard it again
        for t in pytree.tree_leaves((args, kwargs, out)):
            if isinstance(t, _TracedTensor):
                torch._C._cuda_hostTraceRefreshContiguous(t)
        records = [rec for _, rec in self.launches[first[0] :]]
        kind = "other"
        if len(self.sites) == first[2]:
            if not any(isinstance(rec, EagerCall) for rec in records):
                kind = "traced"
            elif len(records) == 1 and type(records[0]) is EagerCall and not records[0].host:
                kind = "eager"
        if kind == "other":
            env.owners[guards:] = [None if o == k else o for o in env.owners[guards:]]
        outputs = tuple(o for o in pytree.tree_leaves(out) if isinstance(o, (torch.Tensor, int, torch.SymInt)))
        spans = (range(first[0], len(self.launches)), range(first[1], len(self.allocs)))
        self.ops.append(OpRec(func, (args, kwargs), *spans, outputs, kind, redo=redo))
        return out

    def _eager_call(self, func: Any, args: tuple, kwargs: dict) -> Any:
        if not isinstance(func, OpOverload):
            raise self.decline(f"{func} is not an operator")
        if func is aten._local_scalar_dense.default:
            raise self.decline(f"{func} reads a traced tensor's value on the host")
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
            if torch.cuda.current_stream(self.device) != self.stream:
                raise self.decline(f"{func} on a stream other than the trace's")
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
        redispatch = (factory and func not in _TRACED_ATEN) or (_pointwise(func) and not any(numbers))
        if hosts and (func in _ZEROS or func is aten.embedding.default or (redispatch and _composite(func) and _out_overload(func) is None)):
            # CompositeExplicitAutograd (embedding, a factory (full,
            # scalar_tensor), a pointwise op's Scalar overload or functional
            # form): its parts, traced, as eager runs them. A
            # functional form over a CUDA out= kernel (abs, logical_not) resizes
            # an empty tensor: the witness of the out= overload instead
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

        elementwise = _pointwise(func) or _elementwise(func, args, kwargs)

        on_roots = [[t for t in tensors if t._root is v._root] for v in written.values() if isinstance(v, _TracedTensor)]

        def shared(full_overlap: bool) -> bool:
            # TensorIterator allows a full overlap: a pointwise op may read the tensor it writes
            return any(len({id(t) for t in on_root} if full_overlap else on_root) > 1 for on_root in on_roots)

        if overlap or shared(elementwise):
            reasons.append(f"{func} writes a storage another operand is of")
        else:
            if self.opaque and shared(False):
                reasons.append(f"{func} writes a tensor it reads: no library kernel")
            for p in self.opaque if not reasons else ():
                if (why := p.accepts(func, args, kwargs)) is None:
                    provider = p
                    break
                reasons.append(why)
            # a traced host's outputs and guards are eager's own: no fake kernel
            if provider is None and hosts and (kernel := _python_kernel(func)) is not None:
                if func.namespace == "aten":
                    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
                    traced = self._traced_host(func, args, kwargs, lambda: kernel(keyset, *args, _fallback=_Fallback(self, func), **kwargs))
                else:
                    traced = self._traced_host(func, args, kwargs, lambda: kernel(*args, **kwargs))
                if not isinstance(traced, str):
                    self.argument_pairs |= pairs
                    return traced
                reasons.append(f"{func}'s kernel declines: {traced}")
            elif provider is None and hosts:
                traced = self._traced_aten(func, args, kwargs) if func in _TRACED_ATEN else None
                if isinstance(traced, str):
                    reasons.append(f"{func}'s traced host declines: {traced}")
                    traced = None
                if traced is None and elementwise and self.device.type == "cuda":
                    traced = self._traced_pointwise(func, args, kwargs)
                    if isinstance(traced, str):
                        reasons.append(f"{func}'s pointwise host declines: {traced}")
                        traced = None
                if traced is None and _kernel_less(func):
                    # a CompositeExplicitAutograd(NonFunctional) op: its body, as
                    # eager runs it, each of its parts routed as any op
                    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)

                    def body() -> Any:
                        try:
                            return func.redispatch(keyset, *args, **kwargs)
                        except RuntimeError as e:
                            # an int-signature body's wrapper (C10_AS_INTARRAYREF_SLOW)
                            if "expected to contain only concrete integers" in str(e):
                                raise declined("its body takes no symbolic size") from e
                            raise

                    traced = self._traced_host(func, args, kwargs, body, parts=True)
                    if isinstance(traced, str):
                        if traced:
                            reasons.append(f"{func}'s body declines: {traced}")
                        traced = None
                if traced is not None:
                    self.argument_pairs |= pairs
                    return traced
        # an eager or library step's outputs: its meta function's metadata on
        # bare meta twins, else the fake kernel's
        routed = func in _META_OPAQUE and (meta := self._opaque_meta(func, leaves, spec)) is not None
        if routed:
            out, twins = meta
        else:
            twins, out = self._fake_call(func, args, kwargs)
        if func is aten._scaled_dot_product_cudnn_attention.default:
            lse = args[4] if len(args) > 4 else kwargs["compute_log_sumexp"]
            if not lse:
                # the fake kernel returns a log-sum-exp the CUDA kernel does
                # not. A metadata fix, not an attention knob: it holds for an
                # eager step as for a bound call, so it is always on
                out = (out[0], None, *out[2:])
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
            binding, values, refusal = bind_at_trace(provider, func, args, kwargs, fakes)
            _guard_each([v == _hint(v) for v in values])
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
            hint = (_EAGER_TAG | ((k + 1) << _ALLOC_SHIFT)) // _ALLOC_ALIGNMENT
            q = self.shape_env.symbol(hint, f"eager{k}.base/{_ALLOC_ALIGNMENT}")
            root = _Root(f"e{k}", _ALLOC_ALIGNMENT * q, "eager")
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
            self.sites.append(record_binding(self, func, provider, binding, operands, scalars, call))
        elif provider is not None:
            bound = provider.bind(trace_key(func, args, kwargs, fakes)[0]) is not None
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
                return twins, func(*twins[0], **twins[1])
            except Exception as e:
                why = f"{type(e).__name__}: {str(e).splitlines()[0]}"
                raise self.decline(f"{func} has no traced metadata ({why})") from e
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

    def _traced_host(self, func: OpOverload, args: tuple, kwargs: dict, kernel: Callable[[], Any], parts: bool = False) -> Any:
        """func's output from kernel(), its Python CUDA kernel (_python_kernel)
        or its composite body on the traced tensors, its calls and launches
        traced as any; why where it declines (or with `parts`, a part runs
        eagerly), which leaves the call to EagerCall: '' where that part runs
        eagerly for no reason, as a plain eager call, and so does the op."""
        env = self.shape_env
        marks = len(self.allocs), len(self.launches), len(self.sites), len(self.eager_outputs), len(self.aten_calls)
        guards, op, env.op, prior, host, host_reads = len(env.owners), env.op, len(self.ops), self.declined, dict(self.host), set(self.host_reads)
        plain_part = False
        try:
            # the Python and global state a torch._native condition reads
            # (node.active, a config flag, the arguments' copy-on-write state)
            # is fixed for the entry under the trace's contract that Python
            # state is fixed at trace time, so it is not guarded
            with _TraceMode(self):
                out = kernel()
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
            del self.allocs[marks[0] :], self.launches[marks[1] :], self.sites[marks[2] :], self.eager_outputs[marks[3] :]
            del self.aten_calls[marks[4] :]
            self.declined, self.host, self.host_reads = prior, host, host_reads
            # what the kernel declined on chose the eager call
            env.owners[guards:] = [None] * (len(env.owners) - guards)
            if plain_part:
                return ""
            return str(e).splitlines()[0] if isinstance(e, Declined) else f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        finally:
            env.op = op

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
            env.owners[guards:] = [None] * (len(env.owners) - guards)
            env.op = op
            return str(e).splitlines()[0]
        except Exception as e:
            del self.allocs[marks[0] :], self.launches[marks[1] :]
            env.op = op
            if isinstance(e, (Declined, AssertionError)):
                raise  # the trace's own verdict, or its bug
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
                dst, src, nbytes = record
                # copy_device_to_device skips a memcpy onto its source
                apart = _sym_expr(dst) - _sym_expr(src)
                if not any(_sym_expr(r.sym).free_symbols & apart.free_symbols for r in roots) and bool(dst == src):
                    continue
                launches.append(Memcpy(str(func), (dst, src), roots, nbytes))
                continue
            function, offsets, params, fields, grid, block, smem, *cpu_scalars = record
            places, values, is_pointer = [], [], []
            for param, offset, width, value, pointer in fields:
                places.append((param, offset, width))
                values.append(value)
                is_pointer.append(pointer)
            launches.append(
                KernelLaunch(
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
                    cpu_scalars=cpu_scalars[0] if cpu_scalars else (),
                )
            )
        if check is not None and not check(out, launches):
            del self.allocs[marks[0] :], self.launches[marks[1] :]
            env.owners[guards:] = [None] * (len(env.owners) - guards)
            return None
        for launch in launches:
            if isinstance(launch, KernelLaunch):
                self.host_reads.update(t.untyped_storage()._cdata for *_, t in launch.cpu_scalars)
            self.record_launch(launch)
        if call is not None:
            self._witnessed(func, *call, out)
        return out

    def _traced_pointwise(self, func: OpOverload, args: tuple, kwargs: dict) -> Any:
        """A pointwise op's traced host (Pointwise.cu) from a witness: the op's
        out= overload (in-place and out= ops, and ops without one: the op) run
        at stand-ins of its tensors in a capture gives its kernels and their
        parameter bytes (the functors', whatever the op's), and the
        TensorIterators it builds each kernel's operands; TensorIteratorSym the
        sizes, strides and addresses in them. The output, or why the host
        declines."""
        schema = func._schema
        if torch.Tag.nondeterministic_seeded in func.tags:
            return "it draws from a generator"
        if not all(isinstance(r.type, torch.TensorType) for r in schema.returns):
            return f"it returns {len(schema.returns)} values"
        values = (*args, *(kwargs.get(a.name, a.default_value) for a in schema.arguments[len(args) :]))
        # a size (full's) sets only the shapes, which the host takes from its tensors
        if any(isinstance(x, _SYM_TYPES) for s, a in zip(schema.arguments, values) if not isinstance(s.type, torch.ListType) for x in pytree.tree_leaves(a)):
            return "a symbolic scalar operand"
        written = [a for s, a in zip(schema.arguments, values) if s.alias_info is not None and s.alias_info.is_write]
        if schema.is_mutable:
            # in-place or out=: it writes and returns its written arguments
            if len(written) != len(schema.returns) or any(r.alias_info is None for r in schema.returns):
                return "it writes other than the tensors it returns"
            op, out_name = func, None
        else:
            # without an out= overload the witness is the op, which allocates its outputs
            written, (op, out_name) = [], _out_overload(func) or (func, None)
        cpu = {}
        for s, a in zip(schema.arguments, values):
            if isinstance(a, torch.Tensor) and self._is_host(a) and a.dim() == 0:
                cpu[id(a)] = a
            elif not isinstance(a, _TracedTensor) and any(isinstance(x, torch.Tensor) for x in pytree.tree_leaves(a)):
                return f"its {s.name} is not a traced tensor"
        if len(cpu) > 1:
            return "it reads two CPU buffers"
        return self._pointwise_host(func, args, kwargs, op, out_name, written)

    def _pointwise_host(self, func: OpOverload | str, args: tuple, kwargs: dict, op: Callable[..., Any], out_name: str | None, written: list) -> Any:
        """_traced_pointwise's host of the witness op(*args, **kwargs), into
        out_name's stand-in where given, writing `written`; func an operator or
        a user jiterator's label."""
        from cuda.bindings import runtime

        from torch.cuda._host_trace_capture import capture_kernel_nodes, KernelNode, MemcpyNode, pack_params
        from torch.cuda._host_trace_cute import _stand_in

        traced: dict[int, tuple[torch.Tensor, _TracedTensor]] = {}  # by id, each stand-in and its tensor

        def stand_in(t: _TracedTensor) -> torch.Tensor:
            s = _stand_in(t)
            traced[id(s)] = (s, t)
            return s

        base = (_ALLOC_TAG | (self.first_alloc + len(self.allocs) + 1) << _ALLOC_SHIFT) & _PLACEHOLDER_LOW
        stand = pytree.tree_map_only(_SYM_TYPES, _hint, pytree.tree_map_only(_TracedTensor, stand_in, (args, kwargs)))
        # a host buffer (_traced_pointwise's one CPU scalar operand) holds no
        # value at trace time: the witness runs at probe values of it
        cpu = next((a for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor)), None)
        probes = [] if cpu is None else _cpu_scalar_probes(cpu.dtype)

        def at_probe(k: int) -> tuple:
            return pytree.tree_map_only(torch.Tensor, lambda t: probes[k] if t is probes[0] else t, stand)

        if cpu is not None:
            stand = pytree.tree_map_only(torch.Tensor, lambda t: probes[0] if t is cpu else t, stand)
        w = None
        if out_name is not None:
            # the witness's output: the meta kernel's at the hints, which the
            # host's launch then matches byte for byte
            try:
                o = _meta_at_hints(func, args, kwargs)
            except Exception as e:
                return f"its meta kernel raised {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            sizes, strides = list(o.shape), list(o.stride())
            extent = 1 + sum((n - 1) * st for n, st in zip(sizes, strides)) if all(sizes) else 0
            storage = torch._C._construct_storage_from_data_pointer(base, self.device, extent * o.element_size())
            with _disable_current_modes():
                w = torch.empty(0, dtype=o.dtype, device=self.device).set_(storage, 0, sizes, strides)
            stand[1][out_name] = w
        spans = [(t.data_ptr(), t.data_ptr() + t.untyped_storage().nbytes()) for t in pytree.tree_leaves(stand) if isinstance(t, torch.Tensor) and t.is_cuda]

        def witness(stand: tuple) -> tuple[list, list, list, list] | str:
            # its nodes, iterators, launch sites' reports (LaunchLayout.h) and outputs
            # a relaxed thread: a witness that allocates declines the op, not the trace
            relaxed = runtime.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed
            mode = _check_cuda_bindings(runtime.cudaThreadExchangeStreamCaptureMode(relaxed))
            made: list[Any] = []
            harvesting = torch._C._cuda_hostTraceSetHarvesting(True)
            torch._C._cuda_hostTraceRecordIterators(True)
            torch._C._cuda_hostTraceRecordLaunches(True)
            try:
                with _disable_current_modes(), torch.cuda.device(self.device):
                    nodes = capture_kernel_nodes(lambda s: made.append(op(*stand[0], **stand[1])), mode="relaxed", memcpy=True)
            except Exception as e:
                return f"its witness raised {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            finally:
                launched = torch._C._cuda_hostTraceRecordLaunches(False)
                iterators = torch._C._cuda_hostTraceRecordIterators(False)
                torch._C._cuda_hostTraceSetHarvesting(harvesting)
                _check_cuda_bindings(runtime.cudaThreadExchangeStreamCaptureMode(mode))
            return nodes, iterators, launched, made

        allocations = torch._C._cuda_hostTraceAllocationCount(self.device.index)
        if isinstance(got := witness(stand), str):
            return got
        nodes, iterators, launched, made = got
        results = made[0] if isinstance(made[0], tuple) else made
        if not all(isinstance(n, (KernelNode, MemcpyNode)) for n in nodes):
            return f"its witness launched {len(nodes)} operations"

        def metadata(iterators: list, nodes: list, k: int) -> list | None:
            # what a CPU scalar's value must not change
            if not all(isinstance(n, KernelNode) for n in nodes):
                return None
            its = [(n, numel, common, reduction, [(o.dtype, o.shape, o.stride(), o.is_cuda or o.data_ptr() == probes[k].data_ptr()) for o in ops]) for ops, n, numel, common, reduction in iterators]
            return [its, [(n.function, n.grid, n.block, n.smem, n.layout) for n in nodes]]

        if cpu is not None:
            if (traced_metadata := metadata(iterators, nodes, 0)) is None:
                return "a memcpy of an op of a CPU buffer"

        def operand(t: torch.Tensor) -> Any:
            # a stand-in, or a broadcast of exactly one (expand_inplace's), is its tensor
            if id(t) in traced:
                return traced[id(t)][1]
            found = set()
            for s, v in traced.values():
                if s.data_ptr() == t.data_ptr() and s.dtype == t.dtype and t.dim() >= s.dim():
                    try:
                        if s.expand(t.shape).stride() == t.stride():
                            found.add(v)
                    except RuntimeError:
                        pass
            return found.pop() if len(found) == 1 else None

        # per iterator: its outputs (a traced tensor, None for the op's output
        # the host allocates, "temp" for a 0-dim temporary, an earlier
        # iterator's (index, output)), their dtypes, its inputs (a traced
        # tensor, an earlier iterator's output, a CPU scalar) and kernel node
        written_by: dict[int, tuple[int, int]] = {}
        plan: list[tuple[list, list[torch.dtype], list, torch.dtype, Any]] = []
        moves: list[tuple[int, int]] = []  # each host allocation's witness address and bytes
        witnessed = 0  # the witness's own allocations (of bytes) among them
        live = iter(nodes)
        empty_reductions: list[Any] = []  # the inputs of zero-element reductions, which launch nothing
        for operands, noutputs, numel, common, reduction in iterators:
            outs, ins = operands[:noutputs], operands[noutputs:]
            if reduction:
                # eager fills a zero-element reduction's output, with a later iterator
                found = [operand(i) for i in ins if i.is_cuda]
                if numel or None in found:
                    return "its witness built a reduction iterator"
                empty_reductions += found
                continue
            if not outs[0].is_cuda:
                continue
            inputs: list[Any] = []
            for i in ins:
                if not i.is_cuda:
                    inputs.append(cpu if cpu is not None and i.data_ptr() == probes[0].data_ptr() else i)
                elif id(i) in written_by:
                    inputs.append(written_by[id(i)])
                elif (t := operand(i)) is not None:
                    inputs.append(t)
                else:
                    return "an iterator reads a tensor that is none of the op's"
            targets: list[Any] = []
            for o in outs:
                if id(o) in written_by:
                    targets.append(written_by[id(o)])
                elif id(o) in traced:
                    if not any(traced[id(o)][1] is x for x in written):
                        return "an iterator writes a tensor the op does not"
                    targets.append(traced[id(o)][1])
                elif any(o is r for r in results) or o.dim() == 0:
                    k = next((k for k, r in enumerate(results) if o is r), None)
                    # a nullary iterator's output (full's, ones_like's) has no operand to take a shape from: the op's
                    targets.append("temp" if k is None else None if any(i.is_cuda for i in ins) else k)
                    moves.append((o.data_ptr(), o.untyped_storage().nbytes()))
                    witnessed += o is not w and moves[-1][1] > 0
                else:
                    return "an iterator writes a temporary"
            node = next(live, None) if numel else None
            if numel and node is None:
                return f"its witness launched {len(nodes)} kernels, not one per nonempty iterator"
            if isinstance(node, MemcpyNode):
                # copy_'s memcpy: the traced copy_ host into the op's output
                if len(outs) != 1 or len(inputs) != 1:
                    return "a memcpy of an iterator of other than one output and input"
                targets = [next((k for k, r in enumerate(results) if outs[0] is r), None) if t is None else t for t in targets]
            written_by.update((id(o), (len(plan), k)) for k, o in enumerate(outs))
            plan.append((targets, [o.dtype for o in outs], inputs, common, node))
        if next(live, None) is not None:
            return f"its witness launched {len(nodes)} kernels, not one per nonempty iterator"
        finals = [written_by.get(id(r)) or (traced[id(r)][1] if id(r) in traced else None) for r in results]
        fakes: list = []
        if None in finals or any(type(t) is int for targets, *_ in plan for t in targets):
            # an output the host allocates at no operand's shape: the fake kernel's
            if not isinstance(func, OpOverload):
                return "an output has no operand's shape"
            _, out = self._fake_call(func, args, kwargs)
            fakes = [out] if len(func._schema.returns) == 1 else list(out)
        # a kernel's unused buffer (log_sigmoid_forward's) is an empty tensor it returns
        empty = [f is None and any(type(n) is int and n == 0 for n in fakes[k].shape) for k, f in enumerate(finals)]
        if any(f is None and not e for f, e in zip(finals, empty)):
            return "no iterator writes its output"
        moves += [(r.data_ptr(), 0) for r, e in zip(results, empty) if e]
        spans += [(a, a + n) for a, n in moves]
        # the witness's outputs, freed before the host's run
        made.clear()
        del got, results, iterators, w
        operands = outs = o = None
        if torch._C._cuda_hostTraceAllocationCount(self.device.index) != allocations + witnessed:
            return "its witness allocated"
        allocated = len(self.allocs)

        # each kernel's byte classes (LaunchLayout.h) where its launch site's
        # report matches it (function, grid, block, parameter sizes and every
        # byte of a class), its bytes of no class zeroed; a kernel without one
        # keeps the witness's bytes, scanned
        kernels = [n for n in nodes if isinstance(n, KernelNode)]
        classes: dict[int, list[str]] = {}
        images = {id(n): n.images for n in kernels}
        j = 0
        for n in kernels:
            if (k := next((k for k in range(j, len(launched)) if launched[k][0] == n.function), None)) is None:
                continue
            _, grid, block, cls, data = launched[k]
            j = k + 1
            if (grid, block) == (n.grid, n.block) and [len(c) for c in cls] == [len(b) for b in n.images] and all(c == "." or x == y for cs, d, b in zip(cls, data, n.images) for c, x, y in zip(cs, d, b)):
                classes[id(n)] = cls
                images[id(n)] = tuple(bytes(0 if c == "." else x for c, x in zip(cs, b)) for cs, b in zip(cls, n.images))
        # each member eager reads from the CPU buffer, (parameter, byte offset,
        # class): its kernel's iterator's CPU scalar, its bytes eager's read of
        # each probe, the kernels' other functor bytes the same at each probe
        scalars: dict[int, list[tuple[int, int, str]]] = {}
        if cpu is not None:
            reading = {id(p[-1]) for p in plan if p[-1] is not None and any(x is cpu for x in p[2])}
            for n in kernels:
                if id(n) not in classes:
                    return f"kernel {n.name} of an op of a CPU buffer has no launch report"
                if id(n) not in reading:
                    continue
                runs = scalars[id(n)] = []
                for p, cs in enumerate(classes[id(n)]):
                    at = 0
                    while at < len(cs):
                        if not (cs[at].isupper() or cs[at].isdigit()):
                            at += 1
                            continue
                        width = len(torch._C._cuda_hostTraceCpuScalarBytes(probes[0], cs[at]))
                        if cs[at : at + width] != cs[at] * width:
                            return f"kernel {n.name}'s CPU scalar member is not of its class's width"
                        runs.append((p, at, cs[at]))
                        at += width
                if not runs:
                    return f"kernel {n.name} reads a CPU buffer at no member its launch report marks"

            def scalar_value(n: KernelNode, image: Sequence[bytes], k: int) -> bool:
                return all(image[p][at : at + len(v)] == v for p, at, c in scalars.get(id(n), ()) for v in [torch._C._cuda_hostTraceCpuScalarBytes(probes[k], c)])

            if not all(scalar_value(n, n.images, 0) for n in kernels):
                return "a kernel's CPU scalar member is not eager's read of the CPU buffer"
            for k in range(1, len(probes)):
                before = torch._C._cuda_hostTraceAllocationCount(self.device.index)
                if isinstance(again := witness(at_probe(k)), str):
                    return f"at another value of its CPU buffer, {again}"
                if metadata(again[1], again[0], k) != traced_metadata:
                    return "a CPU buffer's value sets its iterators or launches"
                for n, m in zip(kernels, again[0]):
                    marked = {(p, at + i) for p, at, c in scalars.get(id(n), ()) for i in range(len(torch._C._cuda_hostTraceCpuScalarBytes(probes[k], c)))}
                    if not scalar_value(n, m.images, k) or any(c not in "kp." and (p, b) not in marked and x != y for p, (cs, ib, jb) in enumerate(zip(classes[id(n)], n.images, m.images)) for b, (c, x, y) in enumerate(zip(cs, ib, jb))):
                        return "a CPU buffer's value sets other than its kernels' CPU scalar members"
                del again
                if torch._C._cuda_hostTraceAllocationCount(self.device.index) - before != witnessed:
                    return "its witness allocated"
        # a functor's pointer member replays as its operand's address
        roots = {id(s): (s.data_ptr(), s.data_ptr() + s.untyped_storage().nbytes(), t) for s, t in traced.values()}
        addresses: dict[int, list[tuple[int, int, _TracedTensor, int]]] = {}
        for n in kernels:
            for p, (cs, b) in enumerate(zip(classes.get(id(n), ()), images[id(n)])):
                if "z" in cs:
                    return f"kernel {n.name}'s functor has a member of the operands' sizes"
                for at in range(0, len(cs) - 7, 8):
                    v = int.from_bytes(b[at : at + 8], "little")
                    if cs[at : at + 8] != "p" * 8 or v == 0:
                        continue
                    if (hit := next(((t, v - lo) for lo, hi, t in roots.values() if lo <= v < hi), None)) is None:
                        return f"kernel {n.name}'s functor points at a tensor that is none of the op's"
                    addresses.setdefault(id(n), []).append((p, at, *hit))

        def check(out_t: _TracedTensor, launches: list) -> bool:
            kernels = [p[-1] for p in plan if p[-1] is not None]
            if len(launches) != len(kernels) or len(self.allocs) - allocated != len(moves):
                return False
            # host allocation j's placeholder address stands for moves[j]
            hosts = [((_ALLOC_TAG | (self.first_alloc + allocated + 1 + j) << _ALLOC_SHIFT) & _PLACEHOLDER_LOW, a, n) for j, (a, n) in enumerate(moves)]

            def address(v: Any) -> int:
                v = _hint(v) & _PLACEHOLDER_LOW
                return next((v - h + a for h, a, n in hosts if h <= v < h + n), v)

            for launch, node in zip(launches, kernels):
                if isinstance(node, MemcpyNode) or isinstance(launch, Memcpy):
                    if not isinstance(launch, Memcpy) or not isinstance(node, MemcpyNode):
                        return False
                    if (*map(address, launch.slots), _hint(launch.nbytes)) != (node.dst, node.src, node.nbytes):
                        return False
                    continue
                if (tuple(map(_hint, launch.grid)), tuple(map(_hint, launch.block)), _hint(launch.smem)) != (node.grid, node.block, node.smem):
                    return False
                pointers = [i in launch.pointers for i in range(len(launch.slots))]
                slots = [address(v) if p else _hint(v) for v, p in zip(launch.slots, pointers)]
                if [bytes(b) for b in pack_params(launch, slots, pointers)] != [bytes(b) for b in images[id(node)]]:
                    return False
                # the witness's bytes the host does not set replay as they are: a
                # tensor's address among them (a functor's pointer), or any CUDA
                # address, would be stale. Of a typed kernel only its functors' bytes
                placed = {(p, b) for p, at, n in launch.fields for b in range(at, at + n)}
                cls = classes.get(id(node))
                for p, image in enumerate(images[id(node)]):
                    for b in range(0, len(image) - 7, 4):
                        v = int.from_bytes(image[b : b + 8], "little")
                        if placed & {(p, b + i) for i in range(8)} or (cls is not None and "p" not in cls[p][b : b + 8]):
                            continue
                        if any(lo <= v < hi for lo, hi in spans) or (v >> 32 and _cuda_address(v)):
                            stale.append(f"kernel {node.name} parameter {p} holds a CUDA address at byte {b}")
                            return False
            return True

        stale: list[str] = []

        declines: list[str] = []
        dynamic = not isinstance(func, OpOverload)  # a user jiterator's launch

        def host() -> Any:
            outs: list[Any] = []
            records: list = []
            if any(t.numel() != 0 for t in empty_reductions):
                declines.append("a zero-element reduction's input is not empty")
                raise NotImplementedError(declines[-1])
            for targets, dtypes, inputs, common, node in plan:
                targets = [
                    torch.empty((), dtype=d, device=self.device) if isinstance(t, str)
                    else outs[t[0]][t[1]] if type(t) is tuple
                    else torch.empty_strided(fakes[t].shape, fakes[t].stride(), dtype=d, device=self.device) if type(t) is int
                    else t
                    for t, d in zip(targets, dtypes)
                ]
                ins = [outs[i[0]][i[1]] if type(i) is tuple else i for i in inputs]
                if isinstance(node, MemcpyNode):
                    o, recs = torch._C._cuda_hostTraceCopy_(targets[0], ins[0])
                    outs.append([o])
                    records += recs
                    continue
                name, function, offsets, image = (node.name, node.function, [o for o, _ in node.layout], list(images[id(node)])) if node else ("", 0, [], [])
                try:
                    o, recs = torch._C._cuda_hostTracePointwise(targets, dtypes, ins, common, dynamic, name, function, offsets, image, addresses.get(id(node), []))
                except NotImplementedError as e:
                    declines.append(str(e).splitlines()[0])
                    raise
                outs.append(o)
                if node is not None and id(node) in scalars:
                    members = tuple((p, at, c, cpu) for p, at, c in scalars[id(node)])
                    recs = [(*r, members) if len(r) == 7 else r for r in recs]
                records += recs
            rets = [outs[f[0]][f[1]] if type(f) is tuple else f if f is not None else torch.empty_strided(fakes[k].shape, fakes[k].stride(), dtype=fakes[k].dtype, device=self.device) for k, f in enumerate(finals)]
            return tuple(rets) if len(rets) > 1 else rets[0], records

        tensors = [a for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, _TracedTensor)]
        out_t = self._run_host(func, host, tensors, (args, kwargs) if isinstance(func, OpOverload) else None, check)
        if out_t is None or isinstance(out_t, str):
            return declines[0] if declines else out_t or (stale[0] if stale else "the host's launch at the hints is not the witness's")
        return out_t

    def _check_as_strided(self, root: _Root, v: torch.Tensor) -> None:
        # eager's setStrided checks (checkAsStridedArgs, checkInBoundsForStorage),
        # which the fake kernel skips for symbolic values
        sizes, strides, offset = list(v.shape), list(v.stride()), v.storage_offset()
        where = f"sizes {[_hint(s) for s in sizes]}, strides {[_hint(s) for s in strides]}, storage offset {_hint(offset)}"
        if not _guard_each([st >= 0 for st in strides]):
            raise RuntimeError(
                f"as_strided: Negative strides are not supported ({where})"
            )
        if not bool(offset >= 0):
            raise RuntimeError(f"Tensor: invalid storage offset ({where})")
        if any(bool(s == 0) for s in sizes):
            return
        needed = _storage_nbytes(sizes, strides, offset, v.dtype.itemsize)
        if root.kind == "allocation":
            a = next(a for a in self.allocs if a.root is root)
            bound = _storage_nbytes(a.sizes, a.strides, 0, a.dtype.itemsize)
            if not bool(needed <= bound):
                raise RuntimeError(
                    f"setStorage: {where} are out of bounds for the storage"
                )
            return
        # neither an input's nor an eager output's storage size is part of the
        # trace: the bound is the bytes up to the input's or output's end
        if root.kind == "argument":
            t = next(i for i in self.inputs if i.root is root)
            sizes, strides, offset, what = t.sizes, t.strides, t.offset, t.name
        else:
            t = next(o for o in self.eager_outputs if o._root is root)
            sizes, strides, offset = list(t.shape), t._sym_strides, t._sym_offset
            what = f"eager output {root.name}"
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
        return None
    if (kernel := opdef._backend_fns.get("cuda")) is not None:
        return kernel
    # a CUDA kernel registered outside the custom op runs ahead of its
    # CompositeExplicitAutograd one
    if torch._C._dispatch_has_kernel_for_dispatch_key(func.name(), "CUDA"):
        return None
    return opdef._backend_fns.get(None)


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


def _hint(v: Any) -> Any:
    return v.node.hint if isinstance(v, _SYM_TYPES) else v


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


def _guard_each(terms: list) -> bool:
    # One guard per term rather than one conjunction: a True answer records
    # every term, a False answer only the first failing term's negation.
    for term in terms:
        if not bool(term):
            return False
    return True


def _storage_nbytes(sizes: list, strides: list, offset: Any, itemsize: int) -> Any:
    # at::detail::computeStorageNbytes, each zero-size test a guard
    extent: Any = 1
    for size, stride in zip(sizes, strides):
        if bool(size == 0):
            return 0
        extent = extent + stride * (size - 1)
    return (offset + extent) * itemsize


def _contiguous_strides(sizes: list) -> list:
    # c10::contiguous_strides: a zero size counts as 1 in the products (guarded
    # on the traced value so no other size's expression changes)
    strides: list = [1] * len(sizes)
    for d in range(len(sizes) - 2, -1, -1):
        size = sizes[d + 1]
        if _hint(size) == 0 and bool(size == 0):
            size = 1
        strides[d] = strides[d + 1] * size
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


def _dense_terms(sizes: list, strides: list) -> list:
    # dense in the permutation the strides have at the trace
    order = sorted(
        range(len(sizes)), key=lambda d: (_hint(sizes[d]) < 2, _hint(strides[d]))
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


def _add(a: torch.Tensor, b: torch.Tensor, alpha: Any) -> Any:
    if type(alpha) not in (int, float):
        raise NotImplementedError("an add of a non-number alpha")
    return torch._C._cuda_hostTraceAdd(a, b, alpha)


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


def _to_copy(x: torch.Tensor, dtype: Any, layout: Any, device: Any, pin_memory: Any, non_blocking: bool, memory_format: Any) -> Any:
    if layout not in (None, torch.strided) or device not in (None, x.device) or pin_memory:
        raise NotImplementedError("a _to_copy to another layout or device, or pinned")
    return torch._C._cuda_hostTraceToCopy(x, x.dtype if dtype is None else dtype, memory_format or torch.preserve_format)


# fill_, untagged in ATen: a FillFunctor gpu_kernel, which writes self and reads
# no tensor; fill is empty_like and fill_
_NULLARY = (aten.fill_, aten.fill)


@functools.cache
def _pointwise(func: OpOverload) -> bool:
    # ATen's declaration (native_functions.yaml's pointwise tag) of the op or
    # of an overload of it: an elementwise kernel, whose functor holds no
    # shape. Which kernel it is the witness shows. An in-place op, often
    # untagged, is its functional op's kernel into self
    packet = func.overloadpacket
    name = func._schema.name.split("::")[1]
    packets = [packet, getattr(aten, name[:-1])] if name.endswith("_") and hasattr(aten, name[:-1]) else [packet]
    return func.namespace == "aten" and (packet in _NULLARY or any(torch.Tag.pointwise in getattr(p, n).tags for p in packets for n in p.overloads()))


def _cpu_scalar_probes(dtype: torch.dtype) -> list[torch.Tensor]:
    """The values a pointwise witness runs a CPU buffer of dtype at: the
    first the trace's."""
    if dtype is torch.bool:
        values: list[Any] = [True, False]
    elif dtype.is_complex:
        values = [0.3 + 0.7j, -2.75 + 1j, 0, 1, -1j, 2, 0.5 - 0.5j]
    elif dtype.is_floating_point:
        values = [0.3, -2.75, 0.0, 1.0, -1.0, 2.0, 0.5]
    elif dtype.is_signed:
        values = [3, -3, 5, 0, 1, -1, 2]
    else:
        values = [3, 5, 0, 1, 2, 7, 200]
    with _disable_current_modes():
        return [torch.tensor(v, dtype=dtype) for v in values]


def _meta_at_hints(func: OpOverload, args: tuple, kwargs: dict) -> Any:
    def at_hints(t: _TracedTensor) -> torch.Tensor:
        return torch.empty_strided([_hint(n) for n in t.shape], [_hint(n) for n in t._sym_strides], dtype=t.dtype, device="meta")

    # a factory's device too: at its own device it would launch, outside the trace
    meta = pytree.tree_map_only(torch.device, lambda _: torch.device("meta"), pytree.tree_map_only(_SYM_TYPES, _hint, (args, kwargs)))
    with _disable_current_modes():
        meta = pytree.tree_map_only(_TracedTensor, at_hints, meta)
        return func(*meta[0], **meta[1])


def _elementwise(func: OpOverload, args: tuple, kwargs: dict) -> bool:
    # an op ATen does not tag pointwise (floor_divide, complex, zero_) whose
    # output has its tensors' broadcast shape: the pointwise host's witness
    # decides the rest, one TensorIterator kernel per iterator over the op's
    # tensors
    if func.namespace != "aten" or {torch.Tag.dynamic_output_shape, torch.Tag.data_dependent_output} & set(func.tags):
        return False
    if not func._schema.returns or not all(isinstance(r.type, torch.TensorType) for r in func._schema.returns):
        return False
    # at the hints; the pointwise host guards its broadcast
    try:
        rets = pytree.tree_leaves(_meta_at_hints(func, args, kwargs))
    except Exception:
        return False
    if not rets or not all(isinstance(r, torch.Tensor) for r in rets):
        return False
    shapes = [[_hint(n) for n in a.shape] for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)]
    out = list(rets[0].shape)

    def fits(s: list) -> bool:
        return len(s) <= len(out) and all(n in (m, 1) for n, m in zip(reversed(s), reversed(out)))

    # an empty operand that does not broadcast to the output is in none of its iterators (an unused buffer)
    shapes = [s for s in shapes if all(s) or fits(s)]
    if not all(fits(s) for s in shapes):
        return False
    # each output size is an operand's, but a factory's (full): the witness decides
    return not shapes or all(m == 1 or any(len(s) >= d and s[-d] == m for s in shapes) for d, m in enumerate(reversed(out), 1))


@functools.cache
def _kernel_less(func: OpOverload) -> bool:
    name = func.name()
    keys = ("CompositeExplicitAutograd", "CompositeExplicitAutogradNonFunctional")
    return func.namespace == "aten" and not _has_kernel(name, "CUDA") and any(_has_kernel(name, k) for k in keys)


@functools.cache
def _composite(func: OpOverload) -> bool:
    name = func.name()
    return not torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA") and torch._C._dispatch_has_kernel_for_dispatch_key(name, "CompositeExplicitAutograd")


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
def _out_overload(func: OpOverload) -> tuple[OpOverload, str] | None:
    """func's out= overload of a CUDA kernel, and its out argument's name."""
    want = [(a.name, str(a.type)) for a in func._schema.arguments]
    for name in func.overloadpacket.overloads():
        op = getattr(func.overloadpacket, name)
        outs = [a.name for a in op._schema.arguments if a.is_out]
        rest = [(a.name, str(a.type)) for a in op._schema.arguments if not a.is_out]
        if len(outs) == 1 and rest == want and torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "CUDA"):
            return op, outs[0]
    return None


# an op's traced host (torch._C._cuda_hostTrace*) on its schema-ordered args
_TRACED_ATEN: dict[OpOverload, Callable[..., Any]] = {
    aten.mul.Tensor: lambda a, b: torch._C._cuda_hostTraceMul(a, b),
    aten.add.Tensor: _add,
    aten.silu.default: lambda a: torch._C._cuda_hostTraceSilu(a),
    aten.gelu.default: lambda a, approximate: torch._C._cuda_hostTraceGelu(a, approximate),
    aten.rsqrt.default: lambda a: torch._C._cuda_hostTraceRsqrt(a),
    aten.where.self: lambda cond, a, b: torch._C._cuda_hostTraceWhere(cond, a, b),
    aten.copy_.default: lambda dst, src, non_blocking: torch._C._cuda_hostTraceCopy_(dst, src),
    aten._to_copy.default: _to_copy,
    aten.clone.default: lambda x, memory_format: torch._C._cuda_hostTraceToCopy(x, x.dtype, memory_format or torch.preserve_format),
    aten.sum.default: lambda x, dtype: _reduce(torch._C._cuda_hostTraceSum, x, [], False, dtype),
    aten.sum.dim_IntList: lambda x, dim, keepdim, dtype: _reduce(torch._C._cuda_hostTraceSum, x, dim, keepdim, dtype),
    aten.mean.default: lambda x, dtype: _reduce(torch._C._cuda_hostTraceMean, x, [], False, dtype),
    aten.mean.dim: lambda x, dim, keepdim, dtype: _reduce(torch._C._cuda_hostTraceMean, x, dim, keepdim, dtype),
    aten.amax.default: lambda x, dim, keepdim: _reduce(torch._C._cuda_hostTraceAmax, x, dim, keepdim),
    aten.amin.default: lambda x, dim, keepdim: _reduce(torch._C._cuda_hostTraceAmin, x, dim, keepdim),
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

_ALLOC_OPS = {
    aten.empty.memory_format,
    aten.empty_strided.default,
    aten.empty_like.default,
    aten.new_empty.default,
    aten.new_empty_strided.default,
}


def _is_non_overlapping_and_dense(t: _TracedTensor) -> bool:
    # c10 answers every other metadata query of a traced tensor from its
    # symbols, each condition a guard; this one is an operator call
    return _guard_each(_dense_terms(list(t.shape), t._sym_strides))


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
        if (tr := current_trace()) is not None and tr.tensors is not None:
            tr.tensors.add(t)
        return t

    def __repr__(self, *, tensor_contents=None) -> str:
        return f"_TracedTensor({self._root.name}, {tuple(self.shape)}, {tuple(self._sym_strides)})"

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
        if func is aten.is_non_overlapping_and_dense.default:
            return _is_non_overlapping_and_dense(args[0])
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
        kwargs = kwargs or {}
        tr = self.trace
        if current_trace() is not tr:
            raise tr.decline(f"{func} on a thread other than the trace's")
        if func is aten.is_non_overlapping_and_dense.default:
            return _is_non_overlapping_and_dense(args[0])
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
        tr.order.append(_order_key(func, args, kwargs, out, _traced_layout))
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
            return self.trace.allocate(func, args, kwargs)
        if func is aten.zero_.default and self.trace.zero(args[0]):
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
        # an empty argument's trace launches nothing for it: a size is zero or not at replay as at the trace
        for size in traced[i].shape:
            bool(size != 0) if _hint(size) else bool(size == 0)
    for i in int_positions:
        traced[i] = tr.int_input(i, args[i])
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


# thread-local, unlike capture_tape's relaxed capture: here the capture is the
# detector, so an operation it refuses must raise on this thread
@contextlib.contextmanager
def _capture(device: torch.device) -> Iterator[None]:
    from cuda.bindings import runtime as _cuda_runtime

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.stream(torch.cuda.Stream(device)), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "The CUDA Graph is empty")
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                yield
            except BaseException:
                with contextlib.suppress(Exception):
                    graph.capture_end()
                raise
            graph.capture_end()
    except torch.AcceleratorError as e:
        if getattr(e, "error_code", None) not in _CAPTURE_ERRORS:
            raise
        raise declined(
            "the host performed an operation a stream capture does not permit "
            f"({str(e).splitlines()[0]}); it would run at the trace and never in a replay"
        ) from e
    raw = graph.raw_cuda_graph()
    _, nodes = _check_cuda_bindings(_cuda_runtime.cudaGraphGetNodes(raw, numNodes=0))
    if nodes:
        raise declined(
            f"the host enqueued {nodes} operations the trace does not record"
        )


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


def _cow_on() -> None:
    torch._C._is_cow_tensor = _traced_is_cow_tensor


def _cow_off() -> None:
    torch._C._is_cow_tensor = _is_cow_tensor


_cow_hold = ProcessHold(_cow_on, _cow_off)

_jiterator_launch = torch._C._cuda_jiterator_compile_and_launch_kernel


def _traced_jiterator_launch(code: str, name: str, return_by_ref: bool, num_outputs: int, tensors: tuple, kwargs: dict | None = None) -> Any:
    tr = current_trace()
    if tr is None:
        return _jiterator_launch(code, name, return_by_ref, num_outputs, tensors, *(() if kwargs is None else (kwargs,)))
    return tr.jiterator(code, name, return_by_ref, num_outputs, tensors, kwargs)


def _jiterator_on() -> None:
    torch._C._cuda_jiterator_compile_and_launch_kernel = _traced_jiterator_launch


def _jiterator_off() -> None:
    torch._C._cuda_jiterator_compile_and_launch_kernel = _jiterator_launch


_jiterator_hold = ProcessHold(_jiterator_on, _jiterator_off)


def _real_layout(t: torch.Tensor) -> tuple:
    if t.layout != torch.strided:
        return (t.layout,)
    return tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype, t.device


def _traced_layout(t: torch.Tensor) -> tuple:
    if not isinstance(t, _TracedTensor):  # a host buffer
        return _real_layout(t)
    sizes = tuple(_hint(s) for s in t.shape)
    strides = tuple(_hint(s) for s in t._sym_strides)
    return sizes, strides, _hint(t._sym_offset), t.dtype, t.device


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


def _call_key(func: Any, args: tuple, kwargs: dict, layout: Callable) -> tuple:
    # an operator call by its tensor arguments' layouts and int arguments
    key: list[Any] = [func]
    for a in pytree.tree_leaves((args, kwargs)):
        if isinstance(a, torch.Tensor):
            key.append(_layout_key(layout(a)))
        elif isinstance(a, (int, torch.SymInt, torch.SymBool)):
            key.append(_hint(a))
    return tuple(key)


def _order_key(func: Any, args: tuple, kwargs: dict, out: Any, layout: Callable) -> tuple:
    # a call of fn's by its key and its tensor outputs' layouts
    outs = tuple(_layout_key(layout(o)) for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor))
    return _call_key(func, args, kwargs, layout), outs


def _order_repr(key: tuple | None) -> str:
    return "no call" if key is None else f"{key[0][0]}{key[0][1:]} returning {key[1]}"


class _Witness(TorchDispatchMode):
    """The warm-up's operator calls: per call key, in call order, the layouts
    of the tensors each returned."""

    @classmethod
    def _should_skip_dynamo(cls):
        return False

    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[tuple, list[list[tuple]]] = {}
        self.order: list[tuple] = []
        # per call its tensor outputs, weakly; after the warm-up those it
        # kept or returned, a leaked traced tensor's twin (trace())
        self.outputs: list[list[Any]] = []

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func in HOST_SEED_OFFSET:
            # the warm-up returns what a replay does
            out = tuple(seed_offset_on_device(func, list(out), args[0].device))
        if func is aten.detach.default:
            return out
        key = _call_key(func, args, kwargs, _real_layout)
        tensors = [o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)]
        self.calls.setdefault(key, []).append([_real_layout(o) for o in tensors])
        if func in _ALLOC_OPS or func.is_view:
            return out
        self.order.append(_order_key(func, args, kwargs, out, _real_layout))
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
    hosts = {id(call) for _, call in tr.aten_calls}
    for _, call in heapq.merge(tr.launches, tr.aten_calls, key=operator.itemgetter(0)):
        if isinstance(call, EagerCall) and not isinstance(call.target, tuple):
            key = _call_key(call.target, call.args, call.kwargs, _traced_layout)
            keyed.append((key, call))
    counts = collections.Counter(key for key, _ in keyed)
    seen: collections.Counter = collections.Counter()
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
            want = _traced_layout(o)
            if got[i] != want and _addressing(got[i]) == _addressing(want):
                # predict the op's own size-1 strides, which a replay checks
                o._sym_strides = [r if n == 1 else s for n, r, s in zip(want[0], got[i][1], o._sym_strides)]
            elif got[i] != want:
                e = declined(
                    f"{call.name} output {i} is (sizes, strides, storage offset, dtype, device) "
                    f"{got[i]} at the warm-up; {what} predicted {want}"
                )
                e.meta_op = None if id(call) in hosts else call.target
                raise e
    for k, (got, want) in enumerate(itertools.zip_longest(witness.order, tr.order) if tr.trusted is None else ()):
        if got != want:
            e = declined(f"fn's call {k} is {_order_repr(want)} in the trace; at the warm-up {_order_repr(got)}")
            e.witness = True
            raise e


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
) -> Tape:
    """Trace one call fn(*args). Tensor arguments become traced tensors and
    int arguments symbols; every other argument is a constant of the tape.
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
    `check_escapes` a traced tensor fn keeps past the trace declines. The
    top-level ops at the indices `eager_ops` maps (in Tape.ops) are eager
    calls, for the reason it maps them to."""
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
    device = args[positions[0]].device
    exact = (torch.Tensor,) if trusted is None else (torch.Tensor, torch.nn.Parameter)
    for i in positions:
        a = args[i]
        if isinstance(a, _TracedTensor):
            raise declined(f"arg{i} is a traced tensor of another trace")
        if type(a) not in exact:
            raise declined(
                f"arg{i} is a {type(a).__name__}; only torch.Tensor arguments are traced"
            )
        if not a.is_cuda or a.device != device:
            raise declined(
                f"arg{i} is on {a.device}; only CUDA tensors on one device are traced"
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
            with _Witness() as witness:
                result = fn(*args)
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
            with _gc_hold, _capture(device), intercepting(), cute_intercepting():
                tr = _Trace(device, trusted, opaque, static_shapes, check_escapes, eager_ops)
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
                    if tensors:
                        # a cycle fn left for the collector is no reference
                        gc.collect()
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
            raise d from e
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
    arguments: per tensor its exact type, dtype, device, rank, layout,
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
            kinds.append((type(a), a.dtype, a.device, a.dim(), a.layout, a.is_neg(), a.is_conj(), rg, rg and grad))
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
        self.sites = tr.sites
        self.outputs = outputs
        self.result_kind = result_kind  # "tensor", "tuple", "list" or "none"
        # how many conditions the host branched on (`guards`)
        env = tr.shape_env
        self.guard_count = len(env.records if isinstance(env, _ir.Env) else env.guards)
        # per guard the op that owns it (OpRec.guards), or None: graph-level
        self.owners = list(tr.shape_env.owners)
        self.ops = tr.ops
        owned: list[list[int]] = [[] for _ in self.ops]
        for i, o in enumerate(self.owners):
            if o is not None:
                owned[o].append(i)
        for op, guards in zip(self.ops, owned):
            op.guards = tuple(guards)
        # (i, j): arguments i < j, disjoint at the trace, where a step not run
        # eagerly writes one and reads the other. A call where they overlap
        # runs eagerly; trusted inputs' aliasing is the caller's (none)
        self.argument_pairs = tuple(sorted(tr.argument_pairs)) if tr.trusted is None else ()

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


def _stand_in(meta: tuple, device: torch.device) -> torch.Tensor:
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


def bind_opaque(tape: Tape, calls: set[int], learn: Callable[..., object] | None = None) -> tuple[Tape, dict[int, KeyedSite]] | None:
    """The tape trace() records at the tape's call where the OpaqueCalls of
    ids `calls` whose keys bind now, after learn(op, provider, (spec, tensor
    positions), key) for each that draws no RNG and does not, had bound then
    (their KeyedSites, their fresh outputs allocations), and each such call's
    site by the call's id;
    the tape's args are the traced call's, or stand-ins at their released
    metadata. None where that trace differs beyond those calls: a key it
    refuses, a guard the tape lacks, a zero_ of a fresh output (a Memset), a
    fresh output at a symbolic storage offset, or an address not its root's
    symbol plus an offset."""
    env = tape.shape_env
    tr = _Trace(tape.device)
    tr.shape_env = env
    # after the tape's allocation symbols, whose hints are its placeholder addresses
    tags = [(_hint(a.q) * _ALLOC_ALIGNMENT ^ _ALLOC_TAG) >> _ALLOC_SHIFT for a in tape.allocs]
    tr.first_alloc = max(tags, default=0)
    marks, owners = len(env.owners), list(env.owners)
    # each launch's and allocation's index in the new tape, for the ops' spans
    at_launch = {id(rec): i for i, (_, rec) in enumerate(tape.launches)}
    at_alloc = {id(a): j for j, a in enumerate(tape.allocs)}
    launch_start, alloc_start = [0] * (len(tape.launches) + 1), [0] * (len(tape.allocs) + 1)
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
                    if learn is not None and rec.generator is None and rec.provider.bind(key := trace_key(rec.target, args, kwargs, fresh)[0]) is None:
                        learn(rec.target, rec.provider, call, key)
                    binding, values, refusal = bind_at_trace(rec.provider, rec.target, args, kwargs, fresh)
                if refusal is not None:
                    return None
                if binding is not None and all(type(t.storage_offset()) is int for t in fresh):
                    _guard_each([v == _hint(v) for v in values])
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
                        converted[id(rec)] = record_binding(tr, rec.target, rec.provider, binding, operands, scalars, call)
                    sites.append(converted[id(rec)])
                    allocs += tr.allocs[first:]
                    launches += tr.launches
                    shift = next(tr._seq) - seq - 1
                    continue
            new = remap(rec)
            if new is None:
                return None
            records[id(rec)] = new
            launches.append((seq + shift, new))
        if len(env.owners) != marks or not converted or _guarded_symbols(tape) & {s.name for s in gone}:
            return None
    except Declined:
        return None
    finally:
        # a guard the binding evaluated again outside any op is graph-level,
        # as in the trace; the tape's env keeps its own record
        owners, env.owners[:marks] = env.owners[:marks], owners
        env.forget(marks)
    launch_start[-1], alloc_start[-1] = len(launches), len(allocs)
    ops = []
    for k, op in enumerate(tape.ops):
        call, outputs = pytree.tree_map_only(_TracedTensor, tensor, (op.call, op.outputs))
        spans = range(launch_start[op.launches.start], launch_start[op.launches.stop]), range(alloc_start[op.allocs.start], alloc_start[op.allocs.stop])
        ops.append(replace(op, call=call, outputs=outputs, launches=spans[0], allocs=spans[1], guards=tuple(i for i, o in enumerate(owners) if o == k)))

    order = {id(a.root): a.seq for a in allocs}
    for s in tape.sites:
        operands = tuple(map(tensor, s.operands))
        sites.append(replace(s, operands=operands, nodes=tuple(records.get(id(n), n) for n in s.nodes)))
    sites.sort(key=lambda s: min(order[id(t._root)] for t, _ in s.scratch.values()))
    out = copy.copy(tape)
    out.allocs, out.launches, out.sites, out.ops, out.owners = allocs, launches, sites, ops, owners
    out.outputs = [replace(o, root=roots[id(o.root)]) if isinstance(o, _OutputRec) and id(o.root) in roots else o for o in tape.outputs]
    positions = {rec.position for rec in tape.inputs}
    out.args = tuple(_stand_in(a, tape.device) if i in positions and not isinstance(a, torch.Tensor) else a for i, a in enumerate(tape.args))
    out.warm_up_result = None
    return out, converted
