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
import functools
import gc
import itertools
import struct
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

import sympy

import torch
from torch._ops import OpOverload
from torch._prims.rng_prims import _impl_graphsafe_rng, graphsafe_run_with_rng_state
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda._host_trace import _TraceShapeEnv, Declined, declined, ProcessHold
from torch.cuda._host_trace_opaque import bind_at_trace, record_binding
from torch.cuda._utils import _check_cuda_bindings
from torch.fx.experimental import _config as fx_config
from torch.fx.experimental.symbolic_shapes import free_symbols, free_unbacked_symbols
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes, TorchDispatchMode
from torch.utils._sympy.value_ranges import ValueRanges


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

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


@dataclass(frozen=True)
class TrustedInputs:
    """What the caller vouches for at every call (under Inductor, Dynamo's
    guards and Inductor's own input handling): the trace records no guard a
    fact below decides, and a replay checks none of it.

    `layouts[i]` is tensor argument i's (sizes, strides), each an int or a
    sympy expression over the caller's symbols, or int argument i's int or
    expression; `ranges` bounds the caller's symbols. Only the caller's symbols
    are symbols of the trace; every other size and stride is a constant. An
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


@dataclass(frozen=True)
class OpaqueCall(EagerCall):
    """An eager call `provider` accepted: a replay binds its key
    (_host_trace_opaque)."""

    provider: Any = None


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
    ) -> None:
        self.device = device
        self.trusted = trusted
        self.opaque = opaque
        # the trace's symbol for each of the caller's, under trusted inputs
        self.given: dict[sympy.Symbol, torch.SymInt] = {}
        # the trace's capturing stream, the only one a launch may be recorded on
        cuda = device.type == "cuda"
        self.stream = torch.cuda.current_stream(device) if cuda else None
        self.shape_env = _TraceShapeEnv(trusted=trusted is not None)
        # computes a view's and an eager call's metadata; without its cache
        # every guard a kernel evaluates is recorded, and without fallback
        # kernels an op with no meta raises instead of running on zeros
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
        self.launches: list[tuple[int, Any]] = []  # (seq, launch or EagerCall)
        self.eager_outputs: list[_TracedTensor] = []
        self.sites: list[KeyedSite] = []  # the calls bound at trace time
        # the CPU tensors the host's steps write, by storage: the trace's own,
        # each step writing them again at a replay
        self.host: dict[int, torch.Tensor] = {}
        # inside graphsafe_run_with_rng_state, the generator its op draws from
        self.generator: torch.Generator | None = None
        # the first decline; kept so that host code that catches it (a
        # try/except around an op) cannot make the trace succeed
        self.declined: Declined | None = None
        # inside a traced ATen host (_traced_aten), whose own ops are not routed again
        self.in_aten = False

    def decline(self, msg: str) -> Declined:
        e = declined(msg)
        if self.declined is None:
            self.declined = e
        return e

    def record_launch(self, record: Any) -> None:
        self.launches.append((next(self._seq), record))

    def input(self, position: int, t: torch.Tensor) -> _TracedTensor:
        env = self.shape_env
        name = f"arg{position}"
        sizes: list[Any] = []
        strides: list[Any] = []
        trusted = self.trusted
        if trusted is None:
            for d in range(t.dim()):
                sizes.append(env.symbol(t.size(d), f"{name}.size({d})", positive=True))
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
        # const_data_ptr leaves a copy-on-write input lazy
        base = t.const_data_ptr() - t.storage_offset() * t.element_size()  # type: ignore[attr-defined]
        sym = env.symbol(_PLACEHOLDER_TAG | (base & _PLACEHOLDER_LOW), f"{name}.base")
        root = _Root(f"p{position}", sym)
        first = t.const_data_ptr()  # type: ignore[attr-defined]
        last = first + (sum((n - 1) * s for n, s in zip(t.shape, t.stride())) + 1) * t.element_size() - 1
        rec = _InputRec(position, name, t.dtype, sizes, strides, offset, root, (first, last))
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
            elif _guard_each(_dense_terms(sizes, src_strides)):
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
        k = len(self.allocs)
        name = f"alloc{k}"
        hint = (_ALLOC_TAG | ((k + 1) << _ALLOC_SHIFT)) // _ALLOC_ALIGNMENT
        q = self.shape_env.symbol(hint, f"{name}.base/{_ALLOC_ALIGNMENT}")
        root = _Root(f"a{k}", _ALLOC_ALIGNMENT * q, "allocation")
        rec = _AllocRec(next(self._seq), name, sizes, strides, dtype, root, q)
        self.allocs.append(rec)
        return _TracedTensor(root, sizes, strides, 0, dtype, self.device)

    def zero(self, t: Any) -> bool:
        """Record t.zero_() as a Memset if t is a whole nonempty allocation
        on the trace's stream; otherwise it is an eager call."""
        root = t._root if isinstance(t, _TracedTensor) else None
        if root is None or root.kind != "allocation" or self.stream is None:
            return False
        if torch.cuda.current_stream(self.device) != self.stream:
            return False
        a = next(a for a in self.allocs if a.root is root)
        # compared as expressions, so no guard: a view other than the whole is not a memset
        exprs = [_sym_expr(v) for v in (*t.shape, *t._sym_strides, t._sym_offset)]
        if exprs != [_sym_expr(v) for v in (*a.sizes, *a.strides, 0)]:
            return False
        if any(_hint(s) == 0 for s in a.sizes):
            return False
        nbytes = _storage_nbytes(a.sizes, a.strides, 0, a.dtype.itemsize)
        self.record_launch(Memset(f"zero_ of {a.name}", (root.sym,), (root,), 0, 1, nbytes, 1, nbytes))
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
        with self.fake_mode:
            try:
                out = func(self._twin(src), *args[1:], **kwargs)
            except Exception as e:
                # the fake kernel's error need not be eager's type; host code
                # that caught it could take a path eager does not
                raise self.decline(f"{func} raised {type(e).__name__}: {e}") from e
        if func is aten.as_strided.default and self.trusted is None:
            self._check_as_strided(src._root, out)

        def wrap(o: torch.Tensor) -> _TracedTensor:
            if o.layout != torch.strided or o.is_conj() or o.is_neg():
                raise self.decline(f"{func} is not a plain strided view")
            sizes, strides, offset = list(o.shape), list(o.stride()), o.storage_offset()
            return _TracedTensor(src._root, sizes, strides, offset, o.dtype, src.device)

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

    def host_step(self, fn: Callable[..., Any], args: tuple, kwargs: dict, name: str) -> None:
        """fn(*args, **kwargs) over the trace's host buffers and constants:
        host code that reads no device memory, which a replay runs before its
        first graph, in the host's order (as eager draws from the CPU
        generator)."""
        for a in pytree.tree_leaves((args, kwargs)):
            if isinstance(a, torch.Tensor) and not self._is_host(a):
                raise self.decline(f"{name} of a tensor other than a CPU buffer of the trace")
            if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                raise self.decline(f"{name} of a {type(a).__name__}")
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
        self.host_step(functools.partial(_run_host_op, func, tuple(held)), args, kwargs, str(func))
        return result[0] if len(schema.returns) == 1 else type(out)(result)

    def eager_call(self, func: Any, args: tuple, kwargs: dict) -> Any:
        if not isinstance(func, OpOverload):
            raise self.decline(f"{func} is not an operator")
        if func is aten._local_scalar_dense.default:
            raise self.decline(f"{func} reads a traced tensor's value on the host")
        packet = func.overloadpacket
        if torch.Tag.inplace_view in func.tags or packet in _METADATA_OPS:
            raise self.decline(f"{func} changes a traced tensor's metadata")
        if self.stream is not None:
            if torch.cuda.current_stream(self.device) != self.stream:
                raise self.decline(f"{func} on a stream other than the trace's")
            if (backend := torch.cuda.get_allocator_backend()) != "native":
                raise self.decline(f"{func} under the {backend} allocator")
        leaves = pytree.tree_leaves((args, kwargs))
        if all(self._is_host(a) for a in leaves if isinstance(a, torch.Tensor)):
            if (result := self._host_op(func, args, kwargs)) is not NotImplemented:
                return result
        for a in leaves:
            if isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor) and not self._is_host(a):
                raise self.decline(f"{func} of a tensor the trace does not track")
            if isinstance(a, (torch.SymFloat, torch.SymBool)):
                raise self.decline(f"{func} of a {type(a).__name__}")
        with self.fake_mode:
            twins = pytree.tree_map_only(_TracedTensor, self._twin, (args, kwargs))
            twins = pytree.tree_map_only(torch.Tensor, lambda t: self._host_fake(t) if self._is_host(t) else t, twins)
            try:
                out = func(*twins[0], **twins[1])
            except Exception as e:
                why = f"{type(e).__name__}: {str(e).splitlines()[0]}"
                raise self.decline(f"{func} has no traced metadata ({why})") from e
        if func is aten._scaled_dot_product_cudnn_attention.default:
            lse = args[4] if len(args) > 4 else kwargs["compute_log_sumexp"]
            if not lse:
                # the fake kernel returns a log-sum-exp the CUDA kernel does
                # not. A metadata fix, not an attention knob: it holds for an
                # eager step as for a bound call, so it is always on
                out = (out[0], None, *out[2:])
        schema = func._schema
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
            if o.device != self.device:
                raise self.decline(f"{func} returns a tensor on {o.device}")
            if o.layout != torch.strided or o.is_conj() or o.is_neg():
                raise self.decline(f"{func} returns a tensor that is not plain strided")

        # the argument each written alias set names
        written: dict[frozenset, Any] = {}
        for i, a in enumerate(schema.arguments):
            if a.alias_info is None or not a.alias_info.is_write:
                continue
            v = args[i] if i < len(args) else kwargs.get(a.name)
            if isinstance(v, torch.Tensor) and self._is_host(v):
                raise self.decline(f"{func} writes a CPU buffer on the device's stream")
            written[frozenset(a.alias_info.before_set)] = v

        reasons, provider, binding = [], None, None
        # a library kernel may assume its output overlaps no input. Arguments
        # may overlap at one call and not at another: a step not run eagerly
        # holds only while its written arguments overlap no other operand
        roots = [t._root for t in pytree.tree_leaves((args, kwargs)) if isinstance(t, _TracedTensor)]
        writes = [self.arguments[id(v._root)] for v in written.values() if isinstance(v, _TracedTensor) and id(v._root) in self.arguments]
        reads = {id(r): self.arguments[id(r)] for r in roots if id(r) in self.arguments}.values()
        others = [(w, o) for w in writes for o in reads if o is not w]
        pairs = {(min(w.position, o.position), max(w.position, o.position)) for w, o in others}
        overlap = any(w.extent[0] <= o.extent[1] and o.extent[0] <= w.extent[1] for w, o in others)
        if overlap or any(roots.count(v._root) > 1 for v in written.values() if isinstance(v, _TracedTensor)):
            reasons.append(f"{func} writes a storage another operand is of")
        else:
            for p in self.opaque:
                if (why := p.accepts(func, args, kwargs)) is None:
                    provider = p
                    break
                reasons.append(why)
            if provider is None and func in _TRACED_ATEN and not self.in_aten:
                if (traced := self._traced_aten(func, args, kwargs, rets[0])) is not None:
                    self.argument_pairs |= pairs
                    return traced
        if provider is not None:
            fakes = [o for r, o in zip(schema.returns, rets) if r.alias_info is None]
            fakes = [o for o in pytree.tree_leaves(fakes) if isinstance(o, torch.Tensor)]
            binding, values, refusal = bind_at_trace(provider, func, args, kwargs, fakes)
            _guard_each([v == _hint(v) for v in values])
            if refusal is not None:
                provider, reasons = None, [refusal]
        if provider is not None:
            self.argument_pairs |= pairs
        made: list[_TracedTensor] = []  # the binding's fresh outputs

        storages = {
            t.untyped_storage()._cdata
            for t in pytree.tree_leaves(twins)
            if isinstance(t, torch.Tensor)
        }

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
            leaves = pytree.tree_leaves((args, kwargs))
            operands = [a for a in leaves if isinstance(a, torch.Tensor)] + made
            scalars = [a for a in leaves if not isinstance(a, torch.Tensor)]
            self.sites.append(record_binding(self, func, provider, binding, operands, scalars))
        elif provider is not None:
            self.record_launch(OpaqueCall(func, args, kwargs, outputs, generator=self.generator, provider=provider))
        else:
            why = "; ".join(reasons) or None
            self.record_launch(EagerCall(func, args, kwargs, outputs, why, self.generator))
        if len(schema.returns) == 1:
            return result[0]
        return type(out)(result) if result else None  # a tuple or a structseq

    def _traced_aten(self, func: OpOverload, args: tuple, kwargs: dict, fake: torch.Tensor) -> Any:
        """func's traced ATen host (_TRACED_ATEN): its output,
        allocated through this trace, and its kernels as KernelLaunches; None
        where the host declines, which leaves the call to EagerCall."""
        from torch.cuda._host_trace_launch import KernelLaunch

        rest = func._schema.arguments[len(args) :]
        args = (*args, *(kwargs.get(a.name, a.default_value) for a in rest))
        if any(isinstance(a, (torch.Tensor, *_SYM_TYPES)) and not isinstance(a, _TracedTensor) for a in args):
            return None
        # a Python number where the schema takes a Tensor (x + 1)
        if any(isinstance(s.type, torch.TensorType) and not isinstance(a, _TracedTensor) for s, a in zip(func._schema.arguments, args)):
            return None
        tensors = [a for a in args if isinstance(a, _TracedTensor)]
        marks = len(self.allocs), len(self.launches)
        self.in_aten = True
        try:
            with _TraceMode(self):
                out, records = _TRACED_ATEN[func](*args)
        except NotImplementedError:
            del self.allocs[marks[0] :], self.launches[marks[1] :]
            return None
        finally:
            self.in_aten = False
        if not isinstance(out, _TracedTensor):
            raise AssertionError(f"{func}'s traced host returned a {type(out)}")
        # a size-1 dim's stride is layout-free, and the fake's may differ from
        # eager's there; the traced host's is eager's
        pairs = [*zip(out.shape, fake.shape), (out._sym_offset, fake.storage_offset())]
        pairs += [(a, b) for a, b, n in zip(out._sym_strides, fake.stride(), fake.shape) if n != 1]
        if out.dtype != fake.dtype or out.dim() != fake.dim() or not _guard_each([a == b for a, b in pairs]):
            raise AssertionError(f"{func}'s traced host allocated {out!r}, its meta {fake.dtype} {fake.shape} {fake.stride()}")
        made = [a.root for a in self.allocs[marks[0] :]]
        roots = tuple({id(r): r for r in (*(t._root for t in (*tensors, out)), *made)}.values())
        for function, offsets, params, fields, grid, block, smem in records:
            places, values, is_pointer = [], [], []
            for param, offset, width, value, pointer in fields:
                places.append((param, offset, width))
                values.append(value)
                is_pointer.append(pointer)
            launch = KernelLaunch(
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
            )
            self.record_launch(launch)
        return out

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


def _hint(v: Any) -> Any:
    return v.node.hint if isinstance(v, _SYM_TYPES) else v


def _sym_expr(v: Any) -> sympy.Expr:
    return v.node.expr if isinstance(v, torch.SymInt) else sympy.Integer(v)


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


def _to_copy(x: torch.Tensor, dtype: Any, layout: Any, device: Any, pin_memory: Any, non_blocking: bool, memory_format: Any) -> Any:
    if layout not in (None, torch.strided) or device not in (None, x.device) or pin_memory:
        raise NotImplementedError("a _to_copy to another layout or device, or pinned")
    return torch._C._cuda_hostTraceToCopy(x, x.dtype if dtype is None else dtype, memory_format or torch.preserve_format)


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
}

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
        raise _declined(f"{func} on a traced tensor outside its trace")


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
        if current_trace() is not self.trace:
            raise self.trace.decline(f"{func} on a thread other than the trace's")
        if func is aten.is_non_overlapping_and_dense.default:
            return _is_non_overlapping_and_dense(args[0])
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
            return self.trace.view(func, args, kwargs)
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
        # trace declines an empty argument, so a replay's must be nonempty too
        for size in traced[i].shape:
            tr.shape_env.evaluate_expr(sympy.Ne(_sym_expr(size), 0))
    for i in int_positions:
        traced[i] = tr.int_input(i, args[i])
    _active.trace = tr
    try:
        # every symbol has its value, so no condition is decided size-obliviously
        with _TraceMode(tr), fx_config.patch(backed_size_oblivious=False):  # type: ignore[attr-defined]
            try:
                out = fn(*traced)
            except Exception as e:
                if tr.declined is not None and e is not tr.declined:
                    raise tr.declined from e
                raise
    finally:
        _active.trace = None
    if tr.declined is not None:
        raise tr.declined
    return out, traced


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


def _call_key(func: Any, args: tuple, kwargs: dict, layout: Callable) -> tuple:
    # an operator call by its tensor arguments' layouts and int arguments
    key: list[Any] = [func]
    for a in pytree.tree_leaves((args, kwargs)):
        if isinstance(a, torch.Tensor):
            key.append(_addressing(layout(a)))
        elif isinstance(a, (int, torch.SymInt)):
            key.append(_hint(a))
    return tuple(key)


class _Witness(TorchDispatchMode):
    """The warm-up's operator calls: per call key, in call order, the layouts
    of the tensors each returned."""

    @classmethod
    def _should_skip_dynamo(cls):
        return False

    def __init__(self) -> None:
        super().__init__()
        self.calls: dict[tuple, list[list[tuple]]] = {}

    def __torch_dispatch__(self, func, types, args: tuple = (), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        key = _call_key(func, args, kwargs, _real_layout)
        leaves = pytree.tree_leaves(out)
        layouts = [_real_layout(o) for o in leaves if isinstance(o, torch.Tensor)]
        self.calls.setdefault(key, []).append(layouts)
        return out


@graphsafe_run_with_rng_state.py_impl(_Witness)
def _graphsafe_rng_witnessed(mode: _Witness, op: Any, *args: Any, rng_state: Any = None, **kwargs: Any) -> Any:
    with mode:
        return _impl_graphsafe_rng(op, *args, rng_state=rng_state, **kwargs)


def _check_witness(launches: list[tuple[int, Any]], witness: _Witness) -> None:
    """Each eager call's fresh outputs, as its fake kernel predicted them at
    the hints, against what the operator returned at the warm-up. The k-th
    call of a key pairs with the warm-up's k-th; a key the warm-up called a
    different number of times (a lazy initialization) is not checked here,
    only at a replay."""
    keyed = []
    for _, call in launches:
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
        if len(got) != len(call.outputs):
            raise declined(
                f"{call.name} returned {len(got)} tensors at the warm-up; its fake kernel {len(call.outputs)}"
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
                    f"{got[i]} at the warm-up; its fake kernel predicted {want}"
                )
                e.meta_op = call.target
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
    the `opaque` providers accepts is an OpaqueCall."""
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
        if a.numel() == 0:
            raise declined(f"arg{i} is empty")
        if a.is_neg() or a.is_conj():
            raise declined(f"arg{i} is a negative or conjugate view")
    if current_trace() is not None:
        raise declined("a trace is already in progress on this thread")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise declined("the current stream is capturing")
        result, witness = None, None
        if warm_up:
            before = [_metadata(args[i]) for i in positions]
            with _Witness() as witness:
                result = fn(*args)
            for i, was in zip(positions, before):
                if _metadata(args[i]) != was:
                    e = declined(
                        f"the warm-up changed the metadata of arg{i} (a resize_, set_ or out= "
                        "inside the call); a trace would describe the changed call"
                    )
                    e.warm_up_ran, e.warm_up_result = True, result
                    raise e
        from torch.cuda._host_trace_cute import intercepting as cute_intercepting
        from torch.cuda._host_trace_triton_launch import intercepting

        try:
            with _gc_hold, _capture(device), intercepting(), cute_intercepting():
                tr = _Trace(device, trusted, opaque)
                out, traced = _symbolic_run(tr, fn, args, positions, int_positions)
                result_kind, outputs = _output_records(out, traced, positions)
                if witness is not None:
                    _check_witness(tr.launches, witness)
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
    return Tape(tr, args, outputs, result_kind, result)


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


def argument_contract(args: tuple) -> tuple:
    """What a call must match before any GPU work, as trace() classifies its
    arguments: per tensor its exact type, dtype, device, rank, layout and
    neg/conj bits, per int its kind, every other argument by value; then the
    global state."""
    kinds: list[Any] = []
    for a in args:
        if isinstance(a, torch.Tensor):
            kinds.append(
                (type(a), a.dtype, a.device, a.dim(), a.layout, a.is_neg(), a.is_conj())
            )
        elif type(a) is int:
            kinds.append(int)
        else:
            kinds.append(_constant(a))
    # the state an allocation or an eager op's output layout depends on
    fill = torch._C._get_deterministic_fill_uninitialized_memory()
    return (tuple(kinds), (torch.get_default_dtype(), torch.are_deterministic_algorithms_enabled(), fill))


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
        self.contract = argument_contract(args) if tr.trusted is None else ()
        self.inputs = tr.inputs
        self.int_inputs = tr.int_inputs
        self.allocs = tr.allocs
        self.launches = tr.launches
        self.sites = tr.sites
        self.outputs = outputs
        self.result_kind = result_kind  # "tensor", "tuple", "list" or "none"
        # every condition the host branched on, in program order; under
        # trusted inputs, only its size-based dispatch decisions
        self.guards = [g.expr for g in tr.shape_env.guards]
        # (i, j): arguments i < j, disjoint at the trace, where a step not run
        # eagerly writes one and reads the other. A call where they overlap
        # runs eagerly; trusted inputs' aliasing is the caller's (none)
        self.argument_pairs = tuple(sorted(tr.argument_pairs)) if tr.trusted is None else ()

    def release_args(self) -> None:
        """Keep only each tensor argument's metadata (sizes, strides, storage
        offset, dtype, base address), so the tape keeps no tensor alive."""
        self.args = tuple(
            _metadata(a) if isinstance(a, torch.Tensor) else a for a in self.args
        )
