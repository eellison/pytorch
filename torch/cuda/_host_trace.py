"""Host tracing for CUDA C++ kernels (private).

A traceable kernel host runs once with symbolic sizes and no data pointers, and
the recorder writes down what it did: the allocations it made, the branches it
took, and every kernel launch with each argument as a value over the input
shapes. Replaying that tape at a new shape evaluates those values, patches the
captured CUDA graph and launches it, without running the host again. A call
whose inputs fail a recorded branch is a miss, never a wrong answer.

    tape = trace(torch.layer_norm, (x, (x.shape[-1],), w, b, 1e-5))
    variant = build(tape, torch.layer_norm, (x, (x.shape[-1],), w, b, 1e-5))
    out = variant.replay((x2, (x2.shape[-1],), w, b, 1e-5))

The symbolic values are ordinary torch.SymInt / SymFloat / SymBool over a
ShapeEnv created for the trace; the C++ host sees them as c10::SymInt and every
branch it takes lands in the ShapeEnv's guards. A traced tensor is a wrapper
subclass on the CUDA device with symbolic sizes and no storage: its raw pointer
raises, its address is read in C++ through sym_const_data_ptr (inputs; in
ordinary mode const_data_ptr, so a copy-on-write input stays lazy) or
sym_mutable_data_ptr (outputs and in-place operands). Views made inside the
host are computed on FakeTensor twins over the same ShapeEnv.

Only the CUDA hosts that were converted to the recorder's types can be traced
(_TRACEABLE); everything else declines by name. Traced and replayed outputs
carry no requires_grad or grad_fn. As with any CUDA graph capture, no thread
may call torch.cuda.synchronize() while a trace or a build is in progress.

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
function writes holds the warm-up's value. torch.cuda.caching_allocator_alloc
inside a trace is invisible to the recorder: such a pointer can only reach a
kernel as a constant.

Every branch on a size is guarded on the value the trace saw, including the
size-1 branches of the view code: a squeeze of a symbolic dim traced at size 1
pins the tape to size 1, and traced at size 8 misses at size 1. Contiguity is
the one exception by construction: it is asked as a single shape-generic
predicate (a size-1 dim has any stride), so a plain layer norm traced at batch
1 serves every batch.

The argument contract is part of the tape: the arity, which positions hold
tensors, and every other argument by value and type. A build or a replay
whose arguments differ in any of these is a miss before it touches the GPU.
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
it is given.

Interim surface. The stable contract is the tape (aten/src/ATen/cuda/host_trace/
Tape.h and the records this module derives from it); the replay here evaluates
the tape in Python per call and will be replaced by a lowering of the tape into
a runtime that already replays parameterized graphs. Private, underscored, and
expected to change.
"""

from __future__ import annotations

import functools
import gc
import json
import math
import struct
import threading
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch._guards import GuardSource, Source
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, is_fake
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils._python_dispatch import _disable_current_modes, TorchDispatchMode
from torch.utils._pytree import tree_map
from torch.utils._sympy.printers import PythonPrinter


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["Declined", "Miss", "Tape", "trace", "build", "Variant"]

aten = torch.ops.aten


def _binding(name: str) -> type[RuntimeError]:
    # A build without CUDA has no recorder bindings; the module still imports
    # (test_public_bindings and test_testing import every torch module) and
    # trace() declines on its first non-CUDA input.
    found = getattr(torch._C, name, None)
    if found is not None:
        return found
    return type(name.removeprefix("_HostTrace"), (RuntimeError,), {})


TapeMismatch = _binding("_HostTraceTapeMismatch")

# The host did something this tracer does not describe. The C++ recorder
# raises the same type (registered as _HostTraceDeclined), so one except
# clause catches a decline from either side.
Declined = _binding("_HostTraceDeclined")
Declined.__doc__ = "The host did something this tracer does not describe."


class Miss(RuntimeError):
    """This call cannot use the tape: run the ordinary host."""


# The CUDA hosts converted to the recorder's types. An op outside this set
# never reaches a kernel under a trace.
_TRACEABLE = {aten.native_layer_norm.default}

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


def _placeholder_address(base: int) -> int:
    """The hint of a traced input's base address.

    A traced tensor has no storage, and its base address exists only as the
    hint of a symbol: the launch under the trace's capture is recorded, never
    executed, and the build binds the symbol to the real address. The hint
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
    # the symbol a value was created as, before any replacement the ShapeEnv
    # made for it (a guard `Eq(s1, 4096)` replaces s1 by 4096 in later
    # expressions; the guard itself still names s1)
    if isinstance(v, _SYM_TYPES) and isinstance(v.node._expr, sympy.Symbol):
        return str(v.node._expr)
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
    # that disjunction rather than a pin (A3, A4). None: not viewable (eager
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
        tr.rec.register_root(t, root.sym, t.element_size(), root.name)
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
        if getattr(_active, "trace", None) is not None and _has_symint(items):
            out = _sym_getitem(self, items)
            if out is not NotImplemented:
                return out
        return super().__getitem__(index)

    @classmethod
    # pyrefly: ignore [bad-override]
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _ROUTED:
            return _routed(func, args, kwargs)
        raise Declined(f"host_trace: {func} on a traced tensor outside its trace")


class _Trace:
    """One trace in progress: the ShapeEnv, the recorder, and the records the
    Python side keeps (inputs, allocations, outputs)."""

    def __init__(self, device: int) -> None:
        self.device = torch.device("cuda", device)
        self.shape_env = ShapeEnv(duck_shape=False, specialize_zero_one=False)
        # keep every float operation in program order (sym_node.py)
        self.shape_env.exact_float_arithmetic = True
        self.fake_mode = FakeTensorMode(shape_env=self.shape_env)
        # a cached fake output is materialized with set_(), whose contiguity
        # refresh guards `size == 1` per dim on the hint; the uncached meta
        # path asks nothing until a view needs it
        self.fake_mode.cache_enabled = False
        self.rec = torch._C._HostTraceRecorder(device)
        self.tensors: list = []
        self.inputs: list[_InputRec] = []
        self.allocs: list[_AllocRec] = []
        self.nsym = 0
        # the real tensor behind each input root (by root name): a host that
        # needs process-lifetime state keyed by the real storage (a symmetric
        # memory handle) looks it up here
        self.real_inputs: dict[str, torch.Tensor] = {}

    def symbol(
        self, value: Any, name: str, *, size: bool = False, fresh: bool = False
    ) -> Any:
        env = self.shape_env
        src = _Src(name)
        if size:
            sym = env.create_symbol(
                value,
                src,
                DimDynamic.DYNAMIC,
                None,
                positive=True,
                do_not_specialize_zero_one=True,
            )
        else:
            sym = env.create_unspecified_symbol(value, src, DimDynamic.DYNAMIC)
        if isinstance(value, float):
            return env.create_symfloatnode(sym, hint=value, source=src)
        return env.create_symintnode(sym, hint=value, source=src)

    def input(self, position: int, t: torch.Tensor) -> _TracedTensor:
        name = f"arg{position}"
        sizes = [
            self.symbol(t.size(d), f"{name}.size({d})", size=True)
            for d in range(t.dim())
        ]
        strides = [
            self.symbol(t.stride(d), f"{name}.stride({d})") for d in range(t.dim())
        ]
        offset = self.symbol(t.storage_offset(), f"{name}.storage_offset()")
        # the const read of the storage base: a copy-on-write input stays lazy
        base = torch._C._host_trace_storage_address(t)
        root = _Root(
            f"p{position}",
            self.symbol(_placeholder_address(base), f"{name}.base"),
            t.element_size(),
        )
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
        else:  # empty_like: at::native::empty_like's dense branch
            src = args[0]
            sizes = list(src.shape)
            dtype = kwargs.get("dtype") or src.dtype
            src_strides = getattr(src, "_sym_strides", None) or list(src.stride())
            if mf in (None, torch.preserve_format) and _guard_each(
                _dense_terms(sizes, src_strides)
            ):
                strides = list(src_strides)
            else:
                strides = _contiguous_strides(sizes)
            mf = None
        if mf not in (None, torch.contiguous_format, torch.preserve_format):
            raise Declined(
                f"host_trace: only contiguous allocations are traced (got {mf})"
            )
        k = len(self.allocs)
        name = f"alloc{k}"
        q = self.symbol(0, f"{name}.base/256")
        root = _Root(f"a{k}", 256 * q, torch.empty((), dtype=dtype).element_size())
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
            if _has_symint(args) or _has_symint(tuple(kwargs.values())):
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
# builds share the process-wide allocation log (Recorder.h alloc_log_begin), so
# they are serialized; traces and replays are per thread
_build_lock = threading.Lock()


def _new_int_symbol(hint: int, name: str) -> torch.SymInt:
    # the recorder's opaque results (Recorder.h opaque()) get a symbol here
    tr = getattr(_active, "trace", None)
    if tr is None:
        raise Declined("host_trace: opaque call outside a trace")
    return tr.symbol(hint, f"opaque {name}")


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
        kwargs = kwargs or {}
        if func in _ROUTED:
            return _routed(func, args, kwargs)
        if func in _ALLOC_OPS:
            return self.trace.allocate(func, args, kwargs)
        if func in _VIEW_OPS:
            return self.trace.view(func, args, kwargs)
        if self.depth == 0 and func in _TRACEABLE:
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
            except Declined:
                raise
            except torch.AcceleratorError:
                raise  # trace() classifies these by CUDA error code
            except RuntimeError as e:
                raise _refused_inside_the_trace(func, e) from e
            finally:
                self.depth -= 1
        # a composite op decomposes under the mode into the ones above
        self.decomposing.append(func)
        try:
            with self:
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
        # the argument contract: arity, which positions are tensors, and every
        # other argument by value and type; a build or a replay with any
        # difference is a miss, before it touches the GPU
        self.nargs = len(args)
        self.positions = _tensor_positions(args)
        self.constants = _constants(args, self.positions)
        # the device class is part of the contract too (checked at build,
        # before any GPU work)
        self.device_identity = _device_identity(self.device.index)
        self.inputs = tr.inputs
        self.allocs = tr.allocs
        self.launches = records["launches"]
        self.opaque = records["opaque"]
        self.rng_increment = records["rng_increment"]
        # every launch on the trace's own capturing stream (no forked side
        # stream): the fact a one-stream consumer needs before lowering
        self.all_on_capture_stream: bool = records["all_on_capture_stream"]
        self.outputs = outputs
        self.guards = [g.expr for g in tr.shape_env.guards]
        self._tensors = tr.tensors  # the symbols' owners stay alive with the tape

    @property
    def num_launches(self) -> int:
        return len(self.launches)

    @property
    def num_allocations(self) -> int:
        return len(self.allocs)

    @property
    def num_guards(self) -> int:
        return len(self.guards)

    @property
    def num_specializations(self) -> int:
        # there is no separate specialization record: an int read of a symbolic
        # value inside the host is a guard like any other branch (see guards)
        return 0

    def to_json(self) -> str:
        """A deterministic rendering for tests and debugging: two traces of the
        same call serialize identically; the `hints` section holds the values
        at the traced call, including addresses."""
        p = PythonPrinter()

        def e(v: Any) -> Any:
            if isinstance(v, _SYM_TYPES):
                return p.doprint(v.node.expr)
            return v

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
                }
                for o in self.opaque
            ],
            "guards": [p.doprint(g) for g in self.guards],
            "outputs": [
                {
                    "name": o.name,
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
        return json.dumps(d, indent=1)


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


def real_input_of(t: torch.Tensor) -> torch.Tensor | None:
    """The real tensor an argument stands for: under a trace, the input tensor
    behind a traced tensor's root (None for a host allocation or a tensor of
    another trace); at a variant's build and outside a trace, `t` itself."""
    if not isinstance(t, _TracedTensor):
        return t
    tr = getattr(_active, "trace", None)
    if tr is None:
        return None
    return tr.real_inputs.get(t._root.name)


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
    initializes on first use happens outside the trace."""
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
    if warm_up:
        # on this thread's current stream, synchronized on that stream only:
        # a device-wide synchronize would invalidate a capture on another thread
        with torch.cuda.device(device):
            fn(*args)
            torch.cuda.current_stream(device).synchronize()
    # The trace capture is thread-local. A CUDAGraph finalized while it is open
    # (an earlier variant's exec and pool, freed by a cyclic collection on this
    # thread) invalidates it, so hold collections until the trace is over. No
    # collection before the capture: a full one costs more than the trace.
    gc_enabled = gc.isenabled()
    gc.disable()
    tr = _Trace(device)
    _active.trace = tr
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
                )
            )
        records = tr.rec.records()
        return Tape(tr, records, outputs, args)
    finally:
        # an exception above would otherwise leave the capture open until the
        # traceback releases the recorder; end() closes it, and is a no-op
        # after a completed trace
        tr.rec.end()
        _active.trace = None
        if gc_enabled:
            gc.enable()


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
_KIND_MASK = {
    "ptr": (1 << 64) - 1,
    "i64": (1 << 64) - 1,
    "u64": (1 << 64) - 1,
    "i32": (1 << 32) - 1,
    "u32": (1 << 32) - 1,
    "i16": (1 << 16) - 1,
    "u8": 255,
}


class _Program:
    """The tape's values as compiled Python: one function per group, bound by
    symbol name. Compiled once per variant; per call it evaluates guards,
    allocation shapes, opaque arguments and every non-constant parameter."""

    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self.printer = PythonPrinter()
        self.ns: dict[str, Any] = {"math": math, "torch": torch, "min": min, "max": max}
        self._code: dict = {}
        input_syms = set()
        for i in tape.inputs:
            for v in (*i.sizes, *i.strides, i.offset, i.root.sym):
                name = _symbol_name(v)
                if name is not None:
                    input_syms.add(name)
        # guards on the inputs alone are checked before anything is touched;
        # the rest (allocation addresses, opaque results) after those exist
        self.early_guards, self.late_guards = [], []
        for g in tape.guards:
            (
                self.early_guards if self._syms(g) <= input_syms else self.late_guards
            ).append(g)
        events: list = [("alloc", a.seq, a) for a in tape.allocs] + [
            ("opaque", o["seq"], o) for o in tape.opaque
        ]
        events.sort(key=lambda e: e[1])
        self.events = events

    @staticmethod
    def _syms(v: Any) -> set:
        if isinstance(v, _SYM_TYPES):
            return {str(s) for s in v.node.expr.free_symbols}
        if hasattr(v, "free_symbols"):
            return {str(s) for s in v.free_symbols}
        return set()

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


class Variant:
    """A tape plus the CUDA graph it was captured into."""

    def __init__(
        self, tape: Tape, fn: Callable[..., Any], args: tuple, device: int | None = None
    ) -> None:
        self.tape = tape
        self.fn = fn
        # the tape's device, not the current one
        self.device = device if device is not None else tape.device.index
        now = _device_identity(self.device)
        for (k, traced), (_, here) in zip(tape.device_identity, now):
            if traced != here:
                raise Miss(
                    f"the tape was traced on a device with {k}={traced}; cuda:{self.device} has {k}={here}"
                )
        self.prog = _Program(tape)
        self.calls = 0
        # replay keeps per-node dirty state (_last); one call at a time
        self._lock = threading.Lock()
        with _build_lock:
            self._build(args)

    # ---- binding

    def _bind_inputs(self, args: tuple) -> dict:
        # the trace's argument contract first: arity, tensor positions, and
        # every other argument by value and type
        tape = self.tape
        if len(args) != tape.nargs:
            raise Miss(f"{len(args)} arguments, the trace had {tape.nargs}")
        positions = _tensor_positions(args)
        if positions != tape.positions:
            raise Miss(
                f"tensor arguments at {positions}, the trace had them at {tape.positions}"
            )
        constants = _constants(args, positions)
        if constants != tape.constants:
            raise Miss(
                f"non-tensor arguments {constants} differ from the traced {tape.constants}"
            )
        env: dict = {}
        for rec in tape.inputs:
            t = args[rec.position]
            if t.dtype != rec.dtype:
                raise Miss(f"{rec.name} is {t.dtype}, the tape traced {rec.dtype}")
            if not t.is_cuda or t.device.index != self.device:
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
            for d in range(t.dim()):
                self._set(env, rec.sizes[d], t.size(d))
                self._set(env, rec.strides[d], t.stride(d))
            self._set(env, rec.offset, t.storage_offset())
            self._set(env, rec.root.sym, torch._C._host_trace_storage_address(t))
        return env

    @staticmethod
    def _set(env: dict, sym: Any, value: Any) -> None:
        name = _symbol_name(sym)
        if name is not None:
            env[name] = value

    def _check(self, guards: list, env: dict) -> None:
        for g in guards:
            try:
                holds = self.prog.ev(g, env)
            except ZeroDivisionError:
                # a guard whose evaluation is undefined at these inputs
                # (a division by a size that is now zero) cannot hold
                raise Miss(
                    f"guard failed: {self.prog.guard_text(g)} is undefined at these inputs"
                ) from None
            if not holds:
                raise Miss(f"guard failed: {self.prog.guard_text(g)} is not true")

    def _events(self, env: dict, allocs: dict) -> None:
        # allocations and opaque calls in host order
        for kind, _seq, rec in self.prog.events:
            if kind == "alloc":
                sizes = [int(self.prog.ev(s, env)) for s in rec.sizes]
                strides = [int(self.prog.ev(s, env)) for s in rec.strides]
                t = torch.empty_strided(
                    sizes,
                    strides,
                    dtype=rec.dtype,
                    device=torch.device("cuda", self.device),
                )
                addr = t.data_ptr()
                if addr % 256 != 0:
                    raise Miss(
                        f"{rec.name} was allocated at an address not aligned to 256 bytes"
                    )
                env[_symbol_name(rec.q)] = addr // 256
                allocs[rec.root.name] = t
            else:
                args = [int(self.prog.ev(a, env)) for a in rec["args"]]
                v = rec["call"](args)
                if rec["kind"] != "rebind" and v != rec["expected"]:
                    raise Miss(
                        f"opaque {rec['fn']} changed from the traced {rec['expected']} to {v}"
                    )
                name = _symbol_name(rec["sym"])
                if name is not None:
                    env[name] = v

    def _launch_state(self, j: int, env: dict) -> tuple:
        L = self.tape.launches[j]
        image = bytearray(L["hint_image"])
        for p in L["params"]:
            v = p["value"]
            if p["kind"] == "rng" or not isinstance(v, _SYM_TYPES):
                continue
            val = self.prog.ev(v, env)
            kind = p["kind"]
            if kind in ("f32", "f64"):
                packed = struct.pack(_KIND_FMT[kind], float(val))
            else:
                packed = struct.pack(_KIND_FMT[kind], int(val) & _KIND_MASK[kind])
            image[p["offset"] : p["offset"] + p["size"]] = packed
        grid = tuple(int(self.prog.ev(g, env)) for g in L["grid"])
        block = tuple(int(self.prog.ev(b, env)) for b in L["block_expr"])
        smem = int(self.prog.ev(L["smem"], env))
        return bytes(image), grid, block, smem

    # ---- build

    def _roots_from_log(self, log: list, env: dict) -> None:
        # bind each allocation root to the address the ordinary call used, by
        # order: the host allocates them in the tape's recorded order
        allocs = self.tape.allocs
        if len(log) < len(allocs):
            raise TapeMismatch(
                f"the tape has {len(allocs)} allocations, the ordinary call made {len(log)}"
            )
        for rec, (addr, _nbytes) in zip(allocs, log[: len(allocs)]):
            if addr % 256 != 0:
                raise TapeMismatch(f"{rec.name} at the build is not 256-byte aligned")
            env[_symbol_name(rec.q)] = addr // 256

    def _build(self, args: tuple) -> None:
        C = torch._C
        tape = self.tape
        # inputs that fail the tape's own guards are a miss before any GPU work
        env = self._bind_inputs(args)
        self._check(self.prog.early_guards, env)
        # everything here stays on the variant's own stream: a device-wide
        # synchronize would invalidate a capture in progress on another thread
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            self.fn(*args)
            self.fn(*args)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        C._host_trace_alloc_log_begin(self.device, stream.cuda_stream)
        # capture_begin/capture_end directly: torch.cuda.graph's prologue
        # synchronizes the whole device and empties the cache, which would
        # invalidate a trace in progress on another thread. Relaxed mode: on
        # this driver a thread-local capture is still invalidated by another
        # thread's allocations and stream syncs; relaxed is not, and the tape
        # is checked byte for byte against what was captured (the trace
        # capture is thread-local, see Recorder.cpp). A device-wide
        # synchronize from any thread still invalidates any capture.
        with torch.cuda.stream(stream):
            graph.capture_begin(capture_error_mode="relaxed")
            try:
                captured = self.fn(*args)
            finally:
                graph.capture_end()
        log = C._host_trace_alloc_log_end()
        stream.synchronize()
        del captured
        exec_ = C._HostTraceExec(graph, self.device)
        if exec_.num_nodes != len(tape.launches):
            raise TapeMismatch(
                f"the tape has {len(tape.launches)} launches, the capture has {exec_.num_nodes} kernel nodes"
            )
        # the tape at the build inputs must reproduce the capture byte for byte
        self._roots_from_log(log, env)
        for kind, _seq, rec in self.prog.events:
            if kind == "opaque":
                v = rec["call"]([int(self.prog.ev(a, env)) for a in rec["args"]])
                if rec["kind"] != "rebind" and v != rec["expected"]:
                    raise TapeMismatch(
                        f"opaque {rec['fn']} is {v} at the build inputs, the tape traced {rec['expected']}"
                    )
                name = _symbol_name(rec["sym"])
                if name is not None:
                    env[name] = v
        self._check(self.prog.late_guards, env)
        self._last: list = []
        for j, L in enumerate(tape.launches):
            name = exec_.kernel_name(j)
            if L["kernel"] not in name:
                raise TapeMismatch(f"launch {j} is {L['kernel']}, node is {name}")
            image, grid, block, smem = self._launch_state(j, env)
            got = exec_.image(j)
            rng = [
                (p["offset"], p["offset"] + p["size"])
                for p in L["params"]
                if p["kind"] == "rng"
            ]
            for b, (x, y) in enumerate(zip(image, got)):
                if x != y and not any(lo <= b < hi for lo, hi in rng):
                    raise TapeMismatch(
                        f"launch {j} ({L['kernel']}): byte {b} differs between the tape and the capture"
                    )
            if (
                tuple(exec_.grid(j)) != grid
                or tuple(exec_.block(j)) != block
                or exec_.smem(j) != smem
            ):
                raise TapeMismatch(
                    f"launch {j} ({L['kernel']}): launch configuration differs from the capture"
                )
            self._last.append((image, grid, block, smem))
        with torch.cuda.stream(stream):
            exec_.instantiate()  # replays once and waits on this stream only
        self.graph = graph
        self.exec = exec_

    # ---- replay

    def replay(self, args: tuple) -> list[torch.Tensor]:
        with self._lock:
            return self._replay(args)

    def _replay(self, args: tuple) -> list[torch.Tensor]:
        env = self._bind_inputs(args)
        prog = self.prog
        self._check(prog.early_guards, env)
        allocs: dict = {}
        self._events(env, allocs)
        self._check(prog.late_guards, env)
        # the node states this call needs; the bookkeeping of what the exec
        # holds (_last) moves only once the push succeeded, and is dropped if
        # the push raised: a later call with the same bindings must push again
        # rather than run the exec with the previous call's state
        updates = []
        new_last = list(self._last)
        for j in range(len(self.tape.launches)):
            state = self._launch_state(j, env)
            if state != self._last[j]:
                new_last[j] = state
                image, grid, block, smem = state
                updates.append((j, image, grid, block, smem))
        with torch.cuda.device(self.device):
            try:
                self.exec.run(updates)
            except BaseException:
                # the exec may hold any mix of old and new node state
                self._last = [None] * len(self._last)
                raise
            self._last = new_last
        self.calls += 1
        outs = []
        for o in self.tape.outputs:
            base = allocs.get(o.root.name)
            if base is None:
                rec = next(i for i in self.tape.inputs if i.root is o.root)
                base = args[rec.position]
            sizes = [int(prog.ev(s, env)) for s in o.sizes]
            strides = [int(prog.ev(s, env)) for s in o.strides]
            offset = int(prog.ev(o.offset, env))
            # the output's own dtype over the root's storage: a view_as_real /
            # view_as_complex output has a dtype and element units of its own
            out = torch.empty((), dtype=o.dtype, device=base.device)
            out.set_(base.untyped_storage(), offset, sizes, strides)
            outs.append(out)
        return outs

    def try_replay(self, args: tuple):
        """The replay, or None when this call cannot use the tape."""
        try:
            return self.replay(args)
        except Miss:
            return None

    @property
    def dirty_nodes(self) -> int:
        return self.exec.dirty_nodes

    # the first revision exposed the evaluator here; kept for its callers
    @property
    def replayer(self) -> Variant:
        return self


def build(
    tape: Tape, fn: Callable[..., Any], args: tuple, device: int | None = None
) -> Variant:
    """Capture `fn` at `args` and check the tape against that capture."""
    return Variant(tape, fn, args, device)
