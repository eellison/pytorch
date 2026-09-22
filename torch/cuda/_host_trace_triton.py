"""Python-launched Triton kernels under a host trace.

A traced host may launch a Triton kernel from Python: torch._native's eager
overrides do (ops/bmm_outer_product serves the K = 1 bmm that way), and a
user's own @triton.jit kernel inside a traced function does. The launch enters
the tape as a launch record like any other (Tape.h LaunchRec, as
tape_records renders it): the function object eager launches (the selected
CompiledKernel's loaded CUfunction, the handle an eager capture's node
holds), the driver's parameter layout, the grid as values over the trace's
symbols, pointer parameters as values over the tape's roots, integer
parameters as values, and Triton's specialization of the arguments as
guards. Nothing runs under the trace.

How the launch is seen. A Python launch is `kernel[grid](*args)`:
KernelInterface.__getitem__ returns the launcher that calls the kernel's run.
The symbolic run intercepts that launcher: the arguments are bound to the
kernel's formals, and the compilation is selected as eager selects it, by
Triton's own binder and cache (JITFunction.run with warmup=True: no launch)
on the traced values, a pointer standing in with its dtype and the address
value's hint, an integer with its hint. The guards of that selection come
from Triton's own binder run a second time on the values of the trace
(_select): the same generated binder, whose per-argument specialize_impl is
replaced by one that reads the symbolic values, so the flags of each
declaration (do_not_specialize, do_not_specialize_on_alignment, a const or a
typed annotation, a constexpr) and the compile key's composition are
Triton's as written, not a list kept here. A pointer stand-in with a symbolic
address goes through Triton's C++ (native_specialize_impl) under a backend
view without native tensor specialization, so Triton's own Python
(get_tensor_specialization) compares the address, and that comparison is a
guard; an integer's equal-to-1 and width classes are the C++'s rules re-read
on the symbolic value (the C++ reads a C long, a concrete read: the two are
compared at the hint here and on every class boundary by the drift test,
test_cudagraph_host_trace_triton_spec.py), its divisibility Triton's own
get_int_specialization; a constexpr's value is pinned; a float or bool has no
value axis. The symbolic run's specialization must equal, argument by
argument, the one Triton computed for the hints, and its compile key must be
the key under which Triton's cache holds the selected compilation. So the
tape guards the selection in the direction the traced call took: the
selected compilation is eager's for every call the tape serves, and a call
whose specialization differs misses and retraces to eager's other
compilation (E24). When the trace had a
warm-up (trace(warm_up=True)), the launcher is observed while it runs and
the k-th launch under the trace must select the k-th observed CompiledKernel
(the object eager launched); a native entry's miss path traces without a
warm-up and the selection stands on the guards alone. The runtime's
DirectTritonOwner (torch/_inductor/runtime/_cudagraph/direct_triton.py) reads
the selected compilation's ABI (which formals hold a slot, their types and
divisibility, the parameter layout), so the record's slot binding is the one
the runtime's own Triton frontend produces.

The hook replaces KernelInterface.__getitem__ for the duration of a trace
(reference counted across threads) and acts on the tracing thread only; a
launch on another thread, or outside a trace, runs the launcher as written.
JITFunction.run and Autotuner.run are untouched: the runtime's own
DirectTriton snapshots and calls them, and a direct `kernel.run(...)` under
a trace is not a launch this hook sees (it fails on the traced tensors'
missing storage, loudly).

Declined by name: a launch the warm-up did not make at that point (or one
it made and the trace did not), a kernel with JIT pre-run hooks or a
launch_metadata hook, an Autotuner, a compilation the runtime's owner refuses
(a float or bool scalar argument, TMA descriptors, host-side scratch), a grid
of more than three axes or outside the launch bounds, a grid that reads
tensor data (the host read declines where it occurs), a launch on another
device or stream than the trace's, a launch option or an integer class
that is a value of the trace where Triton reads a constant, a process whose
Triton adds a pipeline hash to the compile key
(knobs.runtime.add_stages_inspection_hook), and a selection that is not the
warm-up's or that the symbolic run does not reproduce (an internal
inconsistency).
"""

from __future__ import annotations

import contextlib
import threading
import types
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.cuda._utils import _check_cuda_bindings


_state = threading.local()  # phase: None | "observe" | "trace"; observations
_patch_lock = threading.Lock()
_patch_count = 0
_original_getitem: Any = None
# (id(CompiledKernel), device) -> DirectTritonOwner: the owner retains the
# compilation (so the id stays unique) and its loaded copy of the cubin for
# the process's lifetime, as Triton's own kernel cache retains the compilation
_owners: dict[tuple[int, int], Any] = {}
_owners_lock = threading.Lock()

_GRID_LIMITS = (2**31 - 1, 65535, 65535)
_INT_BITS = {"i32": 32, "i64": 64}
_SYM_TYPES = (torch.SymInt, torch.SymFloat, torch.SymBool)


@dataclass(frozen=True)
class _Observed:
    jit: Any
    binary: Any  # the CompiledKernel the ordinary launch selected
    options: tuple  # launch options (num_warps, ...), by name
    device: int


def _value_hint(v: Any) -> Any:
    return v.node.hint if isinstance(v, _SYM_TYPES) else v


class _PointerStandIn:
    """What Triton's specialization reads of a pointer argument
    (native_specialize_impl: the dtype, and data_ptr() where the declaration
    specializes it): the dtype, and the address as a value of the trace for
    the symbolic run of the binder or as its hint for the concrete one."""

    __slots__ = ("dtype", "_address")

    def __init__(self, dtype: torch.dtype, address: Any) -> None:
        self.dtype, self._address = dtype, address

    def data_ptr(self) -> Any:
        return self._address

    def hinted(self) -> _PointerStandIn:
        return _PointerStandIn(self.dtype, _value_hint(self._address))


class _TensorSpecializationInPython:
    """Triton's backend as native_specialize_impl reads it, without native
    tensor specialization: the C++ then hands a tensor argument to the
    backend's own get_tensor_specialization, whose `data_ptr() % 16 == 0` on
    a symbolic address is a guard of the trace."""

    supports_native_tensor_specialization = False

    def __init__(self, backend: Any) -> None:
        self.get_tensor_specialization = backend.get_tensor_specialization


class _LazyIntType:
    """The width class of a symbolic integer, decided (each comparison a
    guard) only where the binder keeps the value's type: a declared
    annotation replaces it, and the class is then no input of the
    selection."""

    __slots__ = ("value",)

    def __init__(self, value: torch.SymInt) -> None:
        self.value = value

    def resolve(self) -> str:
        return _int_type(self.value)


def _int_type(v: Any) -> str:
    # native_specialize_impl's integer classes, on the value
    if bool(v >= -(2**31)) and bool(v <= 2**31 - 1):
        return "i32"
    if bool(v >= -(2**63)) and bool(v <= 2**63 - 1):
        return "i64"
    if bool(v >= 2**63) and bool(v <= 2**64 - 1):
        return "u64"
    raise OverflowError("integer to be specialized too large to represent")


class _SymbolicSpecialization:
    """Triton's per-argument specialization (native_specialize_impl, the
    binder's specialize_impl) on the values of the trace. A pointer stand-in
    with a symbolic address goes through the C++ under
    _TensorSpecializationInPython; an integer's equal-to-1 and width classes
    are the C++'s rules re-read on the symbolic value, its divisibility the
    backend's own get_int_specialization; a float or bool has no value axis;
    a tuple is specialized element-wise as the C++ does; anything else is a
    constant of the call and is read as the C++ reads it. Where a declaration
    disables a specialization or its alignment, the value is no input of the
    result (Triton's rule is `... and align`) and the hint is read."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(
        self, backend: Any, arg: Any, is_const: bool, specialize: bool, align: bool
    ) -> Any:
        return self.specialize(arg, is_const, specialize, align)

    def specialize(
        self, arg: Any, is_const: bool, specialize: bool, align: bool
    ) -> Any:
        from triton._C.libtriton import native_specialize_impl

        if type(arg) is _PointerStandIn:
            if specialize and align and isinstance(arg._address, _SYM_TYPES):
                view = _TensorSpecializationInPython(self.backend)
                return native_specialize_impl(view, arg, is_const, specialize, align)
            return native_specialize_impl(
                self.backend, arg.hinted(), is_const, specialize, align
            )
        if type(arg) is torch.SymInt:
            if specialize and bool(arg == 1):
                return ("constexpr", 1)
            ty = _LazyIntType(arg)
            if not specialize:
                return (ty, None)
            value = arg if align else arg.node.hint
            return (ty, self.backend.get_int_specialization(value, align=align))
        if isinstance(arg, _SYM_TYPES):
            return native_specialize_impl(
                self.backend, arg.node.hint, is_const, specialize, align
            )
        if isinstance(arg, tuple) and _has_symbolic(arg):
            parts = [self.specialize(a, is_const, specialize, align) for a in arg]
            return (tuple(p[0] for p in parts), tuple(p[1] for p in parts))
        return native_specialize_impl(self.backend, arg, is_const, specialize, align)


def _has_symbolic(v: Any) -> bool:
    if isinstance(v, _SYM_TYPES) or type(v) is _LazyIntType:
        return True
    if isinstance(v, tuple):
        return any(_has_symbolic(a) for a in v)
    return False


def _resolve(entry: Any) -> Any:
    # the symbolic run's specialization entry as Triton's key reads it: a
    # width class decided now (the binder kept the value's type), a
    # constexpr's value pinned to its hint (each a guard)
    if type(entry) is _LazyIntType:
        return entry.resolve()
    if isinstance(entry, _SYM_TYPES):
        hint = entry.node.hint
        held = bool(entry) is hint if type(entry) is torch.SymBool else entry == hint
        if not held:
            raise AssertionError(f"host_trace: {entry} is not {hint}")
        return hint
    if isinstance(entry, tuple) and _has_symbolic(entry):
        return tuple(_resolve(a) for a in entry)
    return entry


@dataclass
class TritonTrace:
    """One trace's Triton launches: the warm-up's observations to check the
    selections against (None when the trace had no warm-up), the records made
    and the roots the kernels may write."""

    observations: list | None
    position: int = 0
    launches: list = field(default_factory=list)
    written_roots: list = field(default_factory=list)
    failure: BaseException | None = None


@dataclass(frozen=True)
class TritonLaunch:
    """What a Triton launch record carries beside the LaunchRec fields, under
    its "triton" key: the runtime's owner of the selected compilation (its ABI
    facts and staleness check), the kernel and the compilation eager launched,
    and per formal how it was bound ("ptr", "i32", "i64", "constexpr")."""

    owner: Any
    jit: Any
    binary: Any
    formals: tuple


def _host_trace() -> Any:
    from torch.cuda import _host_trace

    return _host_trace


def _options(jit: Any, kwargs: dict) -> tuple:
    return tuple((k, v) for k, v in kwargs.items() if k not in jit.arg_names)


def _getitem(self: Any, grid: Any) -> Any:
    # KernelInterface.__getitem__: the launcher `kernel[grid]`, with the
    # calling thread's phase read at the launch
    def launch(*args: Any, **kwargs: Any) -> Any:
        phase = getattr(_state, "phase", None)
        if phase is None:
            return self.run(grid=grid, warmup=False, *args, **kwargs)  # noqa: B026
        from triton.runtime.jit import JITFunction

        if phase == "observe":
            binary = self.run(grid=grid, warmup=False, *args, **kwargs)  # noqa: B026
            if type(self) is JITFunction and binary is not None:
                _state.observations.append(
                    _Observed(
                        self,
                        binary,
                        _options(self, kwargs),
                        torch.cuda.current_device(),
                    )
                )
            return binary
        tr = getattr(_host_trace()._active, "trace", None)
        try:
            if type(self) is not JITFunction:
                raise _host_trace().Declined(
                    f"host_trace: Triton {type(self).__name__} {getattr(getattr(self, 'fn', None), 'fn', self).__name__}: "
                    "an autotuned launch under a trace is not recorded (declined)"
                )
            return _intercept(self, args, grid, kwargs)
        except BaseException as error:
            if tr is not None and tr.triton.failure is None:
                tr.triton.failure = error
            raise

    return launch


@contextlib.contextmanager
def hooked() -> Any:
    """The Python launcher observed or intercepted on the calling thread's
    say-so while any trace runs (reference counted across threads); Triton's
    own again when the last trace ends. A process without Triton has nothing
    to hook."""
    global _patch_count, _original_getitem
    with _patch_lock:
        if _patch_count == 0:
            try:
                from triton.runtime.jit import KernelInterface
            except ImportError:
                KernelInterface = None
            if KernelInterface is not None:
                _original_getitem = KernelInterface.__getitem__
                KernelInterface.__getitem__ = _getitem
        _patch_count += 1
    try:
        yield
    finally:
        with _patch_lock:
            _patch_count -= 1
            if _patch_count == 0 and _original_getitem is not None:
                from triton.runtime.jit import KernelInterface

                KernelInterface.__getitem__ = _original_getitem
                _original_getitem = None


@contextlib.contextmanager
def observing() -> Any:
    """The warm-up: every ordinary Triton launch on this thread is observed,
    in order, into the yielded list."""
    previous = (getattr(_state, "phase", None), getattr(_state, "observations", None))
    observations: list = []
    _state.phase, _state.observations = "observe", observations
    try:
        yield observations
    finally:
        _state.phase, _state.observations = previous


@contextlib.contextmanager
def tracing() -> Any:
    """The symbolic run: every Triton launch on this thread is intercepted and
    recorded on the active trace (_host_trace._active.trace.triton)."""
    previous = getattr(_state, "phase", None)
    _state.phase = "trace"
    try:
        yield
    finally:
        _state.phase = previous


def merge(tr: Any, records: dict) -> None:
    """The trace's Triton launches into the recorder's records, in host order
    with the C++ launches; the roots they may write into written_roots. A
    warm-up launch the symbolic run never reached declines: the two runs made
    different calls."""
    tt = tr.triton
    if tt is None:
        return
    if tt.failure is not None:
        raise _host_trace().Declined(
            f"host_trace: a Triton launch failed under the trace: {tt.failure}"
        ) from tt.failure
    if tt.observations is not None and tt.position != len(tt.observations):
        left = tt.observations[tt.position]
        raise _host_trace().Declined(
            f"host_trace: the warm-up launched Triton kernel {left.jit.fn.__name__} "
            f"({len(tt.observations)} Triton launches in all), the trace reached {tt.position}: "
            "the symbolic run made a different call (declined)"
        )
    if not tt.launches:
        return
    _host_trace()._merge_launches(records, tt.launches)
    written = list(records["written_roots"])
    for name in tt.written_roots:
        if name not in written:
            written.append(name)
    records["written_roots"] = written


def _owner(binary: Any, jit: Any, device: int, decline: Any) -> Any:
    from torch._inductor.runtime._cudagraph.direct_triton import (
        DirectTritonDeclined,
        DirectTritonOwner,
    )

    key = (id(binary), device)
    with _owners_lock:
        owner = _owners.get(key)
        try:
            if owner is None:
                owner = DirectTritonOwner(binary, jit, device)
                _owners[key] = owner
            else:
                owner.check()
        except DirectTritonDeclined as error:
            _owners.pop(key, None)
            decline(str(error))
    return owner


def _intercept(jit: Any, args: tuple, grid: Any, kwargs: dict) -> Any:
    ht = _host_trace()
    tr = getattr(ht._active, "trace", None)
    if tr is None:
        return jit.run(grid=grid, warmup=False, *args, **kwargs)  # noqa: B026
    name = jit.fn.__name__

    def decline(why: str) -> Any:
        raise ht.Declined(f"host_trace: Triton kernel {name}: {why} (declined)")

    tt = tr.triton
    if jit.pre_run_hooks:
        decline("the kernel has JIT pre-run hooks, which the trace does not run")
    if jit.launch_metadata is not None:
        decline("the kernel has a launch_metadata hook, which the trace does not run")
    options = _options(jit, kwargs)
    device = tr.device.index
    observed = None
    if tt.observations is not None:
        if tt.position >= len(tt.observations):
            decline(
                f"launched under the trace but the warm-up made {len(tt.observations)} Triton "
                "launches in all: the symbolic run made a different call"
            )
        observed = tt.observations[tt.position]
        tt.position += 1
        if observed.jit is not jit:
            decline(
                f"the warm-up launched {observed.jit.fn.__name__} at this point: the launch "
                "order differs"
            )
        if options != observed.options:
            decline(
                f"launch options {dict(options)} differ from the warm-up's {dict(observed.options)}"
            )
        if observed.device != device:
            decline(
                f"the warm-up launched it on cuda:{observed.device}; the trace is on cuda:{device}"
            )
    if torch.cuda.current_device() != device:
        decline(
            f"launched on cuda:{torch.cuda.current_device()}; the trace is on cuda:{device}"
        )
    if torch.cuda.current_stream(device) != tr.stream:
        decline("launched on a stream other than the trace's capturing stream")
    from torch._inductor.runtime._cudagraph.direct_triton import _bind, _POINTER_DTYPES
    from torch._native.const_tensor_wrapper import ConstTensorWrapper

    try:
        bound = _bind(jit, args, kwargs)
    except TypeError as error:
        decline(f"the arguments do not bind to the kernel's signature: {error}")
    binary = _select(jit, bound, options, device, decline, ht)
    if observed is not None and binary is not observed.binary:
        decline(
            "the compilation Triton selects for the traced values is not the one the warm-up "
            "launched"
        )
    owner = _owner(binary, jit, device, decline)
    if owner.descriptors:
        decline("TMA descriptor arguments are not traced")
    if any(spec.size for spec in owner.scratch):
        decline("a nonzero launcher scratch allocation is not traced")
    layout = tuple(owner.abi_layout)
    params, written, formals = [], [], []
    for row in owner.formals:
        value = bound[row.formal]
        if row.abi_index is None:
            # a declared constexpr, or an integer Triton specialized to 1 and
            # dropped from the ABI: the compiled constant, which the symbolic
            # run of the binder pinned (_select)
            hint = ht._hint(value)
            if type(hint) is not type(row.constant) or hint != row.constant:
                decline(
                    f"constexpr {row.formal} is {hint!r} at the trace; the compilation holds "
                    f"{row.constant!r}"
                )
            formals.append((row.formal, "constexpr"))
            continue
        offset, size = layout[row.abi_index]
        if row.triton_type.startswith("*"):
            const = type(value) is ConstTensorWrapper
            if const:
                value = value._tensor
            if type(value) is not ht._TracedTensor:
                decline(
                    f"pointer argument {row.formal} is a {type(value).__name__}, not a tensor "
                    "of the trace"
                )
            dtype = _POINTER_DTYPES.get(row.triton_type)
            if dtype is None or value.dtype != dtype:
                decline(
                    f"pointer argument {row.formal} is {value.dtype}; the warm-up compiled "
                    f"{row.triton_type}"
                )
            if value.device != tr.device:
                decline(f"pointer argument {row.formal} is on {value.device}")
            if size != 8:
                decline(f"pointer argument {row.formal} has a {size}-byte slot")
            address = value._root.sym + value._sym_offset * value.element_size()
            params.append(
                {
                    "offset": offset,
                    "size": size,
                    "kind": "ptr",
                    "value": address,
                    "name": row.formal,
                    # a ConstTensorWrapper is the host's declaration that the
                    # kernel only reads the argument; any other pointer may be
                    # written (the root joins the tape's written roots)
                    "access": "r" if const else "rw",
                }
            )
            formals.append((row.formal, "ptr"))
            if not const:
                written.append(value._root.name)
            continue
        bits = _INT_BITS.get(row.triton_type)
        if bits is None:
            decline(
                f"argument {row.formal} of type {row.triton_type} has no slot on the tape "
                "(pointers and i32 / i64 integers do)"
            )
        if size != bits // 8:
            decline(
                f"integer argument {row.formal} has a {size}-byte slot for {row.triton_type}"
            )
        if type(value) is not int and type(value) is not torch.SymInt:
            decline(f"integer argument {row.formal} is a {type(value).__name__}")
        # the slot's signed width holds whatever decided the type (the value's
        # class, a guard of the selection, or a declared annotation)
        low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        if not (bool(value >= low) and bool(value <= high)):
            decline(
                f"integer argument {row.formal} is outside the {bits}-bit range of its slot "
                "at the trace"
            )
        params.append(
            {
                "offset": offset,
                "size": size,
                "kind": row.triton_type,
                "value": value,
                "name": row.formal,
                "access": "",
            }
        )
        formals.append((row.formal, row.triton_type))
    for k, index in enumerate(owner.scratch_abi_indices):
        # an advertised zero-size launcher scratch slot: the null pointer eager
        # passes, a constant of the launch
        offset, size = layout[index]
        params.append(
            {
                "offset": offset,
                "size": size,
                "kind": "ptr",
                "value": 0,
                "name": f"__launcher_scratch_{k}",
                "access": "",
            }
        )
    try:
        dims = grid(bound) if callable(grid) else grid
    except ht.Declined as error:
        text = str(error).removeprefix("host_trace: ")
        raise ht.Declined(
            f"host_trace: the grid of Triton kernel {name}: {text}"
        ) from error
    if type(dims) not in (tuple, list) or not 1 <= len(dims) <= 3:
        decline("the grid must be one to three dimensions")
    dims = (*dims, *((1,) * (3 - len(dims))))
    for axis, (extent, limit) in enumerate(zip(dims, _GRID_LIMITS)):
        if type(extent) is not int and type(extent) is not torch.SymInt:
            decline(f"grid axis {axis} is a {type(extent).__name__}")
        # the launch bounds on the grid's values: a launch-configuration
        # check, kernel-tagged as the hosts' grid checks are
        with torch._C._HostTraceKernelChoice():
            inside = bool(extent >= 1) and bool(extent <= limit)
        if not inside:
            decline(f"grid axis {axis} is outside the launch bounds at the traced call")
    with torch.cuda.device(device):
        # eager's loaded module for this compilation (a no-op after its first
        # launch): the function handle an eager capture's node holds
        binary._init_handles()
    function = int(binary.function)
    from cuda.bindings import driver

    for index, (offset, size) in enumerate(layout):
        # the layout the owner read from its own load of the cubin holds for
        # Triton's load too (one cubin); read back rather than assumed
        got = _check_cuda_bindings(driver.cuFuncGetParamInfo(function, index))
        if tuple(got) != (offset, size):
            decline(
                "the parameter layout of eager's loaded function differs from the compilation's"
            )
    num_warps = int(binary.metadata.num_warps)
    smem = int(binary.metadata.shared)
    if owner.module.num_warps != num_warps or owner.module.shared != smem:
        decline(
            "the launch configuration of the compilation differs from its loaded module's"
        )
    total = max(offset + size for offset, size in layout) if layout else 0
    image = bytearray(total)
    for p in params:
        image[p["offset"] : p["offset"] + p["size"]] = ht._pack(
            p["kind"], ht._hint(p["value"])
        )
    block = (num_warps * 32, 1, 1)
    tt.launches.append(
        {
            "seq": tr.rec.next_seq(),
            "kernel": binary.name,
            "func": function,
            "param_layout": list(layout),
            "grid": dims,
            "block": block,
            "block_expr": block,
            "smem": smem,
            "params": params,
            "hint_image": bytes(image),
            "triton": TritonLaunch(owner, jit, binary, tuple(formals)),
        }
    )
    for root in written:
        if root not in tt.written_roots:
            tt.written_roots.append(root)
    return binary


def _probe_address(tensor: Any) -> Any:
    # the address Triton's specialization reads, as a value of the tape: an
    # input's full address (its base symbol, a pointer slot of the predicate,
    # plus the view's offset); an allocation's offset alone, its base a
    # multiple of 256 by construction (the trace's build and the replay's
    # planner both hold it) with no predicate source
    offset_bytes = tensor._sym_offset * tensor.element_size()
    return offset_bytes if tensor._root.allocation else tensor._root.sym + offset_bytes


def _select(
    jit: Any, bound: dict, options: tuple, device: int, decline: Any, ht: Any
) -> Any:
    # the compilation eager selects for these values: Triton's own binder and
    # cache on the values' hints, no launch (warmup=True); then the same
    # binder on the values of the trace, its per-argument specialization
    # reading the symbolic values, whose result must be the hints' entry by
    # entry and whose compile key must be the one the cache holds the
    # compilation under
    from triton import knobs
    from triton.runtime.jit import compute_cache_key

    from torch._native.const_tensor_wrapper import ConstTensorWrapper

    if knobs.runtime.add_stages_inspection_hook is not None:
        decline(
            "triton.knobs.runtime.add_stages_inspection_hook is set: the compile key carries "
            "a pipeline hash the trace does not derive"
        )
    for name, value in options:
        if isinstance(value, _SYM_TYPES):
            decline(f"launch option {name} is a value of the trace")
    hinted, symbolic = {}, {}
    for name, value in bound.items():
        if type(value) is ConstTensorWrapper:
            value = value._tensor
        if type(value) is ht._TracedTensor:
            probe = _probe_address(value)
            hinted[name] = _PointerStandIn(value.dtype, ht._hint(probe))
            symbolic[name] = _PointerStandIn(value.dtype, probe)
        elif isinstance(value, _SYM_TYPES):
            hinted[name], symbolic[name] = ht._hint(value), value
        else:
            hinted[name] = symbolic[name] = value
    launch_options = dict(options)
    with torch.cuda.device(device):
        binary = jit.run(grid=None, warmup=True, **hinted, **launch_options)
        if binary is None:
            decline(
                "Triton compiled nothing for the traced values (a jit cache hook intervened)"
            )
        kernel_cache, kernel_key_cache, _, backend, binder = jit.device_caches[device]
    # JITFunction.run's own key for the hints: it must hold the compilation
    kwargs = dict(launch_options)
    kwargs["debug"] = kwargs.get("debug", jit.debug) or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    _, concrete, opts = binder(**hinted, **kwargs)
    key = compute_cache_key(kernel_key_cache, concrete, opts)
    if kernel_cache.get(key) is not binary:
        decline(
            "the compile key of the traced values does not select the compilation Triton "
            "returned for them"
        )
    symbolic_binder = types.FunctionType(
        binder.__code__,
        {**binder.__globals__, "specialize_impl": _SymbolicSpecialization(backend)},
        binder.__name__,
        binder.__defaults__,
        binder.__closure__,
    )
    # the specialization's comparisons on the values of the trace decide the
    # compilation launched: a kernel choice (plan item 38 stage 0), kernel-tagged
    # as the converted hosts' launch decisions are
    with torch._C._HostTraceKernelChoice():
        _, entries, symbolic_opts = symbolic_binder(**symbolic, **kwargs)
        if len(entries) != len(concrete) or len(entries) != len(jit.params):
            decline(
                "the binder's specialization does not cover the kernel's parameters"
            )
        resolved = [_resolve(entry) for entry in entries]
    for param, got, expected in zip(jit.params, resolved, concrete):
        if got != expected:
            decline(
                f"Triton's specialization of argument {param.name} for the traced values "
                f"({expected!r}) is not the symbolic run's ({got!r})"
            )
    if compute_cache_key({}, resolved, symbolic_opts) != key:
        decline("the compile key of the symbolic run is not the traced values'")
    return binary
