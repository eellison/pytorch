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
value's hint, an integer with its hint. Triton's specialization is a function
of exactly the facts it reads from those values (a pointer's dtype and
16-byte alignment, an integer equal to 1, divisible by 16, inside the int32
range, a constexpr's value, the launch options), and every one of them is
evaluated on the symbolic value afterwards, so the tape guards the selection
in the direction the traced call took: the selected compilation is eager's
for every call the tape serves, and a call whose specialization differs
misses and retraces to eager's other compilation (E24). When the trace had a
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
device or stream than the trace's, and a selection that is not the warm-up's
or does not hold on the symbolic values (an internal inconsistency).
"""

from __future__ import annotations

import contextlib
import threading
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


@dataclass(frozen=True)
class _Observed:
    jit: Any
    binary: Any  # the CompiledKernel the ordinary launch selected
    options: tuple  # launch options (num_warps, ...), by name
    device: int


class _PointerStandIn:
    """What Triton's binder reads of a pointer argument when selecting a
    compilation: the dtype and the address (its hint here)."""

    __slots__ = ("dtype", "_address")

    def __init__(self, dtype: torch.dtype, address: int) -> None:
        self.dtype, self._address = dtype, address

    def data_ptr(self) -> int:
        return self._address


@dataclass
class TritonTrace:
    """One trace's Triton launches: the warm-up's observations to check the
    selections against (None when the trace had no warm-up), the records made
    and the roots the kernels may write."""

    observations: list | None
    position: int = 0
    launches: list = field(default_factory=list)
    written_roots: list = field(default_factory=list)


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

        if type(self) is not JITFunction:
            # an Autotuner (or another KernelInterface): its benchmark launches
            # are no part of the call at the warm-up, and under the trace it
            # declines by name
            if phase == "trace":
                raise _host_trace().Declined(
                    f"host_trace: Triton {type(self).__name__} {getattr(getattr(self, 'fn', None), 'fn', self).__name__}: "
                    "an autotuned launch under a trace is not recorded (declined)"
                )
            return self.run(grid=grid, warmup=False, *args, **kwargs)  # noqa: B026
        if phase == "observe":
            binary = self.run(grid=grid, warmup=False, *args, **kwargs)  # noqa: B026
            if binary is not None:
                _state.observations.append(
                    _Observed(
                        self,
                        binary,
                        _options(self, kwargs),
                        torch.cuda.current_device(),
                    )
                )
            return binary
        return _intercept(self, args, grid, kwargs)

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
    alignments = {row.formal: row.alignment for row in owner.pointers}
    layout = tuple(owner.abi_layout)
    params, written, formals = [], [], []
    for row in owner.formals:
        value = bound[row.formal]
        if row.abi_index is None:
            # a declared constexpr, or an integer Triton specialized to 1 and
            # dropped from the ABI: the compiled constant, as a guard on a value
            _constant_guard(row, value, decline, ht)
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
            offset_bytes = value._sym_offset * value.element_size()
            address = value._root.sym + offset_bytes
            _alignment_guard(
                row.formal,
                value,
                address,
                offset_bytes,
                alignments.get(row.formal, 1),
                decline,
                jit.params[row.source_arg_index],
            )
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
        _int_guards(row, value, bits, decline, jit.params[row.source_arg_index])
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
        if not (bool(extent >= 1) and bool(extent <= limit)):
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


def _select(
    jit: Any, bound: dict, options: tuple, device: int, decline: Any, ht: Any
) -> Any:
    # the compilation eager selects for these values: Triton's own binder and
    # cache on the values' hints, no launch (warmup=True); a pointer stands in
    # with what the binder reads of it
    from torch._native.const_tensor_wrapper import ConstTensorWrapper

    values = {}
    for name, value in bound.items():
        if type(value) is ConstTensorWrapper:
            value = value._tensor
        if type(value) is ht._TracedTensor:
            address = value._root.sym + value._sym_offset * value.element_size()
            value = _PointerStandIn(value.dtype, ht._hint(address))
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = ht._hint(value)
        values[name] = value
    with torch.cuda.device(device):
        binary = jit.run(grid=None, warmup=True, **values, **dict(options))
    if binary is None:
        decline(
            "Triton compiled nothing for the traced values (a jit cache hook intervened)"
        )
    return binary


def _constant_guard(row: Any, value: Any, decline: Any, ht: Any) -> None:
    constant = row.constant
    if type(value) is torch.SymInt:
        if type(constant) is not int or type(constant) is bool:
            decline(
                f"constexpr {row.formal} is symbolic at the trace; the warm-up compiled {constant!r}"
            )
        if not bool(value == constant):
            decline(
                f"constexpr {row.formal} is {ht._hint(value)} at the trace; the warm-up compiled "
                f"{constant}"
            )
        return
    if type(value) is not type(constant) or value != constant:
        decline(
            f"constexpr {row.formal} is {value!r} at the trace; the warm-up compiled {constant!r}"
        )


def _int_guards(row: Any, value: Any, bits: int, decline: Any, parameter: Any) -> None:
    # Record only specialization choices enabled by the JIT declaration.
    # Selected ABI attributes and signed parameter widths still always hold.
    if (
        not parameter.do_not_specialize
        and bool(value == 1)
        and not parameter.annotation_type
    ):
        decline(
            f"integer argument {row.formal} is 1 at the trace, which the warm-up compiled as a slot"
        )
    alignment = max((1, *(v for _, v in row.attributes)))
    modulus = alignment if alignment > 1 else 16
    specializes_alignment = not (
        parameter.do_not_specialize or parameter.do_not_specialize_on_alignment
    )
    if (alignment > 1 or specializes_alignment) and (
        bool(value % modulus == 0) != (alignment > 1)
    ):
        decline(
            f"integer argument {row.formal}'s divisibility by {modulus} at the trace differs from "
            "the warm-up's"
        )
    if bits == 32:
        if not (bool(value >= -(2**31)) and bool(value <= 2**31 - 1)):
            decline(
                f"integer argument {row.formal} is outside the int32 range the warm-up compiled"
            )
    elif parameter.annotation_type:
        if not (bool(value >= -(2**63)) and bool(value <= 2**63 - 1)):
            decline(
                f"integer argument {row.formal} is outside its declared int64 range"
            )
    elif not (bool(value > 2**31 - 1) or bool(value < -(2**31))):
        decline(
            f"integer argument {row.formal} is inside the int32 range; the warm-up compiled i64"
        )


def _alignment_guard(
    formal: str,
    tensor: Any,
    address: Any,
    offset_bytes: Any,
    alignment: int,
    decline: Any,
    parameter: Any,
) -> None:
    # Triton specializes a pointer on its 16-byte alignment (tt.divisibility
    # 16). An input's address is its base symbol plus the view's offset, both
    # values of the tape (the base a pointer slot of the predicate). An
    # allocation's base is a multiple of 256 by construction (the trace's
    # build and the replay's planner both hold it), so only the offset
    # decides; the base symbol has no predicate source and stays out of the
    # guard
    if alignment < 16 and (
        parameter.do_not_specialize or parameter.do_not_specialize_on_alignment
    ):
        return
    probe = offset_bytes if tensor._root.allocation else address
    if bool(probe % 16 == 0) != (alignment >= 16):
        decline(
            f"pointer argument {formal}'s 16-byte alignment at the trace differs from the warm-up's"
        )
