"""Triton JIT launches under a host trace.

While a trace runs, JITFunction.run is replaced (reference counted across
threads; Triton's own again when the last trace ends) and acts on the tracing
thread only. A launch `kernel[grid](*args)` is not run: Triton selects and
compiles the kernel as eager would, on the hints of the traced values
(JITFunction.run with warmup=True), and then runs its own generated binder a
second time on the values of the trace, with the per-argument specialize_impl
replaced by one that reads the symbolic values. Each specialization Triton
makes (an address or an integer divisible by 16, an integer equal to 1, an
integer's i32 / i64 width) is then a comparison of the trace, recorded as a
guard, and the declaration flags (do_not_specialize, annotations, constexpr)
are Triton's as written. The one piece not Triton's own code: the C++ reads
an integer as a C long, so its equal-to-1 and width rules are re-read here on
the symbolic value (_int_type), checked against the C++ on every class
boundary by the drift test.

The launch is recorded as a KernelLaunch on the trace; a compile-only call
(warmup=True) records its guards and no launch. Launch options computed from
sizes, and an autotuned kernel's key, are pinned to their hints (an Eq guard);
the autotuned launch is its cached config's, so the key must have run eagerly
(trace's warm-up call does that). What this module cannot describe of the
call itself declines the trace: autotuner hooks, a real tensor argument, a
tensor launch option, a SymFloat argument, a launch on another device or
stream. A launch whose compilation it does not describe (launch
hooks, whatever the compilation's ABI, _host_trace_triton.triton_abi,
declines, the grid's limits) is an EagerCall instead: a replay launches it
through Triton, as eager does.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import struct
import types
import weakref
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch.cuda._host_trace import Declined, declined, ProcessHold
from torch.cuda._host_trace_launch import _GRID_LIMITS, _probe_address, KernelLaunch
from torch.cuda._host_trace_tape import _hint, _Root, _TracedTensor, current_trace, EagerCall
from torch.cuda._host_trace_triton import param_layout, triton_abi
from torch.cuda._utils import _check_cuda_bindings
from torch.utils._triton import has_triton_package


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torch.cuda._host_trace_tape import _Trace
    from torch.cuda._host_trace_triton import TritonABI


_SYM_TYPES = (torch.SymInt, torch.SymFloat, torch.SymBool)
_INT_BITS = {"i32": 32, "i64": 64}


class OwnedModule:
    """A tape's own load of a compiled kernel's cubin: an explicit unload of
    the kernel's (CompiledKernel.close, a CachingAutotuner's
    release_benchmark_artifacts) leaves the tape's function loaded."""

    def __init__(self, binary: Any) -> None:
        from cuda.bindings import driver

        self._unload = driver.cuModuleUnload
        self.module = _check_cuda_bindings(driver.cuModuleLoadData(binary.asm["cubin"]))
        self.function = int(_check_cuda_bindings(driver.cuModuleGetFunction(self.module, binary.metadata.name.encode())))
        # Triton's opt-in to more than 48 KiB of dynamic shared memory
        attr = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        limit = _check_cuda_bindings(driver.cuFuncGetAttribute(attr, int(binary.function)))
        _check_cuda_bindings(driver.cuFuncSetAttribute(self.function, attr, limit))

    def __del__(self) -> None:
        self._unload(self.module)


_OWNED: weakref.WeakKeyDictionary[Any, OwnedModule] = weakref.WeakKeyDictionary()


def owned_module(binary: Any) -> OwnedModule:
    if binary not in _OWNED:
        _OWNED[binary] = OwnedModule(binary)
    return _OWNED[binary]


class _PointerStandIn:
    """What Triton's specialization reads of a pointer argument: its dtype and
    data_ptr(), the address as a value of the trace or its hint."""

    __slots__ = ("dtype", "_address")

    def __init__(self, dtype: torch.dtype, address: Any) -> None:
        self.dtype, self._address = dtype, address

    def data_ptr(self) -> Any:
        return self._address


class _TensorSpecializationInPython:
    """The backend as native_specialize_impl reads it, without native tensor
    specialization: the C++ hands a tensor to the backend's own Python
    get_tensor_specialization, whose `data_ptr() % 16 == 0` on a symbolic
    address is a guard."""

    supports_native_tensor_specialization = False

    def __init__(self, backend: Any) -> None:
        self.get_tensor_specialization = backend.get_tensor_specialization


class _LazyIntType:
    """A symbolic integer's width class, decided (a guard) only where the
    binder keeps it: a type annotation replaces it."""

    __slots__ = ("value",)

    def __init__(self, value: torch.SymInt) -> None:
        self.value = value


def _int_type(v: Any) -> str:
    # native_specialize_impl's integer classes
    if bool(v >= -(2**31)) and bool(v <= 2**31 - 1):
        return "i32"
    if bool(v >= -(2**63)) and bool(v <= 2**63 - 1):
        return "i64"
    if bool(v >= 2**63) and bool(v <= 2**64 - 1):
        return "u64"
    raise OverflowError("integer to be specialized too large to represent")


class _SymbolicSpecialization:
    """The binder's specialize_impl on the values of the trace. Where a
    declaration disables a specialization or its alignment the value is no
    input of the result, and the hint is read."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(
        self, backend: Any, arg: Any, is_const: bool, specialize: bool, align: bool
    ) -> Any:
        from triton._C.libtriton import native_specialize_impl

        if type(arg) is _PointerStandIn:
            if specialize and align and isinstance(arg._address, torch.SymInt):
                view = _TensorSpecializationInPython(self.backend)
                return native_specialize_impl(view, arg, is_const, specialize, align)
            hinted = _PointerStandIn(arg.dtype, _hint(arg._address))
            return native_specialize_impl(
                self.backend, hinted, is_const, specialize, align
            )
        if type(arg) is torch.SymInt:
            if specialize and bool(arg == 1):
                return ("constexpr", 1)
            ty = _LazyIntType(arg)
            if not specialize:
                return (ty, None)
            value = arg if align else _hint(arg)
            return (ty, self.backend.get_int_specialization(value, align=align))
        # a SymFloat or SymBool has no value axis (its slot declines); as a
        # constexpr its value is pinned by _resolve
        return native_specialize_impl(
            self.backend, _hint(arg), is_const, specialize, align
        )


def _resolve(entry: Any) -> Any:
    # a specialization entry of the symbolic run as Triton's key reads it: a
    # width class decided, a constexpr's value pinned to its hint
    if type(entry) is _LazyIntType:
        return _int_type(entry.value)
    if isinstance(entry, _SYM_TYPES):
        hint = _hint(entry)
        if type(entry) is torch.SymBool:
            held = bool(entry) is hint
        else:
            held = bool(entry == hint)
        if not held:
            raise AssertionError(f"host_trace: {entry} is not {hint}")
        return hint
    if isinstance(entry, (tuple, list)):
        return type(entry)(_resolve(e) for e in entry)
    return entry


def _float_bits(v: float, ty: str) -> int:
    # the launcher's packing (driver.c extractFP16 and so on) as a signed
    # integer of the slot's width
    if ty == "fp16":
        return struct.unpack("<h", struct.pack("<e", v))[0]
    if ty == "fp64":
        return struct.unpack("<q", struct.pack("<d", v))[0]
    f32 = struct.unpack("<i", bytes(ctypes.c_float(v)))[0]  # a C cast: inf on overflow
    return f32 >> 16 if ty == "bf16" else f32  # bf16 truncates


def _traced_leaf(v: Any) -> bool:
    if isinstance(v, (torch.Tensor, *_SYM_TYPES)):
        return True
    return isinstance(v, (tuple, list)) and any(_traced_leaf(e) for e in v)


def _intercept(
    tr: _Trace, jit: Any, args: tuple, grid: Any, warmup: bool, kwargs: dict
) -> Any:
    from triton import knobs

    name = jit.fn.__name__

    # phase (a), the call: its declines are the trace's
    def decline(why: str) -> NoReturn:
        raise tr.decline(f"Triton kernel {name}: {why}")

    if knobs.runtime.add_stages_inspection_hook is not None:
        decline("add_stages_inspection_hook adds a pipeline hash to the compile key")
    device = tr.device.index
    if (current := torch.cuda.current_device()) != device:
        decline(f"launched on cuda:{current}, not the trace's device")
    if torch.cuda.current_stream(device) != tr.stream:
        decline("launched on a stream other than the trace's")

    # a launch option computed from sizes is pinned to its hint
    options = {k: _resolve(v) for k, v in kwargs.items() if k not in jit.arg_names}
    for k, v in options.items():
        if _traced_leaf(v):
            decline(f"launch option {k} is a tensor")
    params = {k: v for k, v in kwargs.items() if k in jit.arg_names}
    try:
        sig = jit.signature.bind(*args, **params)
    except TypeError as e:
        decline(f"the arguments do not bind: {e}")
    sig.apply_defaults()
    bound = dict(sig.arguments)

    hinted, symbolic = {}, {}
    for arg, v in bound.items():
        if isinstance(v, _TracedTensor):
            probe = _probe_address(v)
            hinted[arg] = _PointerStandIn(v.dtype, _hint(probe))
            symbolic[arg] = _PointerStandIn(v.dtype, probe)
        elif isinstance(v, torch.Tensor):
            decline(f"argument {arg} is a tensor the trace does not track")
        elif isinstance(v, torch.SymFloat):
            decline(f"float argument {arg} is a SymFloat")
        elif isinstance(v, _SYM_TYPES):
            hinted[arg], symbolic[arg] = _hint(v), v
        elif _traced_leaf(v):
            decline(f"tuple argument {arg} holds tensors or values of the trace")
        else:
            hinted[arg] = symbolic[arg] = v
    dims = None
    if not warmup:
        dims = grid(bound) if callable(grid) else grid
        if not isinstance(dims, (tuple, list)) or not 1 <= len(dims) <= 3:
            decline("the grid must have one to three dimensions")
        dims = (*dims, 1, 1)[:3]
        for axis, extent in enumerate(dims):
            if type(extent) not in (int, torch.SymInt):
                decline(f"grid axis {axis} is a {type(extent).__name__}")

    # the launch of an op that runs eagerly (_Trace.eager_ops)
    reason = None if dims is None else tr.eager_ops.get(len(tr.ops))
    try:
        if reason is None:
            return _launch(tr, jit, bound, hinted, symbolic, options, dims)
    except Declined as e:
        # a compile-only call has no launch to run eagerly
        if dims is None or e is tr.declined:
            raise
        reason = str(e)
    target = ("triton", jit, dims, options)
    # a SymBool (a constexpr) is pinned to its value, a guard
    args = tuple(bool(v) if isinstance(v, torch.SymBool) else v for v in bound.values())
    tr.record_launch(EagerCall(target, args, {}, (), reason))
    return None


def _launch(
    tr: _Trace,
    jit: Any,
    bound: dict,
    hinted: dict,
    symbolic: dict,
    options: dict,
    dims: tuple | None,
) -> Any:
    from triton import knobs

    name = jit.fn.__name__
    device = tr.device.index

    # phase (b), the compilation and its launch: a decline makes the launch an
    # EagerCall (or, for a compile-only call, the trace's decline)
    def decline(why: str) -> NoReturn:
        raise declined(f"Triton kernel {name}: {why}")

    # an eager launch runs them at every replay
    if jit.pre_run_hooks:
        decline("JIT pre-run hooks run at the trace, never at a replay")
    hooks = (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    if any(getattr(h, "calls", h) for h in hooks):  # a HookChain or a callable
        decline("launch hooks run at the trace, never at a replay")
    binary = _ORIGINALS["jit"](jit, grid=None, warmup=True, **hinted, **options)
    if binary is None:
        decline("Triton compiled nothing (a jit cache hook intervened)")
    _, _, _, backend, binder = jit.device_caches[device]
    # JITFunction.run's own options for its binder
    opts = dict(options)
    opts["debug"] = opts.get("debug", jit.debug) or knobs.runtime.debug
    opts["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    _, concrete, _ = binder(**hinted, **opts)
    symbolic_globals = binder.__globals__ | {
        "specialize_impl": _SymbolicSpecialization(backend)
    }
    symbolic_binder = types.FunctionType(
        binder.__code__, symbolic_globals, binder.__name__, binder.__defaults__
    )
    _, entries, _ = symbolic_binder(**symbolic, **opts)
    for param, entry, want in zip(jit.params, entries, concrete, strict=True):
        got = _resolve(entry)
        if got != want:
            decline(
                f"argument {param.name}: Triton specializes {want!r}, the symbolic run {got!r}"
            )
    if dims is None:
        # compile-only (JITFunction.warmup): the binary, fixed by the guards
        # above, and no launch
        return binary

    abi = triton_abi(binary.src, binary.metadata)
    binary._init_handles()
    owned = owned_module(binary)
    record_launch(tr, name, owned.function, owned, abi, bound, dims)
    return binary


def record_launch(
    tr: _Trace,
    name: str,
    function: int,
    owner: Any,
    abi: TritonABI,
    values: Mapping[str, Any],
    dims: tuple[Any, Any, Any],
) -> None:
    """Record the launch of `function` on the grid `dims`, each argument of
    the ABI read from `values` by name (a constexpr absent from `values` is
    the compiled one). The caller has guarded every specialization, or
    vouches for it. `owner` keeps `function` loaded while the tape lives."""

    def decline(why: str) -> NoReturn:
        raise declined(f"Triton kernel {name}: {why}")

    # under trusted inputs the caller sized the slots and grids it compiled:
    # their limits are read at the hint, never guarded
    vouched = _hint if tr.trusted is not None else lambda v: v
    slots: list[Any] = [0] * abi.num_slots  # the launcher's scratch pointers stay null
    roots: list[_Root] = []
    for a in abi.args:
        v = values.get(a.name, a.constant)
        if a.slot is None:
            # a constexpr, or an integer specialized to 1; a SymInt's value
            # was pinned by the caller
            h = _hint(v)
            if type(h) is not type(a.constant) or h != a.constant:
                decline(f"constexpr {a.name} is {h!r}; compiled {a.constant!r}")
        elif a.is_pointer:
            if not isinstance(v, _TracedTensor):
                decline(f"pointer argument {a.name} is a {type(v).__name__}")
            slots[a.slot] = v.data_ptr()
            if not any(r is v._root for r in roots):
                roots.append(v._root)
        elif a.triton_type not in _INT_BITS:
            if type(v) not in (int, float):
                decline(f"float argument {a.name} is a {type(v).__name__}")
            slots[a.slot] = _float_bits(v, a.triton_type)
        else:
            bits, h = _INT_BITS[a.triton_type], vouched(v)
            if not (bool(h >= -(2 ** (bits - 1))) and bool(h <= 2 ** (bits - 1) - 1)):
                decline(f"integer argument {a.name} is outside its {bits}-bit slot")
            slots[a.slot] = v

    empty = False
    for axis, (extent, limit) in enumerate(zip(dims, _GRID_LIMITS)):
        h = vouched(extent)
        if not (bool(h >= 0) and bool(h <= limit)):
            decline(f"grid axis {axis} is outside [0, {limit}]")
        empty = empty or bool(extent == 0)
    if empty:
        # the launcher skips a launch whose grid has no blocks
        return
    # the launcher's volume check multiplies C ints
    gx, gy, gz = map(vouched, dims)
    if not bool(gx * gy * gz <= 2**31 - 1):
        decline("the grid has more than 2**31 - 1 blocks")

    layout = param_layout(abi, function)
    block = (abi.num_warps * 32, 1, 1)
    launch = KernelLaunch(
        name,
        function,
        abi,
        layout,
        dims,
        block,
        abi.shared,
        tuple(slots),
        tuple(roots),
        owner,
    )
    tr.record_launch(launch)


def _jit_run(self: Any, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
    tr = current_trace()
    if tr is None:
        return _ORIGINALS["jit"](self, *args, grid=grid, warmup=warmup, **kwargs)
    try:
        run = functools.partial(_intercept, tr, self, args, grid, warmup, kwargs)
        # the grid with the arguments, its SymInts renamed for a redo as theirs
        redo = lambda a, k: _jit_run(self, *a, warmup=warmup, **k)  # noqa: E731
        return tr.op(self, args, dict(kwargs, grid=grid), run, host=True, redo=redo)
    except Declined as e:
        if tr.declined is None:
            tr.declined = e
        raise
    except Exception as e:
        if torch.cuda._host_trace.raise_unexpected:
            raise
        raise tr.decline(f"Triton kernel {self.fn.__name__} raised {e!r}") from e


def _autotuner_run(self: Any, *args: Any, **kwargs: Any) -> Any:
    tr = current_trace()
    if tr is None:
        return _ORIGINALS["autotuner"](self, *args, **kwargs)
    run = functools.partial(_autotune, tr, self, args, kwargs)
    return tr.op(self, args, kwargs, run, host=True, redo=lambda a, k: _autotuner_run(self, *a, **k))


def _autotune(tr: _Trace, self: Any, args: tuple, kwargs: dict) -> Any:
    name = self.base_fn.__name__

    def decline(why: str) -> NoReturn:
        raise tr.decline(f"Triton kernel {name}: {why}")

    if self.user_defined_pre_hook or self.user_defined_post_hook:
        decline("autotuner pre_hook and post_hook are not traced")
    if self.reset_to_zero or self.restore_value:
        decline("autotuner reset_to_zero and restore_value are not traced")
    config = self.configs[0]
    if len(self.configs) > 1:
        # Autotuner.run's key: the key arguments' values, then every dtype;
        # each value is pinned to its hint
        named = {**dict(zip(self.arg_names, args)), **kwargs}
        named = {k: v for k, v in named.items() if k in self.arg_names}
        key = [_resolve(named[k]) for k in self.keys if k in named]
        if _traced_leaf(key):
            decline("the autotune key holds a tensor")
        key += [str(v.dtype) for v in named.values() if hasattr(v, "dtype")]
        if (config := self.cache.get(tuple(key))) is None:
            miss = f"autotune key {tuple(key)} is not in the cache"
            e = tr.decline(f"Triton kernel {name}: {miss}")
            e.retry = True  # the eager fallback tunes it
            raise e
    if config.pre_hook is not None:
        decline("a config pre_hook runs at the trace, never at a replay")
    self.best_config = config
    return self.fn.run(*args, **kwargs, **config.all_kwargs())


# Triton's own run methods, saved at the first hook and never cleared: a
# thread may still be inside a hooked run after the last trace ends
_ORIGINALS: dict[str, Any] = {}


def _const_tensor(cls: type, tensor: torch.Tensor) -> Any:
    # torch._native's read-only view of a traced tensor is the traced tensor,
    # bound as any tensor argument
    return tensor if isinstance(tensor, _TracedTensor) else object.__new__(cls)


def _hook() -> None:
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    from torch._native.const_tensor_wrapper import ConstTensorWrapper

    _ORIGINALS["jit"], _ORIGINALS["autotuner"] = JITFunction.run, Autotuner.run
    JITFunction.run, Autotuner.run = _jit_run, _autotuner_run
    # never removed: a class whose assigned __new__ is deleted again rejects
    # its constructor's arguments
    ConstTensorWrapper.__new__ = staticmethod(_const_tensor)  # type: ignore[method-assign, assignment]


def _unhook() -> None:
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    JITFunction.run, Autotuner.run = _ORIGINALS["jit"], _ORIGINALS["autotuner"]


_hooks = ProcessHold(_hook, _unhook)


@contextlib.contextmanager
def intercepting() -> Iterator[None]:
    """Triton's JIT launches intercepted on the tracing thread while any trace
    runs. A process without Triton has nothing to hook."""
    if not has_triton_package():
        yield
        return
    with _hooks:
        yield
