"""Triton JIT launches under a host trace.

While a trace runs, JITFunction.run is replaced (reference counted across
threads; Triton's own again when the last trace ends) and acts on the tracing
thread only. A launch `kernel[grid](*args)` is not run: Triton's own
generated binder runs on the values of the trace, with the per-argument
specialize_impl replaced by one that reads the symbolic values. Each
specialization Triton makes (an address or an integer divisible by 16, an
integer equal to 1, an integer's i32 / i64 width) is then a comparison of the
trace, recorded as a guard, and the declaration flags (do_not_specialize,
annotations, constexpr) are Triton's as written. The kernel is the one
JITFunction.run looks up or compiles for that specialization key; no value is
read at its hint. The one piece not Triton's own code: the C++ reads an
integer as a C long, so its equal-to-1 and width rules are re-read here on
the symbolic value (_int_type), checked against the C++ on every class
boundary by the drift test.

The launch is recorded as a KernelLaunch on the trace; a compile-only call
(warmup=True) records its guards and no launch. Launch options computed from
sizes, and an autotuned kernel's key, are pinned (an Eq guard);
the autotuned launch is its cached config's, so the key must have run eagerly
(trace's warm-up call does that). What this module cannot describe of the
call itself declines the trace: autotuner hooks, a real tensor argument, a
tensor launch option, a SymFloat argument, a launch on another device or
stream. A launch whose compilation it does not describe (launch
hooks, whatever the compilation's ABI, _host_trace_triton.triton_abi,
declines, the grid's limits) is an EagerCall instead: a replay launches it
through Triton, as eager does. The capture checks each record against the
kernel's own launcher at the traced call (KernelLaunch.witness).
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
from torch.cuda._host_trace_launch import _driver_version, _GRID_LIMITS, _probe_address, KernelLaunch, tma_edits, TmaDescriptor
from torch.cuda._host_trace_tape import _PLACEHOLDER_LOW, _Root, _TracedTensor, current_trace, EagerCall
from torch.cuda._host_trace_triton import param_layout, triton_abi
from torch.utils._triton import has_triton_package


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from torch.cuda._host_trace_tape import _Trace
    from torch.cuda._host_trace_triton import TritonABI


_SYM_TYPES = (torch.SymInt, torch.SymFloat, torch.SymBool)
_INT_BITS = {"i32": 32, "i64": 64}


class OwnedModule:
    """A tape's own load of a compiled kernel's cubin: an explicit unload of
    the kernel's (CompiledKernel.close, a CachingAutotuner's
    release_benchmark_artifacts) leaves the tape's function loaded. Loaded as
    CompiledKernel._init_handles loads it, so the function carries Triton's
    own attributes (its shared memory opt-in and cache preference)."""

    def __init__(self, binary: Any) -> None:
        from triton.runtime import driver

        utils = driver.active.utils
        self._unload = utils.unload_module
        device = driver.active.get_current_device()
        self.module, function, *_ = utils.load_binary(binary.name, binary.kernel, binary.metadata.shared, device)
        self.function = int(function)

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


class _Answer:
    """An integer each of whose tests answers `answer` (arithmetic on it is
    itself); counts the tests a rule makes of it."""

    __slots__ = ("answer", "tests")

    def __init__(self, answer: bool) -> None:
        self.answer, self.tests = answer, 0

    def __mod__(self, other: Any) -> _Answer:
        return self

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        self.tests += 1
        return self.answer

    __ne__ = __eq__  # type: ignore[assignment]
    __hash__ = None  # type: ignore[assignment]


def _unread(rule: Callable[[Any], Any], value: Any) -> Any:
    """rule(value) with no guard where its result does not depend on the
    value: a rule that tests it once and gives one result either way (Triton's
    divisibility rule with align off); else on the value itself, each test a
    guard."""
    results, tests = set(), set()
    for answer in (True, False):
        x = _Answer(answer)
        try:
            results.add(rule(x))
        except TypeError:  # the rule does more than test the value
            return rule(value)
        tests.add(x.tests)
    return results.pop() if len(results) == 1 and tests == {1} else rule(value)


class _SymbolicSpecialization:
    """The binder's specialize_impl on the values of the trace: what it reads
    of a value is a guard. Where a declaration disables a specialization or
    its alignment the rule's result does not depend on the value, and no
    guard is recorded (_unread)."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(
        self, backend: Any, arg: Any, is_const: bool, specialize: bool, align: bool
    ) -> Any:
        from triton._C.libtriton import native_specialize_impl

        if type(arg) is _PointerStandIn and isinstance(arg._address, torch.SymInt):
            view = _TensorSpecializationInPython(self.backend)

            def pointer(address: Any) -> Any:
                return native_specialize_impl(view, _PointerStandIn(arg.dtype, address), is_const, specialize, align)

            return _unread(pointer, arg._address)
        if type(arg) is torch.SymInt:
            if specialize and bool(arg == 1):
                return ("constexpr", 1)
            ty = _LazyIntType(arg)
            if not specialize:
                return (ty, None)
            return (ty, _unread(lambda v: self.backend.get_int_specialization(v, align=align), arg))
        if isinstance(arg, _SYM_TYPES):
            # a SymBool's slot (u1) is not traced; as a constexpr the binder
            # passes it by and _resolve pins it
            raise declined(f"a {type(arg).__name__} argument has no traced slot")
        return native_specialize_impl(self.backend, arg, is_const, specialize, align)


def _pinned(v: Any) -> Any:
    # a value of the trace as a constant: int() / bool() / float() records the
    # guard that pins it
    if isinstance(v, torch.SymInt):
        return int(v)
    if isinstance(v, torch.SymBool):
        return bool(v)
    return float(v) if isinstance(v, torch.SymFloat) else v


def _resolve(entry: Any) -> Any:
    # a specialization entry of the symbolic run as Triton's key reads it: a
    # width class decided, a constexpr's value pinned
    if type(entry) is _LazyIntType:
        return _int_type(entry.value)
    if isinstance(entry, _SYM_TYPES):
        return _pinned(entry)
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
    from triton.tools.tensor_descriptor import TensorDescriptor

    name = jit.fn.__name__

    # phase (a), the call: its declines are the trace's
    def decline(why: str) -> NoReturn:
        raise tr.decline(f"Triton kernel {name}: {why}")

    if knobs.runtime.add_stages_inspection_hook is not None:
        decline("add_stages_inspection_hook adds a pipeline hash to the compile key")
    if (current := torch.cuda.current_device()) != tr.device.index:
        decline(f"launched on cuda:{current}, not the trace's device")
    tr.check_stream(f"Triton kernel {name}: launched")

    def foreign(v: Any) -> bool:
        if isinstance(v, _SYM_TYPES):
            return v.node.shape_env is not tr.shape_env
        return isinstance(v, (tuple, list)) and any(foreign(e) for e in v)

    for k, v in kwargs.items():
        if foreign(v):
            decline(f"{k} is a value of another trace")
    # a launch option computed from sizes is pinned
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

    symbolic = {}
    for arg, v in bound.items():
        if foreign(v) or (isinstance(v, TensorDescriptor) and foreign([*v.shape, *v.strides])):
            decline(f"argument {arg} holds a value of another trace")
        if isinstance(v, _TracedTensor):
            symbolic[arg] = _PointerStandIn(v.dtype, _probe_address(v))
        elif isinstance(v, torch.Tensor):
            decline(f"argument {arg} is a tensor the trace does not track")
        elif isinstance(v, torch.SymFloat):
            decline(f"float argument {arg} is a SymFloat")
        elif _traced_leaf(v) and not isinstance(v, _SYM_TYPES):
            decline(f"tuple argument {arg} holds tensors or values of the trace")
        else:
            symbolic[arg] = v
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
            return _launch(tr, jit, bound, symbolic, options, dims)
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
    from triton.runtime.jit import compute_cache_key

    kernel_cache, kernel_key_cache, _, backend, binder = jit.device_caches[device]
    # JITFunction.run's own options for its binder
    opts = dict(options)
    opts["debug"] = opts.get("debug", jit.debug) or knobs.runtime.debug
    opts["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    symbolic_globals = binder.__globals__ | {
        "specialize_impl": _SymbolicSpecialization(backend)
    }
    symbolic_binder = types.FunctionType(
        binder.__code__, symbolic_globals, binder.__name__, binder.__defaults__
    )
    _, entries, _ = symbolic_binder(**symbolic, **opts)
    # the specialization key, each value it read a guard; then JITFunction.run's
    # lookup and compile of that key, a constexpr's value (and an integer
    # specialized to 1) the key's
    specialization = [_resolve(e) for e in entries]
    values = {
        p.name: entry[1] if entry[0] == "constexpr" else bound[p.name]
        for p, entry in zip(jit.params, specialization, strict=True)
    }
    key = compute_cache_key(kernel_key_cache, specialization, opts)
    binary = kernel_cache.get(key)
    if binary is None:
        parsed, signature, constexprs, attrs = jit._pack_args(backend, opts, values, specialization, opts)
        binary = jit._do_compile(key, signature, device, constexprs, parsed, attrs, True)
    if binary is None:
        decline("Triton compiled nothing (a jit cache hook intervened)")
    if dims is None:
        # compile-only (JITFunction.warmup): the binary, fixed by the guards
        # above, and no launch
        return binary

    abi = triton_abi(binary.src, binary.metadata)
    binary._init_handles()
    owned = owned_module(binary)

    def launcher(stream: int, args: dict, grid: tuple[int, int, int]) -> None:
        # JITFunction.run's launch of the binary, with no hooks (declined above)
        binary.run(*grid, stream, binary.function, binary.packed_metadata, None, None, None, *args.values())

    record_launch(tr, name, owned.function, owned, abi, bound, dims, launcher)
    return binary


def _descriptor(a: Any, v: Any, trusted: bool, decline: Any, edits: bool) -> tuple[int, int, tuple[Any, ...], tuple[Any, ...]]:
    """The descriptor argument `a` at the TensorDescriptor `v`, encoded as
    Triton's launcher encodes it: its CUtensorMapDataType, its out-of-bounds
    fill, its launcher slots (shape, then strides) and its encode's slots
    (address, extents innermost first, byte strides of all but the
    innermost), each condition of a valid encode a guard (under `trusted`
    inputs the caller vouches for its layouts, never its address). Without `edits`
    (tma_library_bits) the launch takes the driver's map as is: only where
    Triton leaves it so."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    d = a.descriptor
    if type(v) is not TensorDescriptor or not isinstance(v.base, _TracedTensor):
        decline(f"descriptor argument {a.name} is a {type(v).__name__} over no traced tensor")
    if tuple(v.block_shape) != tuple(reversed(d.box)) or v.base.element_size() != d.elem_size:
        decline(f"descriptor argument {a.name} is not the compiled block and element size")
    if len(v.shape) != d.rank or len(v.strides) != d.rank:
        decline(f"descriptor argument {a.name} is not of rank {d.rank}")

    def require(cond: Any, why: str) -> None:
        if not bool(cond):
            decline(f"descriptor argument {a.name}: {why}")

    e = d.elem_size
    shape, strides = list(v.shape), list(v.strides)
    extents = shape[::-1]
    byte_strides = [e * x for x in strides[-2::-1]]
    require(_probe_address(v.base) % 16 == 0, "the address is not a multiple of 16")
    if not trusted:
        require(strides[-1] == 1, "the innermost stride is not 1")
        for x in extents:
            require(x >= 1 and x <= 2**31 - 1, "an extent is outside [1, 2**31 - 1]")
        for x in byte_strides:
            require(x % 16 == 0 and x >= 0 and x <= 2**40 - 1, "a byte stride is not a multiple of 16 in [0, 2**40 - 1]")
    if _driver_version() <= 13010 and not edits:
        # fillTMADescriptorTiled (as CUTLASS) clears bit 21 of the map's
        # second word where the C int max byte index + 1 is under 128 KiB
        from torch.fx.experimental.symbolic_shapes import sym_and

        index = sum((x - 1) * b for x, b in zip(extents, [e, *byte_strides])) % 2**32
        require(sym_and(index >= 2**17 - 1, index < 2**31), "Triton clears bit 21 of its map (tma_library_bits off)")
    from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST, TMA_TF32

    # make_tensordesc_arg's: a tf32-rounded element type is mapped too
    dtype = TMA_DTYPE_DEVICE_TO_HOST[TMA_TF32 if v.round_f32_to_tf32 else d.elem_type]
    fill = 1 if v.padding == "nan" else 0  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    return dtype, fill, (*shape, *strides), (v.base.data_ptr(), *extents, *byte_strides)


def record_launch(
    tr: _Trace,
    name: str,
    function: int,
    owner: Any,
    abi: TritonABI,
    values: Mapping[str, Any],
    dims: tuple[Any, Any, Any],
    launch_with: Callable[[int, dict[str, Any], tuple[int, int, int]], None],
) -> None:
    """Record the launch of `function` on the grid `dims`, each argument of
    the ABI read from `values` by name (a constexpr absent from `values` is
    the compiled one). The caller has guarded every specialization, or
    vouches for it. `owner` keeps `function` loaded while the tape lives.
    `launch_with(stream, args, grid)` is the launch through its own launcher
    (Triton's, or the runner an Inductor launcher calls) on the raw stream
    handle `stream`, `args` by name: the capture checks the record against it
    at the traced call (KernelLaunch.witness)."""

    def decline(why: str) -> NoReturn:
        raise declined(f"Triton kernel {name}: {why}")

    # under trusted inputs the caller sized the slots and grids it compiled:
    # their limits are not checked
    trusted = tr.trusted is not None
    slots: list[Any] = [0] * abi.num_slots  # the launcher's scratch pointers stay null
    roots: list[_Root] = []
    pointers: set[int] = set()
    descriptors: list[tuple[int, Any, tuple[Any, ...]]] = []
    own = torch.cuda._host_trace.triton_tma_encode and getattr(torch._C, "_host_trace_tma_flavors", False)
    edits = tma_edits()
    for a in abi.args:
        v = values.get(a.name, a.constant)
        if a.descriptor is not None:
            if not own:
                decline("its descriptors launch as Triton encodes them only with triton_tma_encode, in a native replay that takes it")
            if edits is None:
                decline("tma_library_bits is off, and the native replay launches no descriptor as the driver encodes it")
            dtype, fill, launcher, encode = _descriptor(a, v, trusted, decline, edits)
            slots[a.slot + 1 : a.slot + 1 + len(launcher)] = launcher
            desc = a.descriptor
            descriptors.append((a.slot, (dtype, desc.box, desc.swizzle, fill, desc.elem_size), encode))
            if not any(r is v.base._root for r in roots):
                roots.append(v.base._root)
        elif a.slot is None:
            # a constexpr, or an integer specialized to 1; a SymInt's value
            # was pinned by the caller
            h = _pinned(v)
            if type(h) is not type(a.constant) or h != a.constant:
                decline(f"constexpr {a.name} is {h!r}; compiled {a.constant!r}")
        elif a.is_pointer:
            if not isinstance(v, _TracedTensor):
                decline(f"pointer argument {a.name} is a {type(v).__name__}")
            slots[a.slot] = v.data_ptr()
            pointers.add(a.slot)
            if not any(r is v._root for r in roots):
                roots.append(v._root)
        elif a.triton_type not in _INT_BITS:
            if type(v) not in (int, float):
                decline(f"float argument {a.name} is a {type(v).__name__}")
            slots[a.slot] = _float_bits(v, a.triton_type)
        else:
            bits = _INT_BITS[a.triton_type]
            if not trusted and not (bool(v >= -(2 ** (bits - 1))) and bool(v <= 2 ** (bits - 1) - 1)):
                decline(f"integer argument {a.name} is outside its {bits}-bit slot")
            slots[a.slot] = v

    empty = False
    for axis, (extent, limit) in enumerate(zip(dims, _GRID_LIMITS)):
        if not trusted and not (bool(extent >= 0) and bool(extent <= limit)):
            decline(f"grid axis {axis} is outside [0, {limit}]")
        empty = empty or bool(extent == 0)
    if empty:
        # the launcher skips a launch whose grid has no blocks
        return
    # the launcher's volume check multiplies C ints
    gx, gy, gz = dims
    if not trusted and not bool(gx * gy * gz <= 2**31 - 1):
        decline("the grid has more than 2**31 - 1 blocks")

    layout = param_layout(abi, function)
    block = (abi.num_warps * 32, 1, 1)
    fields, tma, address_slots, params = None, [], None, []
    if descriptors:
        # each other parameter a field, then each descriptor's encode slots
        params = sorted(set(range(abi.num_slots)) - {p for p, *_ in descriptors})
        fields = tuple((p, 0, layout[p][1]) for p in params)
        address_slots = {i for i, p in enumerate(params) if p in pointers}
        slots = [slots[p] for p in params]
        for param, args, encode in descriptors:
            address_slots.add(len(slots))
            tma.append(TmaDescriptor(param, len(slots), *args, edits=bool(edits)))
            slots += encode
    # each launcher argument from the record's slots at a call (in their
    # order after the descriptors' moves), or the record's constant
    at = {p: i for i, p in enumerate(params)} if descriptors else {p: p for p in range(abi.num_slots)}
    firsts = {d.param: d.first for d in tma}
    by_name = {a.name: a for a in abi.args}

    def arg_at(k: str, v: Any, slots_at: Sequence[int]) -> Any:
        from triton.tools.tensor_descriptor import TensorDescriptor

        a = by_name.get(k)
        if a is None:
            return v
        if a.slot is None:
            return a.constant
        if a.descriptor is not None:
            rank = a.descriptor.rank
            shape = [slots_at[at[a.slot + 1 + i]] for i in range(rank)]
            strides = [slots_at[at[a.slot + 1 + rank + i]] for i in range(rank)]
            # the encode's address, without a placeholder's top (TmaDescriptor.encode)
            base = _PointerStandIn(v.base.dtype, slots_at[firsts[a.slot]] & _PLACEHOLDER_LOW)
            return TensorDescriptor(base, shape, strides, list(v.block_shape), v.padding, v.round_f32_to_tf32)
        if a.is_pointer or a.triton_type in _INT_BITS:
            return slots_at[at[a.slot]]
        return v  # a float, a constant of the trace

    def witness(stream: int, slots_at: Sequence[int], grid: tuple[int, int, int]) -> None:
        # a value of the trace that is no kernel parameter (an Inductor
        # launcher's grid argument) has no value at the call: not passed
        args = {k: arg_at(k, v, slots_at) for k, v in values.items() if k in by_name or not _traced_leaf(v)}
        launch_with(stream, args, grid)

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
        fields,
        tuple(tma),
        pointers=None if address_slots is None else frozenset(address_slots),
        programmatic=abi.pdl,
        witness=witness,
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
