"""CuTe DSL compiled functions under a host trace.

A function compiled with `cute.compile(..., options="--enable-tvm-ffi")` and
called with traced tensors (also one compiled without TVM-FFI and called with
`from_dlpack` tensors of traced tensors) is not run: its launches are recorded
as KernelLaunch records with `fields`, or the call is an EagerCall that a
replay runs as eager does. A trace-finalize hook on cute.compile prints the
host function with MLIR's printer (a cute.experimental one after the DSL's own
passes that add its TMA descriptors); MLIR's parser loads it into a
context of ours, where the DSL's dialects are not registered, and a generic
interpreter over its arith/scf/cute/cuda ops makes it a _Program: per launch
the kernel, its config and parameter fields as expressions over the call's
arguments, and the compile's specialization (static layout leaves, {div=N},
align<N>, scf.if conditions) as predicates the trace guards. At every trace
the call also runs over stand-in tensors at the traced addresses, captured
and never replayed: its kernel nodes give each function and parameter layout,
and must hold the evaluation's launch config and bytes. A tiled TMA atom's
parameter is a CUtensorMap the replay encodes from its memref's address,
extents and strides.

`export_to_c` writes the host function beside the object as `<object>.cute_host`
(and under the object's content hash);
a TVM-FFI function looked up in a module loaded from the object is intercepted
as the compiled one was, or is an EagerCall naming why.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import hashlib
import inspect
import json
import math
import operator
import os
import re
import struct
import sys
import threading
import traceback
import weakref
from dataclasses import dataclass, field
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch.cuda import _host_trace_ir as _ir
from torch.cuda._host_trace import Declined, declined, ProcessHold
from torch.cuda._host_trace_capture import capture_kernel_nodes, explicit_attributes, KernelNode, pack_params
from torch.cuda._host_trace_launch import _GRID_LIMITS, _probe_address, KernelLaunch, tma_edits, TmaDescriptor
from torch.cuda._host_trace_tape import (
    _PLACEHOLDER_LOW,
    _placeholder,
    _sym_key,
    _TracedTensor,
    current_trace,
    EagerCall,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch.cuda._host_trace_tape import _Root, _Trace


# host ops of these dialects have effects; a described call holds only these
_EFFECT_DIALECTS = ("cuda", "func", "llvm", "gpu")
_LAUNCH_ATTRS = {
    "cuda.launch_cfg.programmatic_stream_serialization_allowed",
    "cuda.launch_cfg.cooperative",
    "cuda.launch_cfg.cluster_dim",
}
_EFFECTS = _LAUNCH_ATTRS | {
    "cuda.launch_cfg.create",
    "cuda.launch_ex",
    "cuda.cast",
    "cuda.return_if_error",
    "func.return",
}
# the attributes the interpreter reads, as Python values
_ATTRS = ("value", "predicate", "mode", "kernel_name", "callee", "num_multicast")
_PASS = {"cute.get_scalars", "cute.to_int_tuple", "arith.extsi", "arith.index_cast", "cute.inttoptr", "cute.recast_iter"}
# the ops bit_op evaluates
_BIT_OPS = {"llvm.intr.ctlz", "arith.extui", "arith.trunci", "arith.shli", "arith.andi", "arith.divui", "arith.floordivsi", "cute.assume"}
# kernel operands of static types that still have a parameter: a TMA atom's
# is its descriptor, an SM100 MMA's is zeros
_PARAM_STATICS = (
    "!cute_nvgpu.atom.non_exec_tiled_tma_",
    "!cute.tiled_mma<!cute_nvgpu.sm100.mma",
)
_TMA_MAKERS = (
    "cute_nvgpu.atom.make_non_exec_tiled_tma_load",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_store",
)
# a TMA atom's format (the DSL's TmaDataFormat) -> (CUtensorMapDataType, element bits)
_TMA_FORMATS = {
    "U8": (0, 8),
    "U16": (1, 16),
    "U32": (2, 32),
    "S32": (3, 32),
    "S64": (5, 64),
    "F16_RN": (6, 16),
    "F32_RN": (7, 32),
    "BF16_RN": (9, 16),
    "TF32_RN": (11, 32),
    "U4": (13, 4),  # CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B: 4-bit elements packed in shared memory
}
# CuTe's layout notation in a cute type's parameters: "(?,(8,?{div=8}))"


class _Undescribed(Exception):
    """What makes a call an EagerCall."""


class _Builtin(str):
    """A builtin MLIR type, read through the IR's type classes: its kind ("int" for a signless
    integer, "float", "index") and width; it prints as MLIR does, for messages."""

    kind: str
    width: int

    def __new__(cls, kind: str, width: int, printed: str) -> _Builtin:
        self = super().__new__(cls, printed)
        self.kind, self.width = kind, width
        return self

    def int_of(self, *widths: int) -> bool:
        return self.kind == "int" and (not widths or self.width in widths)


@dataclass(eq=False)
class _Op:
    """A host op as the interpreter reads it: each operand the index of its
    definition in `_Program.defs` (None for a value it does not model), each
    result's type, and its `_ATTRS`."""

    name: str
    operands: list[int | None] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    regions: list[list[_Op]] = field(default_factory=list)


@dataclass
class _Param:
    """A kernel parameter: `fields` (value, bytes, pointer) packed in order,
    or a CUtensorMap encoded as `tma` (dtype, box, swizzle) over `tma_values`
    (address, extents, byte strides), or an SM100 MMA atom's `zeros`, or a
    `stateless` copy atom's (no runtime state: its bytes are whatever the
    launch left there)."""

    fields: list[tuple[Any, int, bool]] = field(default_factory=list)
    tma: tuple[int, tuple[int, ...], int] | None = None
    tma_values: tuple[Any, ...] = ()
    zeros: tuple[Any, ...] = ()
    stateless: bool = False


@dataclass
class _Launch:
    """The seam: one launch of a call as its host function computes it, all
    over the call's traced values. What consumes it (`_describe`) sees nothing
    of how it was computed."""

    kernel: str
    grid: tuple[Any, ...]
    block: tuple[Any, ...]
    smem: Any
    cluster: tuple[Any, ...]
    attributes: dict[str, tuple[Any, ...]]
    params: list[_Param]


def _flat(v: Any) -> list[Any]:
    if isinstance(v, (tuple, list)):
        return [x for e in v for x in _flat(e)]
    return [v]


def _dynamic(leaf: Any) -> bool:
    # a leaf of a layout or int tuple the DSL evaluated (_evaluate_types): an int, {"dyn": width}, or
    # a coordinate tensor's static basis stride {"basis": [scale, mode]}
    return isinstance(leaf, dict) and "dyn" in leaf


def _tuples(x: Any) -> Any:
    # a descriptor's nested lists (JSON) as tuples
    return tuple(map(_tuples, x)) if isinstance(x, list) else x


def _evaluate_types(module: Any, name: str) -> dict[str, Any]:
    """Each layout-like type of the host function `name` (a memref's, a layout's, a coordinate
    tensor's layout, an int tuple, shape or stride, a tile) as the DSL's own layout algebra
    evaluates it, in a copy of the module so the live one is untouched: {"tuple": t}, {"layout":
    [shape, stride]} or {"tile": [layout per mode]}, each leaf an int or {"dyn": width}; a type it
    cannot evaluate is {"error": why}; a pointer type is {"ptr": {"align": bytes, "gmem": bool}}.
    The leaves come from the DSL's get_leaves of each value (unpack_x_tuple is only ever given
    those)."""
    from cutlass import Integer
    from cutlass._mlir import ir
    from cutlass._mlir._mlir_libs._cutlass_ir import _cute, _cute_nvgpu
    from cutlass._mlir.dialects import cute_nvgpu as nv
    from cutlass._mlir.dialects import cute as cute_ir
    from cutlass.cute import AddressSpace, core

    copy = ir.Module.parse(module.operation.get_asm(print_generic_op_form=True), module.context)
    fn = next(o.operation for o in copy.body.operations if o.operation.name == "func.func" and o.operation.attributes["sym_name"].value == name)
    entry = fn.regions[0].blocks[0]
    # (values, index, type) of each block argument and result: a value is only taken (which runs the
    # DSL's caster for its type) where its type is one evaluated here, never an atom's or a copy's
    found: list[tuple[Any, int, Any]] = [(entry.arguments, i, t) for i, t in enumerate(entry.arguments.types)]

    tmas: list[Any] = []  # the TMA atoms' makers

    def walk(block: Any) -> None:
        for op in block.operations:
            results = op.operation.results
            found.extend((results, i, t) for i, t in enumerate(results.types))
            if op.operation.name in _TMA_MAKERS:
                tmas.append(op.operation)
            for region in op.operation.regions:
                for b in region.blocks:
                    walk(b)

    walk(entry)
    kinds = (_cute.MemRefType, _cute.CoordTensorType, _cute.LayoutType, _cute.IntTupleType, _cute.ShapeType, _cute.StrideType, _cute.TileType, _cute.PtrType, _cute.CoordType, _cute.ComposedLayoutType)

    def leaf(x: Any) -> Any:
        if type(x) is int or x is None:  # None: a coordinate's "_"
            return x
        if isinstance(x, Integer) and x.width in (32, 64):
            return {"dyn": x.width}
        if isinstance(x, core.ScaledBasis):
            # its scale an int, or a Ratio as [numerator, denominator] (a sub-byte recast's 1/2@0)
            s = x.value
            if type(s) is int:
                return {"basis": [s, x.mode]}
            if isinstance(s, core._Ratio):
                return {"basis": [[s.numerator, s.denominator], x.mode]}
        raise TypeError(f"a leaf {x!r}")

    def tree(x: Any) -> Any:
        return [tree(e) for e in x] if isinstance(x, tuple) else leaf(x)

    def tile_mode(m: Any) -> Any:
        # a layout's [shape, stride]; None for "_" (the mode untiled); a nested tile's modes as {"tile": [...]}
        if m is None:
            return None
        if isinstance(m, tuple):
            return {"tile": [tile_mode(e) for e in m]}
        return [tree(m.shape), tree(m.stride)]

    def tma_box(op: Any, fmt: str) -> list[int]:
        # CuTe's construct_tma_gbasis: the smem index -> gmem mode map (the CTA value map composed
        # with the right inverse of the smem layout's non-swizzle portion, coalesced, by the DSL),
        # up to its first mode of a scale other than 1; through the gmem strides in TMA elements,
        # coalesced to modes of at most 256; then a size-1 mode per gmem mode it leaves out
        def value(v: Any) -> Any:
            return v if isinstance(v, ir.Value) else v.value

        memref, smem, vmap = (value(op.operands[i]) for i in range(3))
        if _cute.ComposedLayoutType.isinstance(smem.type):
            smem = value(cute_ir.composed_get_outer(smem))
        shape, stride = layout(value(cute_ir.coalesce(value(cute_ir.composition(vmap, value(cute_ir.right_inverse(smem)))))))
        shape, stride = (shape, stride) if isinstance(shape, list) else ([shape], [stride])
        gshape, gstride = layout(value(cute_ir.get_layout(memref)))
        leaf_of: dict[tuple[int, ...], int] = {}

        def index(t: Any, path: tuple[int, ...]) -> None:
            if isinstance(t, list):
                for i, e in enumerate(t):
                    index(e, (*path, i))
            else:
                leaf_of[path] = len(leaf_of)

        index(gshape, ())
        flat_shape, flat_stride = flatten(gshape), flatten(gstride)
        gbits, tbits = _cute.MemRefType(memref.type).value_type.width, _TMA_FORMATS[fmt][1]
        modes, covered = [], set()
        for n, e in zip(shape, stride):
            if not (isinstance(n, int) and isinstance(e, dict) and "basis" in e and e["basis"][0] == 1):
                break
            j = leaf_of[tuple(e["basis"][1])]
            covered.add(j)
            d = flat_stride[j]
            # recast to TMA elements: an upcast divides the contiguous mode's size, the others' strides
            if tbits != gbits:
                r = tbits // gbits
                if tbits % gbits or (d == 1 and n % r):
                    raise ValueError(f"a TMA recast of {gbits}-bit elements to {tbits}")
                n, d = (n // r, 1) if d == 1 else (n, d // r if isinstance(d, int) and d % r == 0 else {"dyn": 0})
            modes.append((n, d))
        if not modes:
            raise ValueError("no smem-contiguous TMA mode")
        box: list[tuple[int, Any]] = []
        for n, d in modes:
            if n == 1:
                continue
            if box and isinstance(d, int) and isinstance(box[-1][1], int) and box[-1][0] * box[-1][1] == d and n * box[-1][0] <= 256:
                box[-1] = (n * box[-1][0], box[-1][1])
            elif box and box[-1][0] == 1:
                box[-1] = (n, d)
            else:
                box.append((n, d))
        sizes = [n for n, _ in box] + [1 for j, (n, d) in enumerate(zip(flat_shape, flat_stride)) if j not in covered and n != 1 and d != 0]
        # TMA's rank limit: the trailing modes grouped
        return sizes if len(sizes) <= 5 else [*sizes[:4], math.prod(sizes[4:])]

    def flatten(t: Any) -> list[Any]:
        return [x for e in t for x in flatten(e)] if isinstance(t, list) else [t]

    def layout(raw: Any) -> list:
        return [tree(core._unpack_x_tuple(cute_ir.get_shape(raw))), tree(core._unpack_x_tuple(cute_ir.get_stride(raw)))]

    out: dict[str, Any] = {}
    # a copy atom whose instances hold no runtime state: a kernel parameter of no meaningful bytes
    stateless = (
        _cute_nvgpu.CopyAtomG2RType,
        _cute_nvgpu.CopyAtomR2GType,
        _cute_nvgpu.CopyAtomS2RType,
        _cute_nvgpu.CopyAtomR2SType,
        _cute_nvgpu.CopyAtomSIMTSyncCopyType,
        _cute_nvgpu.CopyAtomSIMTAsyncCopyType,
        _cute_nvgpu.CopyAtomLdsmType,
        _cute_nvgpu.CopyAtomStsmType,
    )
    # a 32-bit fast divmod divisor (no accessor: compared with the DSL's own type; the power-of-2 one is not)
    divisor32 = _cute.FastDivmodDivisorType.get(32, False, context=module.context)
    for _, _, t in found:
        if any(k.isinstance(t) for k in stateless):
            out[str(t)] = {"atom": "stateless_copy"}
        elif t == divisor32:
            out[str(t)] = {"fast_divmod": 32}
    for op in tmas:
        # its format: the maker's tma_format attribute, else the DSL's default for its value type
        # (the attribute has no accessor: it is compared with each format's attribute)
        t = op.results.types[0]
        kind = next((k for k in (_cute_nvgpu.CopyAtomNonExecTiledTmaLoadType, _cute_nvgpu.CopyAtomNonExecTiledTmaStoreType) if k.isinstance(t)), None)
        if kind is None:
            out[str(t)] = {"error": "a TMA atom of another kind"}
            continue
        if "tma_format" in op.attributes:
            given = op.attributes["tma_format"]
            fmt = next((f for f in nv.TmaDataFormat if given == ir.Attribute.parse(f"#cute_nvgpu.tma_data_format<{f}>", module.context)), None)
        else:
            fmt = nv.TmaDataFormat(_cute_nvgpu.get_default_tma_format(kind(t).value_type, False))
        out[str(t)] = {"tma": {"format": None if fmt is None else str(fmt)}}
    with ir.InsertionPoint(list(entry.operations)[-1]), ir.Location.unknown(module.context):
        for container, i, t in found:
            key = str(t)
            if key in out or not any(kind.isinstance(t) for kind in kinds):
                continue
            try:
                if _cute.PtrType.isinstance(t):
                    # from the type alone: no value is taken
                    ptr = _cute.PtrType(t)
                    out[key] = {"ptr": {"align": ptr.alignment or 1, "gmem": ptr.address_space == AddressSpace.gmem.value}}
                    continue
                v = container[i]
                raw = v if isinstance(v, ir.Value) else v.value
                if _cute.MemRefType.isinstance(t):
                    # its element's bits, and its iterator's alignment (bytes)
                    align = _cute.PtrType(cute_ir.get_iter(raw).type).alignment
                    out[key] = {"layout": layout(cute_ir.get_layout(raw)), "bits": _cute.MemRefType(t).value_type.width, "align": align}
                elif _cute.ComposedLayoutType.isinstance(t):
                    # a swizzled layout's swizzle (bits, base, shift), the inner function of the composition
                    inner = cute_ir.composed_get_inner(raw).type
                    sw = _cute.SwizzleType(inner) if _cute.SwizzleType.isinstance(inner) else None
                    out[key] = {"composed": {"swizzle": None if sw is None else [sw.num_bits, sw.num_base, sw.num_shift]}}
                elif _cute.CoordTensorType.isinstance(t):
                    out[key] = {"layout": layout(cute_ir.get_layout(raw))}
                elif _cute.LayoutType.isinstance(t):
                    out[key] = {"layout": layout(raw)}
                elif _cute.IntTupleType.isinstance(t) or _cute.ShapeType.isinstance(t) or _cute.StrideType.isinstance(t) or _cute.CoordType.isinstance(t):
                    out[key] = {"tuple": tree(core._unpack_x_tuple(raw))}
                elif _cute.TileType.isinstance(t):
                    modes = core._unpack_x_tuple(raw)
                    single = not isinstance(modes, tuple)
                    out[key] = {"tile": [tile_mode(m) for m in ((modes,) if single else modes)], "single": single}
            except Exception as e:
                out[key] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}
        for op in tmas:
            tma = out.get(str(op.results.types[0]), {}).get("tma")
            if tma is not None:
                try:
                    tma["box"] = tma_box(op, tma["format"])
                except Exception as e:
                    tma["box"] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}
    return out




_DUNDERS: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "int_floordiv": operator.floordiv,
    "mod": operator.mod,
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "sym_max": torch.sym_max,
    "sym_min": torch.sym_min,
    "lshift": operator.lshift,
}


def _bin(method: str, x: Any, y: Any) -> Any:
    # x op y; on the IR backend the node's op itself, not SymInt's dunder
    r = _ir.binary(method, x, y)
    return _DUNDERS[method](x, y) if r is NotImplemented else r


def _neg(x: Any) -> Any:
    n = _ir.direct(x)
    return -x if n is None else torch.SymInt(n.neg())


def _div(x: Any, y: Any) -> Any:
    # C division (toward zero); a traced operand only when nonnegative
    if type(x) is int and type(y) is int:
        q = abs(x) // abs(y)
        return q if (x < 0) == (y < 0) else -q
    if not (bool(_bin("ge", x, 0)) and bool(_bin("gt", y, 0))):
        raise NotImplementedError("a division of traced values of either sign")
    return _bin("int_floordiv", x, y)


def _ceil_div(x: Any, y: Any) -> Any:
    return _neg(_bin("int_floordiv", _neg(x), y))


_BIN: dict[str, Callable[[Any, Any], Any]] = {
    "cute.tuple_add": functools.partial(_bin, "add"),
    "cute.tuple_sub": functools.partial(_bin, "sub"),
    "cute.tuple_mul": functools.partial(_bin, "mul"),
    "cute.tuple_div": _div,
    "cute.ceil_div": _ceil_div,
    "arith.addi": functools.partial(_bin, "add"),
    "arith.subi": functools.partial(_bin, "sub"),
    "arith.muli": functools.partial(_bin, "mul"),
    "arith.divsi": _div,
    "arith.ceildivsi": _ceil_div,
    "arith.remsi": lambda x, y: _bin("sub", x, _bin("mul", _div(x, y), y)),
    "arith.maxsi": functools.partial(_bin, "sym_max"),
    "arith.minsi": functools.partial(_bin, "sym_min"),
}
_FLOAT_BIN: dict[str, Callable[[float, float], float]] = {
    "arith.addf": operator.add,
    "arith.subf": operator.sub,
    "arith.mulf": operator.mul,
    "arith.divf": operator.truediv,
}
# arith.cmpi's signed predicates
_CMP: dict[int, Callable[[Any, Any], Any]] = {
    k: functools.partial(_bin, m) for k, m in enumerate(("eq", "ne", "lt", "le", "gt", "ge"))
}


def _wrap(widths: tuple[int, ...] | None, v: Any, predicates: list[tuple[Any, str]]) -> Any:
    """An integer result at its type's bit widths (_Program.widths), two's
    complement as eager computes it; a traced result predicated in range."""
    if widths is None:
        return v
    out = []
    for x, bits in zip(_flat(v), widths, strict=True):
        half = 2 ** (bits - 1)
        if type(x) is int:
            x = (x + half) % (2 * half) - half
        else:
            # a bound the IR's declared domains imply is a predicate it folds
            # to true, no guard: not built
            n = _ir.direct(x)
            lo, hi = (None, None) if n is None else n.env.ctx.bounds(n.node)
            if lo is None or lo < -half:
                predicates.append((_bin("ge", x, -half), f"an i{bits} result >= {-half}"))
            if hi is None or hi > half - 1:
                predicates.append((_bin("le", x, half - 1), f"an i{bits} result < {half}"))
        out.append(x)
    return _unflatten(v, out) if isinstance(v, tuple) else out[0]


def _elementwise(f: Callable[[Any, Any], Any], x: Any, y: Any) -> Any:
    if isinstance(x, tuple):
        ys = y if isinstance(y, tuple) else (y,) * len(x)
        if f is _ceil_div:
            ys += (1,) * (len(x) - len(ys))  # a tile of fewer modes leaves the rest whole
        if len(ys) != len(x):
            raise _Undescribed(f"an elementwise op of {len(x)} modes by {len(ys)}")
        return tuple(_elementwise(f, a, b) for a, b in zip(x, ys))
    return f(x, y)


def _unflatten(like: Any, values: Sequence[Any]) -> Any:
    it = iter(values)

    def go(t: Any) -> Any:
        return tuple(go(e) for e in t) if isinstance(t, tuple) else next(it)

    return go(like)


def _require(cond: Any, what: str) -> None:
    # a guard of the trace
    if not bool(cond):
        raise _Undescribed(f"{what} does not hold")


def _pin(v: Any) -> int:
    # a guard that returns the value
    return int(v)


class _Program:
    """A compile's host function, loaded from MLIR's generic form into a
    context of ours by MLIR's parser and read through the IR: its formals'
    types, its ops with SSA uses as definition indices, its launches.

    The DSL's cute types are opaque in our context, so none of the DSL's
    value casters (whose tensor caster emits ops) applies; only values of
    builtin integer, float, index and integer vector types are ever
    materialized, whose casters only wrap them."""

    def __init__(self, text: str, name: str, spec: list, types: dict[str, Any]) -> None:
        from cutlass._mlir._mlir_libs._cutlass_ir._mlir import ir

        # spec: TVM-FFI's argument spec (_encode_spec), whose checks guard each call; types: each
        # layout-like type as the DSL evaluated it at the compile (_evaluate_types)
        self.text, self.name, self.spec, self.types = text, name, spec, types

        def type_of(t: Any) -> str:
            if isinstance(t, ir.OpaqueType):
                return f"!{t.dialect_namespace}.{t.data}"
            if isinstance(t, ir.IntegerType) and t.is_signless:
                return _Builtin("int", t.width, str(t))
            if isinstance(t, ir.FloatType):
                return _Builtin("float", t.width, str(t))
            return _Builtin("index", 64, str(t)) if isinstance(t, ir.IndexType) else str(t)

        def modeled(t: Any) -> bool:
            if isinstance(t, ir.VectorType):
                t = t.element_type
                return isinstance(t, ir.IntegerType) and t.width in (1, 8, 16, 32, 64)
            return isinstance(t, (ir.OpaqueType, ir.IndexType, ir.IntegerType, ir.FloatType))

        def attr(a: Any) -> Any:
            if isinstance(a, (ir.BoolAttr, ir.IntegerAttr, ir.FloatAttr)):
                return a.value
            if isinstance(a, ir.DenseI32ArrayAttr):
                return tuple(a)
            if isinstance(a, ir.SymbolRefAttr):
                return tuple(a.value)
            return a

        self.defs: list[tuple[_Op | None, int, str]] = []  # (op, result) or (None, formal), and its type
        self.all: list[_Op] = []
        uses: dict[tuple[Any, int], int] = {}
        reads: list[tuple[Any, _Op]] = []

        def define(owner: _Op | None, i: int, t: Any, value: Callable[[], Any]) -> None:
            self.defs.append((owner, i, type_of(t)))
            if modeled(t):
                for use in value().uses:
                    uses[(use.owner.operation, use.operand_number)] = len(self.defs) - 1

        def load(block: Any, into: list[_Op]) -> None:
            for view in block.operations:
                o = view.operation
                op = _Op(o.name)
                self.all.append(op)
                into.append(op)
                reads.append((o, op))
                for k in _ATTRS:
                    with contextlib.suppress(KeyError):
                        op.attrs[k] = attr(o.attributes[k])
                results = o.results
                for i, t in enumerate(results.types):
                    op.types.append(type_of(t))
                    define(op, i, t, lambda i=i: ir.OpResult(results[i]))
                for region in o.regions:
                    op.regions.append([])
                    for b in region.blocks:
                        if len(b.arguments):
                            raise _Undescribed(f"a host region of {o.name} with block arguments")
                        load(b, op.regions[-1])

        ctx = ir._Context()
        ctx.allow_unregistered_dialects = True
        with ctx, ir.Location.unknown(ctx):
            module = ir.Module.parse(text)
            funcs = [o.operation for o in module.body.operations]
            if len(funcs) != 1 or funcs[0].name != "func.func" or funcs[0].attributes["sym_name"].value != name:
                raise _Undescribed(f"the host function {name} did not load")
            entry = funcs[0].regions[0].blocks[0]
            args = entry.arguments
            for j, t in enumerate(args.types):
                define(None, j, t, lambda j=j: ir.BlockArgument(args[j]))
            self.formals = [ty for _, _, ty in self.defs]
            self.ops: list[_Op] = []  # the top level
            load(entry, self.ops)
            for o, op in reads:
                op.operands = [uses.get((o, n)) for n in range(len(o.operands))]
        self.launches = [op for op in self.all if op.name == "cuda.launch_ex"]
        for op in self.all:
            if op.name.split(".")[0] in _EFFECT_DIALECTS and op.name not in _EFFECTS and op.name not in _BIT_OPS:
                raise _Undescribed(f"its host function holds {op.name}")
            if op.name == "cuda.launch_ex" and op not in self.ops:
                raise _Undescribed("a launch under host control flow")
            if op.name == "cuda.launch_cfg.create" and not (
                len(op.operands) == 8
                and (d := op.operands[7]) is not None
                and self.defs[d][0] is None
                and self.defs[d][2] == "!cuda.stream"
            ):
                raise _Undescribed("a launch on a stream the call does not pass")
        self.kernels = [op.attrs["callee"][-1] for op in self.launches]

    def described(self, ty: str, kind: str) -> Any:
        d = self.types.get(ty)
        if d is None or kind not in d:
            raise _Undescribed(f"its compile did not evaluate {ty} as a {kind}" + (f": {d['error']}" if d and "error" in d else ""))
        return _tuples(d[kind])

    def static(self, ty: str) -> Any:
        """A static int tuple's or coordinate's value, a static layout's ({"shape", "stride"}), or a
        static tile's (each mode's size); else None."""
        d = self.types.get(ty, {})
        if "tuple" in d:
            v = _tuples(d["tuple"])
            return v if all(type(x) is int or x is None for x in _flat(v)) else None
        if "layout" in d and "bits" not in d and not ty.startswith("!cute.coord_tensor"):
            shape, stride = _tuples(d["layout"])
            return {"shape": shape, "stride": stride} if all(type(x) is int for x in _flat((shape, stride))) else None
        elif "tile" in d:
            modes = _tuples(d["tile"])
            if not all(isinstance(m, tuple) for m in modes):
                return None
            sizes = tuple(shape for shape, _ in modes)
            v = sizes[0] if d["single"] else sizes
        else:
            return None
        return v if all(type(x) is int for x in _flat(v)) else None

    def widths(self, ty: str) -> tuple[int, ...] | None:
        if isinstance(ty, _Builtin):
            return (ty.width,) if ty.int_of() and ty.width != 1 else None
        if "tuple" not in self.types.get(ty, {}):
            return None
        return tuple(x["dyn"] if _dynamic(x) else 32 for x in _flat(_tuples(self.types[ty]["tuple"])))

    def runtime(self, ty: str) -> bool:
        # a kernel formal of a fully static type (a layout, swizzle, tiled copy, MMA atom) has no parameter
        if isinstance(ty, _Builtin) or ty.startswith("!cute.memref"):
            return True
        d = self.types.get(ty, {})
        if "ptr" in d:
            return True
        return any(_dynamic(x) for kind in ("tuple", "layout", "tile") if kind in d for x in _flat(_tuples(d[kind])))

    def evaluate(
        self, bound: tuple[list[Any], list[tuple[Any, str]]], smem: dict[str, int]
    ) -> tuple[list[_Launch], list[tuple[Any, str]]]:
        """Each launch of a call over its `_bind` formals, given each kernel's
        dynamic shared memory as launched, and the predicates it holds under."""
        formals, predicates = bound
        ev = _Eval(self, formals, smem, list(predicates))
        return [ev.launch(op) for op in self.launches], ev.predicates


def _bind(program: _Program, args: Sequence[Any], pointees: dict[int, _Root]) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Each formal's value: a memref's view, an integer's value, a pointer's
    address, None for a stream; with the compile's specialization of each
    memref and pointer. `pointees`: per pointer argument (by id) its root."""
    streams = [i for i, ty in enumerate(program.formals) if ty == "!cuda.stream"]
    if len(args) == len(program.formals):
        passed = list(args)
    elif len(args) == len(program.formals) - len(streams):
        # TVM-FFI's environment stream
        it = iter(args)
        passed = [None if i in streams else next(it) for i in range(len(program.formals))]
    else:
        raise _Undescribed(f"{len(args)} arguments for {len(program.formals)} parameters")
    formals: list[Any] = []
    predicates: list[tuple[Any, str]] = []
    for i, (ty, a) in enumerate(zip(program.formals, passed)):
        if ty == "!cuda.stream":
            formals.append(None)
        elif ty.startswith("!cute.memref"):
            if not isinstance(a, _TracedTensor):
                raise _Undescribed(f"memref parameter {i} is passed a {type(a).__name__}")
            formals.append(_memref(program.described(ty, "layout"), a, f"arg{i}"))
        elif isinstance(ty, _Builtin) and ty.int_of(16, 32, 64):
            if type(a) not in (int, torch.SymInt):
                raise _Undescribed(f"integer parameter {i} is passed a {type(a).__name__}")
            formals.append(a)
        elif "ptr" in (ptr := program.types.get(ty, {})):
            root, align = pointees.get(id(a)), ptr["ptr"]["align"]
            if root is None or not ptr["ptr"]["gmem"]:
                raise _Undescribed(f"pointer parameter {i} of type {ty} is passed {a}")
            # the alignment the compile assumed; an allocation's or eager output's base is 256-byte
            # aligned, and its symbol no guard's input
            if root.kind != "argument" and 256 % align:
                raise _Undescribed(f"pointer parameter {i} assumes {align}-byte alignment")
            offset = a if root.kind == "argument" else a - root.sym
            if align > 1:
                predicates.append((offset % align == 0, f"pointer parameter {i}'s address % {align} == 0"))
            formals.append(a)
        elif isinstance(ty, _Builtin) and ty.kind == "float" and ty.width == 32:
            # a float the trace holds as a constant, as it holds an ATen call's
            if type(a) is not float:
                raise _Undescribed(f"f32 parameter {i} is passed a {type(a).__name__}")
            formals.append(a)
        else:
            raise _Undescribed(f"a parameter of type {ty}")
    return formals, predicates


def _memref(layout: tuple[Any, Any], t: _TracedTensor, what: str) -> dict[str, Any]:
    # the call's view of the tensor: its sizes and strides nested as the layout's (the spec guards them)
    view: dict[str, Any] = {}
    for kind, pat, vals in zip(("shape", "stride"), layout, (t.shape, t._sym_strides)):
        leaves = _flat(pat)
        if isinstance(pat, tuple) and len(leaves) != len(pat) or len(leaves) != len(vals):
            raise _Undescribed(f"{what}'s {kind} {pat} is not flat over {len(vals)} dims")
        view[kind] = _unflatten(pat, list(vals))
    view["ptr"] = t.data_ptr()
    return view


class _Eval:
    """A host function's values over a call's formals, each op evaluated
    once on demand. Values are ints or SymInts, bools or SymBools, nested
    tuples of them, and views (a dict of ptr, shape, stride)."""

    def __init__(
        self,
        program: _Program,
        formals: Sequence[Any],
        smem: dict[str, int],
        predicates: list[tuple[Any, str]],
    ) -> None:
        self.program = program
        self.formals = formals  # per formal its value; a stream's is None
        self.smem = smem  # per kernel its dynamic shared memory, as launched
        self.predicates = predicates
        self.results: dict[_Op, list[Any]] = {}

    def value(self, d: int) -> Any:
        owner, i, _ = self.program.defs[d]
        if owner is None:
            return self.formals[i]
        if owner not in self.results:
            statics = [self.program.static(ty) for ty in owner.types]
            if len(statics) == 1 and statics[0] is not None:
                self.results[owner] = statics
            else:
                r = self.op(owner)
                self.results[owner] = list(r) if len(owner.types) > 1 else [r]
        return self.results[owner][i]

    def operand(self, op: _Op, k: int) -> Any:
        if (d := op.operands[k]) is None:
            raise _Undescribed(f"operand {k} of {op.name} is a value of a type the interpreter does not model")
        return self.value(d)

    def operand_type(self, op: _Op, k: int) -> str:
        return "" if (d := op.operands[k]) is None else self.program.defs[d][2]

    def attr(self, op: _Op, name: str, kind: type) -> Any:
        v = op.attrs.get(name)
        if not isinstance(v, kind):
            raise _Undescribed(f"{op.name} with {name} = {v}")
        return v

    def op(self, op: _Op) -> Any:
        n = op.name
        if n == "scf.if":
            cond = self.operand(op, 0)
            if isinstance(cond, torch.SymBool) and (vals := self.select(op, cond)) is not None:
                return vals[0] if len(vals) == 1 else tuple(vals)
            taken = bool(cond)
            self.predicates.append((cond if taken else torch.sym_not(cond), f"a host condition is {taken}"))
            region = op.regions[0 if taken else 1] if len(op.regions) == 2 else []
            yields = [o for o in region if o.name == "scf.yield"]
            vals = [self.operand(yields[0], k) for k in range(len(yields[0].operands))] if yields else []
            return vals[0] if len(vals) == 1 else tuple(vals)
        if n == "arith.constant":
            return self.attr(op, "value", (int, float))
        if n == "cute.kernel_smem_size":
            return self.smem[self.attr(op, "kernel_name", tuple)[-1]]
        if n in _TMA_MAKERS:
            # its layouts are static, in its operand types
            return self.tma(op, self.operand(op, 0))
        a = [self.operand(op, k) for k in range(len(op.operands))]
        if n in _PASS:
            return _wrap(self.program.widths(op.types[0]), a[0], self.predicates)
        if n in ("vector.from_elements", "cute.make_atom"):
            return tuple(a)  # an atom as its runtime state
        if n == "cute.make_tiled_mma":
            return a[0]
        if n in ("cute.make_shape", "cute.make_stride", "cute.make_int_tuple"):
            # the type's pattern, its dynamic leaves the operands' in order
            pat, given = self.program.described(op.types[0], "tuple"), iter(_flat(tuple(a)))
            if sum(map(_dynamic, _flat(pat))) != len(_flat(tuple(a))):
                raise _Undescribed(f"{n} of {len(a)} operands into {op.types[0]}")
            return _unflatten(pat, [next(given) if _dynamic(x) else x for x in _flat(pat)])
        if n == "arith.cmpi":
            pred = self.attr(op, "predicate", int)
            if pred not in _CMP:
                raise _Undescribed(f"arith.cmpi of predicate {pred}")
            return _CMP[pred](a[0], a[1])
        if n == "cute.get_iter":
            return a[0]["ptr"]
        if n in ("cute.make_arith_tuple_iter", "cute.deref_arith_tuple_iter"):
            # a counting iterator as its base coordinate; a coordinate tensor's parameter is its shape
            # alone, so the base must be static
            if not all(type(x) is int for x in _flat(a[0])):
                raise _Undescribed(f"{n} of the coordinate {a[0]}")
            return a[0]
        if n == "cute.make_identity_tensor":
            # a coordinate tensor at the zero coordinate over the shape, its strides the type's static bases
            pat = self.program.described(op.types[0], "layout")
            shape, fixed = _flat(a[0]), _flat(pat[0])
            if len(shape) != len(fixed):
                raise _Undescribed(f"{n} of {a[0]} into {op.types[0]}")
            leaves = [s if _dynamic(x) else x for s, x in zip(shape, fixed)]
            return {"ptr": _unflatten(pat[0], [0] * len(fixed)), "shape": _unflatten(pat[0], leaves), "stride": pat[1]}
        if n == "cute.get_layout":
            return {"shape": a[0]["shape"], "stride": a[0]["stride"]}
        if n == "cute.make_view":
            return {"ptr": a[0], "shape": a[1]["shape"], "stride": a[1]["stride"]}
        if n == "cute.make_layout" and len(a) == 2:
            return {"shape": a[0], "stride": a[1]}
        if n == "cute.fast_divmod.create_divisor":
            self.predicates.append((_bin("ge", a[0], 1), "a fast divmod divisor >= 1"))
            return a[0]
        if n == "cute.slice":
            return _slice(a[0], a[1], self.program.types.get(self.operand_type(op, 0), {}).get("bits"))
        if n == "cute.zipped_divide":
            return _zipped_divide(a[0], self.program.types.get(self.operand_type(op, 1), {}))
        if n == "cute.tile_to_shape" and isinstance(a[0], dict):
            v = _tile_to_shape(a[0], _flat(a[1]), _flat(a[2]))
            # a leaf the type fixes (a size-1 mode's stride 0) is the type's
            pat = self.program.described(op.types[0], "layout")
            got, fixed = _flat(v["shape"]) + _flat(v["stride"]), _flat(pat[0]) + _flat(pat[1])
            if len(got) != len(fixed):
                raise _Undescribed(f"{n} into {op.types[0]}")
            leaves = [y if _dynamic(x) else x for x, y in zip(fixed, got)]
            k = len(_flat(pat[0]))
            return {"shape": _unflatten(pat[0], leaves[:k]), "stride": _unflatten(pat[1], leaves[k:])}
        if n == "cute.make_ordered_layout" or (n == "cute.make_layout" and len(a) == 1):
            # compact strides taken in the order's mode order (one-operand make_layout's: column-major); a
            # leaf the type fixes (a size-1 mode's 0) is the type's
            shape = _flat(a[0])
            order = _flat(a[1]) if len(a) == 2 else list(range(len(shape)))
            pat = self.program.described(op.types[0], "layout")
            if len(order) != len(shape) or not all(type(o) is int for o in order):
                raise _Undescribed(f"{n} of the order {a[1]}")
            strides, acc = [0] * len(shape), 1
            for i in sorted(range(len(shape)), key=lambda i: order[i]):
                strides[i], acc = acc, _bin("mul", acc, shape[i])
            fixed = _flat(pat[1])
            return {"shape": _unflatten(pat[0], shape), "stride": _unflatten(pat[1], [s if _dynamic(x) else x for s, x in zip(strides, fixed)])}
        if n in ("cute.get_shape", "cute.get_stride"):
            return a[0]["shape" if n == "cute.get_shape" else "stride"]
        if n == "cute.select":
            modes = self.attr(op, "mode", tuple)
            if isinstance(a[0], dict):
                return {k: tuple(a[0][k][i] for i in modes) for k in ("shape", "stride")}
            return tuple(a[0][i] for i in modes)
        if n == "cute.get_leaves":
            leaves = _flat(a[0])
            if len(leaves) != len(op.types):
                raise _Undescribed(f"{n} of {len(leaves)} leaves into {len(op.types)} results")
            return leaves if len(leaves) > 1 else leaves[0]
        if n == "cute.size":
            x = a[0]["shape"] if isinstance(a[0], dict) else a[0]
            for k in op.attrs.get("mode", ()):
                x = x[k]
            return math.prod(_flat(x))
        if n in _BIN:
            return _wrap(self.program.widths(op.types[0]), _elementwise(_BIN[n], a[0], a[1]), self.predicates)
        if n in _FLOAT_BIN:
            # the call's float scalars are Python floats, never traced
            if not all(type(x) is float for x in a) or (n == "arith.divf" and a[1] == 0):
                raise _Undescribed(f"{n} of {a}")
            r = _FLOAT_BIN[n](a[0], a[1])
            return struct.unpack("<f", struct.pack("<f", r))[0] if op.types[0].width == 32 else r
        if n in _BIT_OPS:
            return self.bit_op(op, a)
        raise _Undescribed(f"unsupported host op {n}")

    def select(self, op: _Op, cond: torch.SymBool) -> list[Any] | None:
        """An scf.if whose branches yield only integers as selects of both
        branches' values, each branch's predicates holding where it is taken;
        None (the caller guards the condition) if either branch fails."""
        inner, stack = set(), list(op.regions)
        while stack:
            for o in stack.pop():
                inner.add(o)
                stack += o.regions
        yields = [[o for o in region if o.name == "scf.yield"] for region in op.regions]
        if len(op.regions) != 2 or any(len(y) != 1 for y in yields):
            return None
        ints = [self.operand_type(y[0], k) for y in yields for k in range(len(y[0].operands))]
        if not all(isinstance(t, _Builtin) and t.int_of() and t.width != 1 for t in ints):
            return None
        arms, conditioned, outer = [], [], self.predicates
        try:
            # values from outside the branches hold unconditionally
            for o in inner:
                for d in o.operands:
                    if d is not None and self.program.defs[d][0] not in inner:
                        self.value(d)
            for y, holds in zip(yields, (cond, torch.sym_not(cond))):
                self.predicates = []
                vals = [self.operand(y[0], k) for k in range(len(y[0].operands))]
                arms.append(vals)
                conditioned += [(torch.sym_not(holds) | p, f"{what} where a host condition is {holds is cond}") for p, what in self.predicates if p is not True]
        except (_Undescribed, NotImplementedError, ZeroDivisionError):
            for o in inner:
                self.results.pop(o, None)
            return None
        finally:
            self.predicates = outer
        outer += conditioned
        return [a if type(a) is int and a == b else torch.cuda._host_trace.select(cond, a, b) for a, b in zip(*arms, strict=True)]

    def bit_op(self, op: _Op, a: list[Any]) -> Any:
        n, preds = op.name, self.predicates
        bits = [t.width if isinstance(t := self.operand_type(op, k), _Builtin) and t.int_of() else None for k in range(len(a))]
        if None in bits or any(isinstance(x, (bool, torch.SymBool)) for x in a):
            raise _Undescribed(f"{n} over {[self.operand_type(op, k) for k in range(len(a))]}")
        x = a[0]
        if n == "cute.assume":
            m = re.fullmatch(r"!cute\.i\d+(?:<divby (\d+)>)?", op.types[0])
            if m is None:
                raise _Undescribed(f"cute.assume into {op.types[0]}")
            if m.group(1) is not None:
                preds.append((_bin("eq", _bin("mod", x, int(m.group(1))), 0), f"an assumed value % {m.group(1)} == 0"))
            return x

        def nonnegative(v: Any, what: str) -> None:
            if type(v) is not int:
                preds.append((_bin("ge", v, 0), f"{what} >= 0"))

        if n == "llvm.intr.ctlz":
            # a traced operand's is the width less its bit length, its sign a guard
            w = bits[0]
            if type(x) is not int:
                return _bin("sub", w, torch.cuda._host_trace.bit_length(x)) if bool(_bin("ge", x, 0)) else 0
            return 0 if x < 0 else w - x.bit_length()
        if n == "arith.extui":
            if type(x) is int:
                return x % 2 ** bits[0]
            nonnegative(x, "a zero-extended value")
            return x
        if n == "arith.trunci":
            # two's complement at the result's width
            span = 2 ** op.types[0].width
            if type(x) is not int:
                return _bin("sub", _bin("mod", _bin("add", x, span // 2), span), span // 2)
            return _bin("sub", x, (x + span // 2) // span * span)
        y = a[1]
        if n == "arith.shli":
            if type(y) is not int:
                preds.append((_bin("ge", y, 0), "a shift amount >= 0"))
                preds.append((_bin("lt", y, bits[0]), f"a shift amount < {bits[0]}"))
            return _wrap(self.program.widths(op.types[0]), _bin("mul", x, 2**y), preds)
        if n == "arith.andi":
            if type(x) is int and type(y) is int:
                return _wrap(self.program.widths(op.types[0]), x & y, preds)
            v, mask = (x, y) if type(y) is int else (y, x)
            if type(mask) is not int or mask < 0 or mask & (mask + 1):
                raise _Undescribed("a mask of a traced value other than 2**k - 1")
            nonnegative(v, "a masked value")
            return _bin("mod", v, mask + 1)
        if n == "arith.divui":
            if type(x) is int and type(y) is int:
                return x % 2 ** bits[0] // (y % 2 ** bits[1])
            nonnegative(x, "an unsigned dividend")
            nonnegative(y, "an unsigned divisor")
            return _div(x, y)
        if n == "arith.floordivsi":
            return _wrap(self.program.widths(op.types[0]), _bin("int_floordiv", x, y), preds)
        raise _Undescribed(f"unsupported host op {n}")

    def tma(self, op: _Op, view: dict[str, Any]) -> tuple[_Param, dict[str, Any]]:
        """A tiled TMA atom as its descriptor's parameter over the memref
        `view`, and its coordinate tensor: the memref's shape, its type's
        basis strides. The descriptor's modes are CuTe's
        fill_tma_gmem_shape_stride over the coordinate tensor's layout, as the
        DSL evaluates it: leaf j of the memref moves TMA coordinate d by its
        stride's scale s_j where that stride is s_j@d, so mode d's extent is
        1 + sum_j (n_j * s_j - max(s_j, 1)) and its stride the memref stride of
        any j over s_j, in TMA elements (a scale below 1 is a sub-element leaf
        of the memref, as fp8 scale factors are of a U16 TMA)."""
        atom, memref, smem_layout = op.types[0], self.operand_type(op, 0), self.operand_type(op, 1)
        fmt = self.program.types.get(atom, {}).get("tma", {}).get("format")
        if fmt not in _TMA_FORMATS:
            raise _Undescribed(f"the TMA atom {atom} of the format {fmt}")
        if self.program.types.get(memref, {}).get("align", 0) % 16:
            raise _Undescribed(f"a TMA over {memref}")
        # a swizzled shared memory layout: S<b,4,3> (CUtensorMapSwizzle b); a plain layout none
        smem = self.program.types.get(smem_layout, {})
        sw = smem["composed"]["swizzle"] if "composed" in smem else [0, 4, 3]
        if "error" in smem or sw is None or sw[1:] != [4, 3]:
            raise _Undescribed(f"a TMA's shared memory swizzle {smem_layout}")
        dtype, bits = _TMA_FORMATS[fmt]
        shape, stride = _flat(view["shape"]), _flat(view["stride"])
        described = self.program.types[memref]
        fixed_shape, fixed_stride = map(_flat, _tuples(described["layout"]))
        pattern, coords = self.program.described(op.types[1], "layout")
        # each memref leaf's TMA mode and scale (p, q: p/q), None where the TMA does not move along it
        scales: list[tuple[int, int, int] | None] = []
        for j, e in enumerate(_flat(coords)):
            if isinstance(e, dict) and len(e["basis"][1]) == 1:
                p, q = e["basis"][0] if isinstance(e["basis"][0], list) else (e["basis"][0], 1)
                if p != 1 and q != 1:
                    raise _Undescribed(f"a TMA coordinate scale {p}/{q}")
                scales.append((e["basis"][1][0], p, q))
            elif e == 0 and j < len(fixed_shape) and (fixed_shape[j] == 1 or fixed_stride[j] == 0):
                scales.append(None)
            else:
                raise _Undescribed(f"the TMA coordinate tensor {op.types[1]} over {memref}")
        rank = 1 + max((x[0] for x in scales if x is not None), default=-1)
        contributions: list[list[tuple[int, int, int]]] = [[] for _ in range(rank)]  # per mode: (leaf, p, q)
        for j, x in enumerate(scales):
            if x is not None:
                contributions[x[0]].append((j, x[1], x[2]))
        if len(scales) != len(shape) or len(fixed_shape) != len(shape) or not all(contributions):
            raise _Undescribed(f"the TMA coordinate tensor {op.types[1]} over {memref}")
        box = self.program.types[atom]["tma"].get("box")
        if not isinstance(box, list) or len(box) != rank:
            raise _Undescribed(f"the TMA atom {atom}'s box {box} over {rank} coordinate modes")
        box = list(box)
        if (n := op.attrs.get("num_multicast", 1)) > 1:
            # each CTA of the multicast loads a slice of the outermost box mode
            i = max(i for i, b in enumerate(box) if b > 1)
            if box[i] % n:
                raise _Undescribed(f"a multicast TMA box {box} over {n} CTAs")
            box[i] //= n

        def require(cond: Any, what: str) -> None:
            self.predicates.append((cond, what))

        dims, byte_strides = [], []
        for d, contrib in enumerate(contributions):
            extent = 1
            for j, p, q in contrib:
                if q > 1:
                    require(_bin("eq", _bin("mod", shape[j], q), 0), f"a TMA leaf's extent % {q} == 0")
                    extent = _bin("add", extent, _bin("sub", _bin("int_floordiv", shape[j], q), 1))
                else:
                    extent = _bin("add", extent, _bin("mul", _bin("sub", shape[j], 1), p))
            # mode d's stride in bits: memref stride * its bits * q / p, the same for each leaf of the mode
            over = [_bin("mul", stride[j], described["bits"] * q) for j, p, q in contrib]
            for (j, p, _), b in zip(contrib, over):
                require(_bin("eq", _bin("mod", b, p), 0), f"a TMA leaf's stride divides by its scale {p}")
            unit = [_bin("int_floordiv", b, p) for (_, p, _), b in zip(contrib, over)]
            for u in unit[1:]:
                require(_bin("eq", u, unit[0]), f"a TMA mode {d}'s leaves stride alike")
            if d == 0:
                require(_bin("eq", unit[0], bits), "a TMA's first mode stride == 1 element")
            else:
                require(_bin("eq", _bin("mod", unit[0], 8), 0), f"a TMA mode {d}'s stride is whole bytes")
                byte_strides.append(_bin("int_floordiv", unit[0], 8))
            dims.append(extent)
        # the DSL's inline encode keeps a byte stride in units of 16, rounded
        # down (a size-1 mode's stride needn't be a multiple of 16)
        strides = [_bin("sub", v, _bin("mod", v, 16)) for v in byte_strides]
        bounds = [(v, 1, 2**32, "extent") for v in dims] + [(v, 0, 2**40 - 1, "byte stride") for v in strides]
        for v, lo, hi, what in bounds:
            require(_bin("ge", v, lo), f"a TMA {what} >= {lo}")
            require(_bin("le", v, hi), f"a TMA {what} <= {hi}")
        encode = (dtype, tuple(box), sw[0])
        return _Param(tma=encode, tma_values=(view["ptr"], *dims, *strides)), {"shape": _unflatten(pattern, shape), "stride": coords}

    def launch(self, op: _Op) -> _Launch:
        d = op.operands[0]
        cfg = None if d is None else self.program.defs[d][0]
        if cfg is None or cfg.name != "cuda.launch_cfg.create" or len(cfg.operands) != 8:
            raise _Undescribed(f"a launch config from {cfg and cfg.name}")
        attributes = {
            o.name: tuple(self.operand(o, k) for k in range(1, len(o.operands)))
            for o in self.program.ops
            if o.name in _LAUNCH_ATTRS and o.operands[0] == d
        }
        params = []
        for k in range(1, len(op.operands)):
            ty = self.operand_type(op, k)
            if self.program.types.get(ty, {}).get("atom") == "stateless_copy":
                params.append(_Param(stateless=True))
                continue
            if "fast_divmod" in self.program.types.get(ty, {}):
                params.append(_Param(_fast_divmod_fields(self.operand(op, k))))
                continue
            if not (self.program.runtime(ty) or ty.startswith(_PARAM_STATICS)):
                continue
            v = self.operand(op, k)
            if ty.startswith("!cute.memref"):
                if not isinstance(v, dict):
                    raise _Undescribed(f"a memref argument evaluated to {type(v).__name__}")
                fields = [(v["ptr"], 8, True)]
                for pat, vals in zip(self.program.described(ty, "layout"), (v["shape"], v["stride"])):
                    fields += _leaf_fields(pat, vals)
                params.append(_Param(fields))
            elif isinstance(ty, _Builtin) and ty.int_of(16, 32, 64):
                params.append(_Param([(v, ty.width // 8, False)]))
            elif "ptr" in (ptr := self.program.types.get(ty, {})):
                if not ptr["ptr"]["gmem"]:
                    raise _Undescribed(f"a kernel argument of type {ty}")
                params.append(_Param([(v, 8, True)]))
            elif isinstance(ty, _Builtin) and ty.kind == "float" and ty.width == 32 and type(v) is float:
                params.append(_Param([(struct.unpack("<i", struct.pack("<f", v))[0], 4, False)]))
            elif ty.startswith("!cute.coord_tensor"):
                params.append(_Param(_leaf_fields(self.program.described(ty, "layout")[0], v["shape"])))
            elif ty.startswith(_PARAM_STATICS[0]):
                params.append(v)
            elif ty.startswith(_PARAM_STATICS[1]):
                params.append(_Param(zeros=tuple(_flat(v))))
            else:
                raise _Undescribed(f"a kernel argument of type {ty}")
        return _Launch(
            op.attrs["callee"][-1],
            grid=tuple(self.operand(cfg, k) for k in range(4, 7)),
            block=tuple(self.operand(cfg, k) for k in range(3)),
            smem=self.operand(cfg, 3),
            cluster=attributes.pop("cuda.launch_cfg.cluster_dim", (1, 1, 1)),
            attributes=attributes,
            params=params,
        )


def _fast_divmod_fields(d: Any) -> list[tuple[Any, int, bool]]:
    """A 32-bit fast divmod divisor's kernel parameter as the DSL encodes it: the divisor, its
    multiplier and two shifts (Granlund and Montgomery's unsigned division by an invariant: with
    l = ceil(log2 d), m = floor(2**32 * (2**l - d) / d) + 1, shifts min(l, 1) and max(l - 1, 0));
    the multiplier as its two's complement int32."""
    l = (d - 1).bit_length() if type(d) is int else torch.cuda._host_trace.bit_length(_bin("sub", d, 1))
    # floor(2**32 * t / d) in two halves of 16 bits: no intermediate past 2**48 (the program's are int64)
    t = _bin("mul", _bin("sub", _bin("lshift", 1, l), d), 2**16)
    q, r = _bin("int_floordiv", t, d), _bin("mod", t, d)
    m = _bin("add", _bin("add", _bin("mul", q, 2**16), _bin("int_floordiv", _bin("mul", r, 2**16), d)), 1)
    m = _bin("sub", m, _bin("mul", 2**32, _bin("int_floordiv", m, 2**31)))
    return [(d, 4, False), (m, 4, False), (_bin("sym_min", l, 1), 1, False), (_bin("sym_max", _bin("sub", l, 1), 0), 1, False)]


def _slice(x: Any, coord: Any, bits: int | None) -> Any:
    """CuTe's slice of a layout or memref view by a coordinate: the modes its "_" leaves keep; its
    integer leaves move a view's pointer by their strides (in elements of `bits`)."""
    shape, stride = x["shape"], x["stride"]

    def keep(c: Any, n: Any, d: Any) -> tuple[Any, Any, Any] | None:
        # (kept shape, kept stride, offset) of a mode under its coordinate
        if c is None:
            return n, d, 0
        if isinstance(c, tuple):
            if not isinstance(n, tuple) or len(c) != len(n):
                raise _Undescribed(f"a slice by {coord} of {shape}")
            parts = [keep(ci, ni, di) for ci, ni, di in zip(c, n, d)]
            offset = functools.reduce(lambda u, v: _bin("add", u, v), [o for *_, o in parts], 0)
            kept = [(k, s) for k, s, _ in parts if k is not None]
            if not kept:
                return None, None, offset
            ks, ss = zip(*kept)
            return (ks[0], ss[0], offset) if len(ks) == 1 else (tuple(ks), tuple(ss), offset)
        if isinstance(n, tuple):
            # an integer coordinate of a nested mode is its colexicographic index
            if type(c) is not int:
                raise _Undescribed(f"a slice by {coord} of {shape}")
            offset, rest = 0, c
            for ni, di in zip(_flat(n), _flat(d)):
                if type(ni) is not int and rest != 0:
                    raise _Undescribed(f"a slice by {coord} of {shape}")
                ci, rest = (rest, 0) if type(ni) is not int else (rest % ni, rest // ni)
                offset = _bin("add", offset, _bin("mul", ci, di))
            return None, None, offset
        return None, None, _bin("mul", c, d)

    s, d, offset = keep(coord, shape, stride)
    out = {"shape": s if isinstance(s, tuple) else (s,), "stride": d if isinstance(d, tuple) else (d,)}
    if "ptr" in x:
        if bits is None or bits % 8:
            raise _Undescribed(f"a slice of a view of {bits}-bit elements")
        out["ptr"] = _bin("add", x["ptr"], _bin("mul", offset, bits // 8))
    elif offset != 0:
        raise _Undescribed(f"a slice of a layout at an offset {offset}")
    return out


def _zipped_divide(x: Any, tile: dict[str, Any]) -> Any:
    """CuTe's zipped_divide of a layout or memref view by a tile of contiguous modes (t:1 each):
    ((the tile), (the rest)), mode i's rest ceil(n_i / t_i) at stride s_i * t_i."""
    modes = _tuples(tile.get("tile", ()))
    shape, stride = x["shape"], x["stride"]
    if not modes or not all(isinstance(m, tuple) and type(m[0]) is int and m[1] == 1 for m in modes) or len(modes) > len(shape):
        raise _Undescribed(f"a zipped_divide by the tile {tile}")
    inner = tuple(t for t, _ in modes)
    rest = tuple(_ceil_div(n, t) for (t, _), n in zip(modes, shape)) + tuple(shape[len(modes) :])
    rest_stride = tuple(_bin("mul", d, t) for (t, _), d in zip(modes, stride)) + tuple(stride[len(modes) :])
    out = {"shape": (inner, rest), "stride": (tuple(stride[: len(modes)]), rest_stride)}
    if "ptr" in x:
        out["ptr"] = x["ptr"]
    return out


def _tile_to_shape(block: dict[str, Any], target: list[Any], order: list[Any]) -> dict[str, Any]:
    """CuTe's tile_to_shape: blocked_product(block, make_ordered_layout(ceil_div(target, block), order)):
    mode i is (block mode i, its repeat), the repeats compact in `order` scaled by the block's cosize."""
    shape, stride = _tuples(block["shape"]), _tuples(block["stride"])
    leaves = list(zip(_flat(shape), _flat(stride)))
    if not all(type(n) is int and type(d) is int for n, d in leaves) or not all(type(o) is int for o in order):
        raise _Undescribed(f"a tile_to_shape of {block}")
    cosize = 1 + sum((n - 1) * abs(d) for n, d in leaves)
    rank = len(target)
    modes = list(zip(shape, stride)) + [(1, 0)] * (rank - len(shape))
    if len(order) != rank or len(modes) != rank:
        raise _Undescribed(f"a tile_to_shape of {block} to {rank} modes")
    # each repeat ceil(target / block), as CuTe's tile_to_shape takes it
    repeat = [_ceil_div(t, math.prod(_flat(n)) if isinstance(n, tuple) else n) for (n, _), t in zip(modes, target)]
    strides, acc = [0] * rank, cosize
    for i in sorted(range(rank), key=lambda i: order[i]):
        strides[i], acc = acc, _bin("mul", acc, repeat[i])
    return {"shape": tuple((n, r) for (n, _), r in zip(modes, repeat)), "stride": tuple((d, r) for (_, d), r in zip(modes, strides))}


def _leaf_fields(pat: Any, vals: Any) -> list[tuple[Any, int, bool]]:
    # a layout's dynamic leaves, each an i32 field unless its type says i64
    leaves, flat = _flat(pat), _flat(vals)
    if len(leaves) != len(flat):
        raise _Undescribed(f"a layout {pat} over {len(flat)} values")
    return [(v, leaf["dyn"] // 8, False) for leaf, v in zip(leaves, flat) if _dynamic(leaf)]


# the thread's last compile's inputs to the DSL's compile_and_cache: (function name, signature, args, kwargs, TVM-FFI)
_compile_inputs = threading.local()
# per compiled function: its host program, or why it did not load
_compiles: weakref.WeakKeyDictionary[Any, _Program | str] = weakref.WeakKeyDictionary()
# per module loaded from an exported object: the object's path
_loaded_from: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
SIDECAR = ".cute_host"
# a sidecar's format: its host function, TVM-FFI's argument spec, the DSL's evaluation of its types
# (pointer types too, from 4) and its object's digest
_SIDECAR_VERSION = 5


def _encode_spec(params: Sequence[Any], tvm_ffi: bool) -> list:
    """TVM-FFI's argument spec as JSON: per parameter ["tensor", shape, strides, dtype, bits,
    alignment, DLPack device type], ["scalar", value], ["tuple", params], ["const", value],
    ["stream"], ["env_stream"] or [its kind]; a value an int or ["var", index, dtype, divisibility],
    one index per Var (a cute.SymInt the compile reuses is one Var, which the binder checks equal).
    Without TVM-FFI a tensor's dtype is None: an element type set over storage of its width (fp8
    over bytes) is no other dtype."""
    from cutlass.base_dsl.tvm_ffi_builder import spec

    index: dict[int, int] = {}

    def value(v: Any) -> Any:
        return v if isinstance(v, int) else ["var", index.setdefault(id(v), len(index)), str(v.dtype), v.divisibility or 1]

    def encode(p: Any) -> list:
        if isinstance(p, spec.Tensor):
            strides = None if p.strides is None else [value(v) for v in p.strides]
            dtype = str(p.dtype) if tvm_ffi else None
            return ["tensor", [value(v) for v in p.shape], strides, dtype, p.dtype.bits, p.data_alignment or 1, p.dlpack_device_type]
        if isinstance(p, spec.Var):
            return ["scalar", value(p)]
        if isinstance(p, spec.TupleParam):
            return ["tuple", [encode(q) for q in p.params]]
        if isinstance(p, (spec.ConstNone, spec.ConstInt, spec.ConstBool, spec.ConstFloat)):
            return ["const", getattr(p, "value", None)]
        return ["env_stream" if isinstance(p, spec.EnvStream) else "stream" if isinstance(p, spec.Stream) else type(p).__name__]

    return [encode(p) for p in params]


def _spec_args(spec: list, args: Sequence[Any]) -> tuple[list[Any], list[tuple[Any, str]], dict[int, Any]]:
    """The call's arguments its host function takes, and as guards each check TVM-FFI's binder makes
    of them (static and shared sizes and strides, divisibility, int bounds, alignment, dtype); a
    check that fails at the trace's values declines the call, which eager then rejects. And per
    DataPointer argument (by position) its address, a traced tensor's data_ptr(): the binder takes
    any int there and checks nothing of it."""
    from cutlass import Numeric

    from torch.fx.experimental.symbolic_shapes import sym_or

    seen: dict[int, tuple[Any, str]] = {}
    taken: list[Any] = []
    guards: list[tuple[Any, str]] = []
    pointers: dict[int, Any] = {}

    def value(v: Any, x: Any, what: str, unless: Any = False) -> None:
        # unless: where the binder skips the check (a stride of a size-1 dim)
        def check(cond: Any, why: str) -> None:
            guards.append((sym_or(unless, cond), why if unless is False else f"{why} unless {unless}"))

        if isinstance(v, int):
            check(x == v, f"{what} == {v}")
            return
        _, k, dtype, div = v
        if k in seen:
            check(x == seen[k][0], f"{what} == {seen[k][1]}")
            return
        seen[k] = (x, what)
        if dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32"):
            info = torch.iinfo(getattr(torch, dtype))
            check(x >= info.min, f"{what} >= {info.min}")
            check(x <= info.max, f"{what} <= {info.max}")
        if div > 1:
            check(x % div == 0, f"{what} % {div} == 0")

    def take(p: list, a: Any, what: str, top: int | None = None) -> None:
        a = a.value if isinstance(a, Numeric) else a
        if p[0] == "const":
            if type(a) is not type(p[1]) and not (type(p[1]) is int and isinstance(a, torch.SymInt)):
                raise _Undescribed(f"{what} is a {type(a).__name__} for the constant {p[1]!r}")
            if p[1] is not None:
                guards.append((a == p[1], f"{what} == {p[1]!r}"))
            return
        if p[0] == "tuple":
            if not isinstance(a, tuple) or len(a) != len(p[1]):
                raise _Undescribed(f"{what} is not a tuple of {len(p[1])}")
            for j, (q, x) in enumerate(zip(p[1], a)):
                take(q, x, f"{what}[{j}]")
            return
        if p[0] == "tensor":
            _, shape, strides, dtype, bits, align, device = p
            if not isinstance(a, _TracedTensor):
                raise _Undescribed(f"{what} is a {type(a).__name__} for a tensor")
            # TVM-FFI's dtype names are torch's, but for packed fp4; it takes uint8 for int8 (MLIR is signless)
            name = "float4_e2m1fn_x2" if dtype == "float4_e2m1fnx2" else dtype or ""
            dtypes = {getattr(torch, n, None) for n in ((name.removeprefix("u"), "u" + name.removeprefix("u")) if "int" in name else (name,))}
            if device != 2 or a.dim() != len(shape) or (a.dtype not in dtypes if dtype else a.dtype.itemsize * 8 != bits):
                raise _Undescribed(f"{what} is a {a.dim()}-d {a.dtype} tensor for a {len(shape)}-d {dtype or f'{bits}-bit'} one on DLPack device {device}")
            for d, v in enumerate(shape):
                value(v, a.shape[d], f"{what}.shape[{d}]")
            if strides is None:
                raise _Undescribed(f"{what} must be contiguous, a check of no strides")
            for d, v in enumerate(strides):
                value(v, a._sym_strides[d], f"{what}.strides[{d}]", shape[d] == 1 if isinstance(shape[d], int) else a.shape[d] == 1)
            if align > 1:
                guards.append((_probe_address(a) % align == 0, f"{what}'s address % {align} == 0"))
        elif p[0] == "scalar":
            if type(a) not in (int, float, bool, torch.SymInt):
                raise _Undescribed(f"{what} is a {type(a).__name__} for a scalar")
            if type(a) in (int, torch.SymInt):
                value(p[1], a, what)
        elif p[0] == "DataPointer" and top is not None:
            # an int address is memory the trace does not track (or null): an eager call
            if not isinstance(a, torch.SymInt):
                raise _Undescribed(f"{what} is a {type(a).__name__} for a TVM-FFI DataPointer")
            pointers[top] = a
        elif p[0] != "stream":
            raise _Undescribed(f"{what} is a TVM-FFI {p[0]} parameter")
        taken.append(a)

    params = [p for p in spec if p[0] != "env_stream"]
    if len(args) != len(params):
        raise _Undescribed(f"{len(args)} arguments for {len(params)} parameters")
    for i, (p, a) in enumerate(zip(params, args)):
        take(p, a, f"argument {i}", i)
    return taken, guards, pointers


def _program(text: str, name: str, spec: list, types: dict[str, Any]) -> _Program | str:
    try:
        return _Program(text, name, spec, types)
    except Exception as e:
        if torch.cuda._host_trace.raise_unexpected and not isinstance(e, _Undescribed):
            raise
        return str(e) if isinstance(e, _Undescribed) else f"its host function did not load: {e!r}"


def _stream_handle(a: Any) -> int | None:
    if isinstance(a, torch.cuda.Stream):
        return a.cuda_stream
    if type(a).__name__ == "CUstream" and type(a).__module__.startswith("cuda.bindings"):
        return int(a)
    return None


def _describe(
    tr: _Trace, launch: _Launch, node: KernelNode, roots: tuple[_Root, ...], owner: Any
) -> KernelLaunch:
    """The launch as a record over the trace's values, checked against its
    kernel node in the stand-in call's capture at the traced call's values."""
    cluster = tuple(map(_pin, launch.cluster))
    attributes = {k: tuple(map(_pin, v)) for k, v in launch.attributes.items()}
    # with trace_pdl off the attribute stays and declines the launch below
    pdl_key = "cuda.launch_cfg.programmatic_stream_serialization_allowed"
    programmatic = torch.cuda._host_trace.trace_pdl and bool(attributes.pop(pdl_key, (0,))[0])
    if any(any(v) for v in attributes.values()):
        raise _Undescribed(f"the launch attributes {attributes}")
    block = tuple(map(_pin, launch.block))
    smem = _pin(launch.smem)
    grid = launch.grid
    for axis, (extent, limit, c) in enumerate(zip(grid, _GRID_LIMITS, cluster)):
        _require(_bin("ge", extent, 1), f"grid axis {axis} >= 1")
        _require(_bin("le", extent, limit), f"grid axis {axis} <= {limit}")
        if c > 1:
            _require(_bin("eq", _bin("mod", extent, c), 0), f"grid axis {axis} % {c} == 0")
    if len(launch.params) != len(node.layout):
        raise _Undescribed(f"{len(launch.params)} runtime arguments for {len(node.layout)} kernel parameters")
    fields, values, pointers, tma, padding = [], [], [], [], []
    for index, (param, (_, size)) in enumerate(zip(launch.params, node.layout)):
        if param.stateless:
            padding.append((index, 0, size))
            continue
        if param.tma is not None:
            if size != 128:
                raise _Undescribed(f"a TMA descriptor parameter of {size} bytes")
            tma.append((index, param))
            # the DSL's inline encode writes the first 64 bytes and leaves
            # these as it found them (host stack bytes); a replay launches
            # the driver's encode's, as every CUDA C++ TMA kernel does
            padding.append((index, 64, 128))
        elif any(_pin(x) != 0 for x in param.zeros):
            raise _Undescribed("an MMA atom's state is not all zeros")
        offset = 0
        for val, width, pointer in param.fields:
            padding.append((index, offset, offset + -offset % width))
            offset += -offset % width
            if not pointer:
                bits = 8 * width
                _require(_bin("ge", val, -(2 ** (bits - 1))), f"a {bits}-bit field >= -2**{bits - 1}")
                _require(_bin("le", val, 2 ** (bits - 1) - 1), f"a {bits}-bit field < 2**{bits - 1}")
            fields.append((index, offset, width))
            values.append(val)
            pointers.append(pointer)
            offset += width
        if param.fields:
            if not offset <= size < offset + max(w for _, w, _ in param.fields):
                raise _Undescribed(f"kernel parameter {index} is {size} bytes; its fields {offset}")
            padding.append((index, offset, size))
    # each descriptor's slots follow the fields
    descriptors = []
    if tma and (edits := tma_edits()) is None:
        raise _Undescribed("tma_library_bits is off, and the native replay launches no descriptor as the driver encodes it")
    for index, param in tma:
        descriptors.append(TmaDescriptor(index, len(values), *param.tma, edits=edits))
        values += param.tma_values
        pointers += [True] + [False] * (len(param.tma_values) - 1)
    if not all(isinstance(v, torch.SymInt) for v, p in zip(values, pointers) if p):
        raise _Undescribed(f"a pointer of kernel {node.name} is not a traced tensor's")
    record = KernelLaunch(
        node.name,
        node.function,
        None,
        node.layout,
        grid,  # type: ignore[arg-type]
        block,  # type: ignore[arg-type]
        smem,
        tuple(values),
        roots,
        owner,
        tuple(fields),
        tuple(descriptors),
        explicit_attributes(node),
        frozenset(i for i, p in enumerate(pointers) if p),
        programmatic=programmatic,
    )
    evaluated = (tuple(map(tr.values, grid)), block, smem, cluster)
    launched = (node.grid, node.block, node.smem, record.cluster)
    if evaluated != launched:
        raise _Undescribed(f"kernel {node.name} launched {launched}; the evaluation {evaluated}")
    slots = [_at_call(tr, v) & _PLACEHOLDER_LOW if p else tr.values(v) for v, p in zip(values, pointers)]
    # the DSL's maps carry its bit even where the replay launches the
    # driver's as is: the compare checks the encode's inputs either way
    dsl = tuple(dataclasses.replace(d, edits=True, last=[]) for d in record.descriptors)
    packed = pack_params(dataclasses.replace(record, descriptors=dsl), slots, pointers)
    # every byte but padding (whatever the launch left there) is the
    # evaluation's: zero outside its fields, as a replay packs it
    for index, lo, hi in padding:
        packed[index][lo:hi] = node.images[index][lo:hi]
    # a map's bit 21 of its second word is its encoder's (the driver's never sets it for a U4 map at any
    # size; CUTLASS and Triton clear it by size): no encode input, so not compared
    maps = {d.param for d in record.descriptors}
    for index, (data, image) in enumerate(zip(packed, node.images)):
        if index in maps:
            data, image = bytearray(data), bytearray(image)
            data[10] &= ~0x20
            image[10] &= ~0x20
        if data != image:
            raise _Undescribed(f"kernel {node.name} parameter {index} holds other bytes than the evaluation's")
    return record


def _stand_in(tr: _Trace, t: _TracedTensor) -> torch.Tensor:
    """A tensor at the traced tensor's sizes, strides and address at the
    traced call (_at_call); the address without its non-canonical top, as
    TmaDescriptor encodes it. For
    an argument's root that is the caller's live data: nothing reads or
    writes it only because every launch of the call lands in the witness's
    capture, TVM-FFI launching on torch's current stream even inside
    tvm_ffi.use_raw_stream."""
    sizes = [tr.values(s) for s in t.shape]
    strides = [tr.values(s) for s in t._sym_strides]
    extent = 1 + sum((n - 1) * s for n, s in zip(sizes, strides)) if all(sizes) else 0
    address = _at_call(tr, t.data_ptr()) & _PLACEHOLDER_LOW
    storage = torch._C._construct_storage_from_data_pointer(address, t.device, extent * t.element_size())
    with _disable_current_modes():
        return torch.empty(0, dtype=t.dtype, device=t.device).set_(storage, 0, sizes, strides)


def _fields(a: Any) -> list[Any]:
    # a NamedTuple argument as TVM-FFI passes it: its fields, flattened
    if isinstance(a, tuple) and hasattr(a, "_fields"):
        return [x for f in a for x in _fields(f)]
    from cutlass import Numeric

    return [a.value if isinstance(a, Numeric) else a]


def _at_call(tr: _Trace, v: Any) -> int:
    """v at the traced call's values (CallValues); an address into an allocation's or eager
    output's storage, which no call supplies, at that root's placeholder, as the trace placed it."""
    if isinstance(v, torch.SymInt) and len(roots := _roots_of(tr, v)) == 1 and roots[0].kind != "argument":
        return _placeholder(roots[0]) + tr.values(v - roots[0].sym)
    return tr.values(v)


def _roots_of(tr: _Trace, v: Any) -> list[_Root]:
    # the trace's roots whose base symbol v holds
    key = _sym_key(v)
    held = set() if isinstance(key, int) else key.free_symbols
    roots = (*(i.root for i in tr.inputs), *(a.root for a in tr.allocs), *(t._root for t in tr.eager_outputs))
    return [r for r in roots if _sym_key(r.sym).free_symbols & held]


def _pointee(tr: _Trace, address: torch.SymInt) -> _Root:
    """The root a DataPointer argument points into: the trace's one root whose
    base symbol the address holds (lowering checks it is that base plus an
    offset, as for any pointer slot)."""
    found = _roots_of(tr, address)
    if len(found) != 1:
        raise _Undescribed(f"the DataPointer address {address} is not in one traced tensor's storage")
    if found[0].host:
        raise _Undescribed("a DataPointer into a pinned host argument")
    return found[0]


def _stand_ins(tr: _Trace, a: Any) -> Any:
    if isinstance(a, tuple) and hasattr(a, "_fields"):
        return type(a)(*(_stand_ins(tr, x) for x in a))
    return _stand_in(tr, a) if isinstance(a, _TracedTensor) else tr.values(a)


# an intercept may run outside a trace entry's ballast
@torch.cuda._host_trace.on_a_fresh_stack_chunk(16400)
def _intercept(
    tr: _Trace, compiled: Any, args: tuple, kwargs: dict, call: Callable[..., Any], cute_args: tuple = ()
) -> None:
    name = compiled.function_name
    leaves = [x for a in args for x in _fields(a)]

    def decline(why: str) -> NoReturn:
        raise tr.decline(f"CuTe function {name}: {why}")

    if (current := torch.cuda.current_device()) != tr.device.index:
        decline(f"called on cuda:{current}, not the trace's device")
    tr.check_stream(f"CuTe function {name}: called")
    streams = []
    for i, a in enumerate((*args, *kwargs.values())):
        if isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor):
            decline(f"argument {i} is a tensor the trace does not track")
        if leaves is not args and isinstance(a, tuple) and any(isinstance(x, torch.Tensor) and not isinstance(x, _TracedTensor) for x in _fields(a)):
            decline(f"argument {i} holds a tensor the trace does not track")
        if (handle := _stream_handle(a)) is not None:
            if handle != tr.stream.cuda_stream:
                decline(f"argument {i} is a stream other than the trace's")
            if i >= len(args):
                decline("a stream passed by keyword")
            streams.append((i, type(a)))
    try:
        if kwargs:
            raise _Undescribed("keyword arguments")
        program = _compiles.get(compiled)
        if program is None:
            raise _Undescribed(
                "its compile was not observed (cute.compile ran before torch.cuda._host_trace_cute was armed)"
            )
        if isinstance(program, str):
            raise _Undescribed(program)
        # a parameter compiled as None is no formal: TVM-FFI takes the None and drops it
        given, guards, pointers = _spec_args(program.spec, args)
        pointees = {id(a): _pointee(tr, a) for a in pointers.values()}
        bound = _bind(program, given, pointees)
        bound[1].extend(guards)
        for i, ty in enumerate(program.formals):
            if ty == "!cuda.stream" and len(given) == len(program.formals) and _stream_handle(given[i]) is None:
                decline(f"argument {i} is a stream as a {type(given[i]).__name__}; pass a torch.cuda.Stream or CUstream")
        roots: list[_Root] = []
        for r in (*(a._root for a in leaves if isinstance(a, _TracedTensor)), *pointees.values()):
            if not any(r is s for s in roots):
                roots.append(r)
        # a DataPointer's at its address, as _stand_in places a traced tensor
        stand_ins = [_at_call(tr, a) & _PLACEHOLDER_LOW if i in pointers else _stand_ins(tr, a) for i, a in enumerate(args)]
        for i, arg in cute_args:
            stand_ins[i] = arg.make(stand_ins[i])

        def run(side: torch.cuda.Stream) -> None:
            for i, ty in streams:
                stand_ins[i] = side if issubclass(ty, torch.cuda.Stream) else ty(side.cuda_stream)
            with _disable_current_modes():
                call(compiled, *stand_ins)

        nodes = capture_kernel_nodes(run)
        if any(not isinstance(n, KernelNode) for n in nodes):
            raise _Undescribed("the call adds a memset")
        if not torch.cuda._host_trace.trace_pdl and any(n.attribute("PROGRAMMATIC_STREAM_SERIALIZATION") for n in nodes):
            raise _Undescribed("the call adds a programmatic dependency (trace_pdl)")
        if [n.name for n in nodes] != program.kernels:
            raise _Undescribed(f"the call launched {[n.name for n in nodes]}; its host function {program.kernels}")
        launches, predicates = program.evaluate(bound, {node.name: node.smem for node in nodes})
        for cond, what in predicates:
            _require(cond, what)
        records = [_describe(tr, launch, node, tuple(roots), compiled) for launch, node in zip(launches, nodes)]
    except Declined:
        raise
    except Exception as e:
        if torch.cuda._host_trace.raise_unexpected and not isinstance(e, _Undescribed):
            raise
        reason = str(e) if isinstance(e, _Undescribed) else f"{type(e).__name__}: {e}"
        target = ("cute", compiled, tuple(streams), cute_args)
        tr.record_launch(EagerCall(target, tuple(args), dict(kwargs), (), f"CuTe function {name}: {reason}"))
        return
    for record in records:
        tr.record_launch(record)


_lock = threading.Lock()
# cutlass's own methods, saved at the first hook and never cleared
_ORIGINALS: dict[Any, Any] = {}


def install() -> None:
    """Observe cute.compile from now on: each compiled function's host
    program is kept for its calls under a trace, exported beside its object
    and loaded with it. A process without cutlass loaded has nothing to observe,
    unless torch._native registered a CuTe DSL override: that imports cutlass
    at its first call, which can be a trace's warm-up, and what it compiles or
    loads then must be observed."""
    if "cutlass" not in sys.modules:
        from torch._native import registry

        if not registry.get_dsl_operations("cutedsl"):
            return
        import cutlass  # noqa: F401
    from cutlass.base_dsl.compiler import CompileCallable
    from cutlass.base_dsl.export.external_binary_module import ExternalBinaryModule
    from cutlass.cutlass_dsl.cutlass import CutlassBaseDSL
    from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionBase

    with _lock:
        if "compile" not in _ORIGINALS:
            _ORIGINALS["compile"] = CompileCallable.__call__
            _ORIGINALS["compile_and_cache"] = CutlassBaseDSL.compile_and_cache
            CutlassBaseDSL.compile_and_cache = _compile_and_cache
            _ORIGINALS["export"] = TVMFFIJitCompiledFunctionBase.export_to_c
            _ORIGINALS["load"] = ExternalBinaryModule.__new__
            _ORIGINALS["lookup"] = ExternalBinaryModule.__getattr__  # __getitem__ calls it
            CompileCallable.__call__ = _compile
            TVMFFIJitCompiledFunctionBase.export_to_c = _export_to_c
            ExternalBinaryModule.__new__ = staticmethod(_load)
            ExternalBinaryModule.__getattr__ = _lookup


def observing() -> bool:
    """Whether cute.compile is observed: what it compiles from now on is
    exported with its host function (quack's jit_cache asks of its entries)."""
    return "compile" in _ORIGINALS


def _concrete(v: Any) -> Any:
    """A trace's SymInt at a compile-time position: its value, guarded, as
    the compile specializes on it (like SymInt.__hash__ under the trace)."""
    if isinstance(v, torch.SymInt) and getattr(v.node.shape_env, "hash_symints_by_value", False):
        return int(v)
    if isinstance(v, tuple) and hasattr(v, "_fields"):  # a NamedTuple takes its fields positionally
        return type(v)(*map(_concrete, v))
    if isinstance(v, (tuple, list)):
        return type(v)(_concrete(x) for x in v)
    return v


def _fake_tensor(self: Any, dtype: Any, shape: Any, *, stride: Any = None, **kwargs: Any) -> None:
    _ORIGINALS["fake"](self, dtype, _concrete(shape), stride=_concrete(stride), **kwargs)


def _read_only(cls: type, tensor: torch.Tensor) -> Any:
    # a read-only export of a traced tensor is the traced tensor: a CuTe call
    # takes it as it is, and a DLPack export of it declines
    if isinstance(tensor, _TracedTensor):
        return tensor
    return _ORIGINALS["read_only"](cls, tensor)


def _cute_tensor(tensor: Any, *args: Any, **kwargs: Any) -> Any:
    # from_dlpack's CuTe tensor, however from_dlpack was imported
    if isinstance(tensor, _TracedTensor):
        return _DLPackArg(tensor, args, kwargs)
    return _ORIGINALS["tensor"](tensor, *args, **kwargs)


@dataclass(eq=False)
class _DLPackArg:
    """from_dlpack of a traced tensor: the traced tensor for _intercept (all
    a TVM-FFI function reads of it), and the export, its layout marks and
    element type sets to make the CuTe tensor of a stand-in or a replay's
    tensor for a function compiled without TVM-FFI."""

    tensor: _TracedTensor | None
    args: tuple
    kwargs: dict
    marks: list[tuple[str, tuple, dict]] = field(default_factory=list)

    def mark_layout_dynamic(self, *args: Any, **kwargs: Any) -> _DLPackArg:
        self.marks.append(("mark_layout_dynamic", args, kwargs))
        return self

    def mark_compact_shape_dynamic(self, *args: Any, **kwargs: Any) -> _DLPackArg:
        self.marks.append(("mark_compact_shape_dynamic", args, kwargs))
        return self

    @property
    def element_type(self) -> Any:
        from cutlass.cute.runtime import from_dlpack

        for mark, args, _ in reversed(self.marks):
            if mark == "element_type":
                return args[0]
        with _disable_current_modes():
            return from_dlpack(torch.empty(0, dtype=self.tensor.dtype)).element_type

    @element_type.setter
    def element_type(self, element_type: Any) -> None:
        # from_dlpack's force_tf32, or a narrow type over byte storage
        self.marks.append(("element_type", (element_type,), {}))

    def __getattr__(self, name: str) -> Any:
        tr = current_trace()
        why = f"reads {name} of a from_dlpack tensor of a traced tensor"
        raise tr.decline(why) if tr is not None else AttributeError(why)

    def make(self, t: torch.Tensor) -> Any:
        from cutlass.cute.runtime import from_dlpack

        cute_tensor = from_dlpack(t, *self.args, **self.kwargs)
        for mark, args, kwargs in self.marks:
            if mark == "element_type":
                cute_tensor.element_type = args[0]
            else:
                cute_tensor = getattr(cute_tensor, mark)(*args, **kwargs)
        return cute_tensor


def _compile(self: Any, func: Any, *args: Any, **kwargs: Any) -> Any:
    texts: dict[str, str] = {}
    types: dict[str, dict[str, Any]] = {}

    def observe(dsl: Any, module: Any, name: str) -> None:
        # MLIR's printer only: reading the DSL's live values would run its casters
        with contextlib.suppress(Exception):
            # cute.experimental's passes ahead of cute-to-nvvm make its TMA
            # atoms and add them to the launch: run them on a copy
            head, nvvm, _ = dsl._get_pipeline(None).partition(", cute-to-nvvm")
            if nvvm:
                from cutlass._mlir import ir
                from cutlass._mlir.passmanager import PassManager

                module = ir.Module.parse(module.operation.get_asm(print_generic_op_form=True), module.context)
                PassManager.parse(head + ")", module.context).run(module.operation)
            for op in module.body.operations:
                op = op.operation
                if op.name == "func.func" and op.attributes["sym_name"].value == name:
                    texts[name] = op.get_asm(print_generic_op_form=True)
            types[name] = _evaluate_types(module, name)

    hooks = kwargs.pop("trace_finalize_hooks", None)
    hooks = () if hooks is None else (hooks,) if callable(hooks) else tuple(hooks)
    args, kwargs = _concrete(args), {k: _concrete(v) for k, v in kwargs.items()}
    _compile_inputs.last = None
    try:
        result = _ORIGINALS["compile"](self, func, *args, trace_finalize_hooks=(*hooks, observe), **kwargs)
    except Exception as ex:
        if (tr := current_trace()) is None:
            raise
        # the call's SymInts reached the compile through the caller's objects
        # (quack's RMSNorm.N); the eager fallback compiles it with integers
        e = tr.decline(f"a CuTe DSL compile under the trace raised {type(ex).__name__}: {str(ex).splitlines()[0]}")
        # a later call finds the compile cached, unless its cache is keyed
        # differently under the trace (quack's rmsnorm_bwd on T_hint): one
        # retry per function and replay. A callable made per compile (quack's
        # RMSNormBackward) stands for its class
        fn = inspect.unwrap(getattr(func, "__func__", func))
        key, seen = getattr(fn, "__code__", type(fn)), tr.declined_compiles
        e.retry = seen is None or key not in seen
        if seen is not None:
            seen.add(key)
        raise e from ex
    name = getattr(result, "function_name", None)
    if name in texts:
        from cutlass.cute._tvm_ffi_args_spec_converter import _tvm_ffi_args_spec_converter

        # the facts the compile assumes of its arguments, as TVM-FFI's binder checks them (a compile
        # without TVM-FFI has no binder: its call is guarded as if it had)
        try:
            *inputs, tvm_ffi = _compile_inputs.last
            _compiles[result] = _program(texts[name], name, _encode_spec(_tvm_ffi_args_spec_converter(*inputs)[0], tvm_ffi), types.get(name, {}))
        except Exception as e:
            _compiles[result] = f"its TVM-FFI argument spec did not convert: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
    return result


def _compile_and_cache(self: Any, *args: Any, **kwargs: Any) -> Any:
    # the DSL's compile of a module, whose inputs make TVM-FFI's argument spec
    original = _ORIGINALS["compile_and_cache"]
    bound = inspect.signature(original).bind(self, *args, **kwargs).arguments
    _compile_inputs.last = (bound["function_name"], bound["signature"], list(bound["full_args"]), bound["full_kwargs"] or {}, self.compile_options.enable_tvm_ffi)
    return original(self, *args, **kwargs)


def _digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _by_content(path: str) -> str:
    return os.path.join(os.path.dirname(path), _digest(path) + SIDECAR)


def _export_to_c(self: Any, object_file_path: str, *args: Any, **kwargs: Any) -> None:
    _ORIGINALS["export"](self, object_file_path, *args, **kwargs)
    program = _compiles.get(self)
    if program is not None:
        record = {"name": program.name, "text": program.text, "spec": program.spec, "types": program.types} if isinstance(program, _Program) else {"error": program}
        record |= {"version": _SIDECAR_VERSION, "object": _digest(object_file_path)}
        for path in (object_file_path + SIDECAR, _by_content(object_file_path)):
            with open(path, "w") as f:
                json.dump(record, f)


def _load(cls: type, file_path: str, *args: Any, **kwargs: Any) -> Any:
    # __new__, not __init__: quack's cute_dsl_elf_fix replaces __init__ without calling it
    self = _ORIGINALS["load"](cls)
    _loaded_from[self] = file_path
    return self


def _record(path: str) -> tuple[dict | None, bool]:
    """The current-format record of this object's content (a sidecar beside a re-exported object
    may be another's), and whether an older-format one was found."""
    digest = _digest(path) if os.path.exists(path) else None
    older = False
    for sidecar in (path + SIDECAR, *([_by_content(path)] if digest else [])):
        try:
            with open(sidecar) as f:
                record = json.load(f)
        except FileNotFoundError:
            continue
        older |= record.get("version") != _SIDECAR_VERSION
        if record.get("version") == _SIDECAR_VERSION and record["object"] == digest:
            return record, older
    return None, older


def has_host_function(path: str) -> bool:
    """Whether the object at `path` has its host function beside it in the current format: quack's
    jit_cache recompiles an entry that has not."""
    return _record(path)[0] is not None


def _exported(path: str | None) -> _Program | str:
    if path is None:
        return "its module was loaded before torch.cuda._host_trace_cute was armed"
    record, older = _record(path)
    if record is not None:
        return record["error"] if "error" in record else _program(record["text"], record["name"], record["spec"], record["types"])
    if older:
        return f"it was loaded from {path} with a host function exported without TVM-FFI's argument spec: compile and export it again (clear a jit cache that holds it, e.g. quack's)"
    return f"it was loaded from {path} without its host function"


def _lookup(self: Any, prefix: str) -> Any:
    fn = _ORIGINALS["lookup"](self, prefix)
    if not self.enable_tvm_ffi:
        return fn
    program = _exported(_loaded_from.get(self))
    loaded = _Loaded(fn, program.name if isinstance(program, _Program) else prefix)
    _compiles[loaded] = program
    return loaded


def _tvm_ffi_arg(a: Any) -> Any:
    return a.tensor if isinstance(a, _DLPackArg) and a.kwargs.get("enable_tvm_ffi") else a


def _traced_call(original: Callable[..., Any], tvm_ffi: bool = True) -> Callable[..., Any]:
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        tr = current_trace()
        if tr is not None and tvm_ffi:
            # TVM-FFI reads only a CuTe tensor's DLPack export, its torch tensor's
            args = tuple(map(_tvm_ffi_arg, args))
            kwargs = {k: _tvm_ffi_arg(v) for k, v in kwargs.items()}
        elif tr is not None:
            # a torch tensor: cutlass's TensorAdapter, from_dlpack(t).mark_layout_dynamic()
            args = tuple(_DLPackArg(a, (), {}).mark_layout_dynamic() if isinstance(a, _TracedTensor) else a for a in args)
        cute_args = tuple((i, a) for i, a in enumerate(args) if isinstance(a, _DLPackArg))
        by_keyword = any(isinstance(a, _DLPackArg) or not tvm_ffi and isinstance(a, _TracedTensor) for a in kwargs.values())
        if tr is None or not (tvm_ffi or cute_args or by_keyword):
            return original(self, *args, **kwargs)
        try:
            if tvm_ffi and cute_args or by_keyword:
                where = "to a TVM-FFI function" if tvm_ffi else "by keyword"
                raise tr.decline(f"CuTe function {self.function_name}: a from_dlpack tensor of a traced tensor passed {where}")
            args = tuple(a.tensor if isinstance(a, _DLPackArg) else a for a in args)
            cute_args = tuple((i, dataclasses.replace(a, tensor=None)) for i, a in cute_args)

            def redo(a: tuple, k: dict) -> Any:
                a = list(a)
                for i, arg in cute_args:
                    a[i] = dataclasses.replace(arg, tensor=a[i])
                return call(self, *a, **k)

            run = functools.partial(_intercept, tr, self, args, kwargs, original, cute_args)
            return tr.op(self, args, kwargs, run, host=True, redo=redo)
        except Declined as e:
            if e is tr.declined:
                raise
            raise tr.decline(str(e).removeprefix("host_trace: ").removesuffix(" (declined)")) from e

    return call


class _Loaded:
    """A TVM-FFI function looked up in a module loaded from an exported object."""

    def __init__(self, fn: Any, function_name: str) -> None:
        self.fn = fn
        self.function_name = function_name

    __call__ = _traced_call(lambda self, *args, **kwargs: self.fn(*args, **kwargs))


def _call_classes() -> tuple[type, ...]:
    from cutlass.cutlass_dsl.tvm_ffi_provider import (
        TVMFFIJitCompiledFunction,
        TVMFFIJitCompiledFunctionWithKwargs,
    )

    return TVMFFIJitCompiledFunction, TVMFFIJitCompiledFunctionWithKwargs


def _constructors() -> tuple[tuple[str, Any, str, Any], ...]:
    from cutlass.cute import runtime

    from torch.utils.dlpack import ReadOnlyTensorWrapper

    # runtime._Tensor, not its __new__: from_dlpack constructs it by its global
    # name, and a class's inherited object.__new__, once set, can't be set back
    return (
        ("fake", runtime._FakeTensor, "__init__", _fake_tensor),
        ("tensor", runtime, "_Tensor", _cute_tensor),
        ("read_only", ReadOnlyTensorWrapper, "__new__", staticmethod(_read_only)),
    )


def _hook() -> None:
    from cutlass.base_dsl.jit_executor import JitCompiledFunction

    for key, owner, name, hook in _constructors():
        _ORIGINALS[key] = vars(owner)[name]
        setattr(owner, name, hook)
    for cls in _call_classes():
        _ORIGINALS[cls] = cls.__call__
        cls.__call__ = _traced_call(cls.__call__)
    # compiled without TVM-FFI: the calls that pass a _DLPackArg
    _ORIGINALS[JitCompiledFunction] = JitCompiledFunction.__call__
    JitCompiledFunction.__call__ = _traced_call(JitCompiledFunction.__call__, tvm_ffi=False)


def _unhook() -> None:
    from cutlass.base_dsl.jit_executor import JitCompiledFunction

    for cls in (*_call_classes(), JitCompiledFunction):
        cls.__call__ = _ORIGINALS[cls]
    for key, owner, name, _ in _constructors():
        setattr(owner, name, _ORIGINALS[key])


_hooks = ProcessHold(_hook, _unhook)


class _Foreign:
    """A TVM-FFI function from outside CuTe DSL: a call with traced tensors is an EagerCall."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.function_name = f"{fn!r} (TVM-FFI, not CuTe DSL)"

    def __call__(self, *args: Any) -> Any:
        return self.fn(*args)


def _intercept_foreign(tr: _Trace, foreign: _Foreign, args: tuple) -> None:
    name = foreign.function_name

    def decline(why: str) -> NoReturn:
        raise tr.decline(f"CuTe function {name}: {why}")

    if (current := torch.cuda.current_device()) != tr.device.index:
        decline(f"called on cuda:{current}, not the trace's device")
    tr.check_stream(f"CuTe function {name}: called")
    streams = []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor):
            decline(f"argument {i} is a tensor the trace does not track")
        if not isinstance(a, torch.Tensor) and any(isinstance(x, torch.Tensor) for x in pytree.tree_leaves(a)):
            decline(f"argument {i} holds tensors in a {type(a).__name__}")
        if (handle := _stream_handle(a)) is not None:
            if handle != tr.stream.cuda_stream:
                decline(f"argument {i} is a stream other than the trace's")
            streams.append((i, type(a)))
    stand_ins = [_stand_in(tr, a) if isinstance(a, _TracedTensor) else tr.values(a) for a in args]
    returned = []

    def run(side: torch.cuda.Stream) -> None:
        for i, ty in streams:
            stand_ins[i] = side if issubclass(ty, torch.cuda.Stream) else ty(side.cuda_stream)
        with _disable_current_modes():
            returned.append(foreign(*stand_ins))

    # the stand-in call, captured and never replayed, shows what eager's call
    # returns: a replay's eager step returns nothing
    try:
        capture_kernel_nodes(run)
    except Exception as e:
        decline(f"its call over stand-in tensors under a capture raised {type(e).__name__}: {e}")
    if returned[0] is not None:
        decline(f"it returns a {type(returned[0]).__name__}")
    target = ("cute", foreign, tuple(streams), None)
    tr.record_launch(EagerCall(target, tuple(args), {}, (), f"CuTe function {name}: a TVM-FFI function from outside CuTe DSL"))


def _foreign_call(self: Any, *args: Any) -> Any:
    tr = current_trace()
    args = tuple(map(_tvm_ffi_arg, args)) if tr is not None else args
    if tr is None or not any(isinstance(a, _TracedTensor) for a in pytree.tree_leaves(args)):
        return _ORIGINALS["ffi_call"](self, *args)
    foreign = _Foreign(self)
    try:
        run = functools.partial(_intercept_foreign, tr, foreign, args)
        return tr.op(foreign, args, {}, run, host=True, redo=lambda a, k: _foreign_call(self, *a))
    except Declined as e:
        if e is tr.declined:
            raise
        raise tr.decline(str(e).removeprefix("host_trace: ").removesuffix(" (declined)")) from e


def _ffi_hook() -> None:
    from tvm_ffi.core import Function

    _ORIGINALS.setdefault("ffi_call", Function.__call__)
    Function.__call__ = _foreign_call


def _ffi_unhook() -> None:
    from tvm_ffi.core import Function

    Function.__call__ = _ORIGINALS["ffi_call"]


_ffi_hooks = ProcessHold(_ffi_hook, _ffi_unhook)


@contextlib.contextmanager
def intercepting() -> Iterator[None]:
    """CuTe compiled calls intercepted on the tracing thread while any
    trace runs."""
    install()
    with contextlib.ExitStack() as stack:
        if "cutlass" in sys.modules:
            stack.enter_context(_hooks)
        if "tvm_ffi" in sys.modules:
            stack.enter_context(_ffi_hooks)
        try:
            yield
        except TypeError as e:
            # make_ptr takes an int address, and a traced tensor's is symbolic
            *_, (frame, _) = traceback.walk_tb(e.__traceback__)
            if frame.f_code.co_name == "make_ptr" and "cutlass" in frame.f_code.co_filename:
                raise declined(f"make_ptr from a traced tensor's data_ptr ({e})") from e
            raise


install()
