"""CuTe DSL compiled functions under a host trace.

A function compiled with `cute.compile(..., options="--enable-tvm-ffi")` and
called with traced tensors is not run: its launches are recorded as
KernelLaunch records with `fields`, or the call is an EagerCall that a replay
runs through TVM-FFI, as eager does. A trace-finalize hook on cute.compile
prints the host function with MLIR's printer; MLIR's parser loads it into a
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

`export_to_c` writes the host function beside the object as `<object>.cute_host`;
a TVM-FFI function looked up in a module loaded from the object is intercepted
as the compiled one was, or is an EagerCall naming why.
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import re
import struct
import sys
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch.cuda._host_trace import Declined, ProcessHold
from torch.cuda._host_trace_capture import capture_kernel_nodes, explicit_attributes, KernelNode, pack_params
from torch.cuda._host_trace_launch import _GRID_LIMITS, _probe_address, KernelLaunch, TmaDescriptor
from torch.cuda._host_trace_tape import (
    _hint,
    _PLACEHOLDER_LOW,
    _TracedTensor,
    current_trace,
    EagerCall,
)
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
_MEMREF_DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i8": torch.int8,
    "ui8": torch.uint8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "f8E4M3FN": torch.float8_e4m3fn,
    "f8E5M2": torch.float8_e5m2,
}
_PASS = {"cute.get_scalars", "cute.to_int_tuple", "arith.extsi", "arith.index_cast"}
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
# per TMA format of an atom type, its CUtensorMapDataType and element bytes
_TMA_FORMATS = {
    "U8": (0, 1),
    "U16": (1, 2),
    "U32": (2, 4),
    "S32": (3, 4),
    "S64": (5, 8),
    "F16_RN": (6, 2),
    "F32_RN": (7, 4),
    "BF16_RN": (9, 2),
    "TF32_RN": (11, 4),
}
# CuTe's layout notation in a cute type's parameters: "(?,(8,?{div=8}))"
_LEAF_RE = re.compile(r"\(|\)|,|\?(?:\{[^}]*\})?|-?\d+")
_INT_TYPE_RE = re.compile(r"i(\d+)")


class _Undescribed(Exception):
    """What makes a call an EagerCall."""


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
    (address, extents, byte strides), or an SM100 MMA atom's `zeros`."""

    fields: list[tuple[Any, int, bool]] = field(default_factory=list)
    tma: tuple[int, tuple[int, ...], int] | None = None
    tma_values: tuple[Any, ...] = ()
    zeros: tuple[Any, ...] = ()


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


def _parse_tuple(s: str) -> Any:
    """'(?,(8,?{div=8}))' -> ('?', (8, '?{div=8}')): dynamic leaves as their text."""
    tokens = _LEAF_RE.findall(s)
    pos = 0

    def parse() -> Any:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok != "(":
            return tok if tok.startswith("?") else int(tok)
        items = []
        while tokens[pos] != ")":
            items.append(parse())
            if tokens[pos] == ",":
                pos += 1
        pos += 1
        return tuple(items)

    return parse()


def _flat(v: Any) -> list[Any]:
    if isinstance(v, (tuple, list)):
        return [x for e in v for x in _flat(e)]
    return [v]


def _dynamic(leaf: Any) -> bool:
    return isinstance(leaf, str)


def _static_value(ty: str) -> Any:
    m = re.fullmatch(r'!cute\.(?:int_tuple|shape|stride)<"(.*)">', ty)
    if m is not None:
        v = _parse_tuple(m.group(1))
        return None if any(map(_dynamic, _flat(v))) else v
    m = re.fullmatch(r'!cute\.tile<"\[(.*)\]">', ty)
    if m is not None and "?" not in m.group(1):
        return tuple(int(x.split(":")[0]) for x in m.group(1).split(";"))
    m = re.fullmatch(r'!cute\.tile<"(\d+):\d+">', ty)
    return None if m is None else int(m.group(1))


def _memref_pattern(ty: str) -> tuple[Any, Any]:
    m = re.search(r'"(.*):(.*)"', ty)
    if m is None:
        raise _Undescribed(f"a memref of layout {ty}")
    return _parse_tuple(m.group(1)), _parse_tuple(m.group(2))


def _is_runtime(ty: str) -> bool:
    # kernel formals of fully static types (layouts, swizzles, tiled copies,
    # MMA atoms) have no parameter
    return (
        ty.startswith("!cute.memref")
        or re.fullmatch(r"[if]\d+|index", ty) is not None
        or "?" in ty
    )


def _div(x: Any, y: Any) -> Any:
    # C division (toward zero); a traced operand only when nonnegative
    if type(x) is int and type(y) is int:
        q = abs(x) // abs(y)
        return q if (x < 0) == (y < 0) else -q
    if not (bool(x >= 0) and bool(y > 0)):
        raise NotImplementedError("a division of traced values of either sign")
    return x // y


_BIN: dict[str, Callable[[Any, Any], Any]] = {
    "cute.tuple_add": lambda x, y: x + y,
    "cute.tuple_sub": lambda x, y: x - y,
    "cute.tuple_mul": lambda x, y: x * y,
    "cute.tuple_div": _div,
    "cute.ceil_div": lambda x, y: -((-x) // y),
    "arith.addi": lambda x, y: x + y,
    "arith.subi": lambda x, y: x - y,
    "arith.muli": lambda x, y: x * y,
    "arith.divsi": _div,
    "arith.ceildivsi": lambda x, y: -((-x) // y),
    "arith.remsi": lambda x, y: x - _div(x, y) * y,
    "arith.maxsi": torch.sym_max,
    "arith.minsi": torch.sym_min,
}
# arith.cmpi's signed predicates
_CMP: dict[int, Callable[[Any, Any], Any]] = {
    0: lambda x, y: x == y,
    1: lambda x, y: x != y,
    2: lambda x, y: x < y,
    3: lambda x, y: x <= y,
    4: lambda x, y: x > y,
    5: lambda x, y: x >= y,
}


def _wrap(ty: str, v: Any, predicates: list[tuple[Any, str]]) -> Any:
    """An integer result at its type's bit width, two's complement as eager
    computes it; a traced result predicated in range."""
    if (m := _INT_TYPE_RE.fullmatch(ty)) is not None and m.group(1) != "1":
        widths = [int(m.group(1))]
    elif (m := re.fullmatch(r'!cute\.int_tuple<"(.*)">', ty)) is not None:
        leaves = _flat(_parse_tuple(m.group(1)))
        widths = [64 if "i64" in str(x) else 32 for x in leaves]
    else:
        return v
    out = []
    for x, bits in zip(_flat(v), widths, strict=True):
        half = 2 ** (bits - 1)
        if type(x) is int:
            x = (x + half) % (2 * half) - half
        else:
            predicates.append((x >= -half, f"an i{bits} result >= {-half}"))
            predicates.append((x < half, f"an i{bits} result < {half}"))
        out.append(x)
    return _unflatten(v, out) if isinstance(v, tuple) else out[0]


def _elementwise(f: Callable[[Any, Any], Any], x: Any, y: Any) -> Any:
    if isinstance(x, tuple):
        ys = y if isinstance(y, tuple) else (y,) * len(x)
        return tuple(_elementwise(f, a, b) for a, b in zip(x, ys, strict=True))
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
    h = _hint(v)
    if not bool(v == h):
        raise AssertionError(f"host_trace: {v} is not {h}")
    return int(h)


class _Program:
    """A compile's host function, loaded from MLIR's generic form into a
    context of ours by MLIR's parser and read through the IR: its formals'
    types, its ops with SSA uses as definition indices, its launches.

    The DSL's cute types are opaque in our context, so none of the DSL's
    value casters (whose tensor caster emits ops) applies; only values of
    builtin integer, float, index and integer vector types are ever
    materialized, whose casters only wrap them."""

    def __init__(self, text: str, name: str) -> None:
        from cutlass._mlir._mlir_libs._cutlass_ir._mlir import ir

        self.text, self.name = text, name

        def type_of(t: Any) -> str:
            if isinstance(t, ir.OpaqueType):
                return f"!{t.dialect_namespace}.{t.data}"
            return f"i{t.width}" if isinstance(t, ir.IntegerType) and t.is_signless else str(t)

        def modeled(t: Any) -> bool:
            if isinstance(t, ir.VectorType):
                t = t.element_type
                return isinstance(t, ir.IntegerType) and t.width in (1, 8, 16, 32, 64)
            return isinstance(t, (ir.OpaqueType, ir.IndexType, ir.IntegerType, ir.FloatType))

        def attr(a: Any) -> Any:
            if isinstance(a, (ir.BoolAttr, ir.IntegerAttr)):
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
            if op.name.split(".")[0] in _EFFECT_DIALECTS and op.name not in _EFFECTS:
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

    def evaluate(
        self, bound: tuple[list[Any], list[tuple[Any, str]]], smem: dict[str, int]
    ) -> tuple[list[_Launch], list[tuple[Any, str]]]:
        """Each launch of a call over its `_bind` formals, given each kernel's
        dynamic shared memory as launched, and the predicates it holds under."""
        formals, predicates = bound
        ev = _Eval(self, formals, smem, list(predicates))
        return [ev.launch(op) for op in self.launches], ev.predicates


def _bind(program: _Program, args: Sequence[Any]) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Each formal's value: a memref's view, an integer's value, None for a
    stream; with the compile's specialization of each memref."""
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
            view, specialization = _memref(ty, a, f"arg{i}")
            formals.append(view)
            predicates += specialization
        elif (m := _INT_TYPE_RE.fullmatch(ty)) is not None and int(m.group(1)) in (16, 32, 64):
            if type(a) not in (int, torch.SymInt):
                raise _Undescribed(f"integer parameter {i} is passed a {type(a).__name__}")
            formals.append(a)
        elif ty == "f32":
            # a float the trace holds as a constant, as it holds an ATen call's
            if type(a) is not float:
                raise _Undescribed(f"f32 parameter {i} is passed a {type(a).__name__}")
            formals.append(a)
        else:
            raise _Undescribed(f"a parameter of type {ty}")
    return formals, predicates


def _memref(ty: str, t: _TracedTensor, what: str) -> tuple[dict[str, Any], list[tuple[Any, str]]]:
    element = re.match(r"!cute\.memref<(\w+),", ty)
    if element is None or _MEMREF_DTYPES.get(element.group(1)) != t.dtype:
        raise _Undescribed(f"{what} is {t.dtype} where the compile has {ty}")
    view: dict[str, Any] = {}
    predicates: list[tuple[Any, str]] = []
    for kind, pat, vals in zip(("shape", "stride"), _memref_pattern(ty), (t.shape, t._sym_strides)):
        leaves = _flat(pat)
        if isinstance(pat, tuple) and len(leaves) != len(pat) or len(leaves) != len(vals):
            raise _Undescribed(f"{what}'s {kind} {pat} is not flat over {len(vals)} dims")
        for d, (leaf, v) in enumerate(zip(leaves, vals)):
            if not _dynamic(leaf):
                predicates.append((v == leaf, f"{what}.{kind}({d}) == {leaf}"))
            elif (div := re.search(r"div=(\d+)", leaf)) is not None:
                predicates.append((v % int(div.group(1)) == 0, f"{what}.{kind}({d}) % {div.group(1)} == 0"))
        view[kind] = _unflatten(pat, list(vals))
    if (align := re.search(r"align<(\d+)>", ty)) is not None:
        n = int(align.group(1))
        predicates.append((_probe_address(t) % n == 0, f"{what}'s address % {n} == 0"))
    view["ptr"] = t.data_ptr()
    return view, predicates


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
            statics = [_static_value(ty) for ty in owner.types]
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
            taken = bool(cond)
            self.predicates.append((cond if taken else torch.sym_not(cond), f"a host condition is {taken}"))
            region = op.regions[0 if taken else 1] if len(op.regions) == 2 else []
            yields = [o for o in region if o.name == "scf.yield"]
            vals = [self.operand(yields[0], k) for k in range(len(yields[0].operands))] if yields else []
            return vals[0] if len(vals) == 1 else tuple(vals)
        if n == "arith.constant":
            return self.attr(op, "value", int)
        if n == "cute.kernel_smem_size":
            return self.smem[self.attr(op, "kernel_name", tuple)[-1]]
        if n in _TMA_MAKERS:
            # its layouts are static, in its operand types
            return self.tma(op, self.operand(op, 0))
        a = [self.operand(op, k) for k in range(len(op.operands))]
        if n in _PASS or (n == "cute.make_int_tuple" and len(a) == 1):
            return _wrap(op.types[0], a[0], self.predicates)
        if n in ("cute.make_int_tuple", "vector.from_elements", "cute.make_atom"):
            return tuple(a)  # an atom as its runtime state
        if n == "cute.make_tiled_mma":
            return a[0]
        if n in ("cute.make_shape", "cute.make_stride"):
            # the type's pattern, its dynamic leaves the operands' in order
            m = re.fullmatch(r'!cute\.\w+<"(.*)">', op.types[0])
            pat, given = _parse_tuple(m.group(1)), iter(_flat(tuple(a)))
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
        if n == "cute.get_layout":
            return {"shape": a[0]["shape"], "stride": a[0]["stride"]}
        if n == "cute.make_view":
            return {"ptr": a[0], "shape": a[1]["shape"], "stride": a[1]["stride"]}
        if n == "cute.make_layout" and len(a) == 2:
            return {"shape": a[0], "stride": a[1]}
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
            return _wrap(op.types[0], _elementwise(_BIN[n], a[0], a[1]), self.predicates)
        raise _Undescribed(f"unsupported host op {n}")

    def tma(self, op: _Op, view: dict[str, Any]) -> tuple[_Param, dict[str, Any]]:
        """A tiled TMA atom as its descriptor's parameter over the memref
        `view`, and its coordinate tensor, the memref's shape."""
        atom, memref, smem_layout = op.types[0], self.operand_type(op, 0), self.operand_type(op, 1)
        fmt = re.search(r"tma_format = (\w+)", atom)
        basis = re.search(r'tma_gbasis = <"\(([\d,]*)\):\((.*?)\)">', atom)
        align = re.search(r"align<(\d+)>", memref)
        swizzle = re.match(r'!cute\.composed_layout<"S<(\d),4,3> o ', smem_layout)
        if fmt is None or fmt.group(1) not in _TMA_FORMATS or basis is None:
            raise _Undescribed(f"the TMA atom {atom}")
        if align is None or int(align.group(1)) % 16:
            raise _Undescribed(f"a TMA over {memref}")
        if swizzle is None and "S<" in smem_layout:
            raise _Undescribed(f"a TMA's shared memory swizzle {smem_layout}")
        dtype, size = _TMA_FORMATS[fmt.group(1)]
        box = [int(x) for x in basis.group(1).split(",")]
        if (n := op.attrs.get("num_multicast", 1)) > 1:
            # each CTA of the multicast loads a slice of the outermost box mode
            i = max(i for i, b in enumerate(box) if b > 1)
            if box[i] % n:
                raise _Undescribed(f"a multicast TMA box {box} over {n} CTAs")
            box[i] //= n
        modes = [int(m) for m in re.findall(r"1@(\d+)", basis.group(2))]
        shape, stride = _flat(view["shape"]), _flat(view["stride"])
        if (
            basis.group(2) != ",".join(f"1@{m}" for m in modes)
            or sorted(modes) != list(range(len(shape)))
            or len(box) != len(shape)
        ):
            raise _Undescribed(f"the TMA basis {basis.group(0)} of {len(shape)} modes")
        self.predicates.append((stride[modes[0]] == 1, f"a TMA's mode {modes[0]} stride == 1"))
        dims = [shape[m] for m in modes]
        strides = [stride[m] * size for m in modes[1:]]
        bounds = [(v, 1, 2**32, "extent") for v in dims]
        for v in strides:
            self.predicates.append((v % 16 == 0, "a TMA byte stride % 16 == 0"))
            bounds.append((v, 0, 2**40 - 1, "byte stride"))
        for v, lo, hi, what in bounds:
            self.predicates.append((v >= lo, f"a TMA {what} >= {lo}"))
            self.predicates.append((v <= hi, f"a TMA {what} <= {hi}"))
        encode = (dtype, tuple(box), int(swizzle.group(1)) if swizzle else 0)
        return _Param(tma=encode, tma_values=(view["ptr"], *dims, *strides)), {"shape": tuple(shape)}

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
            if not (_is_runtime(ty) or ty.startswith(_PARAM_STATICS)):
                continue
            v = self.operand(op, k)
            if ty.startswith("!cute.memref"):
                if not isinstance(v, dict):
                    raise _Undescribed(f"a memref argument evaluated to {type(v).__name__}")
                fields = [(v["ptr"], 8, True)]
                for pat, vals in zip(_memref_pattern(ty), (v["shape"], v["stride"])):
                    fields += _leaf_fields(pat, vals)
                params.append(_Param(fields))
            elif (m := _INT_TYPE_RE.fullmatch(ty)) is not None and int(m.group(1)) in (16, 32, 64):
                params.append(_Param([(v, int(m.group(1)) // 8, False)]))
            elif ty == "f32" and type(v) is float:
                params.append(_Param([(struct.unpack("<i", struct.pack("<f", v))[0], 4, False)]))
            elif ty.startswith("!cute.coord_tensor"):
                m = re.fullmatch(r'!cute\.coord_tensor<"([^"?]*)", "([^:]*):([^"?]*)">', ty)
                if m is None:
                    raise _Undescribed(f"a kernel argument of type {ty}")
                params.append(_Param(_leaf_fields(_parse_tuple(m.group(2)), v["shape"])))
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


def _leaf_fields(pat: Any, vals: Any) -> list[tuple[Any, int, bool]]:
    # a layout's dynamic leaves, each an i32 field unless its type says i64
    leaves, flat = _flat(pat), _flat(vals)
    if len(leaves) != len(flat):
        raise _Undescribed(f"a layout {pat} over {len(flat)} values")
    return [(v, 8 if "i64" in leaf else 4, False) for leaf, v in zip(leaves, flat) if _dynamic(leaf)]


# per compiled function: its host program, or why it did not load
_compiles: weakref.WeakKeyDictionary[Any, _Program | str] = weakref.WeakKeyDictionary()
# per module loaded from an exported object: the object's path
_loaded_from: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
SIDECAR = ".cute_host"


def _program(text: str, name: str) -> _Program | str:
    try:
        return _Program(text, name)
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
    launch: _Launch, node: KernelNode, roots: tuple[_Root, ...], owner: Any
) -> KernelLaunch:
    """The launch as a record over the trace's values, checked against its
    kernel node in the stand-in call's capture at the hints."""
    cluster = tuple(map(_pin, launch.cluster))
    attributes = {k: tuple(map(_pin, v)) for k, v in launch.attributes.items()}
    if any(any(v) for v in attributes.values()):
        raise _Undescribed(f"the launch attributes {attributes}")
    block = tuple(map(_pin, launch.block))
    smem = _pin(launch.smem)
    grid = launch.grid
    for axis, (extent, limit, c) in enumerate(zip(grid, _GRID_LIMITS, cluster)):
        _require(extent >= 1, f"grid axis {axis} >= 1")
        _require(extent <= limit, f"grid axis {axis} <= {limit}")
        if c > 1:
            _require(extent % c == 0, f"grid axis {axis} % {c} == 0")
    if len(launch.params) != len(node.layout):
        raise _Undescribed(f"{len(launch.params)} runtime arguments for {len(node.layout)} kernel parameters")
    fields, values, pointers, tma, padding = [], [], [], [], []
    for index, (param, (_, size)) in enumerate(zip(launch.params, node.layout)):
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
                _require(val >= -(2 ** (bits - 1)), f"a {bits}-bit field >= -2**{bits - 1}")
                _require(val <= 2 ** (bits - 1) - 1, f"a {bits}-bit field < 2**{bits - 1}")
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
    for index, param in tma:
        descriptors.append(TmaDescriptor(index, len(values), *param.tma))
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
    )
    evaluated = (tuple(map(_hint, grid)), block, smem, cluster)
    launched = (node.grid, node.block, node.smem, record.cluster)
    if evaluated != launched:
        raise _Undescribed(f"kernel {node.name} launched {launched}; the evaluation {evaluated}")
    slots = [_hint(v) & _PLACEHOLDER_LOW if p else _hint(v) for v, p in zip(values, pointers)]
    packed = pack_params(record, slots, pointers)
    # every byte but padding (whatever the launch left there) is the
    # evaluation's: zero outside its fields, as a replay packs it
    for index, lo, hi in padding:
        packed[index][lo:hi] = node.images[index][lo:hi]
    for index, (data, image) in enumerate(zip(packed, node.images)):
        if data != image:
            raise _Undescribed(f"kernel {node.name} parameter {index} holds other bytes than the evaluation's")
    return record


def _stand_in(t: _TracedTensor) -> torch.Tensor:
    """A tensor at the traced tensor's hint sizes, strides and address; the
    address without its non-canonical top, as TmaDescriptor encodes it. For
    an argument's root that is the caller's live data: nothing reads or
    writes it only because every launch of the call lands in the witness's
    capture, TVM-FFI launching on torch's current stream even inside
    tvm_ffi.use_raw_stream."""
    sizes = [_hint(s) for s in t.shape]
    strides = [_hint(s) for s in t._sym_strides]
    extent = 1 + sum((n - 1) * s for n, s in zip(sizes, strides)) if all(sizes) else 0
    address = _hint(t.data_ptr()) & _PLACEHOLDER_LOW
    storage = torch._C._construct_storage_from_data_pointer(address, t.device, extent * t.element_size())
    with _disable_current_modes():
        return torch.empty(0, dtype=t.dtype, device=t.device).set_(storage, 0, sizes, strides)


def _intercept(
    tr: _Trace, compiled: Any, args: tuple, kwargs: dict, call: Callable[..., Any]
) -> None:
    name = compiled.function_name

    def decline(why: str) -> NoReturn:
        raise tr.decline(f"CuTe function {name}: {why}")

    if (current := torch.cuda.current_device()) != tr.device.index:
        decline(f"called on cuda:{current}, not the trace's device")
    if torch.cuda.current_stream() != tr.stream:
        decline("called on a stream other than the trace's")
    streams = []
    for i, a in enumerate((*args, *kwargs.values())):
        if isinstance(a, torch.Tensor) and not isinstance(a, _TracedTensor):
            decline(f"argument {i} is a tensor the trace does not track")
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
        given = [a for a in args if a is not None]
        bound = _bind(program, given)
        for i, ty in enumerate(program.formals):
            if ty == "!cuda.stream" and len(given) == len(program.formals) and _stream_handle(given[i]) is None:
                decline(f"argument {i} is a stream as a {type(args[i]).__name__}; pass a torch.cuda.Stream or CUstream")
        roots: list[_Root] = []
        for a in args:
            if isinstance(a, _TracedTensor) and not any(r is a._root for r in roots):
                roots.append(a._root)
        stand_ins = [_stand_in(a) if isinstance(a, _TracedTensor) else _hint(a) for a in args]

        def run(side: torch.cuda.Stream) -> None:
            for i, ty in streams:
                stand_ins[i] = side if issubclass(ty, torch.cuda.Stream) else ty(side.cuda_stream)
            with _disable_current_modes():
                call(compiled, *stand_ins)

        nodes = capture_kernel_nodes(run)
        if any(not isinstance(n, KernelNode) or n.attribute("PROGRAMMATIC_STREAM_SERIALIZATION") for n in nodes):
            raise _Undescribed("the call adds a memset or a programmatic dependency")
        if [n.name for n in nodes] != program.kernels:
            raise _Undescribed(f"the call launched {[n.name for n in nodes]}; its host function {program.kernels}")
        launches, predicates = program.evaluate(bound, {node.name: node.smem for node in nodes})
        for cond, what in predicates:
            _require(cond, what)
        records = [_describe(launch, node, tuple(roots), compiled) for launch, node in zip(launches, nodes)]
    except Declined:
        raise
    except Exception as e:
        if torch.cuda._host_trace.raise_unexpected and not isinstance(e, _Undescribed):
            raise
        reason = str(e) if isinstance(e, _Undescribed) else f"{type(e).__name__}: {e}"
        target = ("cute", compiled, tuple(streams), None)
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
    from cutlass.cute import runtime
    from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionBase

    from torch.utils.dlpack import ReadOnlyTensorWrapper

    with _lock:
        if "compile" not in _ORIGINALS:
            _ORIGINALS["compile"] = CompileCallable.__call__
            _ORIGINALS["export"] = TVMFFIJitCompiledFunctionBase.export_to_c
            _ORIGINALS["load"] = ExternalBinaryModule.__new__
            _ORIGINALS["lookup"] = ExternalBinaryModule.__getattr__  # __getitem__ calls it
            _ORIGINALS["fake"] = runtime._FakeTensor.__init__
            _ORIGINALS["from_dlpack"] = runtime.from_dlpack
            _ORIGINALS["read_only"] = ReadOnlyTensorWrapper.__new__
            CompileCallable.__call__ = _compile
            TVMFFIJitCompiledFunctionBase.export_to_c = _export_to_c
            ExternalBinaryModule.__new__ = staticmethod(_load)
            ExternalBinaryModule.__getattr__ = _lookup
            runtime._FakeTensor.__init__ = _fake_tensor
            runtime.from_dlpack = _from_dlpack
            ReadOnlyTensorWrapper.__new__ = staticmethod(_read_only)


def observing() -> bool:
    """Whether cute.compile is observed: what it compiles from now on is
    exported with its host function (quack's jit_cache asks of its entries)."""
    return "compile" in _ORIGINALS


def _concrete(v: Any) -> Any:
    """A trace's SymInt at a compile-time position: its value, guarded, as
    the compile specializes on it (like SymInt.__hash__ under the trace)."""
    if isinstance(v, torch.SymInt) and getattr(v.node.shape_env, "hash_symints_by_value", False):
        return int(v)
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


def _from_dlpack(tensor: Any, *args: Any, **kwargs: Any) -> Any:
    # a TVM-FFI function takes a torch tensor where it takes a CuTe tensor, so
    # a traced tensor bound for one stays traced for _intercept
    if isinstance(tensor, _TracedTensor) and kwargs.get("enable_tvm_ffi") and not kwargs.get("force_tf32"):
        return tensor
    return _ORIGINALS["from_dlpack"](tensor, *args, **kwargs)


def _compile(self: Any, func: Any, *args: Any, **kwargs: Any) -> Any:
    texts: dict[str, str] = {}

    def observe(dsl: Any, module: Any, name: str) -> None:
        # MLIR's printer only: reading the DSL's live values would run its casters
        with contextlib.suppress(Exception):
            for op in module.body.operations:
                op = op.operation
                if op.name == "func.func" and op.attributes["sym_name"].value == name:
                    texts[name] = op.get_asm(print_generic_op_form=True)

    hooks = kwargs.pop("trace_finalize_hooks", None)
    hooks = () if hooks is None else (hooks,) if callable(hooks) else tuple(hooks)
    args, kwargs = _concrete(args), {k: _concrete(v) for k, v in kwargs.items()}
    try:
        result = _ORIGINALS["compile"](self, func, *args, trace_finalize_hooks=(*hooks, observe), **kwargs)
    except Exception as ex:
        if (tr := current_trace()) is None:
            raise
        # the call's SymInts reached the compile through the caller's objects
        # (quack's RMSNorm.N); the eager fallback compiles it with integers
        e = tr.decline(f"a CuTe DSL compile under the trace raised {type(ex).__name__}: {str(ex).splitlines()[0]}")
        e.retry = True  # a later call finds the compile cached
        raise e from ex
    name = getattr(result, "function_name", None)
    if name in texts:
        _compiles[result] = _program(texts[name], name)
    return result


def _export_to_c(self: Any, object_file_path: str, *args: Any, **kwargs: Any) -> None:
    _ORIGINALS["export"](self, object_file_path, *args, **kwargs)
    program = _compiles.get(self)
    if program is not None:
        record = {"name": program.name, "text": program.text} if isinstance(program, _Program) else {"error": program}
        with open(object_file_path + SIDECAR, "w") as f:
            json.dump(record, f)


def _load(cls: type, file_path: str, *args: Any, **kwargs: Any) -> Any:
    # __new__, not __init__: quack's cute_dsl_elf_fix replaces __init__ without calling it
    self = _ORIGINALS["load"](cls)
    _loaded_from[self] = file_path
    return self


def _exported(path: str | None) -> _Program | str:
    if path is None:
        return "its module was loaded before torch.cuda._host_trace_cute was armed"
    try:
        with open(path + SIDECAR) as f:
            record = json.load(f)
    except FileNotFoundError:
        return f"it was loaded from {path} without its host function"
    return record["error"] if "error" in record else _program(record["text"], record["name"])


def _lookup(self: Any, prefix: str) -> Any:
    fn = _ORIGINALS["lookup"](self, prefix)
    if not self.enable_tvm_ffi:
        return fn
    program = _exported(_loaded_from.get(self))
    loaded = _Loaded(fn, program.name if isinstance(program, _Program) else prefix)
    _compiles[loaded] = program
    return loaded


def _tvm_ffi_call(original: Callable[..., Any]) -> Callable[..., Any]:
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        tr = current_trace()
        if tr is None:
            return original(self, *args, **kwargs)
        try:
            run = functools.partial(_intercept, tr, self, args, kwargs, original)
            return tr.op(self, args, kwargs, run, host=True, redo=lambda a, k: call(self, *a, **k))
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

    __call__ = _tvm_ffi_call(lambda self, *args, **kwargs: self.fn(*args, **kwargs))


def _call_classes() -> tuple[type, ...]:
    from cutlass.cutlass_dsl.tvm_ffi_provider import (
        TVMFFIJitCompiledFunction,
        TVMFFIJitCompiledFunctionWithKwargs,
    )

    return TVMFFIJitCompiledFunction, TVMFFIJitCompiledFunctionWithKwargs


def _hook() -> None:
    for cls in _call_classes():
        _ORIGINALS[cls] = cls.__call__
        cls.__call__ = _tvm_ffi_call(cls.__call__)


def _unhook() -> None:
    for cls in _call_classes():
        cls.__call__ = _ORIGINALS[cls]


_hooks = ProcessHold(_hook, _unhook)


@contextlib.contextmanager
def intercepting() -> Iterator[None]:
    """TVM-FFI compiled calls intercepted on the tracing thread while any
    trace runs."""
    install()
    if "cutlass" not in sys.modules:
        yield
        return
    with _hooks:
        yield


install()
