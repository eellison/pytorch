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
    _hint,
    _PLACEHOLDER_LOW,
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
_PASS = {"cute.get_scalars", "cute.to_int_tuple", "arith.extsi", "arith.index_cast"}
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


@functools.cache
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


@functools.cache
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


@functools.cache
def _memref_pattern(ty: str) -> tuple[Any, Any]:
    m = re.search(r'"(.*):(.*)"', ty)
    if m is None:
        raise _Undescribed(f"a memref of layout {ty}")
    return _parse_tuple(m.group(1)), _parse_tuple(m.group(2))


@functools.cache
def _is_runtime(ty: str) -> bool:
    # kernel formals of fully static types (layouts, swizzles, tiled copies,
    # MMA atoms) have no parameter
    return isinstance(ty, _Builtin) or ty.startswith("!cute.memref") or "?" in ty


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
# arith.cmpi's signed predicates
_CMP: dict[int, Callable[[Any, Any], Any]] = {
    k: functools.partial(_bin, m) for k, m in enumerate(("eq", "ne", "lt", "le", "gt", "ge"))
}


@functools.cache
def _int_widths(ty: str) -> tuple[int, ...] | None:
    if isinstance(ty, _Builtin) and ty.int_of() and ty.width != 1:
        return (ty.width,)
    if (m := re.fullmatch(r'!cute\.int_tuple<"(.*)">', ty)) is not None:
        return tuple(64 if "i64" in str(x) else 32 for x in _flat(_parse_tuple(m.group(1))))
    return None


def _wrap(ty: str, v: Any, predicates: list[tuple[Any, str]]) -> Any:
    """An integer result at its type's bit width, two's complement as eager
    computes it; a traced result predicated in range."""
    if (widths := _int_widths(ty)) is None:
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

    def __init__(self, text: str, name: str, spec: list) -> None:
        from cutlass._mlir._mlir_libs._cutlass_ir._mlir import ir

        # spec: TVM-FFI's argument spec (_encode_spec), whose checks guard each call
        self.text, self.name, self.spec = text, name, spec

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
            formals.append(_memref(ty, a, f"arg{i}"))
        elif isinstance(ty, _Builtin) and ty.int_of(16, 32, 64):
            if type(a) not in (int, torch.SymInt):
                raise _Undescribed(f"integer parameter {i} is passed a {type(a).__name__}")
            formals.append(a)
        elif isinstance(ty, _Builtin) and ty.kind == "float" and ty.width == 32:
            # a float the trace holds as a constant, as it holds an ATen call's
            if type(a) is not float:
                raise _Undescribed(f"f32 parameter {i} is passed a {type(a).__name__}")
            formals.append(a)
        else:
            raise _Undescribed(f"a parameter of type {ty}")
    return formals, predicates


def _memref(ty: str, t: _TracedTensor, what: str) -> dict[str, Any]:
    # the call's view of the tensor: its sizes and strides nested as the layout's (the spec guards them)
    view: dict[str, Any] = {}
    for kind, pat, vals in zip(("shape", "stride"), _memref_pattern(ty), (t.shape, t._sym_strides)):
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
            if isinstance(cond, torch.SymBool) and (vals := self.select(op, cond)) is not None:
                return vals[0] if len(vals) == 1 else tuple(vals)
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
        if n in ("vector.from_elements", "cute.make_atom"):
            return tuple(a)  # an atom as its runtime state
        if n == "cute.make_tiled_mma":
            return a[0]
        if n in ("cute.make_shape", "cute.make_stride", "cute.make_int_tuple"):
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
            m = re.fullmatch(r"!cute\.i\d+<divby (\d+)>", op.types[0])
            if m is None:
                raise _Undescribed(f"cute.assume into {op.types[0]}")
            preds.append((_bin("eq", _bin("mod", x, int(m.group(1))), 0), f"an assumed value % {m.group(1)} == 0"))
            return x

        def nonnegative(v: Any, what: str) -> None:
            if type(v) is not int:
                preds.append((_bin("ge", v, 0), f"{what} >= 0"))

        if n == "llvm.intr.ctlz":
            # a traced operand's is the width less its bit length, guarded to its sign
            h, w = _hint(x), bits[0]
            if type(x) is not int:
                preds.append((_bin("ge" if h >= 0 else "lt", x, 0), f"a ctlz operand's sign is {h >= 0}"))
                return _bin("sub", w, torch.cuda._host_trace.bit_length(x)) if h >= 0 else 0
            return 0 if h < 0 else w - h.bit_length()
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
            return _wrap(op.types[0], _bin("mul", x, 2**y), preds)
        if n == "arith.andi":
            if type(x) is int and type(y) is int:
                return _wrap(op.types[0], x & y, preds)
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
            return _wrap(op.types[0], _bin("int_floordiv", x, y), preds)
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
        # modes the box leaves out are fine when the memref fixes them at 1
        leaves = _flat(_memref_pattern(memref)[0])
        unit = [m for m in range(len(shape)) if m not in modes and len(leaves) == len(shape) and leaves[m] == 1]
        if (
            basis.group(2) != ",".join(f"1@{m}" for m in modes)
            or sorted(modes + unit) != list(range(len(shape)))
            or len(box) != len(modes)
        ):
            raise _Undescribed(f"the TMA basis {basis.group(0)} of {len(shape)} modes")
        self.predicates.append((_bin("eq", stride[modes[0]], 1), f"a TMA's mode {modes[0]} stride == 1"))
        dims = [shape[m] for m in modes]
        # the DSL's inline encode keeps a byte stride in units of 16, rounded
        # down (a size-1 mode's stride needn't be a multiple of 16)
        strides = [_bin("sub", v, _bin("mod", v, 16)) for v in (_bin("mul", stride[m], size) for m in modes[1:])]
        bounds = [(v, 1, 2**32, "extent") for v in dims] + [(v, 0, 2**40 - 1, "byte stride") for v in strides]
        for v, lo, hi, what in bounds:
            self.predicates.append((_bin("ge", v, lo), f"a TMA {what} >= {lo}"))
            self.predicates.append((_bin("le", v, hi), f"a TMA {what} <= {hi}"))
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
            elif isinstance(ty, _Builtin) and ty.int_of(16, 32, 64):
                params.append(_Param([(v, ty.width // 8, False)]))
            elif isinstance(ty, _Builtin) and ty.kind == "float" and ty.width == 32 and type(v) is float:
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


# the thread's last compile's inputs to the DSL's compile_and_cache: (function name, signature, args, kwargs, TVM-FFI)
_compile_inputs = threading.local()
# per compiled function: its host program, or why it did not load
_compiles: weakref.WeakKeyDictionary[Any, _Program | str] = weakref.WeakKeyDictionary()
# per module loaded from an exported object: the object's path
_loaded_from: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
SIDECAR = ".cute_host"
# a sidecar's format: its host function, TVM-FFI's argument spec and its object's digest
_SIDECAR_VERSION = 2


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


def _spec_args(spec: list, args: Sequence[Any]) -> tuple[list[Any], list[tuple[Any, str]]]:
    """The call's arguments its host function takes, and as guards each check TVM-FFI's binder makes
    of them (static and shared sizes and strides, divisibility, int bounds, alignment, dtype); a
    check that fails at the trace's values declines the call, which eager then rejects."""
    from cutlass import Numeric

    from torch.fx.experimental.symbolic_shapes import sym_or

    seen: dict[int, tuple[Any, str]] = {}
    taken: list[Any] = []
    guards: list[tuple[Any, str]] = []

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

    def take(p: list, a: Any, what: str) -> None:
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
        elif p[0] != "stream":
            raise _Undescribed(f"{what} is a TVM-FFI {p[0]} parameter")
        taken.append(a)

    params = [p for p in spec if p[0] != "env_stream"]
    if len(args) != len(params):
        raise _Undescribed(f"{len(args)} arguments for {len(params)} parameters")
    for i, (p, a) in enumerate(zip(params, args)):
        take(p, a, f"argument {i}")
    return taken, guards


def _program(text: str, name: str, spec: list) -> _Program | str:
    try:
        return _Program(text, name, spec)
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
    evaluated = (tuple(map(_hint, grid)), block, smem, cluster)
    launched = (node.grid, node.block, node.smem, record.cluster)
    if evaluated != launched:
        raise _Undescribed(f"kernel {node.name} launched {launched}; the evaluation {evaluated}")
    slots = [_hint(v) & _PLACEHOLDER_LOW if p else _hint(v) for v, p in zip(values, pointers)]
    # the DSL's maps carry its bit even where the replay launches the
    # driver's as is: the compare checks the encode's inputs either way
    dsl = tuple(dataclasses.replace(d, edits=True, last=[]) for d in record.descriptors)
    packed = pack_params(dataclasses.replace(record, descriptors=dsl), slots, pointers)
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


def _fields(a: Any) -> list[Any]:
    # a NamedTuple argument as TVM-FFI passes it: its fields, flattened
    if isinstance(a, tuple) and hasattr(a, "_fields"):
        return [x for f in a for x in _fields(f)]
    from cutlass import Numeric

    return [a.value if isinstance(a, Numeric) else a]


def _stand_ins(a: Any) -> Any:
    if isinstance(a, tuple) and hasattr(a, "_fields"):
        return type(a)(*map(_stand_ins, a))
    return _stand_in(a) if isinstance(a, _TracedTensor) else _hint(a)


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
        given, guards = _spec_args(program.spec, args)
        bound = _bind(program, given)
        bound[1].extend(guards)
        for i, ty in enumerate(program.formals):
            if ty == "!cuda.stream" and len(given) == len(program.formals) and _stream_handle(given[i]) is None:
                decline(f"argument {i} is a stream as a {type(given[i]).__name__}; pass a torch.cuda.Stream or CUstream")
        roots: list[_Root] = []
        for a in leaves:
            if isinstance(a, _TracedTensor) and not any(r is a._root for r in roots):
                roots.append(a._root)
        stand_ins = [_stand_ins(a) for a in args]
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
        records = [_describe(launch, node, tuple(roots), compiled) for launch, node in zip(launches, nodes)]
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
            _compiles[result] = _program(texts[name], name, _encode_spec(_tvm_ffi_args_spec_converter(*inputs)[0], tvm_ffi))
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
        record = {"name": program.name, "text": program.text, "spec": program.spec} if isinstance(program, _Program) else {"error": program}
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
        return record["error"] if "error" in record else _program(record["text"], record["name"], record["spec"])
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
    stand_ins = [_stand_in(a) if isinstance(a, _TracedTensor) else _hint(a) for a in args]
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
