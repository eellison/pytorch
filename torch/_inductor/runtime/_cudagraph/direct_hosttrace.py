"""Lower a host-tracing Tape (torch.cuda._host_trace) into the shared native replay.

The tape describes what a converted CUDA C++ host did once at symbolic shapes: its
inputs and allocations, every kernel launch as byte ranges of the parameter image
bound to SymInt expressions, and the guards the host branched on. This module is a
function from that tape to the runtime's replay records (owned buffers, physical
calls over exact parameter layouts, a numeric plan, output slots and a compiled
dispatch predicate), plus a thin entry that calls the native boxed replay. It does
not nest the two frontends' tracing modes: the tape is complete when it arrives.

Boxed layout: the tape's tensor inputs, in position order, then (for the standalone
entry) the family's arena, one uint8 tensor whose bytes hold the tape's temporaries at
the planned offsets (hosttrace_arena); the escaping outputs stay the runtime's buffers.
Sizes and strides bind through the numeric plan's `size` / `stride` loads and the
predicate's Tensor fact sources (size, stride, rank, dtype, device, math bits), so
no Python derives a shape per call; storage offsets through `storage_offset`;
addresses through pointer sources. A tape pointer is a storage base plus an
element offset, a runtime input pointer is the tensor's data pointer, so every
pointer expression is rebased by subtracting `itemsize * storage_offset`.
"""

import contextlib
import ctypes
import dataclasses
import functools
import operator
import os
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from typing import Any

import sympy

import torch
from torch._inductor.codecache import CppCodeCache
from torch._inductor.runtime._cudagraph.cuda_tape_import import recorded_guard_pins
from torch._inductor.runtime._cudagraph.hosttrace_allocseq import (
    AllocatorSequence,
    plan_sequence,
)
from torch._inductor.runtime._cudagraph.hosttrace_arena import (
    OutputRing,
    plan_arena,
    plan_outputs,
    round_capacity,
)
from torch._inductor.runtime.cudagraph_arg_mapping import (
    bind_output_slots,
    BorrowedInputOutput,
    BufferSource,
    ExpressionSource,
    InputSource,
    IntegerSource,
    IntExpr,
    OutputReference,
    OwnedBuffer,
    ParameterSource,
    PointerSource,
    storage_roots,
    TensorViewOutput,
)
from torch._inductor.runtime.cudagraph_boxed_replay import (
    _KernelModule,
    _make_replay,
    _NumericProgram,
    _ParameterProgram,
    _PhysicalCall,
    _PhysicalField,
)
from torch._inductor.runtime.cudagraph_compiled_evaluation import (
    _is_integer_payload,
    integer_payload_contract_from_guards,
    simplify_integer_payload,
)
from torch._inductor.runtime.cudagraph_launch_association import (
    RecordedGraphNode,
    RecordedKernelLaunch,
    UnsupportedCapture,
)
from torch._inductor.runtime.cudagraph_preparation import _Preparation
from torch.cuda._host_trace import (
    _constant,
    _PartialTrace,
    Entry as _Entry,
    Float32,
    TopologyMiss as _TopologyMiss,
)
from torch.cuda._utils import _check_cuda_bindings
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import (
    CeilDiv,
    CeilToInt,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    Max as _TMax,
    Min as _TMin,
    Mod as _Mod,
    PythonMod,
    ToFloat,
    TruncToInt,
)


_MODS = (sympy.Mod, _Mod, PythonMod)
# the planned arena (hosttrace_arena): the tape's temporaries in one boxed arena input
# instead of a runtime buffer each; off for the runtime team's consumer entries
_ARENA_DEFAULT = os.environ.get("TORCH_HOST_TRACE_ARENA", "1") != "0"
# the output arena (hosttrace_arena.OutputRing): the escaping outputs as views of one
# refcounted block per call, boxed after the arena, instead of a runtime buffer each.
# Opt-in on this line (HostTraceReplay(output_arena=True) or the variable): the raw
# `entry(box)` surface takes the tape's tensors alone and the runtime team's tests
# drive it (runtime_review/integration_06_outarena/OUTPUT_ARENA.md); the ring keeps
# this many blocks (a hold deeper than that costs one allocation per call)
_OUTPUT_ARENA_DEFAULT = os.environ.get("TORCH_HOST_TRACE_OUTPUT_ARENA", "0") == "1"
_OUTPUT_BLOCKS_DEFAULT = int(os.environ.get("TORCH_HOST_TRACE_OUTPUT_BLOCKS", "2"))
_ARENA_CHECK_DEFAULT = os.environ.get("TORCH_HOST_TRACE_ARENA_CHECK", "0") == "1"
# the allocator-order replay (hosttrace_allocseq, E31): the temporaries as roots the
# family binds by replaying the tape's allocation sequence through the caching
# allocator in a private pool before the launch, in place of the planned arena (the
# output arena composes with it: its block is boxed before the roots); its
# mode is "hold" (the blocks kept while the sizes fit) or "replay" (a sequence per call)
_ALLOCSEQ_DEFAULT = os.environ.get("TORCH_HOST_TRACE_ALLOCSEQ", "0") == "1"
_ALLOCSEQ_MODE_DEFAULT = os.environ.get("TORCH_HOST_TRACE_ALLOCSEQ_MODE", "hold")
# the predicate's entry points: the facts and guards alone, with the arena terms, with
# the region selects (the registered dispatch predicate is every part)
_MODE_FACTS, _MODE_ARENA, _MODE_ALL = 0, 1, 3
_INT64_MAX = 2**63 - 1
# ShapeEnv.simplify() compares a Min / Max's arguments pairwise with a static
# evaluation each; a wide cat's output length (a Max over one product per input)
# with 30 symbols cost 8-11 s per payload, 346 such payloads 223 s of a 340 s leaf
# (embed_cat), while every payload up to 16 symbols cost 10 s together. The tape
# lowering keeps a wider payload as written, as it did before the frozen context.
_PAYLOAD_SIMPLIFY_SYMBOLS = 16
# the payload roots' signed-range obligations (`_Lowering.payload`); the switch is
# the A/B measurement's (runtime_review/integration_06_followups), not a mode
_PAYLOAD_BOUNDS = os.environ.get("TORCH_HOST_TRACE_PAYLOAD_BOUNDS", "1") != "0"
_FLOAT_NODES = (sympy.Float, sympy.Rational, ToFloat, FloatPow, FloatTrueDiv, Float32)


def _predicate_uses_float(expression):
    if isinstance(expression, sympy.Integer):
        return False
    if isinstance(expression, _FLOAT_NODES):
        return True
    if isinstance(expression, Identity):
        return _predicate_uses_float(expression.args[0])
    if isinstance(expression, sympy.Pow):
        exponent = expression.exp
        return not (exponent.is_Integer and exponent >= 0)
    if isinstance(expression, (sympy.Add, sympy.Mul)):
        return any(_predicate_uses_float(arg) for arg in expression.args)
    return isinstance(expression, sympy.Symbol) and expression.is_integer is False


def _checked(op, operands):
    # the predicate's integer sums and products are int64 (the ABI of every value
    # it reads); an overflowing one sets the function's `bad` flag, which fails the
    # term it is part of (the ledger's signed-range obligation, row 3d)
    text = operands[0]
    for operand in operands[1:]:
        text = f"{op}({text}, {operand}, bad)"
    return text


# the predicate's preamble: checked int64 arithmetic over one failure flag
_PREDICATE_PREAMBLE = (
    "#include <algorithm>",
    "#include <cmath>",
    "#include <cstdint>",
    "#include <cstring>",
    "#include <vector>",
    "#include <c10/util/generic_math.h>",
    "static inline int64_t ck_add(int64_t a, int64_t b, bool& bad) {",
    "  int64_t r; bad |= __builtin_add_overflow(a, b, &r); return r;",
    "}",
    "static inline int64_t ck_mul(int64_t a, int64_t b, bool& bad) {",
    "  int64_t r; bad |= __builtin_mul_overflow(a, b, &r); return r;",
    "}",
)
_MINS = (
    _TMin,
    sympy.Min,
)  # the ShapeEnv uses torch's Min/Max; sympy's arrive from callers
_MAXS = (_TMax, sympy.Max)


class HostTraceLoweringDeclined(UnsupportedCapture):
    pass


_SYM_TYPES = (torch.SymInt, torch.SymFloat, torch.SymBool)
_INT_KINDS = {"i32": 4, "i64": 8, "u32": 4, "u64": 8}
_FLOAT_KINDS = {"f32": 4, "f64": 8}


def _as_i32(lowering, expression):
    """A u32 value re-expressed as the int32 with the same four bytes (their 32-bit
    scalar slot is range-checked as signed)."""
    wrap = lowering.node("add", args=(expression, lowering.const(-(2**32))))
    high = lowering.node("ge", args=(expression, lowering.const(2**31)))
    return lowering.node("select", args=(high, wrap, expression))


def _double_bits(value):
    return struct.unpack("<q", struct.pack("<d", float(value)))[0]


def _raw_expr(value):
    # a SymInt's symbol as created: the ShapeEnv replaces a symbol its guards pin
    # (flash's split count under `s > 1` and `s <= 2` becomes 2 in every record), but
    # the guards keep naming it
    if isinstance(value, _SYM_TYPES):
        return value.node._expr
    return _expr(value)


def _rounded_int_div(e):
    """`CeilToInt(IntTrueDiv(a, b))` / `FloorToInt(IntTrueDiv(a, b))` with an integer
    `b` (arange's length ceil((end - start) / step)) as the integer division the plan
    has: the divisor made positive, and when the numerator's sign is decided by the
    symbols' ranges, the rounding on the side the runtime's division admits
    (ceil(a / b) = -floor(-a / b)); None when it is not such an expression."""
    if not (
        isinstance(e, (CeilToInt, FloorToInt)) and isinstance(e.args[0], IntTrueDiv)
    ):
        return None
    numerator, divisor = e.args[0].args
    if not (isinstance(divisor, sympy.Integer) and divisor != 0):
        return None
    if divisor < 0:
        numerator, divisor = -numerator, -divisor
    ceil = isinstance(e, CeilToInt)
    if numerator.is_nonnegative:
        return (CeilDiv if ceil else FloorDiv)(numerator, divisor)
    if numerator.is_nonpositive:
        return -(FloorDiv if ceil else CeilDiv)(-numerator, divisor)
    return None


def _expr(value):
    if isinstance(value, _SYM_TYPES):
        return value.node.expr
    if isinstance(value, bool):
        return sympy.true if value else sympy.false
    if isinstance(value, int):
        return sympy.Integer(value)
    return value


@dataclass(frozen=True)
class _Property:
    kind: str  # size, stride, offset, base, alloc
    index: int  # input position or allocation sequence
    dim: int = -1


class _Symbols:
    """Every symbol the tape mentions, classified by the source that created it: the
    runtime team's `HostTraceSymbolMapping` over the tape is the one correspondence
    (ledger row 1a); the classification here is read from it. A bare object in place
    of a tape (the runtime team's CPU tests) has no mapping and no symbols."""

    def __init__(self, tape, mapping=None):
        from torch._inductor.runtime.cudagraph_host_trace_mapping import (
            HostTraceSymbolMapping,
        )

        self.tape = tape
        self.by_symbol: dict[sympy.Symbol, _Property] = {}
        inputs = getattr(tape, "inputs", ())
        self.positions = [rec.position for rec in inputs]
        self.tensor_index = {rec.position: index for index, rec in enumerate(inputs)}
        self.itemsize = {index: rec.root.itemsize for index, rec in enumerate(inputs)}
        self.opaque = {}
        if mapping is None and inputs:
            try:
                mapping = HostTraceSymbolMapping(tape, input_indices=self.tensor_index)
            except UnsupportedCapture as e:
                raise HostTraceLoweringDeclined(f"host_trace lowering: {e}") from e
        self.mapping = mapping
        if mapping is None:
            return
        # a consumer's mapping (the runtime team's terminal frontend, their tests) may
        # carry only the metadata symbols: what it has is what is classified
        for symbol, md in getattr(mapping, "metadata_symbols", {}).items():
            kind = "offset" if md.property == "storage_offset" else md.property
            dim = -1 if md.dimension is None else md.dimension
            self.by_symbol[symbol] = _Property(kind, md.index, dim)
        alloc_index = {rec.name: k for k, rec in enumerate(getattr(tape, "allocs", ()))}
        host_roots = getattr(mapping, "host_original_symbols", {})
        address_symbols = getattr(mapping, "address_symbols", {})
        for symbol, value in getattr(mapping, "substitutions", {}).items():
            if symbol in host_roots:
                self.by_symbol[symbol] = _Property("hbuf", host_roots[symbol])
                continue
            roots = value.free_symbols & address_symbols.keys()
            if len(roots) != 1:
                raise HostTraceLoweringDeclined(
                    f"host_trace lowering: storage root {symbol} maps to {len(roots)} runtime roots"
                )
            source = address_symbols[next(iter(roots))]
            if type(source) is InputSource:
                self.by_symbol[symbol] = _Property("base", source.index)
            else:
                self.by_symbol[symbol] = _Property("alloc", alloc_index[source.name])
        opaque_index = {id(rec): k for k, rec in enumerate(getattr(tape, "opaque", ()))}
        for symbol, rec in getattr(mapping, "opaque_symbols", {}).items():
            self.by_symbol[symbol] = _Property("opaque", opaque_index.get(id(rec), -1))
            self.opaque[symbol] = rec

    def tensors_of(self, args):
        """The tensors of a call in the tape's positions (the box before the arena)."""
        return [args[position] for position in self.positions]

    def prop(self, symbol):
        prop = self.by_symbol.get(symbol)
        if prop is None:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: symbol {symbol} has no input or allocation source"
            )
        return prop

    def load(self, symbol):
        """The numeric plan's Tensor load for a size or stride symbol."""
        prop = self.prop(symbol)
        if prop.kind not in ("size", "stride"):
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: {symbol} ({prop.kind}) is not a size or stride"
            )
        return IntExpr(prop.kind, prop.index, (IntExpr("constant", prop.dim),))


class _Lowering:
    """sympy -> the runtime's IntExpr numeric plan and its C++ predicate source."""

    # n-ary min/max as one plan node; False selects the pairwise select fold
    nary_minmax = True
    # one plan node per (host function, arguments); False keeps one per record
    dedupe_calls = True

    def __init__(
        self,
        symbols,
        opaque_values=None,
        payload_contract=None,
        *,
        integer_sources=None,
        strict_payload=True,
    ):
        self.symbols = symbols
        self.payload_contract = payload_contract
        self.strict_payload = strict_payload
        self.integer_sources = {} if integer_sources is None else integer_sources
        # opaque results of kind "guard" are pinned to their traced value: the
        # predicate re-runs the host's function and misses when it differs
        self.opaque_values = dict(opaque_values or {})
        # hash-consing at two levels: one IntExpr per distinct sympy subexpression
        # (_memo), and one IntExpr per distinct (op, value, operand identities) across
        # the whole lowering (_nodes). The second makes structurally equal DAGs the same
        # object even when sympy hands over distinct expressions (a 129-input cat lowers
        # its second launch's n-ary Max as a fresh select chain whose first 127 steps
        # equal the first launch's), so the numeric plan's memo hits by identity and the
        # shared prefix takes one set of slots.
        self._memo = {}
        self._memo_float = {}
        self._guarded_payloads = set()
        self._nodes = {}
        # the callable kept alive per host function address (the plan's owner)
        self._call_owners = {}
        # obligations the lowering relies on beyond the tape's guards, as sympy
        # relations over the tape's symbols: the domains of the plan's divisions
        # (numerator >= 0, divisor > 0) and the launch bounds of symbolic grids.
        # Rendered into the predicate beside the guards, and read by the runtime
        # team's consumers as `LoweredTape.extra_guards`
        self.guards = (
            []
            if payload_contract is None
            else [
                symbols.mapping.translate(guard, preserve_operations=True)
                for guard in payload_contract.additional_guards
            ]
        )
        # the payload roots (`payload`) that carry a signed-range obligation
        self._bounded = OrderedSet()

    def require(self, relation):
        """An obligation: a relation that folds to False is a recorded contradiction
        (declined by name), one that folds to True says nothing."""
        if relation is sympy.false or relation == False:  # noqa: E712
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: the obligation {relation} cannot hold"
            )
        if not (relation is sympy.true or relation == True):  # noqa: E712
            self.guards.append(relation)

    def node(self, op, value=None, args=()):
        """The canonical IntExpr for (op, value, args); args must be canonical already."""
        key = (op, value, tuple(id(a) for a in args))
        out = self._nodes.get(key)
        if out is None:
            out = IntExpr(op, value, tuple(args))
            self._nodes[key] = out
        return out

    def intern(self, expression):
        return self.node(expression.op, expression.value, expression.args)

    def call(self, impl, owner, args):
        """The plan's pointer-ABI call (`pcall`) of a host's opaque function at
        `impl` on `args`. A traced host's opaque function is a pure function of its
        integer arguments by construction (E23), so two records of the same call
        on the same arguments are one node: the owner kept alive is the first
        record's callable, which is what makes the node's key equal."""
        if self.dedupe_calls:
            owner = self._call_owners.setdefault(impl, owner)
        return self.node("pcall", (impl, owner), args)

    def const(self, value):
        return self.node("constant", int(value))

    def subst(self, e):
        e = _expr(e)
        if self.opaque_values and getattr(e, "free_symbols", None):
            e = e.xreplace(self.opaque_values)
        return e

    def lower_float(self, e):
        """A SymFloat expression as the plan's float ops, one node per sympy node, in
        the recorded order; the value slot holds the double's bit pattern."""
        e = self.subst(e)
        out = self._memo_float.get(e)
        if out is None:
            out = self._lower_float(e)
            self._memo_float[e] = out
        return out

    def _lower_float(self, e):
        if isinstance(e, (sympy.Float, sympy.Rational, sympy.Integer)):
            return self.node("fconst", _double_bits(e))
        if isinstance(e, sympy.Symbol):
            return self.node("ffromint", args=(self.lower(e),))
        if isinstance(e, Identity):
            return self.lower_float(e.args[0])
        if isinstance(e, ToFloat):
            return self.node("ffromint", args=(self.lower(e.args[0]),))
        if isinstance(e, Float32):
            # the recorder's float32 rounding point (ht::round_float32): the
            # plan's fround32 op rounds the double through a float
            return self.node("fround32", args=(self.lower_float(e.args[0]),))
        if isinstance(e, (FloatTrueDiv,)):
            a, b = e.args
            return self.node("fdiv", args=(self.lower_float(a), self.lower_float(b)))
        if isinstance(e, FloatPow):
            a, b = e.args
            return self.node("fpow", args=(self.lower_float(a), self.lower_float(b)))
        if isinstance(e, sympy.Add):
            result = self.lower_float(e.args[0])
            for arg in e.args[1:]:
                result = self.node("fadd", args=(result, self.lower_float(arg)))
            return result
        if isinstance(e, sympy.Mul):
            result = self.lower_float(e.args[0])
            for arg in e.args[1:]:
                result = self.node("fmul", args=(result, self.lower_float(arg)))
            return result
        if isinstance(e, sympy.Pow):
            base, exponent = e.args
            one = self.node("fconst", _double_bits(1.0))
            if exponent == -1:
                return self.node("fdiv", args=(one, self.lower_float(base)))
            if exponent == sympy.Rational(1, 2):
                return self.node("fsqrt", args=(self.lower_float(base),))
            if exponent == sympy.Rational(-1, 2):
                return self.node(
                    "fdiv",
                    args=(one, self.node("fsqrt", args=(self.lower_float(base),))),
                )
            return self.node(
                "fpow", args=(self.lower_float(base), self.lower_float(exponent))
            )
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: no float lowering for {type(e).__name__} in {e}"
        )

    def __call__(self, expression):
        return self.payload(expression)

    def value(self, value):
        return self.payload(value)

    def payload(self, e):
        """A payload root: an expression the numeric plan evaluates per call (a kernel's
        integer field, a layout's size or stride, a byte count, a grid, a pointer
        displacement). Besides its plan node, every sum, product and power in it (the
        root and its intermediates, once per distinct expression over the lowering)
        gets the signed-range obligation `Le(e, 2**63 - 1)`, unevaluated (sympy would
        fold it by the symbols' signs; the term's work is its rendering): the
        predicate renders it with its checked int64 arithmetic in the plan's order, so
        an intermediate the plan would overflow (sympy's `(-n)**3` is `-(n**3)`: the
        positive cube first) misses before selection instead of failing the native
        evaluation after it (A290's second half). The runtime team's `__int128`
        printer checks a term's final value only, hence the obligations on the
        intermediates too."""
        e = self.subst(e)
        out = self.lower(e)
        if _PAYLOAD_BOUNDS and isinstance(e, sympy.Basic):
            for sub in sympy.preorder_traversal(e):
                if (
                    isinstance(sub, (sympy.Add, sympy.Mul, sympy.Pow))
                    and sub not in self._bounded
                ):
                    self._bounded.add(sub)
                    self.guards.append(sympy.Le(sub, _INT64_MAX, evaluate=False))
        return out

    def floating(self, expression):
        return self.lower_float(expression)

    def lower(self, e):
        e = _expr(e)
        # the memo answers for the expression as recorded too: a wide cat lowers
        # the same size and stride terms thousands of times (25.8k calls, 1.1k
        # distinct, in embed_cat's cat test), and the simplification below costs
        # about 2 ms even for a two-symbol product
        out = self._memo.get(e)
        if out is not None:
            return out
        recorded = e
        original = self.subst(e)
        if self.payload_contract is not None and (
            self.strict_payload
            or (
                _is_integer_payload(e)
                and len(e.free_symbols) <= _PAYLOAD_SIMPLIFY_SYMBOLS
            )
        ):
            try:
                e, _ = simplify_integer_payload(
                    e, self.payload_contract.shape_env, self.payload_contract
                )
            except ValueError as error:
                raise HostTraceLoweringDeclined(
                    f"Host integer payload cannot be simplified: {error}"
                ) from error
        e = self.subst(e)
        # Value simplification must not erase partial-operation domains.
        if original != e and original not in self._guarded_payloads:
            self.guards.extend(
                (
                    sympy.Ge(original, -(2**63), evaluate=False),
                    sympy.Le(original, _INT64_MAX, evaluate=False),
                )
            )
            self._guarded_payloads.add(original)
        out = self._memo.get(e)
        if out is None:
            out = self._lower(e)
            self._memo[e] = out
        self._memo[recorded] = out
        return out

    def _balanced(self, op, nodes):
        while len(nodes) > 1:
            nodes = [
                self.node(op, args=(nodes[i], nodes[i + 1]))
                if i + 1 < len(nodes)
                else nodes[i]
                for i in range(0, len(nodes), 2)
            ]
        return nodes[0]

    def _lower(self, e):
        e = self.subst(e)
        if isinstance(e, sympy.Integer):
            return self.const(e)
        if isinstance(e, sympy.Symbol):
            if e in self.integer_sources:
                return self.intern(self.integer_sources[e])
            if self.symbols is None:
                raise HostTraceLoweringDeclined(
                    f"host_trace lowering: symbol {e} has no integer source"
                )
            prop = self.symbols.prop(e)
            if prop.kind == "offset":
                return self.node("storage_offset", prop.index)
            if prop.kind in ("size", "stride"):
                return self.intern(self.symbols.load(e))
            if prop.kind == "base":
                # a storage base address as a value (an opaque lookup's argument, E19:
                # symm's handle from the buffer's address): the boxed tensor's data
                # pointer less its offset's bytes, as the tape's root symbol values it
                index = prop.index
                offset_bytes = self.node(
                    "multiply",
                    args=(
                        self.node("storage_offset", index),
                        self.const(-self.symbols.itemsize[index]),
                    ),
                )
                return self.node(
                    "add", args=(self.node("pointer", index), offset_bytes)
                )
            if prop.kind == "opaque":
                # a rebind: the host's own function, re-run natively per call on the
                # lowered arguments (guards were substituted by their traced value)
                rec = self.symbols.opaque[e]
                if not rec.get("impl"):
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: opaque {rec['fn']} carries no host function address"
                    )
                return self.call(
                    int(rec["impl"]),
                    rec["call"],
                    tuple(self.lower(a) for a in rec["args"]),
                )
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: {e} ({prop.kind}) inside an integer expression"
            )
        if isinstance(e, Identity):
            return self.lower(e.args[0])
        rounded = _rounded_int_div(e)
        if rounded is not None:
            return self.lower(rounded)
        if isinstance(e, TruncToInt):
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: float-to-int truncation in an integer expression {e}"
            )
        if isinstance(e, (sympy.Add, sympy.Mul)):
            # a balanced tree: an output-plan offset sums one term per size class of
            # the blocks before it, and the runtime's binder reads an expression to
            # depth 16
            return self._balanced(
                "add" if isinstance(e, sympy.Add) else "multiply",
                [self.lower(arg) for arg in e.args],
            )
        if isinstance(e, (FloorDiv, CeilDiv)):
            # the runtime's division admits a non-negative numerator and a positive
            # divisor and raises otherwise (a symbolic divisor is a size, positive by
            # the predicate's facts, or a stride the host already divided by)
            # the operands first: a nested division's own domain then precedes the
            # relation over its value (the predicate evaluates the obligations in order)
            numerator, denominator = e.args
            operands = (self.lower(numerator), self.lower(denominator))
            self.require(sympy.Ge(numerator, 0))
            self.require(sympy.Gt(denominator, 0))
            op = "floordiv" if isinstance(e, FloorDiv) else "ceildiv"
            return self.node(op, args=operands)
        if isinstance(e, _MODS):
            a, b = e.args
            operands = (self.lower(a), self.lower(b))
            self.require(sympy.Ge(a, 0))
            self.require(sympy.Gt(b, 0))
            # a - (a // b) * b, on the non-negative domain the runtime's floordiv admits
            quotient = self.node("floordiv", args=operands)
            return self.node(
                "add",
                args=(
                    self.lower(a),
                    self.node(
                        "multiply",
                        args=(
                            quotient,
                            self.node("multiply", args=(self.lower(b), self.const(-1))),
                        ),
                    ),
                ),
            )
        if isinstance(e, sympy.Pow) and e.exp.is_Integer and e.exp >= 0:
            # an exact integer power as repeated multiplication (the plan has no pow);
            # its signed-range obligation is the payload's (`payload`: every sum,
            # product and power of a payload root, the intermediates included, so an
            # overflowing (-n)**3 = -(n**3) misses before selection)
            base, exponent = self.lower(e.base), int(e.exp)
            if exponent == 0:
                return self.const(1)
            result = base
            for _ in range(exponent - 1):
                result = self.node("multiply", args=(result, base))
            return result
        if isinstance(e, _MINS + _MAXS):
            operands = tuple(self.lower(arg) for arg in e.args)
            if len(operands) == 1:
                return operands[0]
            if self.nary_minmax:
                # one n-ary node: a cat's per-launch grid is a max over every
                # input's size, and as a pairwise select chain its shared
                # subexpressions cost the plan's memo 2^depth to compare
                op = "min" if isinstance(e, _MINS) else "max"
                return self.node(op, args=operands)
            # the pairwise select fold, kept for a plan without n-ary max/min
            op = "lt" if isinstance(e, _MINS) else "gt"
            result = operands[0]
            for other in operands[1:]:
                cond = self.node(op, args=(result, other))
                result = self.node("select", args=(cond, result, other))
            return result
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: no numeric-plan lowering for {type(e).__name__} in {e}"
        )

    # ---- predicate source

    def cpp(self, e, names):
        """The C++ text of an integer, float or boolean expression over the predicate's
        named value slots; opaque rebinds render as inline host-function calls."""
        e = self.subst(e)
        if e is sympy.true:
            return "1"
        if e is sympy.false:
            return "0"
        if isinstance(e, sympy.Integer):
            value = int(e)
            if value == -(2**63):
                return "INT64_MIN"  # the literal 9223372036854775808 is not an int64
            if not -(2**63) < value < 2**63:
                raise HostTraceLoweringDeclined(
                    f"host_trace lowering: the constant {value} is outside int64"
                )
            return f"int64_t({value})"
        if isinstance(e, (sympy.Float, sympy.Rational)):
            return f"double({float(e)!r})"
        if isinstance(e, ToFloat):
            return f"double({self.cpp(e.args[0], names)})"
        if isinstance(e, Float32):
            return f"double(float({self.cpp(e.args[0], names)}))"
        if isinstance(e, FloatTrueDiv):
            a, b = e.args
            return f"(double({self.cpp(a, names)}) / double({self.cpp(b, names)}))"
        if isinstance(e, FloatPow):
            a, b = e.args
            return (
                f"std::pow(double({self.cpp(a, names)}), double({self.cpp(b, names)}))"
            )
        if isinstance(e, sympy.Pow):
            base, exponent = e.args
            if exponent.is_Integer and exponent >= 0:
                # an exact integer power stays an integer (a double would not divide)
                if exponent == 0:
                    return "int64_t(1)"
                return _checked("ck_mul", [self.cpp(base, names)] * int(exponent))
            if exponent == sympy.Rational(1, 2):
                return f"std::sqrt(double({self.cpp(base, names)}))"
            return f"std::pow(double({self.cpp(base, names)}), double({self.cpp(exponent, names)}))"
        if isinstance(e, sympy.Symbol):
            if e in names:
                return names[e]
            if e in self.symbols.opaque:
                # a rebind read by a guard before the predicate hoisted it into a
                # local (lower_tape declares one per rebind at its first use)
                rec = self.symbols.opaque[e]
                if not rec.get("impl"):
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: opaque {rec['fn']} carries no host function address"
                    )
                args = [f"int64_t({self.cpp(a, names)})" for a in rec["args"]]
                return _host_call_expr(int(rec["impl"]), args)
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: predicate symbol {e} has no boxed source"
            )
        if isinstance(e, Identity):
            return self.cpp(e.args[0], names)
        rounded = _rounded_int_div(e)
        if rounded is not None:
            return self.cpp(rounded, names)
        if isinstance(e, sympy.Add):
            if _predicate_uses_float(e):
                return "(" + " + ".join(self.cpp(a, names) for a in e.args) + ")"
            return _checked("ck_add", [self.cpp(a, names) for a in e.args])
        if isinstance(e, sympy.Mul):
            if _predicate_uses_float(e):
                return "(" + " * ".join(self.cpp(a, names) for a in e.args) + ")"
            return _checked("ck_mul", [self.cpp(a, names) for a in e.args])
        if isinstance(e, FloorDiv):
            a, b = e.args
            return f"c10::div_floor_integer({self.cpp(a, names)}, {self.cpp(b, names)})"
        if isinstance(e, CeilDiv):
            a, b = e.args
            return f"(-c10::div_floor_integer(-({self.cpp(a, names)}), {self.cpp(b, names)}))"
        if isinstance(e, _MODS):
            a, b = e.args
            sa, sb = self.cpp(a, names), self.cpp(b, names)
            return f"(((({sa}) % ({sb})) + ({sb})) % ({sb}))"
        if isinstance(e, _MINS):
            return (
                "std::min<int64_t>({"
                + ", ".join(f"int64_t({self.cpp(a, names)})" for a in e.args)
                + "})"
            )
        if isinstance(e, _MAXS):
            return (
                "std::max<int64_t>({"
                + ", ".join(f"int64_t({self.cpp(a, names)})" for a in e.args)
                + "})"
            )
        relations = {
            sympy.Eq: "==",
            sympy.Ne: "!=",
            sympy.Lt: "<",
            sympy.Le: "<=",
            sympy.Gt: ">",
            sympy.Ge: ">=",
        }
        for kind, op in relations.items():
            if isinstance(e, kind):
                return f"(({self.cpp(e.args[0], names)}) {op} ({self.cpp(e.args[1], names)}))"
        if isinstance(e, sympy.And):
            return "(" + " && ".join(self.cpp(a, names) for a in e.args) + ")"
        if isinstance(e, sympy.Or):
            return "(" + " || ".join(self.cpp(a, names) for a in e.args) + ")"
        if isinstance(e, sympy.Not):
            return f"(!({self.cpp(e.args[0], names)}))"
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: no predicate lowering for {type(e).__name__} in {e}"
        )


def _parameter_of(call, offset, size):
    # an absolute byte range of a launch image -> (parameter index, offset inside it)
    for index, (start, width) in enumerate(call.module.parameter_layout):
        if start <= offset and offset + size <= start + width:
            return index, offset - start
    raise HostTraceLoweringDeclined(
        f"host_trace lowering: byte range {offset}+{size} is outside the parameter layout"
    )


def _function_by_symbol(host_symbol):
    # cuda.bindings has no cudaGetFuncBySymbol; the runtime library torch loaded has
    major = (torch.version.cuda or "").split(".")[0]
    if not major:
        raise HostTraceLoweringDeclined("host_trace lowering: a CUDA build is required")
    cudart = ctypes.CDLL(f"libcudart.so.{major}")
    function = ctypes.c_void_p()
    error = cudart.cudaGetFuncBySymbol(
        ctypes.byref(function), ctypes.c_void_p(host_symbol)
    )
    if error != 0 or not function.value:
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: cudaGetFuncBySymbol failed ({error}) for the traced kernel"
        )
    return function.value


class _HostTraceKernelModule(_KernelModule):
    """A kernel of libtorch_cuda: the function handle, layout, block and shared bytes
    the recorder read back from the traced capture. Nothing to borrow: the module is
    the process's own CUDA library."""

    def __init__(self, host_symbol, layout, block, shared, name, device):
        from cuda.bindings import driver

        # the tape records the runtime's host symbol (cudaKernelNodeParams.func);
        # the driver's launch and the graph inspector speak CUfunction
        self.host_symbol = host_symbol
        self._function = _function_by_symbol(host_symbol)
        self._layout = tuple(layout)
        self._block = tuple(block)
        self._shared = shared
        self.name = name
        # the tape's device and its context: a launch from another device or context
        # would not be the traced kernel's
        self.device_index = device
        with torch.cuda.device(device):
            self.context = int(_check_cuda_bindings(driver.cuCtxGetCurrent()))
        if not self.context:
            raise HostTraceLoweringDeclined(
                "host_trace lowering: the tape's device has no CUDA context"
            )

    @property
    def function(self):
        return self._function

    @property
    def parameter_layout(self):
        return self._layout

    @property
    def parameter_sizes(self):
        return tuple(size for _, size in self._layout)

    @property
    def shared(self):
        return self._shared

    @property
    def block(self):
        return self._block

    def check(self):
        if type(self._function) is not int or self._function <= 0:
            raise HostTraceLoweringDeclined(
                "host_trace lowering: launch lost its kernel handle"
            )

    def _borrow_for_cudagraph(self):
        return self

    def launch(self, images, grid, stream, shared=None, block=None):
        from cuda.bindings import driver

        self.check()
        if tuple(len(image) for image in images) != self.parameter_sizes:
            raise HostTraceLoweringDeclined(
                "host_trace lowering: launch images differ from the parameter layout"
            )
        if torch.cuda.current_device() != self.device_index:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: kernel {self.name} is launched on device {torch.cuda.current_device()}, the tape's is {self.device_index}"
            )
        if int(_check_cuda_bindings(driver.cuCtxGetCurrent())) != self.context:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: kernel {self.name} is launched in another CUDA context than the tape's"
            )
        if (
            _check_cuda_bindings(driver.cuStreamIsCapturing(stream))
            != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE
        ):
            raise HostTraceLoweringDeclined(
                "host_trace lowering: kernels are launched only inside the preparation capture"
            )
        storage = tuple(
            ctypes.create_string_buffer(image, len(image)) for image in images
        )
        arguments = (ctypes.c_void_p * len(storage))(
            *(ctypes.addressof(image) for image in storage)
        )
        _check_cuda_bindings(
            driver.cuLaunchKernel(
                self._function,
                *grid,
                *(self._block if block is None else block),
                self._shared if shared is None else shared,
                stream,
                ctypes.addressof(arguments),
                0,
            )
        )


class _HostTraceCuTeDescModule(_HostTraceKernelModule):
    """A CuTe DSL program's kernel recorded from its launch descriptor (torch/
    cuda/_host_trace_cute_desc.py): eager's own function handle, read off a
    capture of eager's call at the trace's warm-up, with the parameter layout
    the driver reported for it; the program's module (the loaded object or
    the in-process compilation) stays referenced for the graph's lifetime."""

    def __init__(self, launch, device):
        from cuda.bindings import driver

        self.record = launch["cute_desc"]
        self._function = int(launch["func"])
        self.check()
        self.host_symbol = None
        self._layout = tuple(tuple(x) for x in launch["param_layout"])
        self._block = tuple(int(b) for b in launch["block"])
        self._shared = int(launch["smem"]) if not isinstance(launch["smem"], _SYM_TYPES) else None
        self.name = launch["kernel"]
        self.device_index = device
        with torch.cuda.device(device):
            self.context = int(_check_cuda_bindings(driver.cuCtxGetCurrent()))
        if not self.context:
            raise HostTraceLoweringDeclined(
                "host_trace lowering: the tape's device has no CUDA context"
            )

    def check(self):
        if type(self._function) is not int or self._function <= 0 or self._function != self.record.function:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: CuTe kernel {getattr(self, 'name', '')} lost eager's function handle"
            )
        if self.record.program.keep_alive is None:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: CuTe kernel {getattr(self, 'name', '')}'s program is no longer loaded"
            )


class _HostTraceTritonModule(_HostTraceKernelModule):
    """A Triton kernel the traced host launched from Python (torch/cuda/
    _host_trace_triton.py): eager's own compilation, launched through the
    function handle Triton loaded for it (the handle an eager capture's node
    holds), with the parameter layout the runtime's DirectTritonOwner read
    from the selected compilation. The owner's loaded copy of the cubin is
    borrowed for the graph's lifetime together with the actual CompiledKernel
    that owns the eager function."""

    def __init__(self, launch, device):
        from cuda.bindings import driver

        self.record = launch
        self._function = launch.binary.function
        self._eager_module = launch.binary.module
        owner = launch.owner
        self.check()
        if owner.device_index != device:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: Triton kernel {launch.binary.name} was compiled for "
                f"cuda:{owner.device_index}, the tape is lowered for cuda:{device}"
            )
        self.host_symbol = None
        self._layout = tuple(owner.abi_layout)
        self._block = (owner.module.num_warps * 32, 1, 1)
        self._shared = int(owner.module.shared)
        self.name = launch.binary.name
        self.device_index = device
        with torch.cuda.device(device):
            self.context = int(_check_cuda_bindings(driver.cuCtxGetCurrent()))
        if not self.context:
            raise HostTraceLoweringDeclined(
                "host_trace lowering: the tape's device has no CUDA context"
            )

    def check(self):
        from torch._inductor.runtime._cudagraph.direct_triton import (
            DirectTritonDeclined,
        )

        try:
            self.record.owner.check()
        except DirectTritonDeclined as error:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: {error} ({self.record.binary.name})"
            ) from error
        function = self.record.binary.function
        if (
            type(function) is not int
            or function <= 0
            or function != self._function
            or self.record.binary.module is None
            or self.record.binary.module != self._eager_module
        ):
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: Triton kernel {self.record.binary.name} lost its loaded function"
            )

    def _borrow_for_cudagraph(self):
        self.check()
        return (
            self.record.binary,
            self.record.owner.module._borrow_for_cudagraph(),
        )


@dataclass(frozen=True)
class _Records:
    input_names: tuple
    integer_inputs: tuple = ()


# predicate value slots after the storage offsets: (kind, input, dimension) rows the
# native dispatcher reads from the boxed Tensors
@dataclass(frozen=True)
class _Fact:
    kind: str  # size, stride, rank, dtype, device, neg, conj
    index: int
    dim: int = 0

    @property
    def name(self):
        return f"{self.kind}{self.index}_{self.dim}"


@dataclass
class LoweredTape:
    tape: object
    symbols: _Symbols
    allocations: tuple
    outputs: tuple
    calls: tuple
    guard_source: str
    predicate: object  # compiled library owner
    predicate_address: int
    pointer_indices: tuple
    offset_indices: tuple
    facts: tuple  # _Fact rows, in predicate slot order
    records: _Records
    # the CUDA device the tape was lowered for (the tape's own unless the caller named
    # another): the kernel handles, the device facts, the preparation and the arena
    device: int = -1
    # (seq, destination PointerSource, byte-count IntExpr, value) memset records
    memsets: tuple = ()
    # _HostTable rows: pinned tables the replay renders per call (H2D)
    host_tables: tuple = ()
    # (seq, source, destination PointerSource, byte-count IntExpr) memcpy records; the
    # source is a host table index or a PointerSource over a pinned CPU input
    memcpys: tuple = ()
    # positions of pinned CPU inputs (the sources of in-graph copies)
    pinned_positions: tuple = ()
    # the philox offsets one replay draws from the device's default generator (IntExpr),
    # and the (call index, parameter, byte offset) of each PhiloxCudaState the kernels
    # read: its seed/offset pointers are the prepared capture's, written at preparation
    rng: object = None
    rng_fields: tuple = ()
    # _Region rows: closed library calls served from the runtime's template registry,
    # and per site the chain class (_chain_class: node kind, programmatic edge) the
    # preparation captured (set by prepare)
    regions: tuple = ()
    region_nodes: dict = None
    region_arena: object = None  # _RegionArena, set by prepare_hosttrace
    # set by prepare_hosttrace from the prepared capture: the tape seq of every node
    # in issue order (a region's nodes share its seq) and the indices, in that order,
    # of the nodes whose incoming edge is programmatic, read from the graph's edge data
    node_seqs: tuple = ()
    programmatic_nodes: tuple = ()
    # the prepared capture's raw (graph, exec) handles, set by prepare_hosttrace before
    # the entry takes the graph: tests read the exec's node states back through them
    capture_handles: tuple = ()
    # what the shared runtime's consumers read (their replay.py / direct_host.py):
    # obligations the lowering emitted beyond the tape's guards (sympy relations over
    # the tape's symbols), the per-launch rng slots (_RngSlot rows), and the compiled
    # guard object of their evaluator (None here: ours is `predicate`)
    extra_guards: tuple = ()
    rng_slots: tuple = ()
    guard: object = None
    # the declared kind of each memcpy record ("h2d" | "d2d"), parallel to `memcpys`
    memcpy_kinds: tuple = ()
    # why the runtime team's compiled guard (`guard`, row 2b) could not be built for
    # this tape, when it could not (their printer's declines); None when it was
    guard_declined: object = None
    # the predicate without its region selects (the same facts, guards and
    # obligations): whether the tape's guards hold at a call, whatever chain it
    # selects (a topology miss is decided by the call, E28)
    facts_address: int = 0
    # the tape's root identity facts, (root name, root name) pairs of storages the
    # host compared and the recorder decided distinct (an allocation against
    # another root): declared facts, never guards
    root_facts: tuple = ()
    # the planned arena (hosttrace_arena.ArenaPlan): the temporaries' offsets in the
    # arena input boxed after the tape's tensors, the requirement expression and the
    # plan's class guards; None when the tape was lowered without one. The predicate
    # entry with the arena terms and no region selects, and the requirement function
    arena: object = None
    arena_address: int = 0
    arena_bytes_address: int = 0
    # the output plan (hosttrace_arena.plan_outputs): the escaping outputs' offsets in
    # the block boxed after the arena, whose views the runtime returns; its
    # requirement function. None when the tape was lowered without an output arena
    output_arena: object = None
    output_bytes_address: int = 0
    # the allocation sequence (hosttrace_allocseq.SequencePlan): the temporaries as
    # roots boxed after the tape's tensors, bound per call by the family's
    # AllocatorSequence; None when the tape was lowered without one
    sequence: object = None
    # boxed positions of the inputs the tape wrote (Tape.written_inputs): a served
    # call materializes a copy-on-write tensor there before the dispatch binds its
    # address, as eager's mutable accessor does at the launch
    written_positions: tuple = ()
    # output positions over allocations no node writes (eager's at::empty returned as
    # it is): served at the plan's shape, their values indeterminate on both paths
    unwritten_outputs: tuple = ()
    # the argument contract, prepared once: the arity, the positions of the non-tensor
    # arguments, and a getter for the tensors in the tape's positions
    nargs: int = dataclasses.field(init=False, repr=False)
    constant_positions: tuple = dataclasses.field(init=False, repr=False)
    tensors: object = dataclasses.field(init=False, repr=False)
    # the predicate's marshaller over a box (torch._C._HostTracePredicate), built on
    # the first check_predicate
    _probe: object = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self):
        tape = self.tape
        self.nargs = tape.nargs
        positions = self.symbols.positions
        boxed = OrderedSet(positions)
        self.constant_positions = tuple(i for i in range(tape.nargs) if i not in boxed)
        if positions == list(range(tape.nargs)):
            self.tensors = tuple  # every argument is a tensor: the call's tuple as is
        elif len(positions) > 1:
            self.tensors = operator.itemgetter(*positions)
        else:
            self.tensors = lambda args: (args[positions[0]],)
        # HostTraceReplay leaves the tensor-position check to the native dispatcher:
        # it refuses a non-Tensor in any predicate source (TypeError) before a variant
        # or the miss handler runs, which needs every boxed input to be one
        fact_sources = OrderedSet([f.index for f in self.facts])
        if positions != getattr(
            tape, "positions", positions
        ) or fact_sources != OrderedSet(range(len(self.records.input_names))):
            raise AssertionError("every boxed input must be a predicate fact source")

    @property
    def input_count(self):
        return len(self.records.input_names)

    @property
    def input_names(self):
        return self.records.input_names

    @property
    def integer_inputs(self):
        return self.records.integer_inputs

    @property
    def registration(self):
        return (
            (),
            self.predicate_address,
            self.predicate,
            self.pointer_indices,
            self.offset_indices,
            tuple((f.kind, f.index, f.dim) for f in self.facts),
        )

    def contract_holds(self, args):
        """The tape's argument contract: arity, tensor positions and every non-tensor
        argument by value and type. Everything about the tensors (dtype, rank, device,
        sizes, strides, offsets, math bits) is the native predicate's."""
        try:
            return (
                len(args) == self.nargs
                and self.constants_hold(args)
                and all(map(isinstance, self.tensors(args), repeat(torch.Tensor)))
            )
        except AttributeError:
            # not a prepared LoweredTape (a bare object over a tape): the tape's form
            from torch.cuda._host_trace import _constants, _tensor_positions

            tape = self.tape
            positions = _tensor_positions(args)
            return (
                len(args) == tape.nargs
                and positions == list(tape.positions)
                and _constants(args, positions) == tape.constants
            )

    def constants_hold(self, args):
        """The non-tensor positions hold the traced constants, by value and type (a
        tensor there is a different value)."""
        constants = tuple([_constant(args[i]) for i in self.constant_positions])
        return constants == self.tape.constants

    def box(self, args, arena=None, outputs=None, sequence=None):
        """The boxed inputs for a call: the tensors in the tape's positions, then the
        family's arena tensor when the tape was lowered with a planned arena, then
        the output block when it was lowered with an output arena, then the sequence
        roots when it was lowered with an allocation sequence."""
        box = list(self.tensors(args))
        if self.arena is not None:
            if arena is None:
                raise ValueError(
                    "host_trace replay: the lowering has a planned arena and the box needs its tensor"
                )
            box.append(arena)
        if self.output_arena is not None:
            if outputs is None:
                raise ValueError(
                    "host_trace replay: the lowering has an output arena and the box needs its block"
                )
            box.append(outputs)
        if self.sequence is not None:
            if sequence is None:
                raise ValueError(
                    "host_trace replay: the lowering has an allocation sequence and the box needs its roots"
                )
            box.extend(sequence)
        return box


@dataclass(frozen=True)
class _HostTable:
    seq: int
    name: str
    nbytes: int
    constants: tuple  # (offset, bytes) written into every slot once
    elements: tuple  # (offset, width, PointerSource | IntExpr) rendered per call


@dataclass(frozen=True)
class _RngSlot:
    """A per-launch philox slot the recorder declared (Tape.rng_slots): the launch,
    the parameter and byte offset of its integer intragraph-offset field, its width,
    and the recorded intragraph prefix; read by the runtime team's mixed rng replay."""

    call_index: int
    parameter: int
    byte_offset: int
    width: int
    prefix: object


@dataclass(frozen=True)
class _Region:
    """A closed library call on the tape (aten.mm / aten.addmm through cuBLAS; an op
    served by a torch._native override, torch/cuda/_host_trace_native.py): its kernel
    nodes come from a template the shared runtime's registry serves per call.
    `sources` are the operands' addresses (inputs then outputs), `metas` their (dtype,
    sizes, strides); `site` is the registry site, `variant` the plan value that reads
    the predicate's selection back."""

    seq: int
    name: str
    op: str
    scalars: tuple
    sources: tuple  # PointerSource per operand
    metas: tuple  # (dtype, sizes, strides) per operand, IntExpr or int entries
    site: int
    variant: object  # IntExpr
    # per operand (sizes, strides, displacement) as tape expressions, for the
    # preparation's registration before the predicate runs
    exprs: tuple

    @property
    def ranks(self):
        return tuple(len(sizes) for _, sizes, _ in self.metas)


def _align_class(address):
    # the largest power of two dividing the address, capped: cuBLAS picks kernels by
    # operand alignment (16..256 bytes matter)
    return min(address & -address, 256) if address else 256


_ALIGN_CPP = "((A) == int64_t(0) ? int64_t(256) : (((A) & -(A)) > int64_t(256) ? int64_t(256) : ((A) & -(A))))"
# the constant facts of the predicate as one memcmp (False: one if-statement each)
_FOLD_CONSTANT_FACTS = True
_HOST_CALL = "reinterpret_cast<int64_t (*)(const int64_t*, size_t)>"


def _host_call_lines(name, impl, arguments):
    """C++ statements declaring `name` as the host function at `impl` applied to the
    rendered `arguments` through the pointer-and-length ABI (a stack array)."""
    if not arguments:
        return [
            f"  const int64_t {name} = {_HOST_CALL}(uintptr_t({impl}))(nullptr, 0);"
        ]
    return [
        f"  const int64_t {name}_a[] = {{{', '.join(arguments)}}};",
        f"  const int64_t {name} = {_HOST_CALL}(uintptr_t({impl}))({name}_a, {len(arguments)});",
    ]


def _host_call_expr(impl, arguments):
    """The same call as one expression (an immediately invoked lambda owns the array)."""
    if not arguments:
        return f"{_HOST_CALL}(uintptr_t({impl}))(nullptr, 0)"
    return (
        f"([&]() -> int64_t {{ const int64_t a[] = {{{', '.join(arguments)}}}; "
        f"return {_HOST_CALL}(uintptr_t({impl}))(a, {len(arguments)}); }})()"
    )


def _region_key(region, sizes, strides, classes):
    """The registry key of a region at concrete operand shapes, without the library
    settings the registry appends: per operand its sizes, strides and alignment class."""
    key = []
    for index in range(len(region.metas)):
        key.extend(int(v) for v in sizes[index])
        key.extend(int(v) for v in strides[index])
        key.append(int(classes[index]))
    return key


def _decode_region_key(region, key):
    settings = len(torch._C._cuda_kernel_template_library_settings())
    values = list(key[: len(key) - settings])
    metas, classes = [], []
    for (dtype, _, _), rank in zip(region.metas, region.ranks):
        metas.append((dtype, tuple(values[:rank]), tuple(values[rank : 2 * rank])))
        classes.append(values[2 * rank])
        values = values[2 * rank + 1 :]
    return tuple(metas), tuple(classes)


def _selected_variant(values):
    # the preparation's value of a region's variant: the selection the predicate made
    # on this thread (the native plan calls selected_kernel_template the same way)
    return torch._C._cuda_kernel_template_selected(values[0])


@dataclass(frozen=True)
class _RegionArena:
    """The scratch a closed region's nodes get at replay in place of the library's own:
    the stream's cuBLAS workspace some kernels bake into their images, and the
    allocations the call makes for itself (its per-call workspace, A147), one buffer
    per allocation index. Owned by the entry and sized at preparation to the library's
    workspace size or the largest template seen there. The addresses are baked into
    every variant registered for the entry's sites (a site belongs to one entry), so a
    later template that needs more or larger scratch is another class of the tape,
    served by the same tape built at those inputs (its arena sized to it)."""

    workspace: torch.Tensor
    buffers: tuple

    @staticmethod
    def for_templates(device, templates):
        cap = torch._C._host_trace_blas_workspace_size()
        sizes = []
        for template in templates:
            for index, n in enumerate(template.scratch or ()):
                if index == len(sizes):
                    sizes.append(cap)
                sizes[index] = max(sizes[index], n)
        with torch.cuda.device(device):
            empty = functools.partial(torch.empty, dtype=torch.uint8, device=device)
            return _RegionArena(empty(cap), tuple(empty(n) for n in sizes))

    @property
    def workspace_address(self):
        return self.workspace.data_ptr()

    @property
    def addresses(self):
        return tuple(b.data_ptr() for b in self.buffers)

    def slot(self, role, const, addresses):
        """The address a classified qword or memset destination takes at these operand
        addresses: a constant, an operand plus a delta, a scratch buffer plus a delta,
        or the workspace (the harvest's roles)."""
        from torch.cuda import _host_trace

        return _host_trace._slot_value(
            role, const, addresses, self.workspace_address, self.addresses
        )

    def check(self, region, template):
        sizes = list(template.scratch or ())
        have = [b.numel() for b in self.buffers]
        if len(sizes) > len(have) or any(n > h for n, h in zip(sizes, have)):
            raise _AnotherClass(
                f"cuBLAS runs {region.name} ({region.op}) with per-call scratch of {sizes} bytes at this shape; this variant's arena holds {have}: build the tape at these inputs"
            )


def _region_template(region, key, device, identity):
    """The cuBLAS variant for `key`: harvested once per process into the
    process-wide template cache (commit 10's key: device, device identity, the
    call, the operand shapes and alignment classes, the library settings). Raises the
    harvest's Miss when the variant is not rebindable."""
    from torch.cuda import _host_trace

    metas, classes = _decode_region_key(region, key)
    spec = (region.op, region.scalars, metas, classes)
    cache_key = (device, identity, *spec, _host_trace._blas_settings())
    template = _host_trace._template(cache_key, spec, device)
    for n in template.nodes:
        if n["kind"] == "memset" and n["elem"] != 1:
            raise _host_trace.Miss(
                f"cuBLAS runs {region.name} ({region.op}) with a memset of {n['elem']}-byte elements at this shape; the shared runtime's template binding drives byte memsets"
            )
    return template


class _AnotherClass(Exception):
    """A harvested template this variant's exec cannot serve: another node chain
    than the site's, or scratch beyond the arena. The call is another class of the
    same tape (E28): the entry builds the tape at the call's inputs."""


def _region_chain(template):
    """A site's node chain: the nodes of the template the build's shapes selected,
    as (kind, template, node) rows; every variant registered for the site has the
    same kinds in order (E28: the chain is part of the variant's class)."""
    return [(n["kind"], template, n) for n in template.nodes]


def _variant_rows(template, arena):
    # the registry's rows, one per node of the template in order: kernel nodes with
    # the operand slots left to the binding and the scratch slots baked to the
    # entry's arena; memset nodes with an operand destination left to the binding,
    # any other destination baked the same way
    rows = []
    for n in template.nodes:
        if n["kind"] == "memset":
            role = n["dst_role"]
            if role is not None and role[0] == "op":
                operand, delta = role[1], role[2]
            else:
                operand, delta = None, arena.slot(role, n["dst"], ())
            rows.append(("memset", operand, delta, 1, n["width"], n["value"]))
            continue
        image = bytearray(n["image"])
        for offset, index, delta in n["scratch_slots"]:
            image[offset : offset + 8] = struct.pack(
                "<Q", arena.addresses[index] + delta
            )
        rows.append(
            (
                n["func"],
                tuple(n["grid"]),
                tuple(n["block"]),
                n["smem"],
                bytes(image),
                tuple(tuple(slot) for slot in n["slots"]),
                tuple(n["ws_slots"]),
                tuple(n["attrs"]),
            )
        )
    return tuple(rows)


def _register_region_variant(region, key, template, chain, arena):
    """Register a harvested template for the region's site, whose chain class is
    `chain` (_host_trace._chain_class: per node its kind and whether its incoming
    edge is programmatic). Raises _AnotherClass when the exec cannot serve it:
    another class, or scratch beyond the arena (the same tape built at the call's
    inputs serves it)."""
    from torch.cuda import _host_trace

    if _host_trace._chain_class(template.nodes) != tuple(chain):
        raise _AnotherClass(
            f"cuBLAS runs {region.name} ({region.op}) as {_host_trace._chain_text(_host_trace._chain_class(template.nodes))} at this shape; this variant's graph holds {_host_trace._chain_text(chain)} (built at another M): build the tape at these inputs"
        )
    arena.check(region, template)
    torch._C._cuda_kernel_template_register(
        region.site, list(key), _variant_rows(template, arena)
    )


def _expand(e):
    # sympy.expand does not accept booleans: expand the arithmetic inside each relation
    if isinstance(e, (sympy.And, sympy.Or, sympy.Not)):
        return e.func(*(_expand(a) for a in e.args))
    if isinstance(e, sympy.core.relational.Relational):
        return e.func(sympy.expand(e.lhs), sympy.expand(e.rhs))
    return sympy.expand(e)


def _pointer_parts(symbols, expr):
    """A tape pointer expression as (root source, sympy byte displacement from the
    runtime's root pointer: an input's data pointer or an allocation's base), through
    the mapping's translation (an input base is its data pointer less the recorded
    offset's bytes, an allocation base is its 256-byte-aligned runtime root)."""
    mapping = symbols.mapping
    try:
        e = sympy.expand(mapping.translate(_expr(expr)))
    except UnsupportedCapture as error:
        raise HostTraceLoweringDeclined(f"host_trace lowering: {error}") from error
    address_symbols = getattr(mapping, "address_symbols", {})
    roots = [s for s in e.free_symbols if s in address_symbols]
    if len(roots) != 1:
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: pointer {_expr(expr)} has {len(roots)} storage roots"
        )
    root = roots[0]
    if e.coeff(root) != 1:
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: pointer {_expr(expr)} is not the root plus a displacement"
        )
    displacement = sympy.expand(e - root)
    if displacement.free_symbols & address_symbols.keys():
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: pointer {_expr(expr)} is not the root plus a displacement"
        )
    return address_symbols[root], displacement


def _pointer_source(lowering, symbols, expr):
    """A tape pointer expression as (root, byte displacement from the runtime's root pointer)."""
    source, displacement = _pointer_parts(symbols, expr)
    return PointerSource(source, lowering.payload(displacement)), source


def _opaque_valued(symbols, expr):
    """Whether a pointer expression is an opaque rebind's result (plus a constant),
    with no storage root: the host looked the pointer up per call (symm's peer table
    and signal pads from the buffer's base address, E19)."""
    free = sympy.expand(_expr(expr)).free_symbols
    return bool(free) and all(
        s in symbols.by_symbol and symbols.by_symbol[s].kind == "opaque" for s in free
    )


def _symbol_values(lowered, boxed):
    """Every size, stride, offset and storage-base symbol of the tape at the boxed
    inputs, and the opaque guards' traced values: substituting them evaluates a tape
    expression at these inputs without the numeric plan."""
    return _values_of(lowered.symbols, boxed)


def _const_data_ptr(t):
    """`t.data_ptr()` through the storage's const accessor: a copy-on-write tensor in
    a position the tape never writes stays lazy, as it does under the native call."""
    return torch._C._data_address(t) + t.element_size() * t.storage_offset()


def _values_of(symbols, boxed):
    values = {}
    for symbol, prop in symbols.by_symbol.items():
        if prop.kind == "size":
            values[symbol] = sympy.Integer(boxed[prop.index].size(prop.dim))
        elif prop.kind == "stride":
            values[symbol] = sympy.Integer(boxed[prop.index].stride(prop.dim))
        elif prop.kind == "offset":
            values[symbol] = sympy.Integer(boxed[prop.index].storage_offset())
        elif prop.kind == "base":
            # the storage's base address, as the tape's root symbol values it (the
            # const accessor: a lazy clone stays lazy)
            values[symbol] = sympy.Integer(torch._C._data_address(boxed[prop.index]))
    for symbol, rec in symbols.opaque.items():
        if rec["kind"] == "guard":
            values[symbol] = sympy.Integer(int(rec["expected"]))
    return values


# PhiloxCudaState as the recorder records it (commit 1 at the cascade-11 tip): the seed
# and offset pointers as one 16-byte `rng` field, the u64 intragraph offset as its own
# param (the per-launch rng slot: constant 0 or a prefix sum), the captured flag as a
# u8 constant, and the struct's tail (padding) as an `rng` field the build never checks
_PHILOX_POINTER_BYTES = 16


def lower_tape(
    tape,
    example_args=None,
    *,
    arena=False,
    output_arena=False,
    allocseq=False,
    device=None,
):
    """The tape's records for the shared runtime, with the dispatch predicate compiled.
    With `arena`, the temporaries are planned into one arena input boxed after the
    tape's tensors (hosttrace_arena), at the shapes of `example_args` (the tape's
    hints without them); with `output_arena`, the escaping outputs are planned into
    one block boxed after it and bound as typed views over that input; with
    `allocseq` (in place of the arena; the output arena composes with it), the
    temporaries are roots boxed after the tape's tensors and the output block that
    the family binds by replaying the tape's allocation sequence (hosttrace_allocseq).
    `device` is the CUDA device the variant will be prepared on (the tape's own by
    default): its kernel handles, device facts and preparation."""
    if allocseq and arena:
        raise ValueError(
            "host_trace lowering: the allocation sequence replaces the planned arena; lower with one or the other"
        )
    if getattr(tape, "all_on_capture_stream", True) is False:
        # the recorder accepted a side stream forked from and joined to the capture
        # stream (E19, O33); this preparation re-issues every record on one stream in
        # host order, a linearization that would replay the branches serialized. A
        # replay of the capture's dependency DAG is the runtime team's (O33)
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: the tape's launches were not all issued on the trace's capturing stream (all_on_capture_stream=False; {len(tape.launches)} launches, {len(tape.memsets)} memsets, {len(tape.memcpys)} copies): the native preparation replays one stream in host order and would serialize the forked branches (a dependency-DAG replay is the runtime team's, O33)"
        )
    symbols = _Symbols(tape)
    # the runtime team's CPU tests lower bare objects without a device: read lazily
    device = (
        getattr(getattr(tape, "device", None), "index", -1)
        if device is None
        else int(device)
    )
    # opaque calls of kind "guard" (a selector the host computed, e.g. flash's split
    # count): the predicate re-runs the host's own function on the call's facts and
    # misses when the result differs from the traced one; everything downstream sees
    # the traced constant. Kind "rebind" (data recomputed per call) needs a plan op.
    opaque_values, opaque_terms = {}, []
    for rec in tape.opaque:
        if rec["kind"] != "guard":
            continue  # a rebind lowers to the plan's call op where its symbol is used
        if not rec.get("impl"):
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: opaque {rec['fn']} carries no host function address"
            )
        opaque_values[_raw_expr(rec["sym"])] = sympy.Integer(int(rec["expected"]))
        opaque_terms.append(rec)
    guards = tuple(g for g in tape.guards if g is not sympy.true)
    try:
        contract = integer_payload_contract_from_guards(
            guards, (s for s in symbols.by_symbol if s.is_integer is True)
        )
    except ValueError as error:
        raise HostTraceLoweringDeclined(
            f"Host integer payload contract: {error}"
        ) from error
    lowering = _Lowering(symbols, opaque_values, contract, strict_payload=False)
    allocations = tuple(
        OwnedBuffer(
            BufferSource(rec.name),
            rec.dtype,
            tuple(_int_or_expr(lowering, v) for v in rec.sizes),
            tuple(_int_or_expr(lowering, v) for v in rec.strides),
        )
        for rec in tape.allocs
    )
    calls = []
    rng_fields = []
    cute = None
    for call_index, L in enumerate(tape.launches):
        if L.get("cute") is not None:
            # a CuTe DSL launch site (torch/cuda/_host_trace_cute.py): its fields,
            # grid, shared bytes and obligations come from the runtime's binder over
            # the tape's symbols, its module is the runtime's kernel owner
            if cute is None:
                from torch._inductor.runtime._cudagraph.hosttrace_cute import (
                    CuTeLowering,
                )

                cute = CuTeLowering(tape, lowering, symbols, device)
            calls.append(cute.lower(L))
            continue
        layout = tuple(L["param_layout"])
        image = bytes(L["hint_image"])
        fields, constants = [], []
        covered = [bytearray(size) for _, size in layout]
        rng_slots_here = [
            r
            for r in getattr(tape, "rng_slots", None) or ()
            if r["launch"] == call_index
        ]

        def parameter_of(offset, size):
            for index, (start, width) in enumerate(layout):
                if start <= offset and offset + size <= start + width:
                    return index, offset - start
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: byte range {offset}+{size} is outside the parameter layout"
            )

        for p in L["params"]:
            index, inner = parameter_of(p["offset"], p["size"])
            covered[index][inner : inner + p["size"]] = b"\1" * p["size"]
            value, kind = p["value"], p["kind"]
            if kind == "rng":
                # a PhiloxCudaState the kernel reads. Its seed / offset pointers are the
                # capture's own (per-capture generator state): the tape holds the trace
                # capture's, the prepared call gets the preparation capture's
                # (prepare_hosttrace), so their bytes are no constant here. The
                # intragraph offset and the captured flag are params of their own
                # (lowered below like any other); the struct's tail is padding, kept
                # as the traced bytes
                if tape.rng_increment is None:
                    raise HostTraceLoweringDeclined(
                        "host_trace lowering: a kernel reads a philox state but the host declared no rng_increment"
                    )
                if p["size"] == _PHILOX_POINTER_BYTES:
                    slot_offset = p["offset"] + _PHILOX_POINTER_BYTES
                    if not any(r["offset"] == slot_offset for r in rng_slots_here):
                        # the recorder declares the intragraph offset as a per-launch
                        # slot beside the pointers; a tape without it would leave the
                        # traced value in the image, and a replay must not draw from it
                        raise HostTraceLoweringDeclined(
                            f"host_trace lowering: the philox state of {L['kernel']} has no per-launch rng slot (a tape without per-launch rng slots)"
                        )
                    rng_fields.append((call_index, index, inner))
                elif p["name"] == "philox_tail":
                    constants.append(
                        (index, inner, image[p["offset"] : p["offset"] + p["size"]])
                    )
                else:
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: an rng field of {p['size']} bytes ({p['name']}) is neither the philox pointers nor the state's tail"
                    )
                continue
            if not isinstance(value, _SYM_TYPES):
                if kind == "ptr" and int(value) != 0:
                    # a pointer the recorder could not attribute to a traced storage:
                    # nothing retains what it points at through the replay's lifetime
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: a constant host pointer field of {L['kernel']} has no retained storage owner"
                    )
                constants.append(
                    (index, inner, image[p["offset"] : p["offset"] + p["size"]])
                )
                continue
            if kind == "ptr":
                if _opaque_valued(symbols, value):
                    # the lookup's result is the field's value, an i64 the plan's call
                    # op computes per call (the runtime's pointer source names a
                    # storage root; this pointer has none)
                    source = ExpressionSource(lowering.lower(value))
                    fields.append(_PhysicalField(index, inner, "i64", source))
                    continue
                source, root = _pointer_source(lowering, symbols, value)
                fields.append(_PhysicalField(index, inner, "pointer", source))
            elif kind in _INT_KINDS:
                if _INT_KINDS[kind] != p["size"]:
                    raise HostTraceLoweringDeclined(
                        "host_trace lowering: integer field width differs from its kind"
                    )
                expression = lowering.payload(value)
                if kind == "u32":
                    expression = _as_i32(lowering, expression)
                source = ExpressionSource(expression)
                fields.append(
                    _PhysicalField(
                        index, inner, "i64" if kind in ("i64", "u64") else "i32", source
                    )
                )
            elif kind in _FLOAT_KINDS:
                if _FLOAT_KINDS[kind] != p["size"]:
                    raise HostTraceLoweringDeclined(
                        "host_trace lowering: float field width differs from its kind"
                    )
                # the plan's float slot holds double bits; a 32-bit field takes the
                # float32 bits (sign-extended into the i32 transport)
                expression = lowering.lower_float(value)
                if kind == "f32":
                    expression = lowering.node("ftobits32", args=(expression,))
                source = ExpressionSource(expression)
                fields.append(
                    _PhysicalField(
                        index, inner, "i32" if kind == "f32" else "i64", source
                    )
                )
            else:
                raise HostTraceLoweringDeclined(
                    f"host_trace lowering: a symbolic {kind} parameter has no scalar binding in the runtime (only i32/i64 and pointers)"
                )
        for index, (start, width) in enumerate(layout):
            # bytes no record names (struct padding, fields the host never wrote): the
            # captured bytes, kept as constants, never assumed to be zero
            span = None
            for b in range(width + 1):
                if b < width and not covered[index][b]:
                    if span is None:
                        span = b
                elif span is not None:
                    constants.append((index, span, image[start + span : start + b]))
                    span = None
        block_exprs = None
        if any(isinstance(b, _SYM_TYPES) for b in L["block_expr"]):
            # a block that depends on shape: bound like the grid (their block binding)
            block_exprs = tuple(_int_expr(lowering, b) for b in L["block_expr"])
        block = tuple(int(b) for b in L["block"])
        smem, shared_expr = L["smem"], None
        if isinstance(smem, _SYM_TYPES):
            shared_expr, smem = lowering.payload(smem), None
        else:
            smem = int(smem)
        grid = tuple(_int_expr(lowering, g) for g in L["grid"])
        for axis, (g, bound) in enumerate(zip(L["grid"], (2**31 - 1, 65535, 65535))):
            if isinstance(g, _SYM_TYPES):
                # a symbolic grid axis: within the launch bounds per call, not only
                # at the preparation shape
                lowering.require(sympy.Gt(_expr(g), 0))
                lowering.require(sympy.Le(_expr(g), bound))
        triton_launch = L.get("triton")
        if triton_launch is not None:
            module = _HostTraceTritonModule(triton_launch, device)
        elif L.get("cute_desc") is not None:
            module = _HostTraceCuTeDescModule(L, device)
        else:
            module = _HostTraceKernelModule(
                int(L["func"]), layout, block, smem, L["kernel"], device
            )
        calls.append(
            _PhysicalCall(
                tuple(fields),
                module,
                grid,
                (),
                tuple(constants),
                shared=shared_expr,
                block=block_exprs,
            )
        )
    memsets = []
    for m in getattr(tape, "memsets", ()):
        dst, _ = _pointer_source(lowering, symbols, m["dst"])
        memsets.append(
            (int(m["seq"]), dst, _int_expr(lowering, m["bytes"]), int(m["value"]))
        )
    from torch.cuda._host_trace import _KIND_FMT, _KIND_MASK

    host_tables = []
    for hb in getattr(tape, "host_buffers", ()):
        constants, elements = [], []
        for q in hb["elements"]:
            offset, width, kind, value = q["offset"], q["size"], q["kind"], q["value"]
            if not isinstance(value, _SYM_TYPES):
                if kind in _FLOAT_KINDS:
                    packed = struct.pack(_KIND_FMT[kind], float(value))
                else:
                    packed = struct.pack(_KIND_FMT[kind], int(value) & _KIND_MASK[kind])
                constants.append((offset, packed))
                continue
            if kind == "ptr":
                if any(
                    symbols.prop(s_).kind == "hbuf"
                    for s_ in sympy.expand(_expr(value)).free_symbols
                ):
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: host table {hb['name']} holds another table's host address"
                    )
                source, _ = _pointer_source(lowering, symbols, value)
                elements.append((offset, width, source))
            elif kind in _INT_KINDS or kind in ("i16", "u8"):
                expression = lowering.payload(value)
                if kind == "u32":
                    expression = _as_i32(lowering, expression)
                elements.append((offset, width, expression))
            elif kind in _FLOAT_KINDS:
                expression = lowering.lower_float(value)
                if kind == "f32":
                    expression = lowering.node("ftobits32", args=(expression,))
                elements.append((offset, width, expression))
            else:
                raise HostTraceLoweringDeclined(
                    f"host_trace lowering: a host table element of kind {kind}"
                )
        host_tables.append(
            _HostTable(
                int(hb["seq"]),
                hb["name"],
                int(hb["nbytes"]),
                tuple(constants),
                tuple(elements),
            )
        )
    memcpys = []
    pinned_positions = []
    written_positions = []
    # the inputs the tape wrote: the producer names them by argument position
    # (`written_inputs`, this line's producer) or by root name (`written_roots`,
    # cascade 14's: the roots read through the mutable accessor)
    written = set(getattr(tape, "written_inputs", ()))
    written_roots = set(getattr(tape, "written_roots", ()))
    for index, rec in enumerate(tape.inputs):
        if rec.device.type == "cpu":
            pinned_positions.append(symbols.positions[index])
        if rec.position in written or rec.root.name in written_roots:
            written_positions.append(index)
    memcpy_kinds = []
    for m in getattr(tape, "memcpys", ()):
        # the record declares its kind (cascade 10b); a record from before that is a
        # host-to-device copy, the only kind the recorder issued then. Nothing is
        # derived from the addresses: another kind declines by name.
        kind = (
            m.get("kind", "h2d") if isinstance(m, dict) else getattr(m, "kind", "h2d")
        )
        if kind not in ("h2d", "d2d"):
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: copy kind {kind!r} is not lowered (h2d and d2d are)"
            )
        src_expr = sympy.expand(_expr(m["src"]))
        table_roots = [
            s_ for s_ in src_expr.free_symbols if symbols.prop(s_).kind == "hbuf"
        ]
        if table_roots:
            if kind != "h2d":
                raise HostTraceLoweringDeclined(
                    f"host_trace lowering: a copy kind {kind!r} record reads a host table"
                )
            root = table_roots[0]
            if src_expr != root:
                raise HostTraceLoweringDeclined(
                    "host_trace lowering: a copy from inside a host table, not from its start"
                )
            source = symbols.prop(root).index
        else:
            source, root = _pointer_source(lowering, symbols, m["src"])
            source_is_host = (
                type(root) is InputSource
                and tape.inputs[root.index].device.type == "cpu"
            )
            if kind == "h2d" and not source_is_host:
                raise HostTraceLoweringDeclined(
                    "host_trace lowering: a copy whose source is neither a host table nor a pinned input"
                )
            if kind == "d2d" and source_is_host:
                raise HostTraceLoweringDeclined(
                    "host_trace lowering: a device-to-device copy record reads a pinned input"
                )
        dst, dst_root = _pointer_source(lowering, symbols, m["dst"])
        if (
            type(dst_root) is InputSource
            and tape.inputs[dst_root.index].device.type == "cpu"
        ):
            raise HostTraceLoweringDeclined(
                "host_trace lowering: a copy into a pinned input"
            )
        memcpys.append((int(m["seq"]), source, dst, _int_expr(lowering, m["bytes"])))
        memcpy_kinds.append(kind)
    # the predicate's pointer slots: one per boxed input (the arena input, then the
    # output block, after the tape's tensors)
    pointer_symbols = {
        index: sympy.Symbol(f"__ptr{index}") for index in range(len(tape.inputs) + 2)
    }

    def key_address(source, displacement):
        # a region operand's address for its alignment class: an input's data pointer
        # plus the displacement; an allocation's base is 256-byte aligned, so its
        # displacement alone
        if type(source) is InputSource:
            return pointer_symbols[source.index] + displacement
        return displacement

    # closed regions: the operands' addresses and shapes, a registry site each, and the
    # plan value that reads the predicate's variant selection back; the predicate's own
    # select terms are rendered with the guards below
    regions, region_exprs = [], []
    selected_address = None
    for r in getattr(tape, "regions", ()):
        # a closed region record the recorder did not write (no operands, no key
        # fields) is never omitted silently: it declines by name
        if not all(hasattr(r, a) for a in ("inputs", "outputs", "seq", "name")):
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: unrecognized closed regions record {r!r}"
            )
        if selected_address is None:
            selected_address = torch._C._cuda_kernel_template_selected_at_address()
        operands = [*r.inputs, *r.outputs]
        sources, metas, exprs, parts = [], [], [], []
        for o in operands:
            source, displacement = _pointer_parts(symbols, o.address)
            sources.append(PointerSource(source, lowering.payload(displacement)))
            metas.append(
                (
                    o.dtype,
                    tuple(_int_or_expr(lowering, v) for v in o.sizes),
                    tuple(_int_or_expr(lowering, v) for v in o.strides),
                )
            )
            exprs.append(
                (
                    [_expr(v) for v in o.sizes],
                    [_expr(v) for v in o.strides],
                    key_address(source, displacement),
                )
            )
            parts.append(
                (
                    tuple(_expr(v) for v in o.sizes),
                    tuple(_expr(v) for v in o.strides),
                    displacement,
                )
            )
        site = torch._C._cuda_kernel_template_new_site()
        variant = lowering.call(
            selected_address, _selected_variant, (lowering.const(site),)
        )
        regions.append(
            _Region(
                int(r.seq),
                r.name,
                r.op,
                tuple(r.scalars),
                tuple(sources),
                tuple(metas),
                site,
                variant,
                tuple(parts),
            )
        )
        region_exprs.append(exprs)
    owned = OrderedSet()
    for region in regions:
        for source in region.sources:
            if type(source.root) is BufferSource:
                owned.add(source.root.name)
    for call in calls:
        for source in (*call.storage_sources, *(field.source for field in call.fields)):
            for root in storage_roots(source):
                if type(root) is BufferSource:
                    owned.add(root.name)
    for _, dst, _, _ in memsets:
        if type(dst.root) is BufferSource:
            owned.add(dst.root.name)
    for _, _, dst, _ in memcpys:
        if type(dst.root) is BufferSource:
            owned.add(dst.root.name)
    for table in host_tables:
        # a kernel reaches these buffers through the device copy of the table
        for _, _, source in table.elements:
            if type(source) is PointerSource and type(source.root) is BufferSource:
                owned.add(source.root.name)
    # an allocation the host returned that no node writes (flash's debug mask
    # without a debug run): a runtime buffer like any other output, allocated per
    # call at the plan's shape and left as eager's at::empty leaves it (its
    # positions are named for the tests: nothing compares their values)
    unwritten_roots = set()
    for rec in tape.allocs:
        if rec.name not in owned and any(o.root is rec.root for o in tape.outputs):
            owned.add(rec.name)
            unwritten_roots.add(id(rec.root))
    unwritten_outputs = tuple(
        k for k, o in enumerate(tape.outputs) if id(o.root) in unwritten_roots
    )
    layout_by_name = {layout.source.name: layout for layout in allocations}
    allocations = tuple(layout for layout in allocations if layout.source.name in owned)
    escapes = OrderedSet(
        rec.name for rec in tape.allocs if any(o.root is rec.root for o in tape.outputs)
    )
    hints = None
    if arena or output_arena or allocseq:
        hints = (
            _values_of(symbols, list(symbols.tensors_of(example_args)))
            if example_args is not None
            else _tape_hints(tape, symbols)
        )
    arena_capacity = sympy.Symbol("__arena_capacity", integer=True, nonnegative=True)
    output_capacity = sympy.Symbol("__output_capacity", integer=True, nonnegative=True)
    arena_index = len(tape.inputs)
    output_index = arena_index + 1 if arena else arena_index
    output_plan = None
    if output_arena:
        # the escaping outputs as one block per call, boxed after the arena: every
        # output over a placed allocation is a typed view of that input (below)
        (
            output_plan,
            allocations,
            calls,
            memsets,
            memcpys,
            host_tables,
            regions,
            region_exprs,
        ) = _arena_pass(
            tape,
            symbols,
            lowering,
            allocations,
            calls,
            memsets,
            memcpys,
            host_tables,
            regions,
            region_exprs,
            escapes,
            hints,
            output_index,
            output_capacity,
            pointer_symbols,
            together=True,
        )
    returned_layouts = set()  # allocations already returned whole (7c)
    by_root = {
        id(rec.root): layout_by_name[rec.name]
        for rec in tape.allocs
        if rec.name in owned
    }
    output_pins = (
        recorded_guard_pins(tape)
        if any(getattr(rec, "identity", None) is not None for rec in tape.outputs)
        else {}
    )

    def identity_expr(value):
        expression = _expr(value)
        return expression.xreplace(output_pins) if output_pins else expression

    outputs = []
    for position, rec in enumerate(tape.outputs):
        identity = getattr(rec, "identity", None)
        if identity is not None:
            if (
                type(identity) is not tuple
                or len(identity) != 2
                or type(identity[1]) is not int
            ):
                raise HostTraceLoweringDeclined("Invalid host trace output identity")
            kind, index = identity
            if kind == "output" and 0 <= index < position:
                target = tape.outputs[index]
                output = OutputReference(index)
            elif kind == "argument" and index in symbols.tensor_index:
                tensor_index = symbols.tensor_index[index]
                target = tape.inputs[tensor_index]
                output = BorrowedInputOutput(InputSource(tensor_index))
            else:
                raise HostTraceLoweringDeclined("Invalid host trace output identity")
            if (
                rec.root is not target.root
                or rec.dtype != target.dtype
                or tuple(map(identity_expr, rec.sizes))
                != tuple(map(identity_expr, target.sizes))
                or tuple(map(identity_expr, rec.strides))
                != tuple(map(identity_expr, target.strides))
                or identity_expr(rec.offset) != identity_expr(target.offset)
            ):
                raise HostTraceLoweringDeclined(
                    "Host trace output identity lost its recorded layout"
                )
            outputs.append(output)
            continue
        if id(rec.root) in by_root:
            layout = by_root[id(rec.root)]
            if output_plan is not None and output_plan.covers(layout.source.name):
                # a view of the output block, in the view's own units over the
                # block's storage (the block's storage offset is zero); two views
                # the host returned of one allocation are two views of the block
                offset = output_plan.offset(layout.source.name) / rec.dtype.itemsize
                outputs.append(
                    TensorViewOutput(
                        InputSource(output_index),
                        tuple(_int_or_expr(lowering, v) for v in rec.sizes),
                        tuple(_int_or_expr(lowering, v) for v in rec.strides),
                        _int_or_expr(
                            lowering, sympy.expand(offset + _expr(rec.offset))
                        ),
                        rec.dtype,
                    )
                )
                continue
            whole = (
                tuple(_int_or_expr(lowering, v) for v in rec.sizes) == layout.size
                and tuple(_int_or_expr(lowering, v) for v in rec.strides)
                == layout.stride
                and _expr(rec.offset) == 0
                and rec.dtype == layout.dtype
            )
            if whole and id(layout) not in returned_layouts:
                # a second whole view of the same allocation stays a view: two equal
                # views the host returned are two Tensors, as eager returns them
                returned_layouts.add(id(layout))
                outputs.append(layout)
                continue
            # the view's sizes, strides and offset are in its own element units, over
            # the allocation's storage (a view_as_real / view_as_complex of it carries
            # its own dtype; the runtime's view binding takes it)
            outputs.append(
                TensorViewOutput(
                    layout.source,
                    tuple(_int_or_expr(lowering, v) for v in rec.sizes),
                    tuple(_int_or_expr(lowering, v) for v in rec.strides),
                    _int_or_expr(lowering, rec.offset),
                    None if rec.dtype == layout.dtype else rec.dtype,
                )
            )
            continue
        index = next(i for i, inp in enumerate(tape.inputs) if inp.root is rec.root)
        inp = tape.inputs[index]
        if rec.dtype == inp.dtype:
            # the runtime adds the input's storage offset to the view's
            offset, dtype = sympy.expand(_expr(rec.offset) - _expr(inp.offset)), None
        else:
            # another dtype: the offset over the input's storage in the view's units,
            # as the tape recorded it (the runtime adds nothing to a typed view's)
            offset, dtype = _expr(rec.offset), rec.dtype
        outputs.append(
            TensorViewOutput(
                InputSource(index),
                tuple(_int_or_expr(lowering, v) for v in rec.sizes),
                tuple(_int_or_expr(lowering, v) for v in rec.strides),
                _int_or_expr(lowering, offset),
                dtype,
            )
        )
    arena_plan = None
    if arena:
        temporaries = OrderedSet(
            layout.source.name
            for layout in allocations
            if layout.source.name not in escapes
        )
        (
            arena_plan,
            allocations,
            calls,
            memsets,
            memcpys,
            host_tables,
            regions,
            region_exprs,
        ) = _arena_pass(
            tape,
            symbols,
            lowering,
            allocations,
            calls,
            memsets,
            memcpys,
            host_tables,
            regions,
            region_exprs,
            temporaries,
            hints,
            arena_index,
            arena_capacity,
            pointer_symbols,
        )
    programmatic_regions = ()
    if allocseq and regions:
        # the free-point rule's input (A326): the regions whose first node the library
        # launches with programmatic stream serialization, from the templates the
        # preparation selects (harvested here when not yet cached). The launch flag
        # is a superset of the driver's edge flags (behind a memset node or first in
        # the graph the driver records a full edge), so the plan is conservative and
        # the preparation checks the driver's flags against it; without example
        # inputs every region counts.
        if example_args is None:
            programmatic_regions = tuple(region.seq for region in regions)
        else:
            from torch.cuda import _host_trace

            tensors = list(symbols.tensors_of(example_args))
            plans = _region_templates(
                regions,
                hints,
                lambda i: _const_data_ptr(tensors[i]) if i < len(tensors) else 0,
                device,
                _host_trace._device_identity(device),
            )
            programmatic_regions = tuple(
                region.seq
                for region, (_, template) in zip(regions, plans)
                if template.nodes[0]["programmatic"]
            )
    sequence_plan = None
    if allocseq:
        temporaries = OrderedSet(
            layout.source.name
            for layout in allocations
            if layout.source.name not in escapes
        )
        (
            sequence_plan,
            allocations,
            calls,
            memsets,
            memcpys,
            host_tables,
            regions,
            region_exprs,
        ) = _sequence_pass(
            tape,
            symbols,
            lowering,
            allocations,
            calls,
            memsets,
            memcpys,
            host_tables,
            regions,
            region_exprs,
            temporaries,
            hints,
            output_index + 1 if output_plan is not None else len(tape.inputs),
            programmatic_regions,
        )
    # the predicate over the boxed Tensors: input data pointers, then storage offsets,
    # then the Tensor facts, in the order the native dispatcher fills int_values
    substitution = {}
    for symbol, prop in symbols.by_symbol.items():
        if prop.kind == "base":
            rec = tape.inputs[prop.index]
            substitution[symbol] = pointer_symbols[
                prop.index
            ] - rec.root.itemsize * _expr(rec.offset)
    used_pointers, used_offsets, terms = OrderedSet(), OrderedSet(), []
    fact_terms = []  # (fact, expected) rows the entry used to check in Python
    positive = []  # size facts that must be > 0
    # every size or stride symbol the tape knows, including the ones the ShapeEnv
    # replaced by a constant in the records (their guards still name the symbol)
    fact_symbols = {
        symbol: _Fact(prop.kind, prop.index, prop.dim)
        for symbol, prop in symbols.by_symbol.items()
        if prop.kind in ("size", "stride")
    }
    for index, rec in enumerate(tape.inputs):
        fact_terms.append((_Fact("rank", index), len(rec.sizes)))
        fact_terms.append((_Fact("dtype", index), _dtype_code(rec.dtype)))
        if rec.device.type == "cpu":
            # a pinned CPU input: the source of an in-graph copy (device fact -1)
            fact_terms.append((_Fact("device", index), -1))
            fact_terms.append((_Fact("pinned", index), 1))
        else:
            fact_terms.append((_Fact("device", index), device))
        fact_terms.append((_Fact("neg", index), 0))
        fact_terms.append((_Fact("conj", index), 0))
        for d, value in enumerate(rec.sizes):
            e = _expr(value)
            fact = _Fact("size", index, d)
            if isinstance(e, sympy.Symbol):
                # the recorder admits non-empty inputs only: every size is positive.
                # (Not a sympy relation: the ShapeEnv's positive assumption folds it away.)
                positive.append(fact)
            elif isinstance(e, sympy.Integer):
                fact_terms.append((fact, int(e)))
            else:
                # a size the ShapeEnv expressed through other symbols: an equality term
                alias = sympy.Symbol(f"__{fact.name}")
                fact_symbols[alias] = fact
                terms.append(sympy.Eq(alias, e, evaluate=False))
        for d, value in enumerate(rec.strides):
            e = _expr(value)
            fact = _Fact("stride", index, d)
            if isinstance(e, sympy.Integer):
                fact_terms.append((fact, int(e)))
            elif not isinstance(e, sympy.Symbol):
                alias = sympy.Symbol(f"__{fact.name}")
                fact_symbols[alias] = fact
                terms.append(sympy.Eq(alias, e, evaluate=False))
    if arena_plan is not None:
        # the arena input: a one-dimensional uint8 CUDA tensor whose length is the
        # capacity the requirement term reads
        fact_terms.append((_Fact("rank", arena_index), 1))
        fact_terms.append((_Fact("dtype", arena_index), _dtype_code(torch.uint8)))
        fact_terms.append((_Fact("device", arena_index), device))
        fact_terms.append((_Fact("neg", arena_index), 0))
        fact_terms.append((_Fact("conj", arena_index), 0))
        fact_symbols[arena_capacity] = _Fact("size", arena_index, 0)
    if output_plan is not None:
        # the output block input: the same form, its length the output requirement's
        fact_terms.append((_Fact("rank", output_index), 1))
        fact_terms.append((_Fact("dtype", output_index), _dtype_code(torch.uint8)))
        fact_terms.append((_Fact("device", output_index), device))
        fact_terms.append((_Fact("neg", output_index), 0))
        fact_terms.append((_Fact("conj", output_index), 0))
        fact_symbols[output_capacity] = _Fact("size", output_index, 0)
    sequence_symbols = ()
    if sequence_plan is not None:
        # the sequence roots: one-dimensional uint8 tensors over their blocks, whose
        # lengths the size terms compare with the bytes the call needs
        sequence_symbols = tuple(
            sympy.Symbol(f"__seq{k}", integer=True, nonnegative=True)
            for k in range(len(sequence_plan.names))
        )
        for k, symbol in enumerate(sequence_symbols):
            fact_symbols[symbol] = _Fact("size", sequence_plan.input_index + k, 0)
    used_facts = OrderedSet(fact for fact, _ in fact_terms)
    used_facts.update(positive)

    def predicate_expr(a):
        # a term as the predicate reads it: opaque guard results as their traced
        # constants, storage bases as the pointer slots less the offsets' bytes
        e = lowering.subst(a)
        return _expand(e.xreplace(substitution)) if substitution else e

    opaque_exprs = []
    for rec in opaque_terms:
        args = [predicate_expr(a) for a in rec["args"]]
        for a in args:
            for s in getattr(a, "free_symbols", ()):
                if s in fact_symbols:
                    used_facts.add(fact_symbols[s])
                elif s in pointer_symbols.values():
                    used_pointers.add(
                        next(i for i, ps in pointer_symbols.items() if ps == s)
                    )
                else:
                    prop = symbols.prop(s)
                    if prop.kind == "offset":
                        used_offsets.add(prop.index)
                    else:
                        raise HostTraceLoweringDeclined(
                            f"host_trace lowering: opaque {rec['fn']} reads {s} ({prop.kind}) with no predicate source"
                        )
        opaque_exprs.append((rec, args))
    # the region keys: sizes, strides and address alignment classes, over the
    # predicate's facts, pointers and offsets (the key address: an input's pointer
    # slot plus the displacement, an allocation's displacement alone)
    region_key_exprs = []
    for exprs in region_exprs:
        rendered = []
        for sizes, strides, address in exprs:
            address = sympy.expand(lowering.subst(address))
            entries = [lowering.subst(v) for v in (*sizes, *strides)] + [address]
            for e in entries:
                for s_ in getattr(e, "free_symbols", ()):
                    if s_ in pointer_symbols.values():
                        used_pointers.add(
                            next(i for i, ps in pointer_symbols.items() if ps == s_)
                        )
                    elif s_ in fact_symbols:
                        used_facts.add(fact_symbols[s_])
                    else:
                        prop = symbols.prop(s_)
                        if prop.kind == "offset":
                            used_offsets.add(prop.index)
                        else:
                            raise HostTraceLoweringDeclined(
                                f"host_trace lowering: a closed region's operand reads {s_} ({prop.kind}) with no predicate source"
                            )
            rendered.append(entries)
        region_key_exprs.append(rendered)

    def rebind_arguments(symbol, seen=()):
        # the free symbols a rebind's arguments read, recursively through rebinds
        rec = symbols.opaque[symbol]
        out = OrderedSet()
        for a in rec["args"]:
            for t in getattr(predicate_expr(a), "free_symbols", ()):
                if t in symbols.opaque and t not in seen:
                    out.update(rebind_arguments(t, (*seen, symbol)))
                else:
                    out.add(t)
        return out

    extra_guards = tuple(dict.fromkeys(lowering.guards))
    alloc_symbols = OrderedSet(
        [symbol for symbol, prop in symbols.by_symbol.items() if prop.kind == "alloc"]
    )
    # the tape's root identity facts (Tape.root_facts): a relation between two roots
    # the recorder decided by identity is a fact of the tape, not a guard; a guard
    # that still reads an allocation address declines below
    root_names = {}
    for symbol, prop in symbols.by_symbol.items():
        if prop.kind == "base":
            root_names[symbol] = tape.inputs[prop.index].root.name
        elif prop.kind == "alloc":
            root_names[symbol] = tape.allocs[prop.index].root.name
    root_facts = tuple(tuple(pair) for pair in getattr(tape, "root_facts", ()))
    root_pairs = OrderedSet(frozenset(pair) for pair in root_facts)
    ordered = []  # (the tape's guard as recorded, the predicate's expression)
    for g in (*guards, *extra_guards):
        # opaque guard results are their traced constants here (the predicate re-runs
        # the host function separately); a guard that folds to True has nothing to say
        e = lowering.subst(g)
        if e is sympy.true or e == True:  # noqa: E712
            continue
        if isinstance(e, sympy.Ne) and not alloc_symbols.isdisjoint(e.free_symbols):
            roots = frozenset(root_names[s] for s in e.free_symbols if s in root_names)
            if len(roots) == 2 and roots in root_pairs:
                continue  # decided by root identity: a declared fact
        e = _expand(e.xreplace(substitution)) if substitution else e
        free = OrderedSet()
        for s in e.free_symbols:
            if s in symbols.opaque:
                free.update(rebind_arguments(s))  # rendered inline by cpp()
            else:
                free.add(s)
        for s in free:
            if s in pointer_symbols.values():
                used_pointers.add(
                    next(i for i, ps in pointer_symbols.items() if ps == s)
                )
            elif s in fact_symbols:
                used_facts.add(fact_symbols[s])
            else:
                prop = symbols.prop(s)
                if prop.kind == "offset":
                    used_offsets.add(prop.index)
                elif prop.kind == "alloc":
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: guard {g} reads an allocation address"
                    )
                else:
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: guard {g} reads {s} ({prop.kind}) with no predicate source"
                    )
        ordered.append((g, e))
    arena_terms = []
    requirements = []  # the capacity terms: (plan, capacity symbol)
    if arena_plan is not None:
        requirements.append((arena_plan, arena_capacity))
    if output_plan is not None:
        requirements.append((output_plan, output_capacity))
    for plan, capacity in requirements:
        # the arena terms: the capacity the call needs, then the plan's class guards
        # (the pair conditions the proofs left, and the hint equalities they used;
        # the output plan is a chain and has none). The capacity term is built
        # unevaluated: sympy's relational evaluation asks the assumptions system about
        # the requirement, seconds for a sum over a hundred sizes (a cat of many inputs)
        capacity_term = sympy.Ge(capacity, lowering.subst(plan.arena), evaluate=False)
        for t in (capacity_term, *(lowering.subst(g) for g in plan.class_guards)):
            for s_ in t.free_symbols:
                if s_ in fact_symbols:
                    used_facts.add(fact_symbols[s_])
                elif symbols.prop(s_).kind == "offset":
                    used_offsets.add(symbols.prop(s_).index)
                else:
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: the arena plan reads {s_} ({symbols.prop(s_).kind}) with no predicate source"
                    )
            arena_terms.append(t)
    if sequence_plan is not None:
        # the size terms: every root covers the bytes the call needs (a call whose
        # temporaries outgrew the bound blocks rebinds them on the miss path)
        for symbol, size in zip(sequence_symbols, sequence_plan.sizes):
            t = sympy.Ge(symbol, lowering.subst(size), evaluate=False)
            for s_ in t.free_symbols:
                if s_ not in fact_symbols:
                    raise HostTraceLoweringDeclined(
                        f"host_trace lowering: the allocation sequence reads {s_} with no predicate source"
                    )
                used_facts.add(fact_symbols[s_])
            arena_terms.append(t)
    input_names = tuple(f"boxed_{i}" for i in range(len(tape.inputs)))
    if arena_plan is not None:
        input_names = (*input_names, "arena")
    if output_plan is not None:
        input_names = (*input_names, "outputs")
    if sequence_plan is not None:
        input_names = (
            *input_names,
            *(f"seq{k}" for k in range(len(sequence_plan.names))),
        )
    names = {}
    declarations = []
    pointer_indices = tuple(sorted(used_pointers))
    offset_indices = tuple(sorted(used_offsets))
    slot = 0
    for index in pointer_indices:
        names[pointer_symbols[index]] = f"p{index}"
        declarations.append(f"  const int64_t p{index} = int_values[{slot}];")
        slot += 1
    for index in offset_indices:
        for symbol, prop in symbols.by_symbol.items():
            if prop.kind == "offset" and prop.index == index:
                names[symbol] = f"o{index}"
        declarations.append(f"  const int64_t o{index} = int_values[{slot}];")
        slot += 1
    facts = tuple(used_facts)
    for fact in facts:
        for symbol, f in fact_symbols.items():
            if f == fact:
                names[symbol] = fact.name
        declarations.append(f"  const int64_t {fact.name} = int_values[{slot}];")
        slot += 1
    # the fixed facts first: rank before any size or stride term can matter. They
    # occupy the first fact slots in fact_terms order (used_facts starts with them),
    # so their check is one memcmp of that slice of int_values against a static
    # array of the expected values: the same equalities, evaluated in one place
    constant_start = len(pointer_indices) + len(offset_indices)
    if facts[: len(fact_terms)] != tuple(fact for fact, _ in fact_terms):
        raise AssertionError("the constant facts must lead the predicate's fact slots")
    if _FOLD_CONSTANT_FACTS and fact_terms:
        expected_values = ", ".join(str(int(expected)) for _, expected in fact_terms)
        checks = [
            f"  static const int64_t expected_facts[{len(fact_terms)}] = {{{expected_values}}};",
            f"  if (std::memcmp(int_values + {constant_start}, expected_facts, sizeof(expected_facts)) != 0) return 0;",
        ]
    else:
        checks = [
            f"  if (!({fact.name} == int64_t({expected}))) return 0;"
            for fact, expected in fact_terms
        ]
    checks.extend(f"  if (!({fact.name} > int64_t(0))) return 0;" for fact in positive)
    # a size or stride the ShapeEnv expressed through other symbols
    checks.extend(f"  if (!({lowering.cpp(t, names)}) || bad) return 0;" for t in terms)
    declared_calls = {}  # (impl, rendered arguments) -> the local holding its value

    def declare_rebinds(expression, into):
        # an opaque rebind a term reads: the host's own function, called once into a
        # local at the term's position (its arguments' domains are the guards before
        # it); its own rebind arguments first
        for t in sorted(expression.free_symbols, key=sympy.default_sort_key):
            if t not in symbols.opaque or t in names:
                continue
            rec = symbols.opaque[t]
            for a in rec["args"]:
                declare_rebinds(predicate_expr(a), into)
            if not rec.get("impl"):
                raise HostTraceLoweringDeclined(
                    f"host_trace lowering: opaque {rec['fn']} carries no host function address"
                )
            rendered = [
                f"int64_t({lowering.cpp(predicate_expr(a), names)})"
                for a in rec["args"]
            ]
            key = (int(rec["impl"]), tuple(rendered))
            if key in declared_calls:
                # the same function on the same arguments: the same value (E23)
                names[t] = declared_calls[key]
                continue
            name = names[t] = declared_calls[key] = f"r{len(names)}"
            into.extend(_host_call_lines(name, key[0], rendered))
            into.append("  if (bad) return 0;")
            if rec.get("domain") == "positive":
                # the host declared the result positive (cascade 13's opaque
                # domains): a value outside it misses
                into.append(f"  if ({name} < 1) return 0;")

    def opaque_term(rec, args, into):
        # an opaque guard: the host's selector re-run on the call's facts must give
        # the traced value
        for a in args:
            declare_rebinds(a, into)
        rendered = [f"int64_t({lowering.cpp(a, names)})" for a in args]
        into.append(
            f"  if (!({_host_call_expr(int(rec['impl']), rendered)} == int64_t({int(rec['expected'])})) || bad) return 0;"
        )

    # the guards in the tape's order, one statement per term (the same short-circuit
    # evaluation as one conjunction, without the single expression a large tape's
    # guards would make); an opaque guard's term goes before the first guard that
    # read its result, the rest after every guard; a term's integer arithmetic is
    # checked (the `bad` flag)
    pending_opaque = list(opaque_exprs)
    for g, t in ordered:
        raw = getattr(g, "free_symbols", OrderedSet())
        for rec, args in list(pending_opaque):
            if _raw_expr(rec["sym"]) in raw:
                opaque_term(rec, args, checks)
                pending_opaque.remove((rec, args))
        declare_rebinds(t, checks)
        checks.append(f"  if (!({lowering.cpp(t, names)}) || bad) return 0;")
    for rec, args in pending_opaque:
        opaque_term(rec, args, checks)
    arena_checks = [
        f"  if (!({lowering.cpp(t, names)}) || bad) return 0;" for t in arena_terms
    ]
    region_checks = []
    if regions:
        # every region selects (the registry records a miss per site, so one ordinary
        # call harvests them all): the selects are combined without short-circuiting,
        # after every guard, so a recorded miss is a key of a call the tape describes
        select_address = torch._C._cuda_kernel_template_select_key_address()
        host_select = "reinterpret_cast<int64_t (*)(int64_t, const int64_t*, size_t)>"
        selects = []
        for k, entries in enumerate(region_key_exprs):
            # the site is a value of the variant's stub (below), not a literal of this
            # source: a rebuild of the same tape shares the compiled predicate; the key
            # is a stack array the registry compares against the site's last key
            arguments = []
            for operand in entries:
                for v in operand:
                    declare_rebinds(v, region_checks)
                arguments.extend(
                    f"int64_t({lowering.cpp(v, names)})" for v in operand[:-1]
                )
                address = f"(int64_t({lowering.cpp(operand[-1], names)}))"
                arguments.append(_ALIGN_CPP.replace("(A)", address))
            region_checks.append(
                f"  const int64_t key{k}[] = {{{', '.join(arguments)}}};"
            )
            selects.append(
                f"int({host_select}(uintptr_t({select_address}))(sites[{k}], key{k}, {len(arguments)}) >= int64_t(0))"
            )
        region_checks.append(f"  if (bad || (({' & '.join(selects)}) == 0)) return 0;")
    # the requirement functions: the arena bytes and the output block bytes this call
    # needs (-1 on overflow), read by the entry on a miss to size the family's
    arena_bytes = tuple(
        line
        for name, plan in (("arena_bytes", arena_plan), ("output_bytes", output_plan))
        if plan is not None
        for line in (
            f'extern "C" int64_t {name}(int64_t* int_values, double* float_values) {{',
            "  (void)float_values;",
            "  bool bad = false;",
            *declarations,
            f"  const int64_t need = {lowering.cpp(lowering.subst(plan.arena), names)};",
            "  return bad ? int64_t(-1) : need;",
            "}",
        )
    )
    sequence_sizes = ()
    if sequence_plan is not None:
        # the sizes function: every root's bytes from the compact fact vector (the
        # size facts the sizes read, then the stride facts, in the plan's slot order)
        compact = OrderedSet()
        for size in sequence_plan.sizes:
            compact.update(
                sorted(lowering.subst(size).free_symbols, key=sympy.default_sort_key)
            )
        ordered = [s_ for s_ in compact if fact_symbols[s_].kind == "size"] + [
            s_ for s_ in compact if fact_symbols[s_].kind == "stride"
        ]
        compact_names = {s_: f"q{i}" for i, s_ in enumerate(ordered)}
        sequence_plan.size_reads = tuple(
            (fact_symbols[s_].index, fact_symbols[s_].dim)
            for s_ in ordered
            if fact_symbols[s_].kind == "size"
        )
        sequence_plan.stride_reads = tuple(
            (fact_symbols[s_].index, fact_symbols[s_].dim)
            for s_ in ordered
            if fact_symbols[s_].kind == "stride"
        )
        sequence_sizes = (
            'extern "C" int64_t seq_sizes(const int64_t* facts, int64_t* out) {',
            "  bool bad = false;",
            *(
                f"  const int64_t {name} = facts[{i}];"
                for i, name in enumerate(compact_names.values())
            ),
            *(
                f"  {{ const int64_t v = {lowering.cpp(lowering.subst(size), compact_names)}; out[{k}] = v > 0 ? v : 1; }}"
                for k, size in enumerate(sequence_plan.sizes)
            ),
            "  return bad ? int64_t(-1) : int64_t(0);",
            "}",
        )
    # the predicate: the tape's terms in one library (cached by source: a rebuild of
    # the same tape at another topology compiles nothing here) and, when the tape has
    # regions, the variant's own stub that binds its site ids and calls it
    entries = (
        ("guard", _MODE_ALL),
        ("guard_facts", _MODE_FACTS),
        ("guard_arena", _MODE_ARENA),
    )

    def entry_lines(call):
        return [
            line
            for name, mode in entries
            for line in (
                f'extern "C" int8_t {name}(int64_t* int_values, double* float_values) {{',
                f"  return {call}(int_values, float_values, {mode}, {'sites' if regions else 'nullptr'});",
                "}",
            )
        ]

    source = "\n".join(
        (
            *_PREDICATE_PREAMBLE,
            'extern "C" int8_t guard_impl(int64_t* int_values, double* float_values, int mode, const int64_t* sites) {',
            "  (void)float_values;",
            "  (void)sites;",
            "  bool bad = false;",
            *declarations,
            *checks,
            *(
                (f"  if (mode & {_MODE_ARENA}) {{", *arena_checks, "  }")
                if arena_checks
                else ()
            ),
            *(
                (f"  if (mode & {_MODE_ALL & ~_MODE_ARENA}) {{", *region_checks, "  }")
                if region_checks
                else ()
            ),
            "  return 1;",
            "}",
            *(entry_lines("guard_impl") if not regions else ()),
            *arena_bytes,
            *sequence_sizes,
            "",
        )
    )
    library = CppCodeCache.load(source)
    if sequence_plan is not None:
        sequence_plan.sizes_address = ctypes.cast(
            library.seq_sizes, ctypes.c_void_p
        ).value
    arena_bytes_address = (
        ctypes.cast(library.arena_bytes, ctypes.c_void_p).value if arena_plan else 0
    )
    output_bytes_address = (
        ctypes.cast(library.output_bytes, ctypes.c_void_p).value if output_plan else 0
    )
    if regions:
        impl = ctypes.cast(library.guard_impl, ctypes.c_void_p).value
        stub = "\n".join(
            (
                "#include <cstdint>",
                f"static const int64_t sites[] = {{{', '.join(str(r.site) for r in regions)}}};",
                "using impl_t = int8_t (*)(int64_t*, double*, int, const int64_t*);",
                f"static const impl_t impl = reinterpret_cast<impl_t>(uintptr_t({impl}));",
                *entry_lines("impl"),
                "",
            )
        )
        # the owner the dispatch keeps: the stub and the library it calls into
        library = (CppCodeCache.load(stub), library)
    functions = library[0] if regions else library
    address = ctypes.cast(functions.guard, ctypes.c_void_p).value
    facts_address = ctypes.cast(functions.guard_facts, ctypes.c_void_p).value
    arena_address = ctypes.cast(functions.guard_arena, ctypes.c_void_p).value
    if not address or not facts_address or not arena_address:
        raise HostTraceLoweringDeclined(
            "host_trace lowering: the compiled predicate has no address"
        )
    # the recorder's per-launch philox slots as rows over the calls' parameter layouts
    # (their u32 field must lie inside one parameter: the compiler-recorded integer)
    rng_slots = []
    for r in getattr(tape, "rng_slots", None) or ():
        call = calls[r["launch"]]
        parameter, inner = _parameter_of(call, r["offset"], r["size"])
        matching = [
            p
            for p in tape.launches[r["launch"]]["params"]
            if p["offset"] == r["offset"]
            and p["size"] == r["size"]
            and p["kind"] in ("u32", "u64")
        ]
        if len(matching) != 1 or r["size"] not in (4, 8):
            raise HostTraceLoweringDeclined(
                "host_trace lowering: an rng slot lost its compiler-recorded integer field"
            )
        rng_slots.append(
            _RngSlot(r["launch"], parameter, inner, r["size"], matching[0]["value"])
        )
    return LoweredTape(
        tape,
        symbols,
        allocations,
        tuple(outputs),
        tuple(calls),
        source,
        library,
        address,
        pointer_indices,
        offset_indices,
        facts,
        _Records(input_names),
        device,
        tuple(memsets),
        tuple(host_tables),
        tuple(memcpys),
        tuple(pinned_positions),
        None if tape.rng_increment is None else lowering.payload(tape.rng_increment),
        tuple(rng_fields),
        tuple(regions),
        rng_slots=tuple(rng_slots),
        memcpy_kinds=tuple(memcpy_kinds),
        extra_guards=extra_guards,
        facts_address=facts_address,
        root_facts=root_facts,
        arena=arena_plan,
        arena_address=arena_address,
        arena_bytes_address=arena_bytes_address,
        output_arena=output_plan,
        output_bytes_address=output_bytes_address,
        sequence=sequence_plan,
        written_positions=tuple(written_positions),
        unwritten_outputs=unwritten_outputs,
    )


def _tape_hints(tape, symbols):
    """The tape's own hints for every size, stride and offset symbol, and the opaque
    guards' traced values (the plan's shapes when no example inputs are given)."""
    values = {
        s: sympy.Integer(int(v))
        for s, v in tape.shape_env.backed_var_to_val.items()
        if v.is_integer and s in symbols.by_symbol
    }
    for symbol, rec in symbols.opaque.items():
        if rec["kind"] == "guard":
            values[symbol] = sympy.Integer(int(rec["expected"]))
    return values


def _storage_nbytes(rec):
    """The bytes of an allocation's storage as the ordinary host asks for them
    (computeStorageNbytes): itemsize * (1 + sum((size - 1) * stride))."""
    extent = sympy.Integer(1)
    for size, stride in zip(rec.sizes, rec.strides):
        extent = extent + (_expr(size) - 1) * _expr(stride)
    return sympy.expand(rec.root.itemsize * extent)


def _buffer_uses(tape, calls, memsets, memcpys, host_tables, regions):
    """Per allocation name the seqs of the events touching it, and the names that
    stay the runtime's buffers whatever the plan: the buffers a host table points
    at (a memcpy endpoint is planned like any other allocation, as the arena has
    held them since the followups)."""
    uses = {}
    excluded = OrderedSet()
    for table in host_tables:
        for _, _, source in table.elements:
            if type(source) is PointerSource and type(source.root) is BufferSource:
                excluded.add(source.root.name)

    def use(source, seq):
        for root in storage_roots(source):
            if type(root) is BufferSource:
                uses.setdefault(root.name, []).append(seq)

    for j, call in enumerate(calls):
        for source in (*call.storage_sources, *(field.source for field in call.fields)):
            use(source, int(tape.launches[j]["seq"]))
    for seq, dst, _, _ in memsets:
        use(dst, seq)
    for seq, source, dst, _ in memcpys:
        use(source, seq)
        use(dst, seq)
    for region in regions:
        for source in region.sources:
            use(source, region.seq)
    return uses, excluded


def _rebase_call(call, rebase):
    mapped = {}

    def source(value):
        if type(value) is not ParameterSource:
            return rebase(value)
        pending = [(value, False)]
        while pending:
            node, ready = pending.pop()
            if id(node) in mapped:
                continue
            if not ready:
                pending.append((node, True))
                pending.extend(
                    (arg, False)
                    for arg in reversed(node.args)
                    if type(arg) is ParameterSource and id(arg) not in mapped
                )
                continue
            mapped[id(node)] = dataclasses.replace(
                node,
                value=rebase(node.value) if node.op == "pointer" else node.value,
                args=tuple(
                    mapped[id(arg)] if type(arg) is ParameterSource else rebase(arg)
                    for arg in node.args
                ),
            )
        return mapped[id(value)]

    return dataclasses.replace(
        call,
        fields=tuple(
            dataclasses.replace(field, source=source(field.source))
            for field in call.fields
        ),
        storage_sources=tuple(rebase(value) for value in call.storage_sources),
    )


def _sequence_pass(
    tape,
    symbols,
    lowering,
    allocations,
    calls,
    memsets,
    memcpys,
    host_tables,
    regions,
    region_exprs,
    candidates,
    hints,
    input_index,
    programmatic=(),
):
    """Sequence the `candidates` (allocation names) as roots boxed from `input_index`
    on and rewrite every pointer source over one to its root plus the displacement:
    the launch fields, memset destinations, memcpy endpoints and region operands.
    Buffers a host table points at, allocations outside the candidates and
    allocations whose size is not a function of the size and stride facts stay the
    runtime's. A region operand's key address stays its displacement: a block's base
    is ALIGN-aligned as an allocation's was. `programmatic` are the seqs of the
    regions whose first node is behind a programmatic edge: their free points move
    (plan_sequence)."""
    uses, excluded = _buffer_uses(tape, calls, memsets, memcpys, host_tables, regions)
    by_name = {rec.name: rec for rec in tape.allocs}
    order = {rec.name: k for k, rec in enumerate(tape.allocs)}
    rows = []
    for layout in allocations:
        name = layout.source.name
        if name not in candidates or name in excluded or name not in uses:
            continue
        size = sympy.expand(lowering.subst(_storage_nbytes(by_name[name])))
        if any(
            symbols.by_symbol.get(s) is None
            or symbols.by_symbol[s].kind not in ("size", "stride")
            for s in size.free_symbols
        ):
            continue
        hint = size.xreplace(hints)
        if hint.free_symbols or int(hint) <= 0:
            continue
        rows.append((name, size))
    nodes = ()
    if programmatic:
        # the launch order the rule walks: every event's seq (the recorder's seqs are
        # not dense over the nodes, so it cannot be read off the uses)
        nodes = sorted(
            {int(launch["seq"]) for launch in tape.launches}
            | {seq for seq, _, _, _ in memsets}
            | {seq for seq, _, _, _ in memcpys}
            | {region.seq for region in regions}
        )
    plan = plan_sequence(rows, uses, hints, input_index, order, programmatic, nodes)

    def rebased(source):
        if (
            type(source) is not PointerSource
            or type(source.root) is not BufferSource
            or not plan.covers(source.root.name)
        ):
            return source
        return PointerSource(
            InputSource(plan.root(source.root.name)), source.byte_offset
        )

    calls = tuple(_rebase_call(call, rebased) for call in calls)
    memsets = [(seq, rebased(dst), n, value) for seq, dst, n, value in memsets]
    memcpys = [
        (seq, rebased(source), rebased(dst), n) for seq, source, dst, n in memcpys
    ]
    regions = [
        dataclasses.replace(
            region, sources=tuple(rebased(source) for source in region.sources)
        )
        for region in regions
    ]
    allocations = tuple(
        layout for layout in allocations if not plan.covers(layout.source.name)
    )
    return (
        plan,
        allocations,
        calls,
        memsets,
        memcpys,
        host_tables,
        regions,
        region_exprs,
    )


def _arena_pass(
    tape,
    symbols,
    lowering,
    allocations,
    calls,
    memsets,
    memcpys,
    host_tables,
    regions,
    region_exprs,
    candidates,
    hints,
    arena_index,
    capacity,
    pointer_symbols,
    together=False,
):
    """Plan the `candidates` (allocation names) into the arena input at `arena_index`
    and rewrite every pointer source over a planned allocation to that root plus the
    block's offset: the launch fields, memset destinations, copy endpoints and region
    operands. With `together`, every candidate is live at once (a call's outputs:
    `plan_outputs`); otherwise the uses give the lifetimes. Buffers a host table points
    at, memcpy endpoints, allocations outside the candidates and allocations whose size
    the inputs do not decide stay the runtime's."""
    uses, excluded = _buffer_uses(tape, calls, memsets, memcpys, host_tables, regions)
    by_name = {rec.name: rec for rec in tape.allocs}
    rows = [
        (
            layout.source.name,
            lowering.subst(_storage_nbytes(by_name[layout.source.name])),
        )
        for layout in allocations
        if layout.source.name in candidates
        and layout.source.name not in excluded
        and (together or layout.source.name in uses)
    ]
    if together:
        plan = plan_outputs(rows, hints, arena_index, capacity)
    else:
        plan = plan_arena(rows, uses, hints, arena_index, capacity)
    root = InputSource(arena_index)
    nodes = {}

    def offset_node(name):
        # offset(b) = offset(support) + rounded(support), one add per block; None at 0
        if name in nodes:
            return nodes[name]
        block = plan.blocks[name]
        if block.support is None:
            node = None
        else:
            below = offset_node(block.support)
            rounded = lowering.payload(plan.blocks[block.support].rounded)
            node = (
                rounded
                if below is None
                else lowering.node("add", args=(below, rounded))
            )
        nodes[name] = node
        return node

    def rebased(source):
        if (
            type(source) is not PointerSource
            or type(source.root) is not BufferSource
            or not plan.covers(source.root.name)
        ):
            return source
        node = offset_node(source.root.name)
        displacement = source.byte_offset
        if node is not None:
            displacement = (
                node
                if displacement.op == "constant" and displacement.value == 0
                else lowering.node("add", args=(node, displacement))
            )
        return PointerSource(root, displacement)

    calls = tuple(_rebase_call(call, rebased) for call in calls)
    memsets = [(seq, rebased(dst), n, value) for seq, dst, n, value in memsets]
    memcpys = [
        (seq, rebased(source), rebased(dst), n) for seq, source, dst, n in memcpys
    ]
    new_regions, new_exprs = [], []
    for region, exprs in zip(regions, region_exprs):
        sources, parts, keyed = [], [], []
        for source, (sizes, strides, displacement), (ksizes, kstrides, _) in zip(
            region.sources, region.exprs, exprs
        ):
            if type(source.root) is BufferSource and plan.covers(source.root.name):
                displacement = sympy.expand(
                    plan.offset(source.root.name) + displacement
                )
                address = pointer_symbols[arena_index] + displacement
            else:
                address = (
                    pointer_symbols[source.root.index] + displacement
                    if type(source.root) is InputSource
                    else displacement
                )
            sources.append(rebased(source))
            parts.append((sizes, strides, displacement))
            keyed.append((ksizes, kstrides, address))
        new_regions.append(
            dataclasses.replace(region, sources=tuple(sources), exprs=tuple(parts))
        )
        new_exprs.append(keyed)
    allocations = tuple(
        layout for layout in allocations if not plan.covers(layout.source.name)
    )
    return (
        plan,
        allocations,
        calls,
        memsets,
        memcpys,
        host_tables,
        new_regions,
        new_exprs,
    )


def _dtype_code(dtype):
    return int(torch._C._cuda_scalar_type_code(dtype))


def _int_or_expr(lowering, value):
    e = _expr(value)
    if isinstance(e, sympy.Integer):
        return int(e)
    return lowering.payload(e)


def _int_expr(lowering, value):
    e = _expr(value)
    if isinstance(e, sympy.Integer):
        return lowering.const(e)
    return lowering.payload(e)


def _fact_value(fact, t):
    if fact.kind == "size":
        return t.size(fact.dim) if fact.dim < t.dim() else -1
    if fact.kind == "stride":
        return t.stride(fact.dim) if fact.dim < t.dim() else -1
    if fact.kind == "rank":
        return t.dim()
    if fact.kind == "dtype":
        return _dtype_code(t.dtype)
    if fact.kind == "device":
        return t.device.index if t.is_cuda else -1
    if fact.kind == "pinned":
        return int(not t.is_cuda and t.is_pinned())
    return int(t.is_neg() if fact.kind == "neg" else t.is_conj())


def _predicate_values(lowered, boxed):
    values = []
    values.extend(_const_data_ptr(boxed[index]) for index in lowered.pointer_indices)
    values.extend(boxed[index].storage_offset() for index in lowered.offset_indices)
    values.extend(_fact_value(fact, boxed[fact.index]) for fact in lowered.facts)
    bits = (ctypes.c_uint64 * max(1, len(values)))(
        *(value % (2**64) for value in values)
    )
    return ctypes.cast(bits, ctypes.POINTER(ctypes.c_int64))


def check_predicate(lowered, boxed, regions=True, *, mode=None):
    """The same evaluation the native dispatcher performs, for the preparation
    inputs; without `regions`, the facts and guards alone (no region select, no
    arena term); `mode` names the entry point directly (_MODE_ARENA: the facts and
    the arena terms)."""
    if mode is None:
        mode = _MODE_ALL if regions else _MODE_FACTS
    address = {
        _MODE_ALL: lowered.predicate_address,
        _MODE_FACTS: lowered.facts_address,
        _MODE_ARENA: lowered.arena_address,
    }[mode]
    probe_class = getattr(torch._C, "_HostTracePredicate", None)
    if probe_class is not None:
        # one C++ pass over the box (the Python marshalling below read 5-25 values
        # per tensor through the Python bindings: milliseconds on a model tape)
        if lowered._probe is None:
            lowered._probe = probe_class(
                lowered.pointer_indices,
                lowered.offset_indices,
                [(f.kind, f.index, f.dim) for f in lowered.facts],
            )
        return lowered._probe(boxed if type(boxed) is list else list(boxed), address)
    predicate = ctypes.CFUNCTYPE(
        ctypes.c_int8, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)
    )(address)
    return predicate(_predicate_values(lowered, boxed), None) == 1


def arena_bytes(lowered, boxed):
    """The arena bytes the call at `boxed` needs, by the lowering's compiled
    requirement function; -1 when its arithmetic overflowed."""
    reader = ctypes.CFUNCTYPE(
        ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)
    )(lowered.arena_bytes_address)
    return reader(_predicate_values(lowered, boxed), None)


def output_block_index(lowered):
    """The boxed position of the output block: after the tape's tensors and the arena
    (the sequence roots come after it)."""
    return len(lowered.tape.inputs) + (lowered.arena is not None)


def _served_block(ring, outputs):
    """The ring block whose storage the call's output views share, as the boxed uint8
    tensor over it (a block allocated for the call alone is not kept by the ring: a
    view over its whole storage stands in)."""
    kept = {block[1]: block[0] for block in ring.blocks}
    storages = [
        o.untyped_storage()
        for o in outputs
        if isinstance(o, torch.Tensor) and o.device == ring.device
    ]
    for storage in storages:
        if storage._cdata in kept:
            return kept[storage._cdata]
    if not storages:
        return ring.blocks[0][0]
    storage = max(storages, key=lambda s: s.nbytes())
    return torch.empty(0, dtype=torch.uint8, device=ring.device).set_(storage)


def output_bytes(lowered, boxed):
    """The output block bytes the call at `boxed` needs, by the lowering's compiled
    requirement function; -1 when its arithmetic overflowed."""
    reader = ctypes.CFUNCTYPE(
        ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)
    )(lowered.output_bytes_address)
    return reader(_predicate_values(lowered, boxed), None)


def _shared_guard(lowered, boxed):
    """Row 2b: the runtime team's compiled guard over the same tape guards and this
    lowering's obligations (translated to the mapping's symbols), the object their
    consumers read as `lowered.guard` (`prepare_host_trace`, their re-export entry).
    It is compiled beside the dispatch predicate (ours carries the region selects and
    the checked arithmetic) and accepts the preparation inputs, as ours did (their
    compiler checks that itself); a tape their printer cannot express is recorded on
    `guard_declined`, not a decline. The standalone `HostTraceReplay` does not compile
    it: nothing on its path reads it, and it costs a second predicate compile per
    variant (1.9 s on the decode chain, 9.9 s on the GPT-2 decoder, 78 s on the GPT-2
    training step)."""
    from torch._inductor.runtime._cudagraph.host_trace_guards import (
        compile_host_trace_guard,
    )

    mapping = lowered.symbols.mapping
    if mapping is None:
        return None
    try:
        extra = tuple(
            mapping.translate(g, preserve_operations=True) for g in lowered.extra_guards
        )
        return compile_host_trace_guard(lowered.tape, mapping, boxed, extra)
    except UnsupportedCapture as e:
        lowered.guard_declined = str(e)
        return None


def _region_templates(regions, values, base_address, device, identity):
    """Per closed region its registry key at `values` (the tape's symbols at the
    inputs) and the template the registry serves for it (harvested once per process
    into the process-wide template cache). Operand shapes and
    displacements are the tape's expressions at `values`; an input operand's
    alignment class is that of its real address, `base_address(index)` plus the
    displacement (a planned root's block is ALIGN-aligned, so the class is the
    displacement's), an allocation's that of its displacement."""
    from torch.cuda import _host_trace

    def concrete(e):
        e = sympy.expand(_expr(e).xreplace(values))
        if not isinstance(e, sympy.Integer):
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: a closed region's operand shape {e} is not decided by the inputs"
            )
        return int(e)

    settings = torch._C._cuda_kernel_template_library_settings()
    plans = []
    for region in regions:
        sizes = [tuple(concrete(v) for v in sz) for sz, _, _ in region.exprs]
        strides = [tuple(concrete(v) for v in st) for _, st, _ in region.exprs]
        classes = []
        for source, (_, _, displacement) in zip(region.sources, region.exprs):
            address = concrete(displacement)
            if type(source.root) is InputSource:
                address += base_address(source.root.index)
            classes.append(_align_class(address))
        key = _region_key(region, sizes, strides, classes) + settings
        try:
            plans.append((key, _region_template(region, key, device, identity)))
        except _host_trace.Miss as e:
            raise HostTraceLoweringDeclined(f"host_trace lowering: {e}") from None
    return plans


def _register_preparation_regions(lowered, boxed, device, identity):
    """Before the predicate runs at preparation: for every closed region, the
    template of the preparation shape (harvested once per process into the
    process-wide template cache); the entry's scratch arena sized to them; then
    each registered for its site, whose chain is that template's node chain (E28: a
    key whose template has another chain is another class of the tape, built as a
    variant of its own). Operand shapes and displacements are the tape's expressions
    at the boxed inputs; an input operand's alignment class is that of its real
    address, an allocation's that of its displacement (the base is 256-byte
    aligned). Returns the chain per region (whose nodes the preparation launches)
    and the arena (None without regions)."""
    from torch.cuda import _host_trace

    if not lowered.regions:
        return [], None
    plans = _region_templates(
        lowered.regions,
        _symbol_values(lowered, boxed),
        lambda index: _const_data_ptr(boxed[index]),
        device,
        identity,
    )
    arena = _RegionArena.for_templates(device, [t for _, t in plans])
    chains = []
    for region, (key, template) in zip(lowered.regions, plans):
        chain = _region_chain(template)
        # the arena was sized to this template: the registration cannot refuse it
        _register_region_variant(
            region, key, template, _host_trace._chain_class(template.nodes), arena
        )
        chains.append(chain)
    return chains, arena


def _prepare_region(region, chain, offset_indices, address_of, arena, stream):
    """Launch a closed region's chain into the preparation capture: its kernel nodes
    with the operand slots at the preparation addresses and the scratch slots at the
    arena, its memset nodes at their destinations there. Returns the node handles in
    launch order and one capture event per node (kind "template")."""
    from cuda.bindings import runtime

    addresses = [
        address_of(source, index)
        for source, index in zip(region.sources, offset_indices)
    ]
    nodes, events = [], []
    for kind, _template, n in chain:
        before = torch._C._cuda_get_capture_frontier(stream)
        if kind == "memset":
            dst = arena.slot(n["dst_role"], n["dst"], addresses)
            _check_cuda_bindings(
                runtime.cudaMemsetAsync(dst, n["value"], n["bytes"], stream)
            )
        else:
            image = bytearray(n["image"])
            for offset, operand, delta in n["slots"]:
                image[offset : offset + 8] = struct.pack(
                    "<Q", addresses[operand] + delta
                )
            for offset, index, delta in n["scratch_slots"]:
                image[offset : offset + 8] = struct.pack(
                    "<Q", arena.addresses[index] + delta
                )
            for offset in n["ws_slots"]:
                image[offset : offset + 8] = struct.pack("<Q", arena.workspace_address)
            torch._C._cuda_launch_kernel_image(
                n["func"],
                tuple(n["grid"]),
                tuple(n["block"]),
                n["smem"],
                stream,
                bytes(image),
                tuple(n["attrs"]),
                programmatic=n["programmatic"],
            )
        after = torch._C._cuda_get_capture_frontier(stream)
        nodes.append(_one_new_node(before, after, "a region's node"))
        events.append(RecordedGraphNode(stream, before, after, "template"))
    return tuple(nodes), events


def _prepare_host_tables(tables, numeric, staging_depth=2):
    """Pinned ring slots per host table (this preparation owns them; retained with the
    entry), constants written once into every slot, and the elements' values as early
    plan values. Shared with the runtime team's mixed replay (their replay.py)."""
    table_slots, table_values = [], []
    for table in tables:
        slots = tuple(
            torch.empty(table.nbytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(staging_depth)
        )
        for slot in slots:
            for offset, data in table.constants:
                slot[offset : offset + len(data)].copy_(
                    torch.frombuffer(bytearray(data), dtype=torch.uint8)
                )
        rows = []
        for offset, width, source in table.elements:
            if type(source) is PointerSource:
                rows.append((offset, width, source, numeric.add(source.byte_offset)))
            else:
                rows.append((offset, width, source, numeric.add(source)))
        table_slots.append(slots)
        table_values.append(rows)
    return table_slots, table_values


def _render_host_table(slot, values, numeric, address_of):
    """Render one table's elements at the plan's current values into a pinned slot."""
    for offset, width, source, index in values:
        value = (
            address_of(source, index)
            if type(source) is PointerSource
            else numeric.values[index]
        )
        data = (value & ((1 << (8 * width)) - 1)).to_bytes(width, "little")
        slot[offset : offset + width].copy_(
            torch.frombuffer(bytearray(data), dtype=torch.uint8)
        )


def _one_new_node(before, after, what):
    # the event produced exactly one new capture node on an active capture
    if (
        after[:3] != before[:3]
        or len(after[3]) != 1
        or any(edge != bytes(8) for _, edge in (*before[3], *after[3]))
    ):
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: {what} did not produce one capture frontier node"
        )
    node = after[3][0][0]
    if not node or any(node == dependency for dependency, _ in before[3]):
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: {what} did not produce a new capture node"
        )
    return node


# a preparation's capture runs one at a time (the recorder's allocation log is process-wide)
_prepare_lock = threading.Lock()


@contextlib.contextmanager
def _capture(graph, stream):
    """capture_begin / capture_end directly on `stream`, relaxed, under the recorder's
    gc hold: torch.cuda.graph's prologue synchronizes the whole device and empties
    both caches, which would invalidate a trace in progress on another thread, and
    its host-cache flush was the two-entry crash's trigger.
    Relaxed: a thread-local capture is invalidated by another thread's allocations
    and stream syncs on this driver; the tape, not the capture, decides what is
    replayed."""
    from torch.cuda._host_trace import _gc_hold

    with _prepare_lock, _gc_hold, torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="relaxed")
        try:
            yield
        finally:
            # CUDAGraph ends a capture only on the stream that began it
            with torch.cuda.stream(stream):
                graph.capture_end()


def _capture_byte_memset(address, byte_count, value, stream):
    """Issue a byte memset into the preparation capture and verify the node's
    parameters against what was asked. Returns (node, capture event)."""
    from cuda.bindings import runtime

    if byte_count <= 0:
        raise HostTraceLoweringDeclined(
            "host_trace lowering: a captured byte memset requires a positive byte count"
        )
    before = torch._C._cuda_get_capture_frontier(stream)
    if before[0] != 1 or not before[1] or not before[2]:
        raise HostTraceLoweringDeclined(
            "host_trace lowering: a memset requires an active CUDA graph capture"
        )
    _check_cuda_bindings(runtime.cudaMemsetAsync(address, value, byte_count, stream))
    after = torch._C._cuda_get_capture_frontier(stream)
    node = _one_new_node(before, after, "a memset")
    params = _check_cuda_bindings(runtime.cudaGraphMemsetNodeGetParams(node))
    if (
        int(params.dst) != address
        or params.width != byte_count
        or params.height != 1
        or params.elementSize != 1
        or (params.value & 255) != (value & 255)
    ):
        raise HostTraceLoweringDeclined(
            "host_trace lowering: the captured byte memset differs from its traced arguments"
        )
    return node, RecordedGraphNode(stream, before, after, "memset")


_MEMCPY_KINDS = {"h2d": "cudaMemcpyHostToDevice", "d2d": "cudaMemcpyDeviceToDevice"}


def _capture_memcpy(destination, source, byte_count, stream, kind):
    """Issue a one-dimensional copy of the record's declared kind ("h2d" from a pinned
    host address, "d2d" between device addresses) into the preparation capture and
    verify the node's parameters against what was asked. Returns (node, capture
    event)."""
    from cuda.bindings import runtime

    if kind not in _MEMCPY_KINDS:
        raise HostTraceLoweringDeclined(
            f"host_trace lowering: a copy of kind {kind!r} is not a host-to-device or device-to-device copy"
        )
    memcpy_kind = getattr(runtime.cudaMemcpyKind, _MEMCPY_KINDS[kind])
    if byte_count <= 0:
        raise HostTraceLoweringDeclined(
            "host_trace lowering: a captured copy requires a positive byte count"
        )
    before = torch._C._cuda_get_capture_frontier(stream)
    if before[0] != 1 or not before[1] or not before[2]:
        raise HostTraceLoweringDeclined(
            "host_trace lowering: a copy requires an active CUDA graph capture"
        )
    _check_cuda_bindings(
        runtime.cudaMemcpyAsync(destination, source, byte_count, memcpy_kind, stream)
    )
    after = torch._C._cuda_get_capture_frontier(stream)
    node = _one_new_node(before, after, "a copy")
    params = _check_cuda_bindings(runtime.cudaGraphMemcpyNodeGetParams(node))
    if (
        int(params.srcPtr.ptr) != source
        or int(params.dstPtr.ptr) != destination
        or params.extent.width != byte_count
        or params.extent.height != 1
        or params.extent.depth != 1
        or params.kind != memcpy_kind
        or int(params.srcArray) != 0
        or int(params.dstArray) != 0
        or (params.srcPos.x, params.srcPos.y, params.srcPos.z) != (0, 0, 0)
        or (params.dstPos.x, params.dstPos.y, params.dstPos.z) != (0, 0, 0)
    ):
        raise HostTraceLoweringDeclined(
            "host_trace lowering: the captured copy differs from its traced arguments"
        )
    return node, RecordedGraphNode(stream, before, after, "memcpy")


def _capture_h2d_copy(destination, source, byte_count, stream):
    """The host-to-device copy of `_capture_memcpy`, under the name the runtime team's
    mixed replay imports."""
    return _capture_memcpy(destination, source, byte_count, stream, "h2d")


def _programmatic_nodes(graph, node_count):
    """The indices, in cudaGraphGetNodes order (the issue order of a one-stream
    capture), of the graph's nodes whose incoming edge is programmatic, read from the
    edge data."""
    from cuda.bindings import driver

    from torch.cuda._utils import _check_cuda_bindings_driver as check

    count = check(driver.cuGraphGetNodes(graph, 0))[-1]
    if count != node_count:
        raise AssertionError(
            f"the prepared graph has {count} nodes, the preparation issued {node_count}"
        )
    nodes = check(driver.cuGraphGetNodes(graph, count))[0]
    index = {int(node): i for i, node in enumerate(nodes)}
    count = check(driver.cuGraphGetEdges(graph, 0))[-1]
    _, to, data, _ = check(driver.cuGraphGetEdges(graph, count))
    kind = int(driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC)
    return tuple(sorted(index[int(t)] for t, d in zip(to, data) if int(d.type) == kind))


def prepare_hosttrace(
    lowered, example_args, *, staging_depth=2, arena=None, outputs=None, sequence=None
):
    """Capture the tape's launches at the example inputs and hand the graph to the
    native replay. Every launch comes from the tape: kernel handle, layout, grid, block,
    shared bytes and parameter bytes; the host itself does not run here. Host tables
    get `staging_depth` pinned slots each; the native replay renders the next slot per
    call once the launch that read it has completed. `arena` is the family's _Arena
    when the lowering planned one (its tensor is boxed after the tape's tensors);
    `outputs` the family's OutputRing when it planned an output arena (a free block of
    it, sized by the caller, is the last boxed input); `sequence` the family's
    AllocatorSequence when it lowered an allocation sequence (its roots, bound by the
    caller at these inputs, are boxed last)."""
    from cuda.bindings import runtime

    from torch._inductor.runtime._cudagraph.replay import _capture_outputs
    from torch.cuda import _host_trace

    if staging_depth < 1:
        raise ValueError("staging_depth must be at least 1")

    tape = lowered.tape
    if not lowered.contract_holds(example_args):
        raise _host_trace.Miss(
            "Host trace preparation arguments differ from the trace contract"
        )
    boxed = lowered.box(
        example_args,
        None if arena is None else arena.tensor,
        None if outputs is None else outputs.take(),
        None if sequence is None else sequence.roots,
    )
    # the positions the tape writes are materialized (a copy-on-write example there,
    # as a served call's); every input address below is read through the const
    # accessor, so an example in a read position stays lazy
    if lowered.written_positions:
        torch._C._host_trace_materialize(boxed, lowered.written_positions)
    device = lowered.device
    identity = _host_trace._device_identity(device)
    for (name, traced), (_, here) in zip(tape.device_identity, identity):
        if traced != here:
            raise HostTraceLoweringDeclined(
                f"host_trace lowering: the tape was traced on a device with {name} {traced}; this device has {here}"
            )
    # the predicate selects every region's variant: the preparation shape's must be
    # registered before it runs
    chains, arena = _register_preparation_regions(lowered, boxed, device, identity)
    lowered.region_arena = arena
    if not check_predicate(lowered, boxed):
        raise HostTraceLoweringDeclined(
            "host_trace lowering: the tape's guards reject the preparation inputs"
        )
    numeric = _NumericProgram(lowered.records, boxed)
    output_slots = bind_output_slots(
        lowered.outputs,
        lowered.allocations,
        lowered.records.input_names,
        OrderedSet(),
        symbolic=True,
    )
    if output_slots is None:
        raise HostTraceLoweringDeclined(
            "host_trace lowering: outputs lost their traced sources"
        )
    for output in lowered.outputs:
        if type(output) is TensorViewOutput:
            # an output view's extents are early values (the shared materializer
            # reads them as prepared values)
            for v in (*output.size, *output.stride, output.offset):
                if type(v) is IntExpr:
                    numeric.add(v)
    dimensions = {}
    for layout in lowered.allocations:
        size, stride = (
            tuple(
                numeric.values[numeric.add(v)] if type(v) is IntExpr else v
                for v in values
            )
            for values in (layout.size, layout.stride)
        )
        dimensions[layout.source] = (size, stride)
    memset_values = [
        (numeric.add(dst.byte_offset), numeric.add(byte_count))
        for _, dst, byte_count, _ in lowered.memsets
    ]
    # host tables: the elements' values are early plan values; the slots are pinned
    # ring buffers this preparation owns, the example values rendered into slot 0
    table_slots, table_values = _prepare_host_tables(
        lowered.host_tables, numeric, staging_depth
    )
    memcpy_values = [
        (
            None if type(source) is int else numeric.add(source.byte_offset),
            numeric.add(dst.byte_offset),
            numeric.add(byte_count),
        )
        for _, source, dst, byte_count in lowered.memcpys
    ]
    if lowered.rng is not None:
        numeric.add(lowered.rng)  # the increment is an early value
    # closed regions: the variant selection and the operand offsets are early values
    region_values = []
    for region in lowered.regions:
        numeric.add(region.variant)
        region_values.append(
            tuple(numeric.add(source.byte_offset) for source in region.sources)
        )
    grids = []
    parameters = None
    buffer_indices = {
        layout.source: len(boxed) + index
        for index, layout in enumerate(lowered.allocations)
    }
    for call in lowered.calls:
        grid = tuple(numeric.values[numeric.add(axis)] for axis in call.grid)
        if not (0 < grid[0] < 2**31 and 0 < grid[1] <= 65535 and 0 < grid[2] <= 65535):
            raise HostTraceLoweringDeclined(
                "host_trace lowering: grid exceeds CUDA launch bounds"
            )
        shared = (
            None if call.shared is None else numeric.values[numeric.add(call.shared)]
        )
        block = (
            None
            if call.block is None
            else tuple(numeric.values[numeric.add(axis)] for axis in call.block)
        )
        grids.append((grid, shared, block))
        for field in call.fields:
            source = field.source
            if type(source) is PointerSource:
                numeric.add(source.byte_offset)
            elif type(source) is ExpressionSource:
                numeric.add(source.expression)
            elif type(source) is ParameterSource:
                if parameters is None:
                    parameters = _ParameterProgram(numeric, len(boxed), buffer_indices)
                parameters.add(source)
    state = _Preparation(lowered, (), tuple(boxed), list(boxed))
    try:
        with torch.cuda.device(device):
            state.stream = torch.cuda.current_stream()
            state.capture_stream = torch.cuda.Stream()
            state.graph = torch.cuda.CUDAGraph(keep_graph=True)
            with _capture(state.graph, state.capture_stream):
                for layout in lowered.allocations:
                    size, stride = dimensions[layout.source]
                    state.buffers[layout.source] = (
                        torch._C._dynamo.guards._empty_strided_cuda(
                            size, stride, layout.dtype
                        )
                    )
                parameter_values = (
                    ()
                    if parameters is None
                    else parameters.evaluate(boxed, state.buffers)
                )
                stream = state.capture_stream.cuda_stream
                philox = None
                rng_slot_fields = {}
                for r in getattr(tape, "rng_slots", None) or ():
                    call = lowered.calls[r["launch"]]
                    parameter, inner = _parameter_of(call, r["offset"], r["size"])
                    symbolic = any(
                        f.parameter == parameter and f.byte_offset == inner
                        for f in call.fields
                    )
                    if symbolic:
                        rng_slot_fields[(r["launch"], parameter, inner)] = True
                    else:
                        for cp, co, data in call.constants:
                            if cp == parameter and co <= inner < co + len(data):
                                rng_slot_fields[(r["launch"], parameter, inner)] = (
                                    int.from_bytes(
                                        data[inner - co : inner - co + r["size"]],
                                        "little",
                                    )
                                )
                                break
                if lowered.rng is not None:
                    # registers the default generator with this capture (as the host's
                    # own philox request would) and returns the per-capture pointers
                    generator = torch.cuda.default_generators[device]
                    philox = torch._C._host_trace_generator_capture_pointers(generator)
                calls = list(lowered.calls)  # prepared calls: the rng pointers added

                def address_of(source, offset_index):
                    root = source.root
                    base = (
                        boxed[root.index]
                        if type(root) is InputSource
                        else state.buffers[root]
                    )
                    return _const_data_ptr(base) + numeric.values[offset_index]

                for k in range(len(lowered.host_tables)):
                    _render_host_table(
                        table_slots[k][0], table_values[k], numeric, address_of
                    )
                # host order: launches, memsets and copies interleave by sequence number
                events = sorted(
                    [(int(L["seq"]), "launch", j) for j, L in enumerate(tape.launches)]
                    + [
                        (seq, "memset", j)
                        for j, (seq, _, _, _) in enumerate(lowered.memsets)
                    ]
                    + [
                        (seq, "memcpy", j)
                        for j, (seq, _, _, _) in enumerate(lowered.memcpys)
                    ]
                    + [
                        (region.seq, "region", j)
                        for j, region in enumerate(lowered.regions)
                    ]
                )
                recorded_memsets, recorded_memcpys, recorded_templates = [], [], []
                # every node of the capture in issue order (kernel launches and the
                # nodes issued here), as the runtime's association reads them
                capture_events, node_seqs, region_firsts = [], [], []
                lowered.region_nodes = {}
                for seq, kind, j in events:
                    if kind == "region":
                        region = lowered.regions[j]
                        nodes, region_events = _prepare_region(
                            region,
                            chains[j],
                            region_values[j],
                            address_of,
                            arena,
                            stream,
                        )
                        capture_events.extend(region_events)
                        region_firsts.append((len(node_seqs), seq))
                        node_seqs.extend([seq] * len(region_events))
                        lowered.region_nodes[region.site] = tuple(
                            (kind, n["programmatic"]) for kind, _, n in chains[j]
                        )
                        recorded_templates.append(
                            (
                                nodes,
                                region.site,
                                region.variant,
                                region.sources,
                                arena.workspace_address,
                            )
                        )
                        continue
                    if kind == "memcpy":
                        _, source, dst, byte_count = lowered.memcpys[j]
                        src_index, dst_index, bytes_index = memcpy_values[j]
                        src_address = (
                            table_slots[source][0].data_ptr()
                            if type(source) is int
                            else address_of(source, src_index)
                        )
                        node, event = _capture_memcpy(
                            address_of(dst, dst_index),
                            src_address,
                            numeric.values[bytes_index],
                            stream,
                            lowered.memcpy_kinds[j],
                        )
                        recorded_memcpys.append((node, source, dst, byte_count))
                        capture_events.append(event)
                        node_seqs.append(seq)
                        continue
                    if kind == "memset":
                        _, dst, byte_count, value = lowered.memsets[j]
                        offset_index, bytes_index = memset_values[j]
                        root = dst.root
                        base = (
                            boxed[root.index]
                            if type(root) is InputSource
                            else state.buffers[root]
                        )
                        address = _const_data_ptr(base) + numeric.values[offset_index]
                        node, event = _capture_byte_memset(
                            address, numeric.values[bytes_index], value, stream
                        )
                        recorded_memsets.append((node, dst, byte_count, value))
                        capture_events.append(event)
                        node_seqs.append(seq)
                        continue
                    call, (grid, shared, block) = calls[j], grids[j]
                    images = [bytearray(size) for size in call.module.parameter_sizes]
                    for parameter, offset, data in call.constants:
                        images[parameter][offset : offset + len(data)] = data
                    for call_index, parameter, offset in lowered.rng_fields:
                        if call_index != j:
                            continue
                        seed, offset_ptr, intragraph = philox
                        if intragraph != 0 and any(
                            slot_call == j and slot is not True
                            for (slot_call, _, _), slot in rng_slot_fields.items()
                        ):
                            # a constant slot is the tape's prefix of the earlier
                            # launches' constant increments from the trace capture's
                            # offset 0 (the first random launch's is 0); it stands
                            # only if the preparation capture starts at 0 as well;
                            # symbolic slots are fields written below
                            raise HostTraceLoweringDeclined(
                                f"host_trace lowering: the preparation capture's intragraph offset is {intragraph}, the tape's constant philox slots count from 0"
                            )
                        # this capture's pointers: constants of the prepared call, so the
                        # runtime's exact-bytes check sees what was launched
                        philox_bytes = seed.to_bytes(8, "little") + offset_ptr.to_bytes(
                            8, "little"
                        )
                        images[parameter][offset : offset + len(philox_bytes)] = (
                            philox_bytes
                        )
                        call = dataclasses.replace(
                            call,
                            constants=(
                                *call.constants,
                                (parameter, offset, philox_bytes),
                            ),
                        )
                    calls[j] = call
                    for field in call.fields:
                        source = field.source
                        if type(source) is PointerSource:
                            root = source.root
                            base = (
                                boxed[root.index]
                                if type(root) is InputSource
                                else state.buffers[root]
                            )
                            value = _const_data_ptr(base) + numeric.prepared_value(
                                source.byte_offset
                            )
                            payload = struct.pack("P", value)
                        else:
                            if type(source) is ParameterSource:
                                value = parameter_values[
                                    parameters.prepared_index(source)
                                ]
                            elif type(source) is IntegerSource:
                                value = source.value
                            else:
                                value = numeric.prepared_value(source.expression)
                            payload = struct.pack(
                                "i" if field.kind == "i32" else "q", value
                            )  # float fields travel as their bit patterns
                        images[field.parameter][
                            field.byte_offset : field.byte_offset + len(payload)
                        ] = payload
                    images = tuple(bytes(image) for image in images)
                    before = torch._C._cuda_get_capture_frontier(stream)
                    call.module.launch(images, grid, stream, shared=shared, block=block)
                    after = torch._C._cuda_get_capture_frontier(stream)
                    state.launches.append(
                        RecordedKernelLaunch(
                            stream, before, after, call.module.function, images
                        )
                    )
                    capture_events.append(state.launches[-1])
                    node_seqs.append(seq)
                # the runtime team's one output materializer (their replay.py, row 10)
                state.outputs = _capture_outputs(
                    lowered.outputs, boxed, state.buffers, numeric
                )
            state.graph.instantiate()
            state.instantiated = True
            _check_cuda_bindings(
                runtime.cudaGraphUpload(
                    state.graph.raw_cuda_graph_exec(), state.stream.cuda_stream
                )
            )
            state.stream.synchronize()
            state.capture_stream.synchronize()
            lowered.capture_handles = (
                state.graph.raw_cuda_graph(),
                state.graph.raw_cuda_graph_exec(),
            )
            lowered.node_seqs = tuple(node_seqs)
            lowered.programmatic_nodes = _programmatic_nodes(
                state.graph.raw_cuda_graph(), len(node_seqs)
            )
            if lowered.sequence is not None:
                # the free points were planned from the templates' launch flags: the
                # driver's edges must not name a region the plan did not (E23)
                flagged = set(lowered.programmatic_nodes)
                unplanned = [
                    seq
                    for first, seq in region_firsts
                    if first in flagged and seq not in lowered.sequence.programmatic
                ]
                if unplanned:
                    raise AssertionError(
                        f"host_trace preparation: the regions at seqs {unplanned} are behind programmatic edges the allocation sequence did not plan for"
                    )
            state.entry = _make_replay(
                state.graph,
                lowered.input_count,
                lowered.allocations,
                output_slots,
                (),
                tuple(calls),
                tuple(state.launches),
                state.buffers,
                state.stream,
                numeric=numeric,
                resources=(lowered, tape, arena, *table_slots),
                capture_inputs=boxed,
                # the boxed positions of the pinned inputs a copy node reads: the owner
                # holds those tensors (the capture's, then each served call's); a CPU
                # input no copy reads is not a used input to the runtime
                pinned_positions=tuple(
                    sorted(
                        source.root.index
                        for _, source, _, _ in lowered.memcpys
                        if type(source) is PointerSource
                        and type(source.root) is InputSource
                        and source.root.index < len(tape.inputs)
                        and tape.inputs[source.root.index].device.type == "cpu"
                    )
                ),
                # the positions the tape never writes are read through the const
                # accessor per call (a copy-on-write tensor there stays lazy, as under
                # eager); the written ones are materialized before the dispatch
                const_positions=tuple(
                    i for i in range(len(boxed)) if i not in lowered.written_positions
                ),
                memsets=tuple(recorded_memsets),
                host_tables=tuple(
                    (
                        tuple(slot.data_ptr() for slot in table_slots[k]),
                        table.nbytes,
                        tuple(
                            (offset, width, source)
                            for offset, width, source, _ in table_values[k]
                        ),
                    )
                    for k, table in enumerate(lowered.host_tables)
                ),
                memcpys=tuple(recorded_memcpys),
                capture_events=tuple(capture_events),
                rng=None
                if lowered.rng is None
                else (torch.cuda.default_generators[device], lowered.rng),
                templates=tuple(recorded_templates),
            )
            if lowered.rng is None:
                state.entry._retire_capture_pool()
            # else: their retirement refuses graphs with captured generator states (O7);
            # the capture pool stays with the entry
            return state.entry
    except BaseException:
        state.abort()
        raise


_current_raw_stream = torch._C._cuda_getCurrentRawStream
_record_host_event = torch._C._host_trace_record_host_event
_storage_address = torch._C._host_trace_storage_address

# pinned buffers the served calls' copies read: storage address -> {entry id: the
# event of that entry's last call reading it} (an entry's calls are ordered on its
# bound stream, so its latest event covers its earlier ones)
_h2d_pending: dict[int, dict[int, torch.cuda.Event]] = {}
_h2d_lock = threading.Lock()


def _h2d_note(addresses, owner, event):
    with _h2d_lock:
        if len(_h2d_pending) > 1024:
            # a caller allocating fresh pinned buffers per call: forget the buffers
            # whose copies are done
            for address in [
                a
                for a, pending in _h2d_pending.items()
                if all(e.query() for e in pending.values())
            ]:
                del _h2d_pending[address]
        for address in addresses:
            _h2d_pending.setdefault(address, {})[id(owner)] = event


def _h2d_wait(address):
    with _h2d_lock:
        pending = _h2d_pending.pop(address, {})
    for event in pending.values():
        event.synchronize()


def wait_for_h2d(pinned):
    """Wait until every entry's served copies from this pinned buffer have read it: a
    caller rewriting a buffer that several entries (or variants) read calls this
    first. An entry's own `wait_for_h2d()` covers its last call, and every entry's
    pending copies from that call's pinned buffers."""
    _h2d_wait(_storage_address(pinned))


_MISSED = object()  # the dispatch's answer when no variant of a family served the call


class _Arena:
    """One family's arena: the uint8 tensor boxed after the tape's tensors, whose
    bytes every variant's temporaries share by the variant's plan (calls of a family
    run one at a time on the entry's stream, and a temporary dies within its call).
    Grown, never shrunk, when a call's requirement exceeds the capacity: a new
    tensor (a rebind of the arena root at the next call), the old one released
    when no box holds it. The base is 512-byte aligned (the plan's granularity)."""

    def __init__(self, device):
        self.device = device
        self.tensor: Any = None  # a uint8 Tensor from the first ensure() on
        self.capacity = 0
        self.grows = 0

    def ensure(self, nbytes):
        """Hold at least `nbytes`; True when a new tensor was taken."""
        if self.tensor is not None and nbytes <= self.capacity:
            return False
        # a small arena doubles (a decode step's temporaries grow with the cache a
        # little every call: 8 growths over 32 TinyLlama steps at the plain rounding),
        # a large one grows to what the call needs
        if self.capacity < 16 << 20:
            nbytes = max(nbytes, 2 * self.capacity)
        capacity = round_capacity(nbytes)
        with torch.cuda.device(self.device):
            raw = torch.empty(capacity + 512, dtype=torch.uint8, device=self.device)
        pad = -raw.data_ptr() % 512
        if self.tensor is not None:
            self.grows += 1
        self.tensor = raw[pad : pad + capacity]
        self.capacity = capacity
        return True


@dataclass
class _Family:
    """The variants of one argument contract (arity, tensor positions, non-tensor
    constants, pinned input positions) share one dispatch: their boxes have the
    same layout. The first variant's
    `lowered` fixes the box, the pinned positions and the fresh outputs for the family,
    since the dispatch does not report which variant served a call. What a served call
    reads is hoisted from `lowered` into the family's own fields."""

    lowered: LoweredTape
    dispatch: Any  # torch._C._CUDAGraphBoxedDispatch, made right after the family
    variants: list
    # the argument contract: the arity, the constants check (None when every position
    # is a tensor) and the getter of the tensors in the box's order
    nargs: int = dataclasses.field(init=False)
    constants_hold: Callable[[tuple], bool] | None = dataclasses.field(init=False)
    tensors: Callable[[tuple], tuple] = dataclasses.field(init=False)
    # positions of the pinned CPU inputs a served call holds; the variants' device
    pinned: tuple = dataclasses.field(init=False)
    device: int = dataclasses.field(init=False)
    # boxed positions any variant of the family writes: a copy-on-write tensor there
    # is materialized before the dispatch reads its address (the union over the
    # family's tapes; the dispatch does not report which variant serves a call)
    written: tuple = dataclasses.field(init=False)
    # the raw stream the family's dispatch is bound to: the current stream at the
    # first variant's preparation (the runtime refuses a hit on another stream by
    # name; a miss there is refused before a trace, `_missed`)
    stream: int = dataclasses.field(init=False)
    # the family's arena (None when its variants were lowered without one), boxed
    # after the tape's tensors, and its output ring (None likewise), a free block of
    # which is boxed after it
    arena: _Arena | None = None
    outputs: OutputRing | None = None
    # the family's allocation sequence (None when its variants were lowered without
    # one): its roots are boxed last, bound per call in its mode
    sequence: AllocatorSequence | None = None
    # the non-tensor arguments by position, as the family's first call passed them:
    # what a box stands for at those positions (`arguments`)
    constants: dict = dataclasses.field(default_factory=dict)
    # the box in one C++ pass over the call's tuple: the tensors at the tape's
    # positions, the written positions materialized, the arena appended; the ring's
    # block and the sequence roots follow it
    boxer: Callable = dataclasses.field(init=False)

    def __post_init__(self):
        lowered = self.lowered
        self.nargs = lowered.nargs
        self.constants_hold = (
            lowered.constants_hold if lowered.constant_positions else None
        )
        self.tensors = lowered.tensors
        self.pinned = lowered.pinned_positions
        self.device = lowered.device
        self.written = lowered.written_positions
        self.stream = _current_raw_stream(self.device)
        positions = lowered.symbols.positions
        every = positions == list(range(lowered.nargs))
        self.boxer = torch._C._HostTraceBoxer(
            None if every else positions, self.written
        )

    def box(self, args):
        if type(args) is not tuple:
            args = tuple(args)
        box = self.boxer(args, None if self.arena is None else self.arena.tensor)
        if self.outputs is not None:
            box.append(self.outputs.take())
        if self.sequence is not None:
            box.extend(self.sequence.take(box))
        return box

    def arguments(self, box):
        """The argument list a caller's box stands for: its tensors at the tape's
        positions, the family's constants elsewhere (a box carrying the arena too
        stands for the same call)."""
        tensors = iter(box)
        constants = self.constants
        return tuple(
            constants[i] if i in constants else next(tensors) for i in range(self.nargs)
        )


@dataclass
class _Variant:
    """One prepared tape of an entry, and the policy object commit 1's `Entry` keeps
    per variant: a call of this variant's argument contract runs its family's
    dispatch, where the C++ predicate tries every variant of the family at once.
    The entry's loop reaches a variant only on a miss of the first family
    (`HostTraceReplay.__call__` serves the hit path from the family directly)."""

    tape: object
    lowered: LoweredTape
    entry: object
    owner: "HostTraceReplay"
    family: _Family

    @property
    def program(self):
        return self.lowered  # the runtime team's name for it

    def matches(self, args):
        family = self.family
        if len(args) != family.nargs:
            return False
        if family.constants_hold is not None and not family.constants_hold(args):
            return False
        return all(map(isinstance, family.tensors(args), repeat(torch.Tensor)))

    def __call__(self, args):
        family = self.family
        box = family.box(args)
        owner = self.owner
        outputs = owner._dispatch(family, box)
        if outputs is _MISSED:
            outputs = owner._missed(family, args)
        else:
            owner._missed_call = False
            if owner.arena_check and (
                family.arena is not None or family.sequence is not None
            ):
                owner._check_arena(family, args, outputs)
        if family.pinned:
            owner._hold_pinned(args, family)
        return outputs


def _why_missed(lowered, boxed):
    """The first fact or recorded branch of a variant the boxed inputs fail, for the miss
    log: dtype and rank, the tape's guards at these sizes, strides, offsets and
    addresses, then the opaque selectors re-run at them. A device term is named
    generically."""
    for index, rec in enumerate(lowered.tape.inputs):
        t = boxed[index]
        if t.dtype != rec.dtype:
            return f"{rec.name} is {t.dtype}, the tape traced {rec.dtype}"
        if t.dim() != len(rec.sizes):
            return (
                f"{rec.name} has rank {t.dim()}, the tape traced rank {len(rec.sizes)}"
            )
    values = _symbol_values(lowered, boxed)
    # an opaque rebind's value at these inputs (its function re-run, in declaration
    # order: a later rebind may read an earlier one), so a guard over it evaluates
    # by substitution like any other; a guard-kind opaque is its traced value
    for symbol, rec in lowered.symbols.opaque.items():
        if rec["kind"] == "guard":
            continue
        try:
            args = [int(_expr(a).xreplace(values)) for a in rec["args"]]
            values[symbol] = sympy.Integer(rec["call"](args))
        except (ArithmeticError, TypeError, ValueError):
            pass
    for kind, relations in (
        ("guard", lowered.tape.guards),
        ("obligation", lowered.extra_guards),
    ):
        for g in relations:
            try:
                failed = _expr(g).xreplace(values) is sympy.false
            except (ArithmeticError, TypeError, ValueError):
                failed = True
            if failed:
                return f"{kind} failed: {g}"
    for rec in lowered.symbols.opaque.values():
        try:
            got = rec["call"]([int(_expr(a).xreplace(values)) for a in rec["args"]])
        except (ArithmeticError, TypeError, ValueError):
            continue
        if rec["kind"] == "guard" and got != int(rec["expected"]):
            return f"opaque {rec['fn']} gives {got} at these inputs, the tape traced {rec['expected']}"
        if rec.get("domain") == "positive" and got < 1:
            return f"opaque {rec['fn']} gives {got} at these inputs, below its declared domain"
    if not check_predicate(lowered, boxed, regions=False):
        # the facts and guards fail under the compiled predicate while none did by
        # substitution: a guard over a rebind this evaluation could not compute, a
        # pointer alignment or a device term
        return "a pointer alignment, device or opaque-rebind term of the predicate"
    if lowered.arena is not None and not check_predicate(
        lowered, boxed, mode=_MODE_ARENA
    ):
        return "the arena's capacity or plan class"
    return "no term of the predicate this evaluation can name"


def _same_sequence(family, lowered):
    plan = lowered.sequence
    if (family.sequence is None) != (plan is None):
        return False
    return plan is None or family.sequence.plan.signature == plan.signature


def _tape_class(tape, reason):
    """The class of calls a lowering or preparation decline covers, as the entry
    remembers a declined trace (Declined.partial): the tape's argument contract, its
    inputs and every guard it recorded. The lowering is a function of the tape, so a
    call that binds and holds the guards reaches the same tape and the same decline."""
    return _PartialTrace(
        tape.nargs,
        list(tape.positions),
        tape.constants,
        tape.inputs,
        tuple(_expr(g) for g in tape.guards),
        tape.device.index,
        reason,
        None,
    )


class HostTraceReplay(_Entry):
    """Trace `fn` once at `example_args`, lower the tape, and serve later calls through
    the native boxed replay. The policy is commit 1's `Entry` (one policy for both
    backends): a call is served by the first variant it matches (its argument
    contract; the family's C++ predicate then selects among the family's variants); a
    call that misses every variant is traced at its own inputs and served by the
    variant built from that tape, never by the ordinary host (`max_variants` bounds
    the variants of one entry and a miss beyond it raises); the ordinary host serves a
    call only when its trace, lowering or preparation declines, warned once per
    declined class, and the class (the guards so far, or the exact inputs) is
    remembered so no call of it is traced again. A miss traces without a warm-up: the
    host executes exactly once per call; the constructor's warm-up is the entry's one
    execution at construction. `build_variant(tape, args=None)` is the native builder
    (`args` defaults to the tape's own call, `Tape.args`). A call whose guards hold
    but whose closed regions select a node chain the variant's exec does not hold
    (E28) is a `TopologyMiss`: the entry builds the same tape at the call's inputs
    (no re-trace) and keeps that variant beside the first.

    The input contract is the caller's (E32): the tape's guards and root facts are
    checked per call, the ambient dispatch state is not (E21). Grad mode in
    particular: a served call is a graph replay, so its outputs carry no autograd
    history and an input requiring grad is bound like any other tensor (eager's
    refusal of an in-place op on a leaf requiring grad is not raised). Call the entry
    under `torch.no_grad()` (the model runners do); a training step goes through a
    functional (make_fx) step whose tape has no grad mode inside. The entry does not
    scan the box for grad mode (O48)."""

    # the entry's warning names our caller: `__call__` here, the entry's, `_miss`
    warn_stacklevel = 6

    def __init__(
        self,
        fn,
        example_args=None,
        *,
        warm_up=True,
        staging_depth=2,
        max_variants=16,
        arena=None,
        arena_check=None,
        output_arena=None,
        output_blocks=None,
        allocseq=None,
        allocseq_mode=None,
        tape=None,
        device=None,
    ):
        super().__init__(
            fn,
            build_variant=self.build_variant,
            warm_up=False,
            max_variants=max_variants,
        )
        self.constructor_warm_up = warm_up
        self.staging_depth = staging_depth
        # the CUDA device the variants are prepared on: the tape's own unless named
        # (a tape traced on one device, served on another; the inputs are that
        # device's, which its device facts check)
        self.device = None if device is None else int(device)
        # the planned arena (hosttrace_arena): the module default unless given; with
        # `arena_check`, every served call re-evaluates its plan at the call's shapes
        # in Python and asserts that no two live blocks overlap (a debug mode)
        self.arena_enabled = _ARENA_DEFAULT if arena is None else bool(arena)
        self.arena_check = (
            _ARENA_CHECK_DEFAULT if arena_check is None else bool(arena_check)
        )
        # the output arena (hosttrace_arena.OutputRing): the module default unless given
        self.output_arena_enabled = (
            _OUTPUT_ARENA_DEFAULT if output_arena is None else bool(output_arena)
        )
        self.output_blocks = (
            _OUTPUT_BLOCKS_DEFAULT if output_blocks is None else int(output_blocks)
        )
        # the allocation sequence (hosttrace_allocseq): the module default unless given;
        # it replaces the planned arena (the output arena composes with it)
        self.allocseq_enabled = (
            _ALLOCSEQ_DEFAULT if allocseq is None else bool(allocseq)
        )
        self.allocseq_mode = (
            _ALLOCSEQ_MODE_DEFAULT if allocseq_mode is None else str(allocseq_mode)
        )
        if self.allocseq_enabled:
            if arena:
                raise ValueError(
                    "host_trace replay: the allocation sequence replaces the planned arena"
                )
            self.arena_enabled = False
        self.arena_grows = 0  # calls that grew a family's arena before being served
        self.output_grows = (
            0  # calls that raised a family's output floor before being served
        )
        self.sequence_rebinds = (
            0  # calls that rebound a family's sequence roots before being served
        )
        self._families: list[_Family] = []
        # the first family, once there is one (None for a boxed entry and after close):
        # the hit path of `__call__` is its contract and its dispatch
        self._hot: _Family | None = None
        # the first variant's prepared native entry (None before it): a plain attribute,
        # so a call through it runs no Python frame of this class
        self.entry = None
        # the dispatch's cold callback tells a Python caller of a family's dispatch (it
        # gets _MISSED and decides) from an outside caller of `entry(box)` (the policy
        # runs in the callback); while that policy runs, the family's dispatch is
        # busy and its Python equivalent serves
        self._python_dispatch = False
        self._entry_call = None
        self.calls = 0
        self.misses = 0  # calls no existing variant served at once
        self.miss_log = []  # (call, why each variant missed, what served the call)
        self.single = None
        # closed regions: (site, key) -> reason, for keys whose template the harvest
        # refused (not rebindable, a multi-byte memset); a key whose template has
        # another node chain than the site's is no refusal but a topology miss
        self.refused = {}
        self._last_event = None
        self._last_pinned = ()  # storage addresses of the last served call's pinned inputs
        # per device, a ring of events for the pinned-input holds (one per call in
        # flight plus one) and the next slot
        self._events: dict[int, list] = {}
        self._streams = {}  # raw stream handle -> torch.cuda.Stream, built once
        self._missed_call = False
        self._traces_at_call = 0
        self._pending_why = None
        # a preparation (a build) runs under the lock; another thread's preparation
        # meanwhile is refused, not queued
        self.lock = threading.Lock()
        self._cold_owner = None
        self.closed = False
        # constructed without example arguments: the boxed entry the runtime team's
        # tests drive, `replay(box)` with the tensors in a list, cleared on return
        self.boxed = example_args is None
        # the constructor's warm-up is the entry's first call (E24): its return value
        # when the trace or the build declined after it ran (`_construct`)
        self.warm_up_outputs = None
        if example_args is not None:
            self._construct(fn, tuple(example_args), warm_up, tape)

    def _construct(self, fn, args, warm_up, tape=None):
        """The constructor's trace and build at `args`. A decline after the warm-up
        ran leaves a usable entry, as a miss's decline does: the class is remembered
        (the ordinary host serves it, warned once) and the warm-up's return value is
        kept on `warm_up_outputs`, since that call already happened and its result
        is the caller's. A decline before anything ran raises: the caller decides. A
        given tape (a test's, the oracle's) is prepared as it is: no trace, nothing
        ran here, and the lowering's own declines raise."""
        from torch.cuda import _host_trace

        exact = _host_trace._exact_class(args)
        ran = outputs = None
        if tape is not None:
            # the lowering's own declines surface: nothing ran here
            self.variants.append(self._prepare_variant(tape, args))
            return
        try:
            self.traces += 1
            tape = _host_trace.trace(fn, args, warm_up=warm_up)
            ran, outputs, tape.warm_up_outputs = warm_up, tape.warm_up_outputs, None
            self.variants.append(self._build(tape, args))
        except _host_trace.Declined as e:
            if ran is None:
                ran, outputs = e.warm_up_ran, e.warm_up_outputs
            if not ran:
                raise
            self.warm_up_outputs = outputs
            if self.single is None:
                self.single = isinstance(outputs, torch.Tensor)
            self._decline(e, args, exact, stacklevel=4)

    # what the first variant fixed for the entry (tests and callers read these);
    # None before a boxed entry's first variant
    @property
    def tape(self):
        return self.variants[0].tape if self.variants else None

    @property
    def lowered(self):
        return self.variants[0].lowered if self.variants else None

    @property
    def dispatch(self):
        return self._families[0].dispatch if self._families else None

    @property
    def pinned_positions(self):
        return self.lowered.pinned_positions

    @property
    def served(self):
        return self.calls - self.ordinary

    @property
    def declines(self):
        """Why this entry declined, once per distinct reason (trace, lowering or
        preparation declines alike): the entry's record."""
        return self.declined_reasons

    @property
    def declined_classes(self):
        """The declined classes the entry remembers: by their guards so far and by
        their exact inputs."""
        return (*self.declined, *self.declined_exact)

    def _prepare_variant(self, tape, args):
        """Lower and prepare `tape` at `args`, admit the variant to the family of its
        argument contract (a new family when none holds it). Raises the lowering's
        own declines."""
        lowered = lower_tape(
            tape,
            args,
            arena=self.arena_enabled,
            output_arena=self.output_arena_enabled,
            allocseq=self.allocseq_enabled,
            device=self.device,
        )
        # the family of the variant's argument contract; with an allocation sequence,
        # of the same sequence plan as well (one sequence object binds every variant
        # of a family, the dispatch not saying which serves a call)
        family = next(
            (
                f
                for f in self._families
                if f.lowered.contract_holds(args)
                and f.pinned == lowered.pinned_positions
                and _same_sequence(f, lowered)
            ),
            None,
        )
        arena = outputs = sequence = None
        if lowered.arena is not None:
            # the family's arena (a new family's is made here), holding the plan's
            # requirement at the preparation inputs
            # (a family lowered without one gets a fresh arena here and is refused
            # below: the dispatch boxes one layout per family)
            arena = family.arena if family is not None else None
            if arena is None:
                arena = _Arena(torch.device("cuda", lowered.device))
            grew = arena.ensure(
                lowered.arena.required_bytes(
                    _symbol_values(lowered, lowered.tensors(args))
                )
            )
            if grew and family is not None:
                family.dispatch.bind((arena.tensor,))
        if lowered.output_arena is not None:
            # the family's output ring (a new family's is made here), its capacity
            # floor at the plan's requirement at the preparation inputs
            outputs = (
                family.outputs
                if family is not None
                else OutputRing(
                    torch.device("cuda", lowered.device), self.output_blocks
                )
            )
            outputs.ensure(
                lowered.output_arena.required_bytes(
                    _symbol_values(lowered, lowered.tensors(args))
                )
            )
        if lowered.sequence is not None:
            # the family's sequence (a new family's is made here), its roots bound at
            # the preparation inputs before the capture reads their addresses
            sequence = (
                family.sequence
                if family is not None
                else AllocatorSequence(lowered.device, self.allocseq_mode)
            )
            sequence.bind(lowered.sequence, lowered.tensors(args))
        entry = prepare_hosttrace(
            lowered,
            args,
            staging_depth=self.staging_depth,
            arena=arena,
            outputs=outputs,
            sequence=sequence,
        )
        try:
            single = len(tape.outputs) == 1
            if self.single is None:
                self.single = single
            elif single != self.single:
                raise HostTraceLoweringDeclined(
                    "host_trace lowering: the traced call returned another number of outputs than the entry's first trace"
                )
            if family is not None:
                first = family.lowered
                if (lowered.arena is None) != (first.arena is None) or (
                    lowered.output_arena is None
                ) != (first.output_arena is None):
                    raise HostTraceLoweringDeclined(
                        "host_trace lowering: the variant and its family differ on the planned arena; the dispatch boxes one layout per family"
                    )
                if lowered.written_positions != family.written:
                    family.written = tuple(
                        sorted({*family.written, *lowered.written_positions})
                    )
                family.dispatch.append(entry, lowered.registration)
            else:
                family = _Family(
                    lowered,
                    None,
                    [],
                    arena=arena,
                    outputs=outputs,
                    sequence=sequence,
                    constants={i: args[i] for i in lowered.constant_positions},
                )
                # the arena is the dispatch's own trailing input: a caller of the
                # dispatch (`entry(box)`, the runtime team's boxed surface) passes the
                # tape's tensors alone (with the output ring or the sequence on, the
                # box carries their block and roots: opt-in surfaces)
                family.dispatch = torch._C._cuda_make_boxed_dispatch(
                    ((entry, lowered.registration),),
                    self._on_miss,
                    () if arena is None else (arena.tensor,),
                )
                self._families.append(family)
        except BaseException:
            entry.close()
            raise
        variant = _Variant(tape, lowered, entry, self, family)
        family.variants.append(variant)
        if self.entry is None:
            self.entry = family.dispatch
        if self._hot is None and not self.boxed:
            self._hot = self._families[0]
        # the variant retains the tape's records (inputs, guards, regions, device); the
        # call the tape was traced at is the preparation's alone
        tape.args = None
        return variant

    def build_variant(self, tape, args=None):
        """The entry's builder: the variant of `tape` at `args` (the tape's own call by
        default). A recognized decline of the lowering or the preparation is raised as
        the trace's `Declined`, with the tape's class on `partial`, so the entry
        remembers it like a declined trace; anything else propagates."""
        if args is None:
            args = tape.args
            if args is None:
                raise RuntimeError(
                    "host_trace replay: the tape's call was released when its variant was prepared"
                )
        args = tuple(args)
        # a preparation runs under the lock; a concurrent preparation from another
        # thread is refused, not queued (served calls are the dispatcher's business,
        # which refuses a concurrent call itself)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Host trace preparation is busy")
        try:
            if self._cold_owner not in (None, threading.get_ident()):
                raise RuntimeError("Host trace preparation is busy")
            variant = self._build(tape, args)
        finally:
            self.lock.release()
        self.miss_log.append(
            (self.calls, self._why_or_contract(), f"variant {len(self.variants) + 1}")
        )
        return variant

    def _build(self, tape, args):
        """`_prepare_variant`, with a recognized decline of the lowering or the
        preparation raised as the trace's `Declined` (the tape's class on `partial`),
        so the entry remembers it like a declined trace; anything else propagates."""
        from torch.cuda import _host_trace

        try:
            return self._prepare_variant(tape, args)
        except (UnsupportedCapture, _host_trace.Miss) as e:
            reason = str(e)
            declined = _host_trace.Declined(reason)
            declined.partial = _tape_class(tape, reason)
            raise declined from e

    def _serve_unmatched(self, args, tape):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Host trace preparation is busy")
        try:
            if self._cold_owner is not None:
                raise RuntimeError("Host trace preparation is busy")
            if self.closed:
                raise RuntimeError("host_trace replay is closed")
            self._cold_owner = threading.get_ident()
        finally:
            self.lock.release()
        try:
            return super()._serve_unmatched(args, tape)
        finally:
            with self.lock:
                self._cold_owner = None

    def _why_or_contract(self):
        if self._pending_why is not None:
            return self._pending_why
        if not self._families:
            return "the entry has no variant yet"
        return "the argument contract (arity, tensor positions or a non-tensor argument) differs from every family's"

    def _on_miss(self, box):
        # the dispatch's cold callback: no variant of the family took the call. A
        # Python caller of the dispatch decides itself, outside the dispatcher's lock;
        # an outside caller of `entry(box)` gets the entry's policy here
        if self._python_dispatch:
            return _MISSED
        return self._entry_miss(box)

    def _dispatch(self, family, box):
        """The family's native dispatch over the full box (the arena appended), or its
        Python equivalent (each variant's predicate, then its entry) while the
        dispatcher runs this entry's cold callback for an `entry(box)` caller."""
        if self._entry_call is not None:
            for variant in family.variants:
                outputs = self._serve(variant, box)
                if outputs is not None:
                    return outputs
            return _MISSED
        self._python_dispatch = True
        try:
            return family.dispatch(box)
        finally:
            self._python_dispatch = False

    def _entry_miss(self, box):
        """The cold callback for a caller of `entry(box)`: the first family's tensors
        in the box's order (the arena is the dispatch's own input). The entry's policy
        over the arguments the box stands for, once; the box is consumed as a served
        call's is, and the outputs come back as the list the surface returns."""
        family = self._families[0]
        args = family.arguments(box)
        self._entry_call = family
        try:
            outputs = self._serve_cold(args)
        finally:
            self._entry_call = None
        box.clear()
        return list(outputs)

    def _serve(self, variant, box):
        # the predicate selects each region's variant on this thread and the entry's
        # plan reads that selection back, so it runs before the direct entry call; None
        # when the variant rejects these inputs
        if not check_predicate(variant.lowered, box):
            return None
        return variant.entry(box)

    def _regrown(self, family, args, box):
        """A variant of `family` whose guards hold at `box` but whose arena terms
        failed: too small an arena grows, too small an output block is replaced and
        outgrown sequence roots are rebound (a rebind, and the dispatch is retried:
        no miss). Returns (outputs, box): the outputs when the retry served, None
        when no variant's guards hold here or the retry missed, and the box the
        retry used (the given one when nothing grew: a caller that goes on to the
        harvest must not hand it the outgrown arena tensor); a call outside a plan's
        class (the arena) is the same tape built at these inputs (_TopologyMiss)."""
        for variant in family.variants:
            lowered = variant.lowered
            if (
                lowered.arena is None
                and lowered.output_arena is None
                and lowered.sequence is None
            ) or not check_predicate(lowered, box, regions=False):
                continue
            if check_predicate(lowered, box, mode=_MODE_ARENA):
                continue
            grew = False
            if lowered.arena is not None:
                need = arena_bytes(lowered, box)
                if need > family.arena.capacity:
                    family.arena.ensure(need)
                    family.dispatch.bind((family.arena.tensor,))
                    self.arena_grows += 1
                    grew = True
            if lowered.output_arena is not None:
                need = output_bytes(lowered, box)
                block = box[output_block_index(lowered)]
                if need > block.numel() and family.outputs.ensure(need):
                    self.output_grows += 1
                    grew = True
            if lowered.sequence is not None:
                family.sequence.bind(lowered.sequence, box)
                self.sequence_rebinds += 1
                grew = True
            if grew:
                box = family.box(args)
                outputs = self._dispatch(family, box)
                return (None if outputs is _MISSED else outputs), box
            why = f"the call's shapes are outside variant {self.variants.index(variant) + 1}'s arena plan class: build the tape at these inputs"
            self._missed_call = True
            self._pending_why = why
            raise _TopologyMiss(why, variant.tape)
        return None, box

    def _missed(self, family, args):
        """The dispatch served no variant of `family` (the cold path of a served call):
        a closed region's new key is harvested and the family retried; otherwise the
        call misses (Miss), and the entry traces it."""
        from torch.cuda import _host_trace

        box = family.box(args)
        for t in box:
            if type(t) not in (torch.Tensor, torch.nn.Parameter):
                # the contract admits any Tensor; the native dispatch serves exact
                # Tensors and Parameters and reports anything else as a miss
                raise TypeError(
                    f"host_trace replay: boxed inputs must be Tensors or Parameters, not {type(t).__name__}"
                )
        stream = _current_raw_stream(family.device)
        if stream != family.stream:
            # a miss on a stream the family is not bound to: a variant prepared here
            # could not join the family's dispatch (one stream per dispatch; a hit
            # elsewhere is refused by name), so it is refused the same way, before a
            # trace, a lowering or a build
            raise RuntimeError(
                f"host_trace replay: the call runs on stream {stream:#x} while the family's variants are bound to stream {family.stream:#x}; a native replay requires its bound device and stream"
            )
        if family.sequence is not None:
            # the variants of one contract sit in one family per sequence plan (the
            # roots differ): another family of this contract may serve the call
            # natively, with its own rebind, before this family's miss path
            for other in self._families:
                if other is family or not other.lowered.contract_holds(args):
                    continue
                other_box = other.box(args)
                outputs = self._dispatch(other, other_box)
                if outputs is _MISSED:
                    outputs, _ = self._regrown(other, args, other_box)
                if outputs is not None:
                    return outputs
        if (
            family.arena is not None
            or family.outputs is not None
            or family.sequence is not None
        ):
            outputs, box = self._regrown(family, args, box)
            if outputs is not None:
                return outputs
        self._missed_call = True
        registered, topology = self._harvest_missed(family, box)
        if registered:
            # a closed region's key was new (a harvest, not a shape class): the
            # variant whose other guards hold serves this call
            for variant in family.variants:
                outputs = self._serve(variant, box)
                if outputs is not None:
                    which = (
                        f"variant {self.variants.index(variant) + 1} after the harvest"
                    )
                    self.miss_log.append(
                        (self.calls, "a closed region's new key", which)
                    )
                    return outputs
        if topology is not None:
            # the tape's guards held and the call selects a node chain this
            # variant's exec does not hold: the same tape built at these inputs
            # is the variant that serves it (E28), which the entry keeps beside
            # this one; no re-trace, and the harvest is cached
            variant, why = topology
            self._pending_why = why
            raise _TopologyMiss(why, variant.tape)
        why = "; ".join(_why_missed(v.lowered, box) for v in family.variants)
        self._pending_why = why
        raise _host_trace.Miss(why)

    def _ordinary(self, args):
        outcome = (
            "declined" if self.traces != self._traces_at_call else "declined class"
        )
        self.miss_log.append((self.calls, self._why_or_contract(), outcome))
        return self._run_ordinary(args)

    def _run_ordinary(self, args):
        self.ordinary += 1
        out = self.fn(*args)
        if self.single is None:
            # no variant fixed the entry's output arity (declined at construction)
            self.single = isinstance(out, torch.Tensor)
        return [out] if isinstance(out, torch.Tensor) else list(out)

    def __call__(self, *args):
        # the hit path: the first family's contract by arity and constants, then its
        # dispatch, which reads every boxed position as an exact Tensor itself (anything
        # else misses every variant) and tries the family's variants at once. This is
        # the first step of the entry's loop with the tensor scan left to the dispatcher;
        # a call it does not serve takes the entry's loop below, which evaluates the
        # first variant's `matches` in full before deciding.
        family = self._hot
        if (
            family is not None
            and len(args) == family.nargs
            and (family.constants_hold is None or family.constants_hold(args))
        ):
            arena = family.arena
            box = family.boxer(args, None if arena is None else arena.tensor)
            ring = family.outputs
            if ring is not None:
                box.append(ring.take())
            sequence = family.sequence
            if sequence is not None:
                box.extend(sequence.take(box))
            self._python_dispatch = True
            try:
                outputs = family.dispatch(box)
            finally:
                self._python_dispatch = False
            if outputs is not _MISSED:
                self.calls += 1
                if family.pinned:
                    self._hold_pinned(args, family)
                if self.arena_check and (arena is not None or sequence is not None):
                    self._check_arena(family, args, outputs)
                return outputs[0] if self.single else tuple(outputs)
        box = None
        if self.boxed and len(args) == 1 and type(args[0]) is list:
            box = args[0]
            args = tuple(box)
        outputs = self._serve_cold(args)
        if box is not None:
            box.clear()
            return list(outputs)
        return outputs[0] if self.single else tuple(outputs)

    def _serve_cold(self, args):
        """The entry's loop over `args` (the later variants, the declined classes, a
        trace and a build), counted as a call and, when it did not serve natively, a
        miss."""
        if self.closed:
            raise RuntimeError("host_trace replay is closed")
        self.calls += 1
        self._missed_call = False
        self._pending_why = None
        self._traces_at_call = self.traces
        ordinary = self.ordinary
        variants = len(self.variants)
        try:
            return _Entry.__call__(self, *args)
        finally:
            # a miss whether the call served, ran ordinary, or raised on the way
            if (
                self._missed_call
                or len(self.variants) != variants
                or self.traces != self._traces_at_call
                or self.ordinary != ordinary
            ):
                self.misses += 1

    def _harvest_missed(self, family, box):
        # the predicate ran every region's select after its guards held: a site
        # with no template for its key recorded the key. Harvest each and register
        # it when its node chain is the site's, so the variant serves the key from
        # now on. A template with another chain (or scratch beyond the arena) is
        # another class of that variant's tape: the first such variant of this
        # family whose guards hold at this call (the facts-only predicate: a key
        # may be left over from an earlier call the variant did not serve) is
        # reported for a topology miss. Returns (any key registered, (variant,
        # reason) of the topology miss or None).
        from torch.cuda import _host_trace

        registered, topology = False, None
        for variant in self.variants:
            for region in variant.lowered.regions:
                key = torch._C._cuda_kernel_template_take_miss(region.site)
                if key is None:
                    continue
                key = tuple(key)
                if (region.site, key) in self.refused:
                    continue
                try:
                    template = _region_template(
                        region,
                        key,
                        variant.lowered.device,
                        tuple(variant.tape.device_identity),
                    )
                    _register_region_variant(
                        region,
                        key,
                        template,
                        variant.lowered.region_nodes[region.site],
                        variant.lowered.region_arena,
                    )
                    registered = True
                except _host_trace.Miss as e:
                    self.refused[(region.site, key)] = str(e)
                except _AnotherClass as e:
                    if (
                        topology is None
                        and variant in family.variants
                        and check_predicate(variant.lowered, box, regions=False)
                    ):
                        topology = (variant, str(e))
        return registered, topology

    def region_stats(self):
        """Counters for tests and debugging: template swaps, swaps that went through
        cuGraphExecUpdate, the process-wide harvest count, the refused keys, and per
        variant and region the site's node chain (E28: the exec holds exactly it)."""
        from torch.cuda import _host_trace

        applies = graph_updates = 0
        for variant in self.variants:
            swaps, updates = variant.entry._template_stats()
            applies += swaps
            graph_updates += updates
        return {
            "applies": applies,
            "graph_updates": graph_updates,
            "harvests": _host_trace.gemm_harvests(),
            "refused": dict(self.refused),
            "sites": [
                {
                    "name": region.name,
                    "site": region.site,
                    "nodes": len(variant.lowered.region_nodes[region.site]),
                    "kinds": [k for k, _ in variant.lowered.region_nodes[region.site]],
                    "programmatic": [
                        p for _, p in variant.lowered.region_nodes[region.site]
                    ],
                }
                for variant in self.variants
                for region in variant.lowered.regions
            ],
        }

    def arena_stats(self):
        """Per family the arena's capacity and growths, per variant the plan: blocks
        placed, buffers left to the runtime, the requirement at the preparation
        shapes, the class guards and how the pair conditions were decided."""
        families = [
            {
                "capacity": f.arena.capacity if f.arena else 0,
                "variants": len(f.variants),
                "output_blocks": len(f.outputs.blocks) if f.outputs else 0,
                "output_capacities": tuple(b[2] for b in f.outputs.blocks)
                if f.outputs
                else (),
                "output_held": f.outputs.held() if f.outputs else 0,
                "output_floor": f.outputs.minimum if f.outputs else 0,
                "output_overflow": f.outputs.overflow if f.outputs else 0,
                "output_replaced": f.outputs.replaced if f.outputs else 0,
            }
            for f in self._families
        ]
        variants = []
        for variant in self.variants:
            plan = variant.lowered.arena
            row = (
                None
                if plan is None
                else {
                    "blocks": len(plan.blocks),
                    "buffers": len(variant.lowered.allocations),
                    "bytes_at_hints": plan.required_bytes(plan.hints),
                    "lower_bound_at_hints": plan.lower_bound,
                    "class_guards": len(plan.class_guards),
                    "equalities": plan.equalities,
                    "proofs": dict(plan.proofs),
                }
            )
            outputs = variant.lowered.output_arena
            if outputs is not None:
                row = dict(row or {"buffers": len(variant.lowered.allocations)})
                row["output_blocks"] = len(outputs.blocks)
                row["output_bytes_at_hints"] = outputs.required_bytes(outputs.hints)
            variants.append(row)
        return {
            "families": families,
            "variants": variants,
            "grows": self.arena_grows,
            "output_grows": self.output_grows,
        }

    def sequence_stats(self):
        """Per family the allocation sequence's roots, sequences run, roots rebound,
        pools made, pressure releases, the last sequence's peak and the pool's
        reserved bytes; per variant the plan: roots, buffers left to the runtime, the
        program length and the live peak and the sum of the roots at the preparation
        shapes."""
        families = [
            None
            if f.sequence is None
            else {
                "mode": f.sequence.mode,
                "roots": len(f.sequence.roots),
                "binds": f.sequence.binds,
                "rebinds": f.sequence.rebinds,
                "pools": f.sequence.pools,
                "pressure_releases": f.sequence.pressure_releases,
                "peak": f.sequence.peak,
                "reserved": f.sequence.reserved(),
            }
            for f in self._families
        ]
        variants = [
            None
            if v.lowered.sequence is None
            else {
                "roots": len(v.lowered.sequence.names),
                "buffers": len(v.lowered.allocations),
                "program": len(v.lowered.sequence.program),
                "programmatic": len(v.lowered.sequence.programmatic),
                "peak_at_hints": v.lowered.sequence.peak_at_hints,
                "sum_at_hints": v.lowered.sequence.sum_at_hints,
            }
            for v in self.variants
        ]
        return {
            "families": families,
            "variants": variants,
            "rebinds": self.sequence_rebinds,
        }

    def _check_arena(self, family, args, outputs=None):
        # the debug assertion: every variant of the family whose arena terms hold at
        # this call (the dispatch does not say which served, and it emptied the box)
        # has its plan re-evaluated at the call's shapes over what served it: the
        # arena tensor, the ring block the outputs view and the roots as bound (a
        # fresh `family.box` would take another ring block and, in replay mode, run
        # another sequence; without the outputs, as the tests call it, the box is
        # built as a call would); no two live blocks may overlap, none may exceed
        # the arena; no two live sequence roots may share bytes
        if outputs is None:
            box = family.box(args)
        else:
            box = list(family.tensors(args))
            if family.written:
                torch._C._host_trace_materialize(box, family.written)
            if family.arena is not None:
                box.append(family.arena.tensor)
            if family.outputs is not None:
                box.append(_served_block(family.outputs, outputs))
            if family.sequence is not None:
                box.extend(family.sequence.roots)
        for variant in family.variants:
            lowered = variant.lowered
            if (
                lowered.arena is None
                and lowered.output_arena is None
                and lowered.sequence is None
            ) or not check_predicate(lowered, box, mode=_MODE_ARENA):
                continue
            if lowered.sequence is not None:
                family.sequence.check(lowered.sequence, box)
            if lowered.arena is None and lowered.output_arena is None:
                continue
            values = _symbol_values(lowered, box)
            if lowered.arena is not None:
                lowered.arena.check(values, family.arena.capacity)
            if lowered.output_arena is not None:
                lowered.output_arena.check(
                    values, box[output_block_index(lowered)].numel()
                )

    def _hold_pinned(self, args, family):
        # the raw current stream: torch.cuda.current_stream() builds a Stream object,
        # which costs more than the whole native call
        device = family.device
        raw_stream = _current_raw_stream(device)
        stream = self._streams.get(raw_stream)
        if stream is None:
            stream = self._streams[raw_stream] = torch.cuda.current_stream(device)
        ring = self._events.get(device)
        if ring is None:
            events = tuple(torch.cuda.Event() for _ in range(self.staging_depth + 1))
            ring = self._events[device] = [events, 0]
        events, slot = ring
        event = events[slot]
        ring[1] = (slot + 1) % len(events)
        event.record(stream)
        self._last_event = event
        addresses = []
        for position in family.pinned:
            # the caching host allocator defers the buffer's reuse to this stream's
            # event, so a pinned input the caller frees is not handed out while the
            # queued copy still reads it; wait_for_h2d() covers in-place rewrites
            _record_host_event(args[position], raw_stream)
            addresses.append(_storage_address(args[position]))
        _h2d_note(addresses, self, event)
        self._last_pinned = addresses

    def wait_for_h2d(self):
        """Wait until the last served call's copies have read their pinned inputs: a
        caller rewriting the same pinned buffer in place calls this first. Covers
        every entry's pending copies from that call's pinned buffers (the
        module-level `wait_for_h2d(pinned)` takes the buffer itself)."""
        if self._last_event is not None:
            self._last_event.synchronize()
        for address in self._last_pinned:
            _h2d_wait(address)

    @property
    def memcpy_stats(self):
        """(memcpy nodes re-parameterized, of which the source moved) so far, summed
        over the variants."""
        nodes = moved = 0
        for variant in self.variants:
            n, m = variant.entry._memcpy_stats()
            nodes += n
            moved += m
        return (nodes, moved)

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Host trace preparation is busy")
        try:
            if self._cold_owner is not None:
                raise RuntimeError("Host trace preparation is busy")
            self.closed = True
            self._hot = None
            self.warm_up_outputs = None
            for family in self._families:
                family.dispatch.close()
                if family.outputs is not None:
                    family.outputs.clear()
                if family.sequence is not None:
                    family.sequence.release()
            # a CuTe launch site's receipt (its loaded kernel, its ordinary borrow)
            # closes with the replay, once its graph has let the kernel go
            for variant in self.variants:
                for call in variant.lowered.calls:
                    release = getattr(call.module, "release", None)
                    if release is not None:
                        release()
            self.variants.clear()
            self._families.clear()
        finally:
            self.lock.release()


def _bind_capture_rng(call, images, fields, philox):
    """Write a preparation capture's philox seed / offset pointers into the launch
    images at `fields` ((parameter, byte offset) rows) and into the call's constants
    that cover them, so the runtime's exact-bytes check sees the prepared state. The
    binder the runtime team's mixed replay imports; our own preparation writes the
    same bytes inline (and the intragraph offset, commit F)."""
    seed, offset_pointer, _ = philox
    payload = seed.to_bytes(8, "little") + offset_pointer.to_bytes(8, "little")
    constants = list(call.constants)
    for parameter, offset in fields:
        images[parameter][offset : offset + len(payload)] = payload
        overlaps_constant = False
        for index, (cp, co, data) in enumerate(constants):
            start = max(co, offset)
            end = min(co + len(data), offset + len(payload))
            if cp == parameter and start < end:
                overlaps_constant = True
                bound = bytearray(data)
                bound[start - co : end - co] = payload[start - offset : end - offset]
                constants[index] = (cp, co, bytes(bound))
        if not overlaps_constant:
            constants.append((parameter, offset, payload))
    return dataclasses.replace(call, constants=tuple(constants))


@dataclass
class _PreparedHostTrace:
    """What the runtime team's re-export module hands out: the prepared entry with
    its lowered tape (`guard` is their evaluator's object, None on this line)."""

    entry: object
    lowered: LoweredTape

    @property
    def guard(self):
        return self.lowered.guard

    @property
    def program(self):
        return self.lowered

    def close(self):
        self.entry.close()


def prepare_host_trace(tape, args):
    lowered = lower_tape(tape, args)
    entry = prepare_hosttrace(lowered, args)
    lowered.guard = _shared_guard(lowered, lowered.box(args))
    return _PreparedHostTrace(entry, lowered)


def lower_host_trace(tape, tensor_examples):
    if len(tensor_examples) != len(tape.inputs):
        raise HostTraceLoweringDeclined(
            "host_trace lowering: the host replay requires every tensor input"
        )
    lowered = lower_tape(tape)
    return lowered, lowered.symbols.mapping, lowered.extra_guards


class _HostIntegers(_Lowering):
    """The lowering over a shared symbol mapping and finalized payload contract."""

    def __init__(self, mapping, payload_contract=None):
        symbols = _Symbols(getattr(mapping, "tape", None), mapping)
        opaque_values = {
            symbol: sympy.Integer(int(rec["expected"]))
            for symbol, rec in symbols.opaque.items()
            if rec["kind"] == "guard"
        }
        super().__init__(symbols, opaque_values, payload_contract)
