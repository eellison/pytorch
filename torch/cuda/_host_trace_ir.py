"""The integer expression IR behind an untrusted trace's symbolic values (private).

With `_host_trace.symbolic = "ir"` the object behind each traced SymInt /
SymFloat / SymBool is `IRSymNode` below, over hash-consed nodes of a small
integer IR, in place of sympy through torch's ShapeEnv (`_TraceShapeEnv`).
c10::SymInt reaches it through PythonSymNodeImpl, whose contract is the method
names IRSymNode implements, so the traced hosts are unchanged.

Nodes are immutable and interned per trace (one node per (op, args); equality
is identity), and every node carries its hint, the value at the traced call.
Integer sums and products are flattened coefficient maps (`add`: constant
term plus (node, coefficient) pairs; `mul`: constant coefficient plus (node,
exponent) pairs), so two syntactic forms of one quantity are one node:

  x // 1 = x, x % 1 = 0, 0 // x = 0 % x = 0, constants fold
  a number times a sum distributes (sympy's rule)
  (g*x) // (g*y) = x // y                    g != 0 by the division's domain guard
  (k*d*x + r) // d = k*x + r // d             floor division, any sign of d
  (k*d + y) % d = y % d, (k*b*x) % b = 0     floor modulo
  (x // a) // b = x // (a*b)                  a, b > 0
  min / max flatten, drop duplicates, fold their constants
  a relation is rel(d, 0) with d = lhs - rhs; eq / ne oriented by the sign of
  the leading coefficient, gt / ge written as lt / le, `not` pushed into it

plus interval bounds from the declared domains (an input size is >= 1, every
other symbol unbounded), which decide a relation without a guard as sympy's
assumptions and the ShapeEnv's ranges do.

Guards are recorded as `_TraceShapeEnv` records them: a boolean the host reads
is decided by its hint and recorded once, as the node or its negation; a value
the host reads is recorded as `eq(node, hint)`; a partial operation records
its divisor's domain when it is created.

What the IR cannot express raises `Unsupported` and is counted in the env's
census; the trace declines on either, so nothing falls back to sympy
mid-trace. The tape's consumers read sympy: `IRSymNode.expr` and the env's
`guards` / `var_to_sources` / `backed_var_to_val` are exported on first read
(`_SympyExport`), a relation in the form the host wrote it.
"""

from __future__ import annotations

import dis
import hashlib
import math
import os
import struct
import sys
from typing import Any

import sympy

import torch
from torch._guards import ShapeGuard, SLoc
from torch.cuda._host_trace_program import f32_bits
from torch.utils._sympy.functions import (
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    CeilToInt,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    Max,
    Min,
    Mod,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    PythonMod,
    ToFloat,
    TruncToInt,
    Where,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges


INT_OPS = frozenset(
    {
        "const",
        "sym",
        "add",
        "mul",
        "floordiv",
        "ceildiv",
        "mod",
        "min",
        "max",
        "nod",
        "pbn",
        "bitlen",
        "f32div",
        "bitand",
        "bitor",
        "bitxor",
        "where",
        "ftrunc",
        "ffloor",
        "fceil",
    }
)
BOOL_OPS = frozenset(
    {"true", "false", "eq", "ne", "lt", "le", "and", "or", "not", "fcmp"}
)
FLOAT_OPS = frozenset(
    {
        "fconst",
        "fsym",
        "ffromint",
        "fadd",
        "fsub",
        "fmul",
        "fdiv",
        "fneg",
        "fsqrt",
        "fpow",
    }
)
_REL_TEXT = {"eq": "==", "ne": "!=", "lt": "<", "le": "<="}
_REL_CLASSES = {
    "Eq": sympy.Eq,
    "Ne": sympy.Ne,
    "Lt": sympy.Lt,
    "Le": sympy.Le,
    "Gt": sympy.Gt,
    "Ge": sympy.Ge,
}
_NEGATED = {"Eq": "Ne", "Ne": "Eq", "Lt": "Ge", "Le": "Gt", "Gt": "Le", "Ge": "Lt"}
_BITWISE = {
    "bitand": (lambda a, b: a & b, BitwiseFn_bitwise_and),
    "bitor": (lambda a, b: a | b, BitwiseFn_bitwise_or),
    "bitxor": (lambda a, b: a ^ b, BitwiseFn_bitwise_xor),
}
_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__)) + os.sep
_NO_SLOC = SLoc(None, None)


class Unsupported(Exception):
    """An operation the IR backend does not express; the trace declines on
    it by name (op, site)."""

    def __init__(self, op: str, site: str) -> None:
        super().__init__(f"{op} at {site}")
        self.op = op
        self.site = site


_COMPARE = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
_TRUTH = ("POP_JUMP_IF_FALSE", "POP_JUMP_IF_TRUE", "UNARY_NOT", "TO_BOOL")
_LEN_USES: dict[tuple, tuple | None] = {}


def _len_use(code: Any, lasti: int) -> tuple | None:
    # (rel, c) when the instruction at lasti is a call of the global `len`
    # whose result only feeds `== c`-style comparisons or a truth test
    instrs = list(dis.get_instructions(code))
    at = next((i for i, x in enumerate(instrs) if x.offset == lasti), None)
    if at is None or instrs[at].opname != "CALL":
        return None
    depth = instrs[at].arg + 2
    for x in reversed(instrs[:at]):
        if x.opcode in dis.hasjrel or x.opcode in dis.hasjabs:
            return None
        arg = x.arg if x.opcode >= dis.HAVE_ARGUMENT else None
        depth -= dis.stack_effect(x.opcode, arg, jump=False)
        if depth <= 0:
            if depth or x.opname != "LOAD_GLOBAL" or x.argval != "len":
                return None
            break
        if x.is_jump_target:
            return None
    else:
        return None
    after = instrs[at + 1 : at + 3]
    if after and after[0].opname in _TRUTH:
        return ("ne", 0)
    if (
        len(after) == 2
        and after[0].opname == "LOAD_CONST"
        and type(after[0].argval) is int
        and after[1].opname == "COMPARE_OP"
    ):
        rel = _COMPARE.get(after[1].argrepr.removeprefix("bool(").removesuffix(")"))
        return None if rel is None else (rel, after[0].argval)
    return None


def _len_compare() -> tuple | None:
    # len(t) of a traced tensor reaches int_ through SymInt.__index__; when
    # the caller only compares it to a constant, the comparison is the guard
    index = sys._getframe(2)
    if index.f_code is not torch.SymInt.__index__.__code__:
        return None
    f = index.f_back
    if f is None or f.f_globals.get("len", f.f_builtins.get("len")) is not len:
        return None
    k = (f.f_code, f.f_lasti)
    if k not in _LEN_USES:
        _LEN_USES[k] = _len_use(f.f_code, f.f_lasti)
    return _LEN_USES[k]


def _site() -> str:
    # the innermost frame outside torch: where the user's code stands
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename.startswith(_TORCH_DIR):
        f = f.f_back
    if f is None:
        return "?"
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"


class Node:
    """One interned expression. `args` name other nodes of the same context;
    a key of the table is the tuple of their ids."""

    __slots__ = ("id", "op", "args", "hint", "__weakref__")

    def __init__(self, id: int, op: str, args: tuple, hint: Any) -> None:
        self.id = id
        self.op = op
        self.args = args
        self.hint = hint

    def __repr__(self) -> str:
        return render(self)

    @property
    def is_int(self) -> bool:
        return self.op in INT_OPS

    @property
    def is_bool(self) -> bool:
        return self.op in BOOL_OPS

    @property
    def is_float(self) -> bool:
        return self.op in FLOAT_OPS

    @property
    def free_symbols(self) -> set[str]:
        out: set[str] = set()
        todo, seen = [self], set()
        while todo:
            n = todo.pop()
            if n.id in seen:
                continue
            seen.add(n.id)
            if n.op in ("sym", "fsym"):
                out.add(n.args[0])
            todo.extend(children(n))
        return out


def children(n: Node) -> list[Node]:
    if n.op in ("add", "mul"):
        return [t for t, _c in n.args[1]]
    if n.op == "fcmp":
        return list(n.args[1:])
    return [a for a in n.args if isinstance(a, Node)]


def _mul_hint(coeff: int, items: tuple) -> int:
    h = coeff
    for k, e in items:
        h *= k.hint**e
    return h


def _dense(sizes: list, strides: list) -> bool:
    # torch.fx.experimental.symbolic_shapes._eval_is_non_overlapping_and_dense
    if len(sizes) == 1:
        return strides[0] == 1 or sizes[0] < 2
    expected = 1
    for length, stride in sorted(zip(sizes, strides), key=lambda p: p[1]):
        if length == 1:
            continue
        if stride != expected:
            return False
        expected *= length
    return True


class Ctx:
    """One trace's expression table and canonical forms."""

    def __init__(self) -> None:
        self.table: dict = {}
        self.nodes: list[Node] = []
        self.symbols: dict[str, Node] = {}
        self.positive: set[str] = set()  # symbols declared >= 1
        # symbol node -> (lower, upper) tightened by a caller (the test
        # oracle's implication pass); empty while tracing
        self.extra_bounds: dict = {}
        # relation node id -> (sympy class name, lhs, rhs) as first written
        self.written: dict[int, tuple] = {}
        # (op, a, b) -> result of add / mul / a comparison; nodes are interned,
        # so a node hashes by identity and an operation is a function of its
        # operands (a comparison also of extra_bounds: memoized only while empty)
        self.memo: dict = {}
        self.consts: dict[int, Node] = {}
        self.bounds_memo: dict[Node, tuple] = {}

    def mk(self, op: str, args: tuple, hint: Any) -> Node:
        key = (op, args)
        n = self.table.get(key)
        if n is None:
            n = Node(len(self.nodes), op, args, hint)
            self.table[key] = n
            self.nodes.append(n)
        return n

    # ---- leaves
    def const(self, v: int) -> Node:
        v = int(v)
        n = self.consts.get(v)
        if n is None:
            n = self.consts[v] = self.mk("const", (v,), v)
        return n

    def sym(self, name: str, hint: int, positive: bool = False) -> Node:
        n = self.symbols.get(name)
        if n is None:
            n = self.mk("sym", (name,), int(hint))
            self.symbols[name] = n
            if positive:
                self.positive.add(name)
        return n

    def fconst(self, v: float) -> Node:
        v = float(v)
        return self.mk("fconst", (struct.pack("<d", v),), v)

    def fsym(self, name: str, hint: float) -> Node:
        n = self.symbols.get(name)
        if n is None:
            n = self.mk("fsym", (name,), float(hint))
            self.symbols[name] = n
        return n

    def true(self) -> Node:
        return self.mk("true", (), True)

    def false(self) -> Node:
        return self.mk("false", (), False)

    def boolean(self, b: bool) -> Node:
        return self.true() if b else self.false()

    # ---- the canonical integer forms
    def as_terms(self, n: Node) -> tuple[int, dict]:
        # (constant, {node: coefficient}) of any integer node
        if n.op == "const":
            return n.args[0], {}
        if n.op == "add":
            return n.args[0], dict(n.args[1])
        if n.op == "mul" and n.args[0] != 1:
            coeff, factors = n.args
            if len(factors) > 1 or factors[0][1] != 1:
                inner = self.mk("mul", (1, factors), _mul_hint(1, factors))
            else:
                inner = factors[0][0]
            return 0, {inner: coeff}
        return 0, {n: 1}

    def as_factors(self, n: Node) -> tuple[int, dict]:
        # (coefficient, {node: exponent}) of any integer node
        if n.op == "const":
            return n.args[0], {}
        if n.op == "mul":
            return n.args[0], dict(n.args[1])
        return 1, {n: 1}

    def add_terms(self, const: int, terms: dict) -> Node:
        if any(
            k.op in ("add", "const") or (k.op == "mul" and k.args[0] != 1)
            for k in terms
        ):
            flat: dict = {}
            for k, v in terms.items():
                c, ts = self.as_terms(k)
                const += c * v
                for t, cf in ts.items():
                    flat[t] = flat.get(t, 0) + cf * v
            terms = flat
        terms = {k: v for k, v in terms.items() if v != 0}
        if not terms:
            return self.const(const)
        if const == 0 and len(terms) == 1:
            ((k, v),) = terms.items()
            return k if v == 1 else self.mul_factors(v, {k: 1})
        items = tuple(sorted(terms.items(), key=lambda kv: kv[0].id))
        hint = const + sum(k.hint * v for k, v in items)
        return self.mk("add", (const, items), hint)

    def mul_factors(self, coeff: int, factors: dict) -> Node:
        if any(k.op in ("mul", "const") for k in factors):
            flat: dict = {}
            for k, v in factors.items():
                c, fs = self.as_factors(k)
                coeff *= c**v
                for f, e in fs.items():
                    flat[f] = flat.get(f, 0) + e * v
            factors = flat
        factors = {k: v for k, v in factors.items() if v != 0}
        if coeff == 0:
            return self.const(0)
        if not factors:
            return self.const(coeff)
        if len(factors) > 1:
            # a sum among several factors gives its content (the common
            # factor of its coefficients, signed so its leading term is
            # positive) to the product, so q*(16*x - 16) and (16*q)*(x - 1)
            # are one node
            for k in list(factors):
                if k.op != "add":
                    continue
                c, ts = k.args
                g = math.gcd(c, *(cf for _t, cf in ts))
                if (ts[0][1] if ts else c) < 0:
                    g = -g
                if g != 1:
                    e = factors.pop(k)
                    k2 = self.add_terms(c // g, {t: cf // g for t, cf in ts})
                    factors[k2] = factors.get(k2, 0) + e
                    coeff *= g**e
        if len(factors) == 1:
            ((k, e),) = factors.items()
            if e == 1:
                if coeff == 1:
                    return k
                if k.op == "add":  # a number times a sum distributes
                    c, ts = k.args
                    return self.add_terms(c * coeff, {t: cf * coeff for t, cf in ts})
        items = tuple(sorted(factors.items(), key=lambda kv: kv[0].id))
        return self.mk("mul", (coeff, items), _mul_hint(coeff, items))

    def add(self, a: Node, b: Node) -> Node:
        key = ("add", a, b)
        r = self.memo.get(key)
        if r is None:
            ca, ta = self.as_terms(a)
            cb, tb = self.as_terms(b)
            for k, v in tb.items():
                ta[k] = ta.get(k, 0) + v
            r = self.memo[key] = self.add_terms(ca + cb, ta)
        return r

    def neg(self, a: Node) -> Node:
        return self.mul(self.const(-1), a)

    def sub(self, a: Node, b: Node) -> Node:
        return self.add(a, self.neg(b))

    def mul(self, a: Node, b: Node) -> Node:
        key = ("mul", a, b)
        r = self.memo.get(key)
        if r is None:
            ca, fa = self.as_factors(a)
            cb, fb = self.as_factors(b)
            for k, v in fb.items():
                fa[k] = fa.get(k, 0) + v
            r = self.memo[key] = self.mul_factors(ca * cb, fa)
        return r

    def pow(self, a: Node, e: int) -> Node:
        if e == 0:
            return self.const(1)
        c, f = self.as_factors(a)
        return self.mul_factors(c**e, {k: v * e for k, v in f.items()})

    def pbn(self, base: int, e: Node) -> Node:
        # base ** e for a symbolic natural exponent: PowByNatural(base, e)
        if e.op == "const":
            return self.const(base ** e.args[0])
        return self.mk("pbn", (self.const(base), e), base**e.hint)

    # ---- bounds over the declared domains
    def bounds(self, n: Node) -> tuple:
        """(lower, upper) of an integer node, None where unknown. Reads the
        declared domains alone (a size symbol is >= 1, every other symbol
        unbounded), never a hint."""
        if self.extra_bounds:
            return self._bounds(n)
        b = self.bounds_memo.get(n)
        if b is None:
            b = self.bounds_memo[n] = self._bounds(n)
        return b

    def _bounds(self, n: Node) -> tuple:
        op = n.op
        if op == "const":
            return n.args[0], n.args[0]
        if op == "sym":
            lo, hi = (1, None) if n.args[0] in self.positive else (None, None)
            extra = self.extra_bounds.get(n)
            if extra is not None:
                elo, ehi = extra
                if elo is not None:
                    lo = elo if lo is None else max(lo, elo)
                if ehi is not None:
                    hi = ehi if hi is None else min(hi, ehi)
            return lo, hi
        if op == "add":
            c, ts = n.args
            lo, hi = c, c
            for t, cf in ts:
                tlo, thi = self.bounds(t)
                if cf < 0:
                    tlo, thi = (
                        None if thi is None else cf * thi,
                        None if tlo is None else cf * tlo,
                    )
                else:
                    tlo, thi = (
                        None if tlo is None else cf * tlo,
                        None if thi is None else cf * thi,
                    )
                lo = None if lo is None or tlo is None else lo + tlo
                hi = None if hi is None or thi is None else hi + thi
                if lo is None and hi is None:
                    return None, None
            return lo, hi
        if op == "mul":
            c, fs = n.args
            lo, hi = 1, 1
            for f, e in fs:
                flo, fhi = self.bounds(f)
                if flo is None or flo < 0:
                    return None, None  # a factor of unknown sign
                lo = None if lo is None else lo * flo**e
                hi = None if hi is None or fhi is None else hi * fhi**e
            if c < 0:
                return (None if hi is None else c * hi), (
                    None if lo is None else c * lo
                )
            return (None if lo is None else c * lo), (None if hi is None else c * hi)
        if op in ("min", "max", "where"):
            args = n.args[1:] if op == "where" else n.args
            bs = [self.bounds(a) for a in args]
            los = [b[0] for b in bs]
            his = [b[1] for b in bs]
            if op == "min":
                lo = None if None in los else min(los)
                known = [h for h in his if h is not None]
                return lo, (min(known) if known else None)
            if op == "where":
                return (None if None in los else min(los)), (
                    None if None in his else max(his)
                )
            known = [x for x in los if x is not None]
            hi = None if None in his else max(his)
            return (max(known) if known else None), hi
        # a division's divisor is nonzero where its result is used (its domain
        # guard precedes every use), so a divisor the domains put at >= 0
        # counts as positive here, as sympy's FloorDiv assumes
        if op == "mod":
            blo, bhi = self.bounds(n.args[1])
            if blo is not None and blo >= 0:
                return 0, (None if bhi is None else max(bhi - 1, 0))
            return None, None
        if op in ("floordiv", "ceildiv"):
            alo, ahi = self.bounds(n.args[0])
            blo, bhi = self.bounds(n.args[1])
            if alo is not None and alo >= 0 and blo is not None and blo >= 0:
                lo = 0 if not bhi else alo // bhi
                hi = None if ahi is None else -(-ahi // max(blo, 1))
                return lo, hi
            return None, None
        if op == "nod":
            return 0, 1
        if op == "pbn":
            return (1, None) if n.args[0].args[0] >= 1 else (None, None)
        if op == "bitlen":
            return 0, None
        return None, None

    def is_nonnegative(self, n: Node) -> bool:
        lo, _ = self.bounds(n)
        return lo is not None and lo >= 0

    # ---- division
    def _below(self, a: Node, b: Node) -> bool:
        alo, ahi = self.bounds(a)
        blo, _bhi = self.bounds(b)
        return (
            alo is not None
            and alo >= 0
            and ahi is not None
            and blo is not None
            and ahi < blo
        )

    def floordiv(self, a: Node, b: Node) -> Node:
        if b.op == "const" and b.args[0] == 0:
            raise ZeroDivisionError("floordiv by zero")
        if a.op == "const" and b.op == "const":
            return self.const(a.args[0] // b.args[0])
        if b.op == "const" and b.args[0] == 1:
            return a
        if b.op == "const" and b.args[0] == -1:
            return self.neg(a)
        if a.op == "const" and a.args[0] == 0:
            return a
        if self._below(a, b):
            return self.const(0)  # 0 <= a < b by the declared domains
        if b.op == "const" and a.op == "add":
            # the terms of a sum that a constant divisor divides leave it:
            # (k*d*x + r) // d == k*x + r // d under floor division
            c, ts = a.args
            d = b.args[0]
            out_terms = {t: cf // d for t, cf in ts if cf % d == 0}
            rest = {t: cf for t, cf in ts if cf % d != 0}
            if out_terms or (c % d == 0 and c != 0):
                quotient = self.add_terms(c // d if c % d == 0 else 0, out_terms)
                remainder = self.add_terms(c if c % d != 0 else 0, rest)
                if remainder.op == "const" and remainder.args[0] == 0:
                    return quotient
                return self.add(quotient, self.floordiv(remainder, b))
        # (x // a) // b == x // (a*b) for positive a and b; a divisor the
        # domains put at >= 0 is positive by its domain guard (see bounds)
        if (
            a.op == "floordiv"
            and self.is_nonnegative(a.args[1])
            and self.is_nonnegative(b)
        ):
            return self.floordiv(a.args[0], self.mul(a.args[1], b))
        # a common factor cancels: (x*g) // (y*g) == x // y for g != 0
        ca, fa = self.as_factors(a)
        cb, fb = self.as_factors(b)
        g = math.gcd(ca, cb)
        common = {k: min(fa[k], fb[k]) for k in fa if k in fb}
        if g > 1 or common:
            a2 = self.mul_factors(
                ca // g, {k: v - common.get(k, 0) for k, v in fa.items()}
            )
            b2 = self.mul_factors(
                cb // g, {k: v - common.get(k, 0) for k, v in fb.items()}
            )
            return self.floordiv(a2, b2)  # no common factor is left: one level
        return self.mk("floordiv", (a, b), a.hint // b.hint)

    def ceildiv(self, a: Node, b: Node) -> Node:
        # ceil(a / b) of integers: its own kind, as the sympy path keeps
        # CeilToInt(IntTrueDiv(a, b)) apart from FloorDiv
        if b.op == "const" and b.args[0] == 0:
            raise ZeroDivisionError("ceildiv by zero")
        if a.op == "const" and b.op == "const":
            return self.const(-(-a.args[0] // b.args[0]))
        if b.op == "const" and b.args[0] == 1:
            return a
        return self.mk("ceildiv", (a, b), -(-a.hint // b.hint))

    def int_ratio(self, f: Node) -> tuple[Node, Node] | None:
        # the integer operands of a float division of integer-valued operands
        # (what int_truediv builds), else None
        if f.op != "fdiv":
            return None
        out = []
        for x in f.args:
            if x.op == "ffromint":
                out.append(x.args[0])
            elif x.op == "fconst" and x.hint == int(x.hint):
                out.append(self.const(int(x.hint)))
            else:
                return None
        return out[0], out[1]

    def mod(self, a: Node, b: Node) -> Node:
        if b.op == "const" and b.args[0] == 0:
            raise ZeroDivisionError("mod by zero")
        if a.op == "const" and b.op == "const":
            return self.const(a.args[0] % b.args[0])
        if b.op == "const" and abs(b.args[0]) == 1:
            return self.const(0)
        if a.op == "const" and a.args[0] == 0:
            return a
        if self._below(a, b):
            return a  # 0 <= a < b by the declared domains
        ca, fa = self.as_factors(a)
        cb, fb = self.as_factors(b)
        if cb != 0 and ca % cb == 0 and all(fa.get(k, 0) >= v for k, v in fb.items()):
            return self.const(0)  # a multiple of the divisor
        if b.op == "const" and a.op == "add":
            # a term that is a multiple of a constant divisor drops out
            c, ts = a.args
            d = b.args[0]
            kept = {t: cf for t, cf in ts if cf % d != 0}
            if len(kept) < len(ts) or c % d != c:
                return self.mod(self.add_terms(c % d, kept), b)
        return self.mk("mod", (a, b), a.hint % b.hint)

    def _minmax(self, op: str, items: list) -> Node:
        flat: dict = {}
        consts: list = []
        stack = list(items)
        while stack:
            x = stack.pop()
            if x.op == op:
                stack.extend(x.args)
            elif x.op == "const":
                consts.append(x.args[0])
            else:
                flat[x.id] = x
        args = sorted(flat.values(), key=lambda n: n.id)
        if consts:
            args.append(self.const((min if op == "min" else max)(consts)))
        if len(args) > 1:
            # an argument the declared domains put on the losing side of
            # another leaves (max(s, 0) is s for a size), as torch's Max / Min
            # decide from assumptions
            bs = [self.bounds(a) for a in args]
            keep = []
            for i, a in enumerate(args):
                lo_i, hi_i = bs[i]
                dominated = False
                for j, (lo_j, hi_j) in enumerate(bs):
                    if j == i:
                        continue
                    if op == "max":
                        dominated = (
                            lo_j is not None and hi_i is not None and lo_j >= hi_i
                        )
                    else:
                        dominated = (
                            hi_j is not None and lo_i is not None and hi_j <= lo_i
                        )
                    if dominated and (bs[j] != bs[i] or j < i):
                        break
                    dominated = False
                if not dominated:
                    keep.append(a)
            args = keep
        if len(args) == 1:
            return args[0]
        f = min if op == "min" else max
        return self.mk(op, tuple(args), f(n.hint for n in args))

    def min(self, *items: Node) -> Node:
        return self._minmax("min", list(items))

    def max(self, *items: Node) -> Node:
        return self._minmax("max", list(items))

    def nod(self, sizes: list, strides: list) -> Node:
        # the non-overlapping-and-dense indicator over sizes and strides, kept
        # as one node like torch's IsNonOverlappingAndDenseIndicator (its
        # constant short circuits included); a host that reads it guards on it
        dim = len(sizes)
        hint = int(_dense([s.hint for s in sizes], [s.hint for s in strides]))
        if dim == 0 or all(n.op == "const" for n in (*sizes, *strides)):
            return self.const(hint)
        if dim == 1:
            st, sz = strides[0], sizes[0]
            if (st.op == "const" and st.args[0] == 1) or (
                sz.op == "const" and sz.args[0] < 2
            ):
                return self.const(1)
        return self.mk("nod", (*sizes, *strides), hint)

    # ---- the functions a traced host builds (torch.cuda._host_trace)
    def bitwise(self, op: str, a: Node, b: Node) -> Node:
        f = _BITWISE[op][0]
        if a.op == "const" and b.op == "const":
            return self.const(f(a.args[0], b.args[0]))
        return self.mk(op, (a, b), f(a.hint, b.hint))

    def bitlen(self, a: Node) -> Node:
        if a.op == "const":
            return self.const(a.args[0].bit_length())
        return self.mk("bitlen", (a,), a.hint.bit_length())

    def f32div(self, a: Node, b: Node) -> Node:
        if a.op == "const" and b.op == "const" and b.args[0] > 0:
            return self.const(f32_bits(a.args[0], b.args[0]))
        return self.mk("f32div", (a, b), f32_bits(a.hint, b.hint))

    def where(self, c: Node, a: Node, b: Node) -> Node:
        if c.op in ("true", "false"):
            return a if c.op == "true" else b
        return self.mk("where", (c, a, b), a.hint if c.hint else b.hint)

    # ---- relations: rel(d, 0) with d = lhs - rhs in add form
    def rel(self, op: str, d: Node) -> Node:
        if d.op == "const":
            v = d.args[0]
            return self.boolean(
                {"eq": v == 0, "ne": v != 0, "lt": v < 0, "le": v <= 0}[op]
            )
        lo, hi = self.bounds(d)
        if op in ("eq", "ne"):
            if (lo is not None and lo > 0) or (hi is not None and hi < 0):
                return self.boolean(op == "ne")
        elif op == "lt":
            if lo is not None and lo >= 0:
                return self.false()
            if hi is not None and hi < 0:
                return self.true()
        else:
            if lo is not None and lo > 0:
                return self.false()
            if hi is not None and hi <= 0:
                return self.true()
        c, ts = self.as_terms(d)
        # the common integer factor of the terms and the constant leaves (an
        # exact division of both sides; torch's _reduce_to_lowest_terms); an
        # equality whose terms' factor does not divide the constant has no
        # integer solution (2*q == 1), as sympy's parity assumptions decide
        g = math.gcd(c, *ts.values())
        if g > 1:
            c //= g
            ts = {t: cf // g for t, cf in ts.items()}
            d = self.add_terms(c, ts)
        elif op in ("eq", "ne") and ts and c % math.gcd(*ts.values()) != 0:
            return self.boolean(op == "ne")
        lead = ts[min(ts, key=lambda n: n.id)] if ts else c
        if op in ("eq", "ne") and lead < 0:
            d = self.neg(d)
        h = d.hint
        hint = {"eq": h == 0, "ne": h != 0, "lt": h < 0, "le": h <= 0}[op]
        return self.mk(op, (d,), bool(hint))

    def _written(self, r: Node, cls: str, a: Node, b: Node) -> Node:
        if r.op not in ("true", "false"):
            self.written.setdefault(r.id, (cls, a, b))
        return r

    def _compare(self, cls: str, a: Node, b: Node) -> Node:
        if self.extra_bounds:
            return self._compare_new(cls, a, b)
        key = (cls, a, b)
        r = self.memo.get(key)
        if r is None:
            r = self.memo[key] = self._compare_new(cls, a, b)
        return r

    def _compare_new(self, cls: str, a: Node, b: Node) -> Node:
        if a.is_float or b.is_float:
            rel, x, y = {
                "Eq": ("eq", a, b),
                "Ne": ("ne", a, b),
                "Lt": ("lt", a, b),
                "Le": ("le", a, b),
                "Gt": ("lt", b, a),
                "Ge": ("le", b, a),
            }[cls]
            r = self.fcmp(rel, x, y)
        elif cls in ("Gt", "Ge"):
            r = self.rel("lt" if cls == "Gt" else "le", self.sub(b, a))
        else:
            r = self.rel(cls.lower(), self.sub(a, b))
        return self._written(r, cls, a, b)

    def eq(self, a: Node, b: Node) -> Node:
        return self._compare("Eq", a, b)

    def ne(self, a: Node, b: Node) -> Node:
        return self._compare("Ne", a, b)

    def lt(self, a: Node, b: Node) -> Node:
        return self._compare("Lt", a, b)

    def le(self, a: Node, b: Node) -> Node:
        return self._compare("Le", a, b)

    def gt(self, a: Node, b: Node) -> Node:
        return self._compare("Gt", a, b)

    def ge(self, a: Node, b: Node) -> Node:
        return self._compare("Ge", a, b)

    def fcmp(self, rel: str, a: Node, b: Node) -> Node:
        a, b = self.to_float(a), self.to_float(b)
        x, y = a.hint, b.hint
        hint = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y}[rel]
        if a.op == "fconst" and b.op == "fconst":
            return self.boolean(hint)
        return self.mk("fcmp", (rel, a, b), bool(hint))

    def not_(self, a: Node) -> Node:
        r = self._not(a)
        w = self.written.get(a.id)
        if w is not None:
            self._written(r, _NEGATED[w[0]], w[1], w[2])
        return r

    def _not(self, a: Node) -> Node:
        op = a.op
        if op == "true":
            return self.false()
        if op == "false":
            return self.true()
        if op == "eq":
            return self.mk("ne", a.args, not a.hint)
        if op == "ne":
            return self.mk("eq", a.args, not a.hint)
        if op == "lt":  # not (d < 0) is -d <= 0
            return self.rel("le", self.neg(a.args[0]))
        if op == "le":
            return self.rel("lt", self.neg(a.args[0]))
        if op == "not":
            return a.args[0]
        if op == "fcmp":
            rel, x, y = a.args
            inv = {
                "eq": ("ne", x, y),
                "ne": ("eq", x, y),
                "lt": ("le", y, x),
                "le": ("lt", y, x),
            }
            return self.mk("fcmp", inv[rel], not a.hint)
        return self.mk("not", (a,), not a.hint)

    def junction(self, op: str, items: list) -> Node:
        absorb, unit = ("false", "true") if op == "and" else ("true", "false")
        flat: dict = {}
        stack = list(items)
        while stack:
            x = stack.pop()
            if x.op == op:
                stack.extend(x.args)
            elif x.op == absorb:
                return self.boolean(op != "and")
            elif x.op != unit:
                flat[x.id] = x
        args = sorted(flat.values(), key=lambda n: n.id)
        if not args:
            return self.boolean(op == "and")
        if len(args) == 1:
            return args[0]
        hint = all(n.hint for n in args) if op == "and" else any(n.hint for n in args)
        return self.mk(op, tuple(args), bool(hint))

    def and_(self, a: Node, b: Node) -> Node:
        return self.junction("and", [a, b])

    def or_(self, a: Node, b: Node) -> Node:
        return self.junction("or", [a, b])

    # ---- the float lane: one node per operation in the host's order
    def to_float(self, a: Node) -> Node:
        if a.is_float:
            return a
        if a.op == "const":
            return self.fconst(float(a.args[0]))
        return self.mk("ffromint", (a,), float(a.hint))

    def fbin(self, op: str, a: Node, b: Node) -> Node:
        a, b = self.to_float(a), self.to_float(b)
        x, y = a.hint, b.hint
        if op == "fadd":
            hint = x + y
        elif op == "fsub":
            hint = x - y
        elif op == "fmul":
            hint = x * y
        elif op == "fdiv":
            hint = x / y
        else:
            hint = x**y
        return self.mk(op, (a, b), float(hint))

    def fun(self, op: str, a: Node) -> Node:
        a = self.to_float(a)
        hint = -a.hint if op == "fneg" else math.sqrt(a.hint)
        return self.mk(op, (a,), float(hint))

    def fint(self, op: str, a: Node) -> Node:
        # int() / math.floor / math.ceil of a float: an integer node over a
        # float operand (the ratio of integers takes floordiv / ceildiv)
        f = {"ftrunc": int, "ffloor": math.floor, "fceil": math.ceil}[op]
        if a.op == "fconst":
            return self.const(f(a.hint))
        return self.mk(op, (a,), f(a.hint))


def render(n: Node) -> str:
    """Python-syntax text of a node, for debugging."""
    op = n.op
    if op == "const":
        return str(n.args[0])
    if op in ("sym", "fsym"):
        return n.args[0]
    if op == "add":
        c, ts = n.args
        parts = [f"{cf}*{render(t)}" if cf != 1 else render(t) for t, cf in ts]
        if c:
            parts.append(str(c))
        return "(" + " + ".join(parts) + ")"
    if op == "mul":
        c, fs = n.args
        parts = [f"{render(f)}**{e}" if e != 1 else render(f) for f, e in fs]
        if c != 1:
            parts.insert(0, str(c))
        return "(" + "*".join(parts) + ")"
    if op == "floordiv":
        return f"({render(n.args[0])} // {render(n.args[1])})"
    if op == "mod":
        return f"({render(n.args[0])} % {render(n.args[1])})"
    if op == "pbn":
        return f"({render(n.args[0])} ** {render(n.args[1])})"
    if op in _REL_TEXT:
        return f"({render(n.args[0])} {_REL_TEXT[op]} 0)"
    if op in ("and", "or"):
        return "(" + f" {op} ".join(render(a) for a in n.args) + ")"
    if op == "not":
        return f"(not {render(n.args[0])})"
    if op in ("true", "false"):
        return op.capitalize()
    if op == "fconst":
        return repr(n.hint)
    if op == "fcmp":
        return f"({render(n.args[1])} {_REL_TEXT[n.args[0]]} {render(n.args[2])})"
    if op in ("fadd", "fsub", "fmul", "fdiv", "fpow"):
        sym = {"fadd": "+", "fsub": "-", "fmul": "*", "fdiv": "/", "fpow": "**"}[op]
        return f"({render(n.args[0])} {sym} {render(n.args[1])})"
    return f"{op}(" + ", ".join(render(a) for a in children(n)) + ")"


class Env:
    """One trace's IR context, symbols, guard record and census: what
    `_TraceShapeEnv` is to the sympy backend, with the sympy views of it the
    tape's consumers read."""

    # a key built from sizes (a compile cache's) is a specialization
    hash_symints_by_value = True
    # a static read decides as eager's concrete value does (_TraceShapeEnv's)
    static_reads_guard = True
    # what torch's helpers ask of a shape env
    _translation_validation_enabled = False
    _replacements_version_counter = 0

    def __init__(self) -> None:
        self.ctx = Ctx()
        # the raw guard record in evaluation order: (node, written form or None)
        self.records: list[tuple[Node, tuple | None]] = []
        self._index: dict[Any, int] = {}  # a record's key -> its index
        # per record, the top-level op that recorded it, or None: graph-level
        # (_TraceShapeEnv.owners)
        self.owners: list[int | None] = []
        # the op a guard recorded now belongs to
        self.op: int | None = None
        self.names: dict[str, str] = {}  # symbol name -> source name, in creation order
        self.unique_ids: set[int] = set()
        # taken id p -> q: every id in [p, q) is taken, so a probe skips the
        # run; a merge into unique_ids only adds taken ids, so a skip stays valid
        self._id_skips: dict[int, int] = {}
        self.census: list[tuple[str, str]] = []  # (op, site) of every Unsupported
        self._export: _SympyExport | None = None

    @property
    def export(self) -> _SympyExport:
        if self._export is None:
            self._export = _SympyExport(self)
        return self._export

    # ---- symbols, named as the ShapeEnv names them (s<id>, zf<id>; the id
    # from the source name, so both backends name a call's symbols alike)
    def _unique_id(self, source: str) -> int:
        attempt = int(hashlib.sha256(source.encode()).hexdigest(), 16) % 100
        path = []
        while attempt in self.unique_ids:
            path.append(attempt)
            attempt = self._id_skips.get(attempt, attempt + 1)
        for p in path:
            self._id_skips[p] = attempt + 1
        self.unique_ids.add(attempt)
        return attempt

    def symbol(self, value: int | float, source: str, *, positive: bool = False) -> Any:
        k = self._unique_id(source)
        if isinstance(value, float):
            name = f"zf{k}"
            out: Any = torch.SymFloat(IRSymNode(self.ctx.fsym(name, value), self, float))
        else:
            name = f"s{k}"
            out = torch.SymInt(IRSymNode(self.ctx.sym(name, value, positive), self, int))
        self.names[name] = source
        return out

    # ---- the guard record
    def _record(self, g: Node, written: tuple | None) -> None:
        # deduped as _TraceShapeEnv dedupes sympy: a relation by the form it
        # was written in (Eq(a, b) and Eq(b, a) are two guards), the rest by node
        if g.op == "true":
            return
        if written is None:
            written = self.ctx.written.get(g.id)
        key = _record_key(g, written)
        i = self._index.get(key)
        if i is None:
            self._index[key] = len(self.records)
            self.records.append((g, written))
            self.owners.append(self.op)
        elif self.owners[i] != self.op:
            self.owners[i] = None

    def forget(self, n: int) -> None:
        """Drops the guards recorded after the first n."""
        for g, written in self.records[n:]:
            del self._index[_record_key(g, written)]
        del self.records[n:], self.owners[n:]
        if self._export is not None:
            del self._export._guards[n:]

    def guard_bool(self, g: Node, written: tuple | None = None) -> bool:
        if g.op in ("true", "false"):
            return g.op == "true"
        hint = bool(g.hint)
        if hint:
            self._record(g, written)
        else:
            self._record(self.ctx.not_(g), negated(written))
        return hint

    def guard_value(self, n: Node) -> Any:
        if n.op in ("const", "fconst"):
            return n.hint
        c = self.ctx.fconst(n.hint) if n.is_float else self.ctx.const(n.hint)
        self._record(self.ctx.eq(n, c), ("Eq", n, c))
        return n.hint

    def domain(self, divisor: Node) -> None:
        # a partial operation's domain, recorded when the operation is created
        # unless the declared domains decide it
        if divisor.hint == 0:
            raise ZeroDivisionError("division by zero")
        zero = self.ctx.const(0)
        self._record(self.ctx.ne(divisor, zero), ("Ne", divisor, zero))

    def unsupported(self, op: str) -> Unsupported:
        e = Unsupported(op, _site())
        self.census.append((e.op, e.site))
        return e

    # ---- what torch's helpers ask of a shape env
    def evaluate_sym_node(
        self, sym_node: Any, size_oblivious: bool = False, fallback_value: Any = None
    ) -> Any:
        return sym_node.evaluate(size_oblivious)

    def _maybe_evaluate_static(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        if expr is sympy.true or expr is sympy.false or expr.is_number:
            return expr
        return None

    def replace(self, e: Any) -> Any:
        return e

    # ---- the sympy views (_TraceShapeEnv's attributes of the same names)
    @property
    def guards(self) -> list[ShapeGuard]:
        return self.export.guards()

    @property
    def var_to_sources(self) -> dict:
        return self.export.symbol_table()[0]

    @property
    def backed_var_to_val(self) -> dict:
        return self.export.symbol_table()[1]

    @property
    def var_to_range(self) -> dict:
        return self.export.symbol_table()[2]


def _record_key(g: Node, written: tuple | None) -> Any:
    return g.id if written is None else (written[0], written[1].id, written[2].id)


def negated(written: tuple | None) -> tuple | None:
    return None if written is None else (_NEGATED[written[0]], *written[1:])


class IRSymNode:
    """The SymNode-shaped backend over an IR node: what torch.SymInt /
    SymFloat / SymBool and c10's PythonSymNodeImpl call."""

    __slots__ = ("node", "env", "pytype", "constant", "written", "__weakref__")
    fx_node = None
    _optimized_summation = False

    def __init__(
        self,
        node: Node,
        env: Env,
        pytype: type,
        constant: Any = None,
        written: tuple | None = None,
    ) -> None:
        self.node = node
        self.env = env
        self.pytype = pytype
        self.constant = constant
        # a relation's (sympy class name, lhs, rhs) as the host wrote it
        self.written = written

    # ---- what the tape's consumers read: sympy, exported on first read
    @property
    def expr(self) -> Any:
        if self.written is not None:
            return self.env.export.relation(self.written)
        return self.env.export.expr(self.node)

    _expr = expr

    @property
    def hint(self) -> Any:
        return self.node.hint

    _hint = hint

    @property
    def shape_env(self) -> Env:
        return self.env

    def symbol_name(self) -> str | None:
        return self.node.args[0] if self.node.op in ("sym", "fsym") else None

    def has_hint(self) -> bool:
        return True

    def require_hint(self, fallback: Any = None) -> Any:
        return self.node.hint

    def is_int(self) -> bool:
        return self.pytype is int

    def is_float(self) -> bool:
        return self.pytype is float

    def is_bool(self) -> bool:
        return self.pytype is bool

    def is_nested_int(self) -> bool:
        return False

    def nested_int(self) -> None:
        return None

    def is_constant(self) -> bool:
        return self.constant is not None

    def is_symbolic(self) -> bool:
        return self.constant is None

    def maybe_as_int(self) -> int | None:
        return self.node.args[0] if self.node.op == "const" else None

    def maybe_as_float(self) -> float | None:
        return self.node.hint if self.node.op == "fconst" else None

    def maybe_as_bool(self) -> bool | None:
        return {"true": True, "false": False}.get(self.node.op)

    def constant_int(self) -> int | None:
        return self.maybe_as_int()

    def constant_bool(self) -> bool | None:
        return self.maybe_as_bool()

    def str(self) -> str:
        return str(self.expr)

    def __str__(self) -> str:
        return str(self.expr)

    def __repr__(self) -> str:
        return f"IRSymNode({render(self.node)}, pytype={self.pytype.__name__}, hint={self.node.hint})"

    def _graph_repr(self) -> str:
        return str(self.expr)

    def _value_eq(self, other: Any) -> bool:
        return (
            isinstance(other, IRSymNode)
            and other.node is self.node
            and other.pytype is self.pytype
        )

    def _value_hash(self) -> int:
        return hash((self.node.id, self.pytype))

    def clone(self) -> IRSymNode:
        return self

    def with_shape_env(self, env: Any) -> IRSymNode:
        return self

    def wrap_int(self, num: int) -> IRSymNode:
        return IRSymNode(self.env.ctx.const(num), self.env, int, constant=num)

    def wrap_float(self, num: float) -> IRSymNode:
        return IRSymNode(self.env.ctx.fconst(num), self.env, float, constant=num)

    def wrap_bool(self, num: bool) -> IRSymNode:
        return IRSymNode(self.env.ctx.boolean(num), self.env, bool, constant=num)

    # ---- arithmetic
    def _out(self, n: Node) -> IRSymNode:
        return IRSymNode(
            n, self.env, float if n.is_float else bool if n.is_bool else int
        )

    def _int(self, n: Node) -> IRSymNode:
        return IRSymNode(n, self.env, int)

    def _flt(self, n: Node) -> IRSymNode:
        return IRSymNode(n, self.env, float)

    def _bool(self, n: Node) -> IRSymNode:
        return IRSymNode(n, self.env, bool)

    def _ints(self, other: IRSymNode, what: str) -> tuple[Node, Node]:
        a, b = self.node, other.node
        if not (a.is_int and b.is_int):
            raise self.env.unsupported(f"{what} of a non-integer")
        return a, b

    def add(self, other: IRSymNode) -> IRSymNode:
        c, a, b = self.env.ctx, self.node, other.node
        return self._out(
            c.fbin("fadd", a, b) if (a.is_float or b.is_float) else c.add(a, b)
        )

    def sub(self, other: IRSymNode) -> IRSymNode:
        c, a, b = self.env.ctx, self.node, other.node
        return self._out(
            c.fbin("fsub", a, b) if (a.is_float or b.is_float) else c.sub(a, b)
        )

    def mul(self, other: IRSymNode) -> IRSymNode:
        c, a, b = self.env.ctx, self.node, other.node
        return self._out(
            c.fbin("fmul", a, b) if (a.is_float or b.is_float) else c.mul(a, b)
        )

    def neg(self) -> IRSymNode:
        c, a = self.env.ctx, self.node
        return self._out(c.fun("fneg", a) if a.is_float else c.neg(a))

    def pos(self) -> IRSymNode:
        return self

    def abs(self) -> IRSymNode:
        n = self.node
        if n.is_float:
            raise self.env.unsupported("abs of a float")
        return self._int(self.env.ctx.max(n, self.env.ctx.neg(n)))

    def mod(self, other: IRSymNode) -> IRSymNode:
        a, b = self._ints(other, "mod")
        self.env.domain(b)
        return self._int(self.env.ctx.mod(a, b))

    def int_floordiv(self, other: IRSymNode) -> IRSymNode:
        a, b = self._ints(other, "floordiv")
        self.env.domain(b)
        return self._int(self.env.ctx.floordiv(a, b))

    floordiv = int_floordiv

    def float_truediv(self, other: IRSymNode) -> IRSymNode:
        self.env.domain(other.node)
        return self._flt(self.env.ctx.fbin("fdiv", self.node, other.node))

    int_truediv = float_truediv
    truediv = float_truediv

    def pow_by_natural(self, other: IRSymNode) -> IRSymNode:
        a, e = self._ints(other, "pow_by_natural")
        if e.op == "const":
            return self._int(self.env.ctx.pow(a, e.args[0]))
        if a.op == "const":
            return self._int(self.env.ctx.pbn(a.args[0], e))
        raise self.env.unsupported("a symbolic base to a symbolic exponent")

    def float_pow(self, other: IRSymNode) -> IRSymNode:
        return self._flt(self.env.ctx.fbin("fpow", self.node, other.node))

    # c10 binds no integer pow (sym_node.py's SymNode.pow)
    pow = float_pow

    def lshift(self, other: IRSymNode) -> IRSymNode:
        # LShift: base * PowByNatural(2, shift)
        a, s = self._ints(other, "lshift")
        if s.op == "const" and s.args[0] < 0:
            raise ValueError("negative shift count")
        c = self.env.ctx
        return self._int(c.mul(a, c.pbn(2, s)))

    def rshift(self, other: IRSymNode) -> IRSymNode:
        # RShift: FloorDiv(base, PowByNatural(2, shift)), no domain guard
        a, s = self._ints(other, "rshift")
        if s.op == "const" and s.args[0] < 0:
            raise ValueError("negative shift count")
        c = self.env.ctx
        return self._int(c.floordiv(a, c.pbn(2, s)))

    def _bitwise(self, op: str, other: IRSymNode) -> IRSymNode:
        a, b = self._ints(other, op)
        return self._int(self.env.ctx.bitwise(op, a, b))

    def bitwise_and(self, other: IRSymNode) -> IRSymNode:
        return self._bitwise("bitand", other)

    def bitwise_or(self, other: IRSymNode) -> IRSymNode:
        return self._bitwise("bitor", other)

    def bitwise_xor(self, other: IRSymNode) -> IRSymNode:
        return self._bitwise("bitxor", other)

    def sym_min(self, other: IRSymNode) -> IRSymNode:
        a, b = self._ints(other, "min")
        return self._int(self.env.ctx.min(a, b))

    def sym_max(self, other: IRSymNode) -> IRSymNode:
        a, b = self._ints(other, "max")
        return self._int(self.env.ctx.max(a, b))

    def sym_sum(self, args: list) -> IRSymNode:
        acc: dict = {}
        c = 0
        for a in args:
            ca, ts = self.env.ctx.as_terms(a.node)
            c += ca
            for t, cf in ts.items():
                acc[t] = acc.get(t, 0) + cf
        return self._int(self.env.ctx.add_terms(c, acc))

    def sym_float(self) -> IRSymNode:
        return self._flt(self.env.ctx.to_float(self.node))

    def sym_int(self) -> IRSymNode:
        return self._int(self.env.ctx.fint("ftrunc", self.node)) if self.node.is_float else self

    trunc = sym_int

    def floor(self) -> IRSymNode:
        if not self.node.is_float:
            return self
        ratio = self.env.ctx.int_ratio(self.node)
        if ratio is not None:
            return self._int(self.env.ctx.floordiv(*ratio))
        return self._int(self.env.ctx.fint("ffloor", self.node))

    def ceil(self) -> IRSymNode:
        if not self.node.is_float:
            return self
        ratio = self.env.ctx.int_ratio(self.node)
        if ratio is not None:
            return self._int(self.env.ctx.ceildiv(*ratio))
        return self._int(self.env.ctx.fint("fceil", self.node))

    def sym_sqrt(self) -> IRSymNode:
        return self._flt(self.env.ctx.fun("fsqrt", self.node))

    def sym_ite(self, t: IRSymNode, f: IRSymNode) -> IRSymNode:
        if self.node.op == "true":
            return t
        if self.node.op == "false":
            return f
        raise self.env.unsupported("sym_ite on a symbolic condition")

    def __getattr__(self, name: str) -> Any:
        # an attribute a SymNode has that this backend lacks (the math
        # functions, round, is_integer): named and counted
        if name.startswith("__"):
            raise AttributeError(name)
        raise self.env.unsupported(name)

    # ---- torch.cuda._host_trace's functions of a traced value
    def bit_length(self) -> torch.SymInt:
        return torch.SymInt(self._int(self.env.ctx.bitlen(self.node)))

    def select(self, a: Any, b: Any) -> torch.SymInt:
        a, b = (self._leaf(x) for x in (a, b))
        return torch.SymInt(self._int(self.env.ctx.where(self.node, a, b)))

    def f32_div(self, a: Any, b: Any) -> torch.SymInt:
        a, b = (self._leaf(x) for x in (a, b))
        return torch.SymInt(self._int(self.env.ctx.f32div(a, b)))

    def _leaf(self, x: Any) -> Node:
        return self.env.ctx.const(x) if isinstance(x, int) else x.node.node

    # ---- relations and boolean algebra
    def _rel(self, cls: str, other: IRSymNode) -> IRSymNode:
        a, b = self.node, other.node
        r = self.env.ctx._compare(cls, a, b)
        written = None if r.op in ("true", "false") else (cls, a, b)
        return IRSymNode(r, self.env, bool, written=written)

    def eq(self, other: IRSymNode) -> IRSymNode:
        return self._rel("Eq", other)

    def ne(self, other: IRSymNode) -> IRSymNode:
        return self._rel("Ne", other)

    def lt(self, other: IRSymNode) -> IRSymNode:
        return self._rel("Lt", other)

    def le(self, other: IRSymNode) -> IRSymNode:
        return self._rel("Le", other)

    def gt(self, other: IRSymNode) -> IRSymNode:
        return self._rel("Gt", other)

    def ge(self, other: IRSymNode) -> IRSymNode:
        return self._rel("Ge", other)

    def _junction(self, r: Node, other: IRSymNode) -> IRSymNode:
        # True & x is x as written (sym_eq's reduce), as sympy's And keeps it
        written = self.written if r is self.node else other.written if r is other.node else None
        return IRSymNode(r, self.env, bool, written=written)

    def sym_and(self, other: IRSymNode) -> IRSymNode:
        return self._junction(self.env.ctx.and_(self.node, other.node), other)

    def sym_or(self, other: IRSymNode) -> IRSymNode:
        return self._junction(self.env.ctx.or_(self.node, other.node), other)

    def sym_not(self) -> IRSymNode:
        r = self.env.ctx.not_(self.node)
        written = None if r.op in ("true", "false") else negated(self.written)
        return IRSymNode(r, self.env, bool, written=written)

    and_ = sym_and
    or_ = sym_or

    # ---- guard reads
    def guard_bool(self, file: Any = "", line: Any = 0) -> bool:
        return self.env.guard_bool(self.node, self.written)

    def guard_int(self, file: Any = "", line: Any = 0) -> int:
        return int(self.env.guard_value(self.node))

    def guard_float(self, file: Any = "", line: Any = 0) -> float:
        return float(self.env.guard_value(self.node))

    expect_true = guard_bool
    statically_known_true = guard_bool
    guard_size_oblivious = guard_bool
    guard_or_false = guard_bool
    guard_or_true = guard_bool

    def expect_size(self, file: Any = "", line: Any = 0) -> bool:
        return self.ge(self.wrap_int(0)).guard_bool()

    def bool_(self) -> bool:
        return self.env.guard_bool(self.node, self.written)

    def int_(self) -> int:
        if self.node.op != "const" and (use := _len_compare()) is not None:
            getattr(self, use[0])(self.wrap_int(use[1])).guard_bool()
            return self.node.hint
        return int(self.env.guard_value(self.node))

    def evaluate(self, size_oblivious: bool = False) -> Any:
        if self.pytype is bool:
            return self.env.guard_bool(self.node, self.written)
        return self.env.guard_value(self.node)

    # ---- the sizes / strides predicates (sym_node.py's sympy versions,
    # over IR nodes; C++ asks them of the first symbolic size or stride)
    def _contig(self, sizes: list, strides: list, order: list) -> IRSymNode:
        c = self.env.ctx
        if len(order) != len(sizes):
            return self._bool(c.false())
        one, z, terms = c.const(1), c.const(1), []
        for d in order:
            terms.append(c.or_(c.eq(sizes[d].node, one), c.eq(strides[d].node, z)))
            z = c.mul(z, sizes[d].node)
        r = c.junction("and", terms) if terms else c.true()
        zero = c.const(0)
        for s in sizes:
            r = c.or_(r, c.eq(s.node, zero))
        return self._bool(r)

    def is_contiguous(self, sizes: list, strides: list) -> IRSymNode:
        return self._contig(sizes, strides, list(range(len(sizes) - 1, -1, -1)))

    def is_channels_last_contiguous_2d(self, sizes: list, strides: list) -> IRSymNode:
        return self._contig(sizes, strides, [1, 3, 2, 0])

    def is_channels_last_contiguous_3d(self, sizes: list, strides: list) -> IRSymNode:
        return self._contig(sizes, strides, [1, 4, 3, 2, 0])

    def _cl_strides(self, sizes: list, strides: list, order: list) -> IRSymNode:
        c = self.env.ctx
        if len(order) != len(sizes):
            return self._bool(c.false())
        zero, one = c.const(0), c.const(1)
        m = zero
        r = c.ne(strides[1].node, zero)
        for d in order:
            r = c.and_(r, c.and_(c.ne(sizes[d].node, zero), c.ge(strides[d].node, m)))
            if d == 0:
                r = c.and_(r, c.ne(m, strides[1].node))
            m = c.mul(strides[d].node, c.max(sizes[d].node, one))
        return self._bool(r)

    def is_channels_last_strides_2d(self, sizes: list, strides: list) -> IRSymNode:
        return self._cl_strides(sizes, strides, [1, 3, 2, 0])

    def is_channels_last_strides_3d(self, sizes: list, strides: list) -> IRSymNode:
        return self._cl_strides(sizes, strides, [1, 4, 3, 2, 0])

    def is_non_overlapping_and_dense_indicator(
        self, sizes: list, strides: list
    ) -> IRSymNode:
        return self._int(
            self.env.ctx.nod([s.node for s in sizes], [s.node for s in strides])
        )

    def is_non_overlapping_and_dense(self, sizes: list, strides: list) -> IRSymNode:
        ind = self.is_non_overlapping_and_dense_indicator(sizes, strides).node
        return self._bool(self.env.ctx.eq(ind, self.env.ctx.const(1)))


class _SympyExport:
    """An IR trace's symbols, guards and values as sympy, built on first read:
    symbols with the names, assumptions, hints and sources the ShapeEnv gives
    them, a relation in the form the host first wrote it (unevaluated: the IR
    decided what the declared domains decide), everything else through
    sympy's and torch's own constructors."""

    def __init__(self, env: Env) -> None:
        self.env = env
        self.symbols: dict[str, sympy.Symbol] = {}
        self._exprs: dict[int, Any] = {}
        self._relations: dict[tuple, Any] = {}
        self._guards: list[ShapeGuard] = []
        self._table: tuple = ({}, {}, {})
        self._size_range = ValueRanges(1, int_oo)
        self._int_range = ValueRanges.unknown_int()
        self._float_range = ValueRanges(-sympy.oo, sympy.oo)

    def symbol(self, name: str) -> sympy.Symbol:
        s = self.symbols.get(name)
        if s is None:
            node = self.env.ctx.symbols[name]
            if node.is_float:
                s = sympy.Symbol(name, real=True)
            else:
                positive = name in self.env.ctx.positive
                s = sympy.Symbol(name, integer=True, positive=positive or None)
            self.symbols[name] = s
        return s

    def symbol_table(self) -> tuple[dict, dict, dict]:
        # (var_to_sources, backed_var_to_val, var_to_range) over every symbol
        from torch.cuda._host_trace import _Src

        sources, values, ranges = self._table
        if len(sources) != len(self.env.names):
            ctx = self.env.ctx
            for name, source in list(self.env.names.items())[len(sources) :]:
                s = self.symbol(name)
                node = ctx.symbols[name]
                sources[s] = [_Src(source)]
                if node.is_float:
                    values[s] = sympy.Float(node.hint)
                    ranges[s] = self._float_range
                else:
                    values[s] = sympy.Integer(node.hint)
                    ranges[s] = self._size_range if name in ctx.positive else self._int_range
        return self._table

    def guards(self) -> list[ShapeGuard]:
        records = self.env.records
        for g, written in records[len(self._guards) :]:
            e = self.expr(g) if written is None else self.relation(written)
            self._guards.append(ShapeGuard(e, _NO_SLOC, False))
        return self._guards

    def relation(self, written: tuple) -> Any:
        cls, a, b = written
        key = (cls, a.id, b.id)
        r = self._relations.get(key)
        if r is None:
            r = _REL_CLASSES[cls](self.expr(a), self.expr(b), evaluate=False)
            self._relations[key] = r
        return r

    def expr(self, n: Node) -> Any:
        r = self._exprs.get(n.id)
        if r is None:
            r = self._expr(n)
            self._exprs[n.id] = r
        return r

    def _sides(self, d: Node) -> tuple:
        # rel(d, 0) as lhs rel rhs: the positive terms against the negated rest
        c, ts = self.env.ctx.as_terms(d)
        lhs = [sympy.Integer(cf) * self.expr(t) for t, cf in ts.items() if cf > 0]
        rhs = [sympy.Integer(-cf) * self.expr(t) for t, cf in ts.items() if cf < 0]
        if c > 0:
            lhs.append(sympy.Integer(c))
        elif c < 0:
            rhs.append(sympy.Integer(-c))
        return sympy.Add(*lhs), sympy.Add(*rhs)

    def _expr(self, n: Node) -> Any:
        from torch.cuda._host_trace import BitLength, F32Div

        op, e, ctx = n.op, self.expr, self.env.ctx
        if op in ("eq", "ne", "lt", "le", "fcmp"):
            w = ctx.written.get(n.id)
            if w is not None:
                cls, a, b = w
                return _REL_CLASSES[cls](e(a), e(b), evaluate=False)
            if op == "fcmp":
                rel, a, b = n.args
                return _REL_CLASSES[rel.capitalize()](e(a), e(b), evaluate=False)
            lhs, rhs = self._sides(n.args[0])
            return _REL_CLASSES[op.capitalize()](lhs, rhs, evaluate=False)
        if op == "const":
            return sympy.Integer(n.args[0])
        if op in ("sym", "fsym"):
            return self.symbol(n.args[0])
        if op == "add":
            c, ts = n.args
            return sympy.Add(
                sympy.Integer(c), *(sympy.Integer(cf) * e(t) for t, cf in ts)
            )
        if op == "mul":
            c, fs = n.args
            return sympy.Mul(sympy.Integer(c), *(e(f) ** ex for f, ex in fs))
        if op == "floordiv":
            # unevaluated: the IR applied the sound folds
            return FloorDiv(e(n.args[0]), e(n.args[1]), evaluate=False)
        if op == "ceildiv":
            return CeilToInt(IntTrueDiv(e(n.args[0]), e(n.args[1])))
        if op == "mod":
            a, b = n.args
            cls = Mod if ctx.is_nonnegative(a) and ctx.is_nonnegative(b) else PythonMod
            return cls(e(a), e(b))
        if op == "min":
            return Min(*(e(a) for a in n.args))
        if op == "max":
            return Max(*(e(a) for a in n.args))
        if op == "nod":
            return IsNonOverlappingAndDenseIndicator(*(e(a) for a in n.args))
        if op == "pbn":
            return PowByNatural(e(n.args[0]), e(n.args[1]))
        if op == "bitlen":
            return BitLength(e(n.args[0]))
        if op == "f32div":
            return F32Div(e(n.args[0]), e(n.args[1]))
        if op in _BITWISE:
            return _BITWISE[op][1](e(n.args[0]), e(n.args[1]))
        if op == "where":
            return Where(*(e(a) for a in n.args))
        if op == "true":
            return sympy.true
        if op == "false":
            return sympy.false
        if op == "and":
            return sympy.And(*(e(a) for a in n.args))
        if op == "or":
            return sympy.Or(*(e(a) for a in n.args))
        if op == "not":
            return sympy.Not(e(n.args[0]))
        if op == "fconst":
            return sympy.Float(n.hint)
        if op == "ffromint":
            return ToFloat(e(n.args[0]))
        if op == "fadd":
            return e(n.args[0]) + e(n.args[1])
        if op == "fsub":
            return e(n.args[0]) - e(n.args[1])
        if op == "fmul":
            return e(n.args[0]) * e(n.args[1])
        if op == "fneg":
            return -e(n.args[0])
        if op == "fdiv":
            ratio = ctx.int_ratio(n)
            if ratio is not None:
                return IntTrueDiv(e(ratio[0]), e(ratio[1]))
            return FloatTrueDiv(e(n.args[0]), e(n.args[1]))
        if op == "fpow":
            return FloatPow(e(n.args[0]), e(n.args[1]))
        if op == "fsqrt":
            return OpaqueUnaryFn_sqrt(e(n.args[0]))
        if op == "ftrunc":
            return TruncToInt(e(n.args[0]))
        if op == "ffloor":
            return FloorToInt(e(n.args[0]))
        if op == "fceil":
            return CeilToInt(e(n.args[0]))
        raise AssertionError(f"host_trace: no sympy form for the IR node kind {op}")
