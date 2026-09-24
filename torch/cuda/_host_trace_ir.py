"""The integer expression IR behind a host trace's symbolic values (private).

The recorder's values are torch.SymInt / SymFloat / SymBool; the object behind
each one is a SymNode-shaped backend. With `_host_trace.symbolic = "ir"` that
backend is `IRSymNode` below, over hash-consed nodes of a small integer IR,
in place of sympy through torch's ShapeEnv (`_TraceShapeEnv`). The hosts,
TensorIteratorSym and the recorder's C++ are unchanged: c10::SymInt reaches
either backend through PythonSymNodeImpl, whose contract is the method names
IRSymNode implements.

Nodes are immutable and interned per trace (one node per (op, args); equality
is identity; `id` is the creation index), and every node carries its hint,
the value at the traced call. Integer sums and products are flattened
coefficient maps (`add`: constant term plus (node, coefficient) pairs; `mul`:
constant coefficient plus (node, exponent) pairs), so two syntactic forms of
one quantity are one node: the eight integer identities below and interval
bounds from the declared domains (an input size is >= 1, an opaque result
declared positive is >= 1, every other symbol unbounded) are all the
canonicalization a tape needs (DECISIONS O57: 0 of 39,000 recorded
operations on the four reference tapes need more).

  x // 1 = x, x % 1 = 0, 0 // x = 0 % x = 0, constants fold
  a number times a sum distributes (sympy's rule)
  (g*x) // (g*y) = x // y                    g != 0 by the division's domain guard
  (k*d*x + r) // d = k*x + r // d             floor division, any sign of d
  (k*d + y) % d = y % d, (k*b*x) % b = 0     floor modulo
  (x // a) // b = x // (a*b)                  a, b > 0
  min / max flatten, drop duplicates, fold their constants
  a relation is rel(d, 0) with d = lhs - rhs; eq / ne oriented by the sign of
  the leading coefficient, gt / ge written as lt / le, `not` pushed into it

Guards come from the hosts' comparisons exactly as under sympy (E27, A204):
a boolean node the host reads is decided by its hint and recorded once, as
the node or its negation; a value the host reads is recorded as `eq(node,
hint)`; a relation the declared domains decide is a constant and no guard,
as sympy's assumptions decided it; a partial operation records its domain
when it is created (A243). The dedupe and pin passes (A205 / A206) are
restated on the IR in `_Pass`. The float lane keeps the host's operation
order (one node per operation, nothing re-associated), which is what
`Identity` and `Float32` guarantee on the sympy path.

What the IR cannot express raises `Unsupported`, which the trace turns into
a decline naming the operation and the site and counts in the trace's census;
nothing falls back to sympy mid-trace. At the tape boundary `_host_trace`
exports the IR to sympy (`_SympyExport`), so the adapter's lowering and the
runtime's payload contract keep receiving sympy at this stage; the C++ form
of this backend (a SymNodeImpl over the same table) is the next stage.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    import builtins


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
        "fround32",
    }
)
_REL_TEXT = {"eq": "==", "ne": "!=", "lt": "<", "le": "<="}
_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__)) + os.sep


# the attribution of a guard recorded outside any op, or on an env no trace
# owns: no op, kernel-choice depth 0, the Python between ops
_NO_ROW: tuple[int, int, str] = (-1, 0, "python")


class Unsupported(NotImplementedError):
    """An operation the IR backend does not express; the trace declines on
    it by name (op, site) and counts it in its census."""

    def __init__(self, op: str, site: str) -> None:
        super().__init__(f"{op} at {site}")
        self.op = op
        self.site = site


def _site() -> str:
    # the innermost frame outside torch: where the user's code stands
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename.startswith(_TORCH_DIR):
        f = f.f_back
    if f is None:
        return "?"
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"


class Node:
    """One interned expression. `args` name other nodes (never expressions
    of another context); a key of the table is the tuple of their ids."""

    __slots__ = ("id", "op", "args", "hint", "__weakref__")

    def __init__(self, id: int, op: str, args: tuple, hint: Any) -> None:
        self.id = id
        self.op = op
        self.args = args
        self.hint = hint

    def __repr__(self) -> str:
        return render(self)

    def __bool__(self) -> bool:
        if self.op == "true":
            return True
        if self.op == "false":
            return False
        raise Unsupported(f"bool() of {self.op}", _site())

    def __getattr__(self, name: str) -> Any:
        # a sympy attribute read off an IR node (a torch helper reached through
        # the fake-mode view fallback): named and counted, never guessed
        if name.startswith("__"):
            raise AttributeError(name)
        raise Unsupported(f"sympy attribute .{name} on {self.op}", _site())

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
            todo.extend(_children(n))
        return out


def _children(n: Node) -> list[Node]:
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
        # symbol node -> (lower, upper) from the dedupe pass's kept guards
        self.extra_bounds: dict = {}

    def _key_of(self, a: Any) -> Any:
        if isinstance(a, Node):
            return a.id
        if isinstance(a, tuple):
            return tuple(self._key_of(x) for x in a)
        return a

    def mk(self, op: str, args: tuple, hint: Any) -> Node:
        key = (op, self._key_of(args))
        n = self.table.get(key)
        if n is None:
            n = Node(len(self.nodes), op, args, hint)
            self.table[key] = n
            self.nodes.append(n)
        return n

    # ---- leaves
    def const(self, v: int) -> Node:
        v = int(v)
        return self.mk("const", (v,), v)

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
    def _as_terms(self, n: Node) -> tuple[int, dict]:
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

    def _as_factors(self, n: Node) -> tuple[int, dict]:
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
                c, ts = self._as_terms(k)
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
                c, fs = self._as_factors(k)
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
        ca, ta = self._as_terms(a)
        cb, tb = self._as_terms(b)
        for k, v in tb.items():
            ta[k] = ta.get(k, 0) + v
        return self.add_terms(ca + cb, ta)

    def neg(self, a: Node) -> Node:
        return self.mul(self.const(-1), a)

    def sub(self, a: Node, b: Node) -> Node:
        return self.add(a, self.neg(b))

    def mul(self, a: Node, b: Node) -> Node:
        ca, fa = self._as_factors(a)
        cb, fb = self._as_factors(b)
        for k, v in fb.items():
            fa[k] = fa.get(k, 0) + v
        return self.mul_factors(ca * cb, fa)

    def pow(self, a: Node, e: int) -> Node:
        if e == 0:
            return self.const(1)
        c, f = self._as_factors(a)
        return self.mul_factors(c**e, {k: v * e for k, v in f.items()})

    # ---- bounds over the declared domains (and the pass's tightened ones)
    def bounds(self, n: Node) -> tuple:
        """(lower, upper) of an integer node, None where unknown. Reads the
        declared domains alone (a size symbol is >= 1, every other symbol
        unbounded) plus the intervals the dedupe pass tightened from kept
        guards while it runs; never a hint. What decides a relation without
        a guard, as sympy's assumptions did."""
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
                t_lo, t_hi = self.bounds(t)
                if cf < 0:
                    t_lo, t_hi = (
                        None if t_hi is None else cf * t_hi,
                        None if t_lo is None else cf * t_lo,
                    )
                else:
                    t_lo, t_hi = (
                        None if t_lo is None else cf * t_lo,
                        None if t_hi is None else cf * t_hi,
                    )
                lo = None if lo is None or t_lo is None else lo + t_lo
                hi = None if hi is None or t_hi is None else hi + t_hi
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
        if op in ("min", "max"):
            bs = [self.bounds(a) for a in n.args]
            los = [b[0] for b in bs]
            his = [b[1] for b in bs]
            if op == "min":
                lo = None if None in los else min(los)
                known = [h for h in his if h is not None]
                return lo, (min(known) if known else None)
            known = [x for x in los if x is not None]
            hi = None if None in his else max(his)
            return (max(known) if known else None), hi
        # a division's divisor is nonzero where its result is used (the
        # domain guard of A243 precedes every use), so a divisor the domains
        # put at >= 0 counts as positive here, as sympy's FloorDiv assumed it
        if op == "mod":
            b_lo, b_hi = self.bounds(n.args[1])
            if b_lo is not None and b_lo >= 0:
                return 0, (None if b_hi is None else max(b_hi - 1, 0))
            return None, None
        if op in ("floordiv", "ceildiv"):
            a_lo, a_hi = self.bounds(n.args[0])
            b_lo, b_hi = self.bounds(n.args[1])
            if a_lo is not None and a_lo >= 0 and b_lo is not None and b_lo >= 0:
                lo = 0 if not b_hi else a_lo // b_hi
                hi = None if a_hi is None else -(-a_hi // max(b_lo, 1))
                return lo, hi
            return None, None
        if op == "nod":
            return 0, 1
        return None, None

    def is_positive(self, n: Node) -> bool:
        lo, _ = self.bounds(n)
        return lo is not None and lo > 0

    def is_nonnegative(self, n: Node) -> bool:
        lo, _ = self.bounds(n)
        return lo is not None and lo >= 0

    # ---- division
    def _below(self, a: Node, b: Node) -> bool:
        a_lo, a_hi = self.bounds(a)
        b_lo, _b_hi = self.bounds(b)
        return (
            a_lo is not None
            and a_lo >= 0
            and a_hi is not None
            and b_lo is not None
            and a_hi < b_lo
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
        ca, fa = self._as_factors(a)
        cb, fb = self._as_factors(b)
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
        # ceil(a / b) of integers: -((-a) // b); its own kind, as the sympy
        # path keeps CeilToInt(IntTrueDiv(a, b)) apart from FloorDiv
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
        ca, fa = self._as_factors(a)
        cb, fb = self._as_factors(b)
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

    # ---- relations: rel(d, 0) with d = lhs - rhs in add form
    def _rel(self, op: str, d: Node) -> Node:
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
        c, ts = self._as_terms(d)
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

    def eq(self, a: Node, b: Node) -> Node:
        if a.is_float or b.is_float:
            return self.fcmp("eq", a, b)
        return self._rel("eq", self.sub(a, b))

    def ne(self, a: Node, b: Node) -> Node:
        if a.is_float or b.is_float:
            return self.fcmp("ne", a, b)
        return self._rel("ne", self.sub(a, b))

    def lt(self, a: Node, b: Node) -> Node:
        if a.is_float or b.is_float:
            return self.fcmp("lt", a, b)
        return self._rel("lt", self.sub(a, b))

    def le(self, a: Node, b: Node) -> Node:
        if a.is_float or b.is_float:
            return self.fcmp("le", a, b)
        return self._rel("le", self.sub(a, b))

    def gt(self, a: Node, b: Node) -> Node:
        return self.lt(b, a)

    def ge(self, a: Node, b: Node) -> Node:
        return self.le(b, a)

    def fcmp(self, rel: str, a: Node, b: Node) -> Node:
        a, b = self.to_float(a), self.to_float(b)
        if a.op == "fconst" and b.op == "fconst":
            x, y = a.hint, b.hint
            return self.boolean(
                {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y}[rel]
            )
        x, y = a.hint, b.hint
        hint = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y}[rel]
        return self.mk("fcmp", (rel, a, b), bool(hint))

    def not_(self, a: Node) -> Node:
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
            return self._rel("le", self.neg(a.args[0]))
        if op == "le":
            return self._rel("lt", self.neg(a.args[0]))
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
            hint = x / y  # raises on zero, as the sympy path's hint arithmetic does
        else:
            hint = x**y
        return self.mk(op, (a, b), float(hint))

    def fun(self, op: str, a: Node) -> Node:
        a = self.to_float(a)
        if op == "fneg":
            hint = -a.hint
        elif op == "fsqrt":
            hint = math.sqrt(a.hint)
        else:  # fround32: the double narrowed to a float
            hint = struct.unpack("<f", struct.pack("<f", a.hint))[0]
        return self.mk(op, (a,), float(hint))

    def fint(self, op: str, a: Node) -> Node:
        # int() / math.floor / math.ceil of a float: an integer node over a
        # float operand (the ratio of integers takes floordiv / ceildiv)
        f = {"ftrunc": int, "ffloor": math.floor, "fceil": math.ceil}[op]
        if a.op == "fconst":
            return self.const(f(a.hint))
        return self.mk(op, (a,), f(a.hint))

    def ftrunc(self, a: Node) -> Node:
        return self.fint("ftrunc", a)

    # ---- substitution (pins read into uses): symbol node -> node
    def subst(self, n: Node, pins: dict, memo: dict | None = None) -> Node:
        """`n` with `pins` read into its symbols, rebuilt through the
        constructors above, so a substituted relation re-canonicalizes (and
        may fold under the bounds in force)."""
        memo = {} if memo is None else memo
        r = memo.get(n.id)
        if r is not None:
            return r
        op = n.op
        if op in ("sym", "fsym"):
            r = pins.get(n, n)
        elif op in ("const", "fconst", "true", "false"):
            r = n
        elif op == "add":
            c, ts = n.args
            acc: dict = {}
            for t, cf in ts:
                s = self.subst(t, pins, memo)
                acc[s] = acc.get(s, 0) + cf
            r = self.add_terms(c, acc)
        elif op == "mul":
            c, fs = n.args
            facc: dict = {}
            for f, e in fs:
                s = self.subst(f, pins, memo)
                facc[s] = facc.get(s, 0) + e
            r = self.mul_factors(c, facc)
        elif op in ("floordiv", "ceildiv", "mod"):
            a, b = (self.subst(x, pins, memo) for x in n.args)
            r = getattr(self, op)(a, b)
        elif op in ("min", "max"):
            r = self._minmax(op, [self.subst(a, pins, memo) for a in n.args])
        elif op == "nod":
            half = len(n.args) // 2
            xs = [self.subst(a, pins, memo) for a in n.args]
            r = self.nod(xs[:half], xs[half:])
        elif op in ("eq", "ne", "lt", "le"):
            r = self._rel(op, self.subst(n.args[0], pins, memo))
        elif op in ("and", "or"):
            r = self.junction(op, [self.subst(a, pins, memo) for a in n.args])
        elif op == "not":
            r = self.not_(self.subst(n.args[0], pins, memo))
        elif op == "fcmp":
            rel, x, y = n.args
            r = self.fcmp(rel, self.subst(x, pins, memo), self.subst(y, pins, memo))
        elif op == "ffromint":
            r = self.to_float(self.subst(n.args[0], pins, memo))
        elif op in ("fadd", "fsub", "fmul", "fdiv", "fpow"):
            x, y = (self.subst(a, pins, memo) for a in n.args)
            r = self.fbin(op, x, y)
        elif op in ("fneg", "fsqrt", "fround32"):
            r = self.fun(op, self.subst(n.args[0], pins, memo))
        elif op in ("ftrunc", "ffloor", "fceil"):
            r = self.fint(op, self.subst(n.args[0], pins, memo))
        else:
            raise AssertionError(f"host_trace ir: no substitution for {op}")
        memo[n.id] = r
        return r


def render(n: Node) -> str:
    """Python-syntax text of a node (what str() of a traced value prints)."""
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
    if op == "ceildiv":
        return f"ceildiv({render(n.args[0])}, {render(n.args[1])})"
    if op == "mod":
        return f"({render(n.args[0])} % {render(n.args[1])})"
    if op in ("min", "max"):
        return f"{op}(" + ", ".join(render(a) for a in n.args) + ")"
    if op == "nod":
        return "dense(" + ", ".join(render(a) for a in n.args) + ")"
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
    if op == "ffromint":
        return f"float({render(n.args[0])})"
    if op in ("fadd", "fsub", "fmul", "fdiv", "fpow"):
        sym = {"fadd": "+", "fsub": "-", "fmul": "*", "fdiv": "/", "fpow": "**"}[op]
        return f"({render(n.args[0])} {sym} {render(n.args[1])})"
    if op in ("ftrunc", "ffloor", "fceil"):
        return f"{op[1:]}({render(n.args[0])})"
    return f"{op}({render(n.args[0])})"


class _Pass:
    """The dedupe pass over an ordered guard list (A205 / A206 on the IR).
    In order, a guard is dropped when it is true once the pins and
    unifications of the kept equalities before it are read into it (the
    substituted relation re-canonicalizes under the bounds the kept guards
    tightened), or when it is already a kept fact (by node; an `and` / `or`
    argument that is a kept fact becomes true). A kept `eq` pins its latest
    unit-coefficient symbol to the rest, a kept `lt` / `le` / `ne` affine in
    one symbol tightens that symbol's interval."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.pins: dict = {}  # symbol node -> node over no pinned symbol
        self._uses: dict = {}  # symbol node -> the pinned symbols whose value names it
        self.facts: set[int] = set()
        self.bounds_extra: dict = {}
        ctx.extra_bounds = self.bounds_extra

    def close(self) -> None:
        self.ctx.extra_bounds = {}

    def resolve(self, g: Node) -> Node:
        e = self.ctx.subst(g, self.pins) if (self.pins or self.bounds_extra) else g
        return self._with_facts(e)

    def _with_facts(self, n: Node) -> Node:
        if n.id in self.facts:
            return self.ctx.true()
        if n.op in ("and", "or"):
            return self.ctx.junction(n.op, [self._with_facts(a) for a in n.args])
        return n

    def feed(self, g: Node) -> Node | None:
        """The kept form of `g` after the guards fed before it, or None."""
        e = self.resolve(g)
        if e.op == "true":
            return None
        self.facts.add(e.id)
        if e.op in ("lt", "le", "ne"):
            self._tighten(e)
        elif e.op == "eq":
            self._pin_or_unify(e)
        return e

    def test(self, g: Node) -> bool:
        """Whether the guards fed so far imply `g` (it would be dropped)."""
        return self.resolve(g).op == "true"

    def _pin_or_unify(self, e: Node) -> None:
        ctx = self.ctx
        d = e.args[0]
        c, ts = ctx._as_terms(d)
        # s*(a - b) == 0 with s a size (positive): a - b == 0
        if len(ts) == 1 and c == 0:
            ((t, cf),) = ts.items()
            if t.op == "mul" and t.args[0] == 1:
                keep = {f: ex for f, ex in t.args[1] if not self._positive_sym(f)}
                if len(keep) < len(t.args[1]):
                    c, ts = ctx._as_terms(ctx.mul_factors(cf, keep))
        elif len(ts) == 2 and c == 0:
            (t1, c1), (t2, c2) = ts.items()
            f1 = dict(ctx._as_factors(t1)[1])
            f2 = dict(ctx._as_factors(t2)[1])
            common = [f for f in f1 if f in f2 and self._positive_sym(f)]
            if common:
                for f in common:
                    del f1[f]
                    del f2[f]
                d = ctx.add(ctx.mul_factors(c1, f1), ctx.mul_factors(c2, f2))
                c, ts = ctx._as_terms(d)
        units = [t for t, cf in ts.items() if t.op == "sym" and cf in (1, -1)]
        if units:
            s = max(units, key=lambda n: n.id)
            cf = ts[s]
            rest = ctx.add_terms(c, {t: v for t, v in ts.items() if t is not s})
            value = ctx.mul(ctx.const(-cf), rest)
        elif len(ts) == 1:
            ((s, cf),) = ts.items()
            if s.op != "sym" or c % cf != 0:
                return
            value = ctx.const(-c // cf)
        else:
            return
        self._pin(s, value)

    def _pin(self, s: Node, value: Node) -> None:
        # the pins stay resolved: a new pin's value is read through the pins
        # before it, and the pins naming the new symbol are read through it
        # (sympy's pass followed the chains at each use instead)
        ctx = self.ctx
        if self.pins:
            value = ctx.subst(value, self.pins)
        if s.args[0] in value.free_symbols:
            return
        self.pins[s] = value
        for k in list(self._uses.pop(s, ())):
            self.pins[k] = ctx.subst(self.pins[k], {s: value})
            for name in self.pins[k].free_symbols:
                self._uses.setdefault(ctx.symbols[name], set()).add(k)
        for name in value.free_symbols:
            self._uses.setdefault(ctx.symbols[name], set()).add(s)

    def _positive_sym(self, n: Node) -> bool:
        return n.op == "sym" and n.args[0] in self.ctx.positive

    def _tighten(self, e: Node) -> None:
        ctx = self.ctx
        c, ts = ctx._as_terms(e.args[0])
        if len(ts) != 1:
            return
        ((s, cf),) = ts.items()
        if s.op != "sym":
            return
        lo, hi = ctx.bounds(s)
        if e.op == "ne":
            if lo is not None and cf * lo + c == 0:
                self.bounds_extra[s] = (lo + 1, hi)
            elif hi is not None and cf * hi + c == 0:
                self.bounds_extra[s] = (lo, hi - 1)
            return
        d = c + 1 if e.op == "lt" else c  # cf*s + d <= 0 over integers
        if cf > 0:
            b = (-d) // cf
            self.bounds_extra[s] = (lo, b if hi is None else min(hi, b))
        else:
            b = -(d // cf)
            self.bounds_extra[s] = (b if lo is None else max(lo, b), hi)


class Env:
    """One trace's IR context, symbols, guard record and census: what
    `_TraceShapeEnv` is to the sympy backend."""

    # what torch's helpers ask of a shape env on the fake-mode view fallback
    _translation_validation_enabled = False
    _replacements_version_counter = 0

    def __init__(self, declined: type[Exception]) -> None:
        self.ctx = Ctx()
        self.declined = declined
        self.guards: list[Node] = []  # the raw record, in evaluation order
        self._recorded: set[int] = set()
        # per raw guard (op index, kernel-choice depth, origin), the trace's
        # attribution at the record (`attribute`, set by the trace; _NO_ROW
        # for an env used outside one)
        self.guard_rows: list[tuple[int, int, str]] = []
        self.attribute: Any = None
        self.symbols: dict[str, Node] = {}  # name -> node, in creation order
        self.sources: dict[str, str] = {}  # name -> source name
        self.unique_ids: set[int] = set()
        # a root's base symbol node -> (root name, is an allocation)
        self.roots: dict[Node, tuple[str, bool]] = {}
        self.root_facts: list[tuple[str, str]] = []
        self.guard_notes: dict[Node, str] = {}
        self.census: list[tuple[str, str]] = []  # (op, site) of every Unsupported
        self.evaluations = 0
        self.replacements: dict = {}

    # ---- symbols, named as the ShapeEnv names them (s<id>, zf<id>; the id
    # from the source name, so both backends name a call's symbols alike)
    def _unique_id(self, source: str) -> int:
        attempt = int(hashlib.sha256(source.encode()).hexdigest(), 16) % 100
        while attempt in self.unique_ids:
            attempt += 1
        self.unique_ids.add(attempt)
        return attempt

    def create_symbol(self, source: str, hint: Any, positive: bool) -> Any:
        k = self._unique_id(source)
        if isinstance(hint, float):
            name = f"zf{k}"
            node = self.ctx.fsym(name, hint)
            # pyrefly: ignore [bad-argument-type]
            out: Any = torch.SymFloat(IRSymNode(node, self, float))
        else:
            name = f"s{k}"
            node = self.ctx.sym(name, int(hint), positive)
            # pyrefly: ignore [bad-argument-type]
            out = torch.SymInt(IRSymNode(node, self, int))
        self.symbols[name] = node
        self.sources[name] = source
        return out

    def note_root(self, sym: Any, name: str, alloc: bool) -> None:
        self.roots[sym.node.node] = (name, alloc)

    def note_pin(self, v: Any, note: str) -> None:
        n = v.node.node
        if n.op != "const":
            self.guard_notes.setdefault(self.ctx.eq(n, self.ctx.const(n.hint)), note)

    # ---- the guard record
    def _record(self, g: Node) -> None:
        if g.op != "true" and g.id not in self._recorded:
            self._recorded.add(g.id)
            self.guards.append(g)
            self.guard_rows.append(
                _NO_ROW if self.attribute is None else self.attribute(self, g)
            )

    def guard_bool(self, g: Node) -> bool:
        """A boolean the host reads: a constant decides itself; a relation
        between two roots' addresses with an allocation among them is decided
        by identity (a root fact); the rest is the hint, recorded."""
        self.evaluations += 1
        if g.op == "true":
            return True
        if g.op == "false":
            return False
        if self.roots and g.op in ("eq", "ne", "lt", "le"):
            fact = self._root_identity(g)
            if fact is not None:
                return fact
        hint = bool(g.hint)
        self._record(g if hint else self.ctx.not_(g))
        return hint

    def guard_value(self, n: Node) -> Any:
        self.evaluations += 1
        if n.op in ("const", "fconst"):
            return n.hint
        c = self.ctx.fconst(n.hint) if n.is_float else self.ctx.const(n.hint)
        self._record(self.ctx.eq(n, c))
        return n.hint

    def domain(self, divisor: Node) -> None:
        # a partial operation's domain (A243), recorded when the operation is
        # created unless the declared domains decide it
        g = self.ctx.ne(divisor, self.ctx.const(0))
        if g.op == "false":
            raise ZeroDivisionError("division by a zero the trace can prove")
        self._record(g)

    def _root_identity(self, rel: Node) -> bool | None:
        # one pointer into each of two roots (an input's address is its
        # symbol, an allocation's 256 times its symbol), whatever common
        # factor the relation's canonical form divided out
        c, ts = self.ctx._as_terms(rel.args[0])
        units = []
        for t, cf in ts.items():
            r = self.roots.get(t)
            if r is not None:
                units.append((cf / (256 if r[1] else 1), r))
        if len(units) != 2 or not any(r[1] for _u, r in units):
            return None
        (u1, _r1), (u2, _r2) = units
        if u1 != -u2:
            return None
        a, b = sorted(r[0] for _u, r in units)
        if rel.op not in ("eq", "ne"):
            raise self.declined(
                f"host_trace: the host ordered the addresses of two roots ({a}, {b}); "
                "only their identity is decided under a trace (declined)"
            )
        if (a, b) not in self.root_facts:
            self.root_facts.append((a, b))
        return rel.op == "ne"

    def unsupported(self, op: str) -> Unsupported:
        e = Unsupported(op, _site())
        self.census.append((e.op, e.site))
        return e

    # ---- what torch's helpers ask of a shape env (the fake-mode fallback)
    def evaluate_sym_node(
        self, sym_node: Any, size_oblivious: bool = False, fallback_value: Any = None
    ) -> Any:
        return sym_node.evaluate(size_oblivious)

    def _maybe_evaluate_static(self, expr: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(expr, Node) and expr.op in ("true", "false", "const", "fconst"):
            return expr.hint
        return None

    def replace(self, e: Any) -> Any:
        return e

    # ---- the tape's guards
    def tape_guards(self) -> tuple[list, dict, dict, list]:
        """The recorded guards in order without those the guards before them
        imply, the pins (symbol node -> node) the tape reads into every use,
        the notes of the kept guards, and the raw index of each kept guard."""
        p = _Pass(self.ctx)
        try:
            out, notes, kept = [], {}, []
            for k, g in enumerate(self.guards):
                e = p.feed(g)
                if e is None:
                    continue
                out.append(e)
                kept.append(k)
                note = self.guard_notes.get(g)
                if note is not None:
                    notes[e] = note
            pins = dict(p.pins)
        finally:
            p.close()
        return out, pins, notes, kept

    def implied(self, kept: list, g: Node) -> bool:
        """Whether the ordered guards `kept` imply `g` by the pass's rules:
        `g` is true, or is one of them, once the pins and bounds all of them
        establish are read into `g` and into each of them."""
        p = _Pass(self.ctx)
        try:
            for k in kept:
                p.feed(k)
            final = {p.resolve(k).id for k in kept}
            r = p.resolve(g)
            return r.op == "true" or r.id in final
        finally:
            p.close()


class IRSymNode:
    """The SymNode-shaped backend over an IR node: what torch.SymInt /
    SymFloat / SymBool and c10's PythonSymNodeImpl call."""

    __slots__ = ("node", "env", "pytype", "constant", "__weakref__")
    fx_node = None
    _optimized_summation = False

    def __init__(
        self, node: Node, env: Env, pytype: type, constant: Any = None
    ) -> None:
        self.node = node
        self.env = env
        self.pytype = pytype
        self.constant = constant

    # ---- what the tracer reads
    @property
    def expr(self) -> Node:
        return self.node

    @property
    def _expr(self) -> Node:
        return self.node

    @property
    def hint(self) -> Any:
        return self.node.hint

    @property
    def _hint(self) -> Any:
        return self.node.hint

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

    def str(self) -> builtins.str:
        return render(self.node)

    def __str__(self) -> builtins.str:
        return render(self.node)

    def __repr__(self) -> builtins.str:
        return f"IRSymNode({render(self.node)}, pytype={self.pytype.__name__}, hint={self.node.hint})"

    def _graph_repr(self) -> builtins.str:
        return render(self.node)

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
        if self.node.is_float or other.node.is_float:
            raise self.env.unsupported("mod of a float")
        self.env.domain(other.node)
        return self._int(self.env.ctx.mod(self.node, other.node))

    def int_floordiv(self, other: IRSymNode) -> IRSymNode:
        if self.node.is_float or other.node.is_float:
            raise self.env.unsupported("floordiv of a float")
        self.env.domain(other.node)
        return self._int(self.env.ctx.floordiv(self.node, other.node))

    floordiv = int_floordiv

    def float_truediv(self, other: IRSymNode) -> IRSymNode:
        self.env.domain(other.node)
        return self._flt(self.env.ctx.fbin("fdiv", self.node, other.node))

    int_truediv = float_truediv
    truediv = float_truediv

    def pow_by_natural(self, other: IRSymNode) -> IRSymNode:
        if other.node.op != "const":
            raise self.env.unsupported("a symbolic integer exponent")
        return self._int(self.env.ctx.pow(self.node, other.node.args[0]))

    def float_pow(self, other: IRSymNode) -> IRSymNode:
        return self._flt(self.env.ctx.fbin("fpow", self.node, other.node))

    def pow(self, other: IRSymNode) -> IRSymNode:
        if self.node.is_float or other.node.is_float:
            return self.float_pow(other)
        if other.node.op == "const" and other.node.args[0] >= 0:
            return self.pow_by_natural(other)
        return self.float_pow(other)

    def lshift(self, other: IRSymNode) -> IRSymNode:
        if other.node.op != "const":
            raise self.env.unsupported("a symbolic shift")
        return self._int(
            self.env.ctx.mul(self.node, self.env.ctx.const(2 ** other.node.args[0]))
        )

    def rshift(self, other: IRSymNode) -> IRSymNode:
        if other.node.op != "const":
            raise self.env.unsupported("a symbolic shift")
        c = self.env.ctx
        return self._int(c.floordiv(self.node, c.const(2 ** other.node.args[0])))

    def sym_min(self, other: IRSymNode) -> IRSymNode:
        if self.node.is_float or other.node.is_float:
            raise self.env.unsupported("min of a float")
        return self._int(self.env.ctx.min(self.node, other.node))

    def sym_max(self, other: IRSymNode) -> IRSymNode:
        if self.node.is_float or other.node.is_float:
            raise self.env.unsupported("max of a float")
        return self._int(self.env.ctx.max(self.node, other.node))

    def sym_sum(self, args: list) -> IRSymNode:
        acc: dict = {}
        c = 0
        for a in args:
            ca, ts = self.env.ctx._as_terms(a.node)
            c += ca
            for t, cf in ts.items():
                acc[t] = acc.get(t, 0) + cf
        return self._int(self.env.ctx.add_terms(c, acc))

    def sym_float(self) -> IRSymNode:
        return self._flt(self.env.ctx.to_float(self.node))

    def sym_int(self) -> IRSymNode:
        return self._int(self.env.ctx.ftrunc(self.node)) if self.node.is_float else self

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

    def round(self, ndigits: Any = None) -> IRSymNode:
        raise self.env.unsupported("round")

    def is_integer(self) -> IRSymNode:
        raise self.env.unsupported("is_integer")

    def sym_ite(self, t: IRSymNode, f: IRSymNode) -> IRSymNode:
        if self.node.op == "true":
            return t
        if self.node.op == "false":
            return f
        raise self.env.unsupported("sym_ite on a symbolic condition")

    def __getattr__(self, name: builtins.str) -> Any:
        # the math functions (sym_cos ...), bitwise ops, xor, sym_log2: named
        # and counted (an attribute a SymNode has that this backend lacks)
        if name.startswith("__"):
            raise AttributeError(name)
        raise self.env.unsupported(name)

    # ---- relations and boolean algebra
    def eq(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.eq(self.node, other.node))

    def ne(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.ne(self.node, other.node))

    def lt(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.lt(self.node, other.node))

    def le(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.le(self.node, other.node))

    def gt(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.gt(self.node, other.node))

    def ge(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.ge(self.node, other.node))

    def sym_and(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.and_(self.node, other.node))

    def sym_or(self, other: IRSymNode) -> IRSymNode:
        return self._bool(self.env.ctx.or_(self.node, other.node))

    def sym_not(self) -> IRSymNode:
        return self._bool(self.env.ctx.not_(self.node))

    and_ = sym_and
    or_ = sym_or

    # ---- guard reads
    def guard_bool(self, file: Any = "", line: Any = 0) -> bool:
        return self.env.guard_bool(self.node)

    def guard_int(self, file: Any = "", line: Any = 0) -> int:
        return int(self.env.guard_value(self.node))

    def guard_float(self, file: Any = "", line: Any = 0) -> float:
        return float(self.env.guard_value(self.node))

    expect_true = guard_bool
    guard_size_oblivious = guard_bool
    guard_or_false = guard_bool
    guard_or_true = guard_bool

    def expect_size(self, file: Any = "", line: Any = 0) -> bool:
        return self.env.guard_bool(self.env.ctx.ge(self.node, self.env.ctx.const(0)))

    def statically_known_true(self, file: Any = "", line: Any = 0) -> bool:
        return self.node.op == "true"

    def bool_(self) -> bool:
        return self.env.guard_bool(self.node)

    def int_(self) -> int:
        return int(self.env.guard_value(self.node))

    def evaluate(self, size_oblivious: bool = False) -> Any:
        if self.pytype is bool:
            return self.env.guard_bool(self.node)
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
