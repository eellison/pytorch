"""Host tracing (private): sympy expressions and guards as rows of an
IntegerProgram, so the one compiled evaluator is also the guard printer.

The evaluator's arithmetic is checked (an overflow or a division outside its
domain is a failing status, which the replay treats as a miss), so a lowered
expression either evaluates to its exact integer value or misses. Divisions
keep Python's floor semantics for every sign; sympy's static sign
assumptions only select a shorter form, and a call where they do not hold
fails a division's domain instead of computing a wrong value.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING

import sympy
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from torch.cuda._host_trace import BitLength, declined, F32Div
from torch.cuda._host_trace_ir import children, Ctx, Node, render
from torch.cuda._host_trace_program import f64_bits, MAX_I64, MIN_I64, OutOfDomain
from torch.utils._sympy.functions import (
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    CeilToInt,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    PowByNatural,
    PythonMod,
    ToFloat,
    Where,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from torch.cuda._host_trace_program import IntegerProgram


# below this in magnitude an integer's double is exact
_EXACT = (1 << 53) - 1
_COMPARISONS = {
    sympy.Eq: "eq",
    sympy.Ne: "ne",
    sympy.Lt: "lt",
    sympy.Le: "le",
    sympy.Gt: "gt",
    sympy.Ge: "ge",
}
# SymInt &, |, ^ (triton.next_power_of_2); >> and << are already floor
# divisions and products by powers of two
_BITWISE = {
    BitwiseFn_bitwise_and: "bitand",
    BitwiseFn_bitwise_or: "bitor",
    BitwiseFn_bitwise_xor: "bitxor",
}
# the float expressions a relation compares as doubles
_FLOATS = (sympy.Float, ToFloat, FloatPow, FloatTrueDiv)
# (row, swapped, negated): no float row holds a nan, so each relation is ==
# or <, or its negation
_FLOAT_COMPARISONS = {
    sympy.Eq: ("feq", False, False),
    sympy.Ne: ("feq", False, True),
    sympy.Lt: ("flt", False, False),
    sympy.Le: ("flt", True, True),
    sympy.Gt: ("flt", True, False),
    sympy.Ge: ("flt", False, True),
}


class Lowering:
    """Lowers sympy integer and boolean expressions into `program`.

    `sources` maps each symbol the call supplies to its value: an
    IntegerProgram leaf row (`("size", i, dim)`, `("boxed", i)`, ...) or the
    index of a row already in the program. A symbol it does not map (an
    allocation's address, a value the host computed) is declined.
    """

    def __init__(
        self, program: IntegerProgram, sources: dict[sympy.Symbol, tuple | int]
    ) -> None:
        self.program = program
        self.sources = sources
        self._memo: dict[sympy.Basic, int] = {}
        self._float_memo: dict[sympy.Basic, int] = {}
        # every guard's row, in lowering order (the oracle compares them)
        self.guard_rows: list[int] = []

    @contextlib.contextmanager
    def total(self) -> Iterator[list[int]]:
        """Rows lowered inside fail at no call; yields the rows that are 1
        where they equal the expressions (IntegerProgram.domains). None is
        reused outside."""
        memo, float_memo = self._memo, self._float_memo
        self._memo, self._float_memo, self.program.domains = dict(memo), dict(float_memo), []
        try:
            yield self.program.domains
        finally:
            self._memo, self._float_memo, self.program.domains = memo, float_memo, None

    def lower(self, e: sympy.Basic) -> int:
        """The row holding e's value (a boolean is 0 or 1)."""
        try:
            return self._lower(e)
        except OutOfDomain as error:
            raise declined(f"{e} fails at the hints") from error

    def predicate(
        self, guards: Iterable[sympy.Basic], addresses: frozenset[sympy.Symbol] = frozenset()
    ) -> int:
        """The row that is 1 when the guards, each true at the hints, and the
        sources' declared signs hold:
        sympy simplified the host's expressions assuming a positive symbol is
        positive, so a call where it is not must miss. A guard false at the
        hints over one of `addresses` (placeholders with the addresses' low
        bits) pinned an address: the host read it as an int."""
        conditions = []
        for symbol in self.sources:
            bound = self._declared_bound(symbol)
            if bound is not None:
                conditions.append(self._emit("ge", self.lower(symbol), self._constant(bound)))
        return self._all((*conditions, *self._guard_rows(guards, addresses)))

    def conjunction(
        self, guards: Iterable[sympy.Basic], addresses: frozenset[sympy.Symbol] = frozenset()
    ) -> int:
        """The row that is 1 when the guards, each true at the hints, hold."""
        return self._all(self._guard_rows(guards, addresses))

    def _guard_rows(self, guards: Iterable[sympy.Basic], addresses: frozenset[sympy.Symbol]) -> tuple[int, ...]:
        guards = tuple(guards)
        rows = tuple(self._lower_guard(g) for g in guards)
        for g, row in zip(guards, rows):
            if self.program.values[row] != 1:
                self._false_at_hints(g, addresses)
        self.guard_rows += rows
        return rows

    def _declared_bound(self, symbol: sympy.Symbol) -> int | None:
        if symbol.is_positive:
            return 1
        return 0 if symbol.is_nonnegative else None

    def _lower_guard(self, g: sympy.Basic) -> int:
        return self.lower(g)

    def _false_at_hints(self, g: sympy.Basic, addresses: frozenset) -> None:
        if g.free_symbols & addresses:
            raise declined(f"the host read a traced address as an int (the guard {g})")
        if g.has(FloatPow):
            # c10's SymFloat::sqrt traces as FloatPow(x, 0.5) with a hint
            # from pow(x, 0.5), which is not sqrt(x) for every x
            raise declined(f"the float guard {g} is false at the hints under sqrt")
        raise AssertionError(f"host_trace: the guard {g} is false at the hints")

    def split(self, e: sympy.Basic, root: sympy.Symbol) -> sympy.Basic | None:
        """e as root plus a displacement free of root, else None."""
        displacement = e - root
        own = root.free_symbols
        if own & e.free_symbols and not own & displacement.free_symbols:
            return displacement
        return None

    def _all(self, rows: Iterable[int]) -> int:
        result = self._constant(1)
        for row in rows:
            result = self._emit("and", result, row)
        return result

    def _chain(self, op: str, rows: list[int]) -> int:
        result = rows[0]
        for row in rows[1:]:
            result = self._emit(op, result, row)
        return result

    def _any(self, rows: list[int]) -> int:
        result = rows[0]
        for row in rows[1:]:
            result = self._emit("select", result, self._constant(1), row)
        return result

    def _not(self, row: int) -> int:
        return self._emit("select", row, self._constant(0), self._constant(1))

    def _float_relation(self, entry: tuple[str, bool, bool], lhs: int, rhs: int) -> int:
        op, swapped, negated = entry
        row = self._emit(op, *((rhs, lhs) if swapped else (lhs, rhs)))
        return self._not(row) if negated else row

    def _emit(self, op: str, *operands: int) -> int:
        return self.program.emit(op, *operands)

    def _constant(self, value: int) -> int:
        return self._emit("constant", value)

    def _negate(self, row: int) -> int:
        return self._emit("multiply", row, self._constant(-1))

    def _lower(self, e: sympy.Basic) -> int:
        row = self._memo.get(e)
        if row is None:
            row = self._memo[e] = self._lower_uncached(e)
        return row

    def _lower_uncached(self, e: sympy.Basic) -> int:
        if isinstance(e, (BooleanTrue, BooleanFalse)):
            return self._constant(int(bool(e)))
        if isinstance(e, sympy.Integer):
            if not MIN_I64 <= int(e) <= MAX_I64:
                raise declined(f"the constant {e} is not an int64")
            return self._constant(int(e))
        if isinstance(e, sympy.Symbol):
            source = self.sources.get(e)
            if source is None:
                raise declined(f"{e} is not read from the call's inputs")
            if e.is_integer is not True:
                raise declined(f"{e} is not an integer")
            return source if isinstance(source, int) else self._emit(*source)
        if type(e) in _COMPARISONS and any(isinstance(x, _FLOATS) for x in e.args):
            lhs, rhs = (self._lower_float(x) for x in e.args)
            return self._float_relation(_FLOAT_COMPARISONS[type(e)], lhs, rhs)
        if type(e) in _COMPARISONS:
            lhs, rhs = (self._lower_integer(x) for x in e.args)
            return self._emit(_COMPARISONS[type(e)], lhs, rhs)
        if isinstance(e, sympy.And):
            return self._chain("and", [self._lower_boolean(x) for x in e.args])
        if isinstance(e, sympy.Or):
            return self._any([self._lower_boolean(x) for x in e.args])
        if isinstance(e, sympy.Not):
            return self._not(self._lower_boolean(e.args[0]))
        if isinstance(e, Identity):
            return self._lower_integer(e.args[0])
        if isinstance(e, Where):
            c, a, b = e.args
            return self._emit("select", self._lower_boolean(c), self._lower_integer(a), self._lower_integer(b))
        if isinstance(e, (sympy.Add, sympy.Mul)):
            op = "add" if isinstance(e, sympy.Add) else "multiply"
            return self._chain(op, [self._lower_integer(x) for x in e.args])
        if isinstance(e, (sympy.Pow, PowByNatural)):
            return self._power(e)
        if isinstance(e, FloorDiv):
            return self._divide(*e.args, floor=True)
        if isinstance(e, (CeilToInt, FloorToInt)) and (ratio := _int_ratio(e.args[0])) is not None:
            self._require_exact(self._lower_integer(ratio[0]))
            return self._divide(*ratio, floor=isinstance(e, FloorToInt))
        if isinstance(e, (PythonMod, Mod)):
            # n - d * (n // d); torch's Mod is the same on its nonnegative domain.
            # A term of n that d divides (an allocation's 256-aligned base, not
            # yet bound where a guard reads it) leaves it unchanged
            n, d = e.args
            n = sympy.Add(*(t for t in sympy.Add.make_args(n) if not (t / d).is_integer))
            quotient = self._divide(n, d, floor=True)
            product = self._emit("multiply", self._lower_integer(d), quotient)
            return self._emit("add", self._lower_integer(n), self._negate(product))
        if isinstance(e, (Min, Max)):
            rows = [self._lower_integer(x) for x in e.args]
            return self._emit("min" if isinstance(e, Min) else "max", *rows)
        if type(e) in _BITWISE:
            lhs, rhs = (self._lower_integer(x) for x in e.args)
            return self._emit(_BITWISE[type(e)], lhs, rhs)
        if isinstance(e, BitLength):
            return self._emit("bitlength", self._lower_integer(e.args[0]))
        if isinstance(e, F32Div):
            n, d = (self._lower_integer(x) for x in e.args)
            return self._emit("f32div", n, d)
        raise declined(f"no integer lowering for {type(e).__name__} in {e}")

    def _lower_integer(self, e: sympy.Basic) -> int:
        if e.is_Boolean or isinstance(e, sympy.Rel):
            raise declined(f"the boolean {e} is used as an integer")
        return self._lower(e)

    def _lower_float(self, e: sympy.Basic) -> int:
        """The row holding the bits of e's double, computed as the host did:
        each operation rounds once, so only the correctly rounded ones lower."""
        row = self._float_memo.get(e)
        if row is None:
            row = self._float_memo[e] = self._lower_float_uncached(e)
        return row

    def _lower_float_uncached(self, e: sympy.Basic) -> int:
        if isinstance(e, (sympy.Float, sympy.Integer)):
            value = float(e)
            exact = e._prec == 53 if isinstance(e, sympy.Float) else value == int(e)
            if not exact or not math.isfinite(value):
                raise declined(f"the constant {e} is not a double")
            return self._constant(f64_bits(value))
        if isinstance(e, ToFloat):
            return self._emit("tofloat", self._lower_integer(e.args[0]))
        if isinstance(e, FloatPow) and e.args[1] == 0.5:
            return self._emit("fsqrt", self._lower_float(e.args[0]))
        if isinstance(e, FloatTrueDiv):
            return self._emit("fdiv", *(self._lower_float(x) for x in e.args))
        raise declined(f"no exact float lowering for {type(e).__name__} in {e}")

    def _lower_boolean(self, e: sympy.Basic) -> int:
        if not (e.is_Boolean or isinstance(e, sympy.Rel)):
            raise declined(f"the integer {e} is used as a boolean")
        return self._lower(e)

    def _power(self, e: sympy.Basic) -> int:
        base, exponent = e.args
        if base == 2 and not isinstance(exponent, sympy.Integer):
            shift = self._lower_integer(exponent)
            return self._emit("lshift", self._constant(1), shift)
        if not (isinstance(exponent, sympy.Integer) and exponent >= 0):
            raise declined(f"{e} is not a natural power")
        return self._natural_power(self._lower_integer(base), int(exponent))

    def _natural_power(self, square: int, n: int) -> int:
        result = self._constant(1)
        while n:
            if n & 1:
                result = self._emit("multiply", result, square)
            n >>= 1
            if n:
                square = self._emit("multiply", square, square)
        return result

    def _require_exact(self, n: int) -> None:
        """The host rounded a double quotient n / d of integers and then took
        its floor or ceiling: below 2**53 in magnitude n's double is exact and
        the rounding moves the quotient by less than its distance (at least
        1 / |d|) to the next integer, so the integer division is exact. A call
        past it misses: under total() a domain, else a division by the
        condition, which fails the evaluation where it is 0."""
        ok = self._emit("and", self._emit("le", n, self._constant(_EXACT)), self._emit("le", self._constant(-_EXACT), n))
        if self.program.values[ok] != 1:
            raise declined(f"a rounded float quotient's numerator {self.program.values[n]} is past 2**53")
        if self.program.domains is not None:
            self.program.domains.append(ok)
        else:
            self._emit("floordiv", self._constant(0), ok)

    def _divide(self, n_expr: sympy.Basic, d_expr: sympy.Basic, *, floor: bool) -> int:
        n, d = self._lower_integer(n_expr), self._lower_integer(d_expr)
        n_sign = 1 if n_expr.is_nonnegative else -1 if n_expr.is_nonpositive else 0
        d_sign = -1 if d_expr.is_negative else 1 if d_expr.is_positive else 0
        return self._divide_rows(n, d, n_sign, d_sign, floor)

    def _divide_rows(self, n: int, d: int, n_sign: int, d_sign: int, floor: bool) -> int:
        """Python's floor (or ceiling) division from the evaluator's divisions,
        which admit a nonnegative numerator and a positive divisor:
        n / d = -n / -d, and for d > 0 and n < 0 floor(n / d) = -ceil(-n / d).
        A sign known statically (n_sign 1 for n >= 0, -1 for n <= 0; d_sign
        1 for d > 0, -1 for d < 0; 0 unknown) needs no select."""
        if d_sign < 0:
            n, d, n_sign = self._negate(n), self._negate(d), -n_sign
        elif d_sign == 0:
            flip = self._emit("lt", d, self._constant(0))
            n = self._emit("select", flip, self._negate(n), n)
            d = self._emit("select", flip, self._negate(d), d)
            n_sign = 0
        same, other = ("floordiv", "ceildiv") if floor else ("ceildiv", "floordiv")
        if n_sign > 0:
            return self._emit(same, n, d)
        if n_sign < 0:
            return self._negate(self._emit(other, self._negate(n), d))
        negative = self._emit("lt", n, self._constant(0))
        magnitude = self._emit("select", negative, self._negate(n), n)
        return self._emit(
            "select",
            negative,
            self._negate(self._emit(other, magnitude, d)),
            self._emit(same, magnitude, d),
        )


def _int_ratio(f: sympy.Basic) -> tuple[sympy.Basic, sympy.Basic] | None:
    """(n, d), integers whose quotient the double f is, rounded once by the
    host: IntTrueDiv, or FloatTrueDiv of operands each a ToFloat or an
    integer-valued double (Ctx.int_ratio of the IR backend), else None."""
    if isinstance(f, IntTrueDiv):
        return f.args
    if not isinstance(f, FloatTrueDiv):
        return None
    out = []
    for x in f.args:
        if isinstance(x, ToFloat):
            out.append(x.args[0])
        elif isinstance(x, sympy.Float) and x._prec == 53 and math.isfinite(v := float(x)) and v == int(v):
            out.append(sympy.Integer(int(v)))
        else:
            return None
    return out[0], out[1]


# a relation's class name as its integer row and as (float row, swapped, negated)
_COMPARISON_ROWS = {"Eq": "eq", "Ne": "ne", "Lt": "lt", "Le": "le", "Gt": "gt", "Ge": "ge"}
_FLOAT_COMPARISON_ROWS = dict(zip(_COMPARISON_ROWS, _FLOAT_COMPARISONS.values()))


class IRLowering(Lowering):
    """Lowers the IR backend's nodes (torch.cuda._host_trace_ir) with no
    sympy step: the rows Lowering gives a node's sympy export, from the node.

    `sources` maps each sym node the call supplies to its value, as
    Lowering's does; a guard is an Env record, (node, written form or None).
    What the export's lowering declines (a float operation other than an
    exact division or a square root, an IsNonOverlappingAndDenseIndicator,
    TruncToInt) this declines too, and a sign the declared domains decide
    (Ctx.bounds) selects the shorter division as sympy's assumptions do."""

    def __init__(self, program: IntegerProgram, sources: dict[Node, tuple | int], ctx: Ctx) -> None:
        super().__init__(program, sources)  # type: ignore[arg-type]
        self.ctx = ctx

    def lower(self, e: Node | int) -> int:  # type: ignore[override]
        try:
            if isinstance(e, int):
                return self._int_constant(e)
            if not self.ctx.owns(e):
                raise declined(f"{render(e)} is a value of another trace")
            return self._lower(e)  # type: ignore[arg-type]
        except OutOfDomain as error:
            raise declined(f"{render(e) if isinstance(e, Node) else e} fails at the hints") from error

    def _declared_bound(self, symbol: Node) -> int | None:  # type: ignore[override]
        return 1 if symbol.args[0] in self.ctx.positive else None

    def _lower_guard(self, g: tuple) -> int:  # type: ignore[override]
        node, written = g
        try:
            return self._relation(*written) if written is not None else self._lower(node)
        except OutOfDomain as error:
            raise declined(f"{render(node)} fails at the hints") from error

    def _false_at_hints(self, g: tuple, addresses: frozenset) -> None:  # type: ignore[override]
        node = g[0]
        text = render(node)
        if node.free_symbols & {a.args[0] for a in addresses}:
            raise declined(f"the host read a traced address as an int (the guard {text})")
        if _contains(node, "fpow"):
            raise declined(f"the float guard {text} is false at the hints under sqrt")
        raise AssertionError(f"host_trace: the guard {text} is false at the hints")

    def split(self, e: Node | int, root: Node) -> Node | None:  # type: ignore[override]
        if not isinstance(e, Node) or not isinstance(root, Node):
            return None
        displacement = self.ctx.sub(e, root)
        own = root.free_symbols
        if own & e.free_symbols and not own & displacement.free_symbols:
            return displacement
        return None

    def _int_constant(self, value: int) -> int:
        if not MIN_I64 <= value <= MAX_I64:
            raise declined(f"the constant {value} is not an int64")
        return self._constant(value)

    def _int(self, n: Node) -> int:
        if n.is_bool:
            raise declined(f"the boolean {render(n)} is used as an integer")
        return self._lower(n)  # type: ignore[arg-type]

    def _bool(self, n: Node) -> int:
        if not n.is_bool:
            raise declined(f"the integer {render(n)} is used as a boolean")
        return self._lower(n)  # type: ignore[arg-type]

    def _lower_uncached(self, n: Node) -> int:  # type: ignore[override]
        op, args = n.op, n.args
        if op == "const":
            return self._int_constant(args[0])
        if op == "sym":
            source = self.sources.get(n)  # type: ignore[call-overload]
            if source is None:
                raise declined(f"{args[0]} is not read from the call's inputs")
            return source if isinstance(source, int) else self._emit(*source)
        if op == "add":
            c, terms = args
            rows = [self._int(t) if cf == 1 else self._emit("multiply", self._int_constant(cf), self._int(t)) for t, cf in terms]
            return self._chain("add", [*([self._int_constant(c)] if c else []), *rows])
        if op == "mul":
            c, factors = args
            rows = [self._power_of(f, e) for f, e in factors]
            return self._chain("multiply", [*([self._int_constant(c)] if c != 1 else []), *rows])
        if op in ("floordiv", "ceildiv"):
            if op == "ceildiv":
                # IRSymNode.ceil of an int ratio
                self._require_exact(self._int(args[0]))
            return self._divide_nodes(*args, floor=op == "floordiv")
        if op == "mod":
            num, den = args
            quotient = self._divide_nodes(num, den, floor=True)
            product = self._emit("multiply", self._int(den), quotient)
            return self._emit("add", self._int(num), self._negate(product))
        if op in ("min", "max"):
            return self._emit(op, *(self._int(a) for a in args))
        if op == "pbn":
            base, exponent = args
            if base.args[0] == 2:
                return self._emit("lshift", self._constant(1), self._int(exponent))
            raise declined(f"{render(n)} is not a natural power")
        if op == "bitlen":
            return self._emit("bitlength", self._int(args[0]))
        if op in ("f32div", "bitand", "bitor", "bitxor"):
            return self._emit(op, self._int(args[0]), self._int(args[1]))
        if op == "where":
            c, a, b = args
            return self._emit("select", self._bool(c), self._int(a), self._int(b))
        if op in ("ffloor", "fceil"):
            ratio = self.ctx.int_ratio(args[0])
            if ratio is not None:
                self._require_exact(self._int(ratio[0]))
                return self._divide_nodes(*ratio, floor=op == "ffloor")
        if op in ("true", "false"):
            return self._constant(int(op == "true"))
        if op in ("eq", "ne", "lt", "le", "fcmp"):
            written = self.ctx.written.get(n.id)
            if written is not None:
                return self._relation(*written)
            if op == "fcmp":
                rel, a, b = args
                return self._relation(rel.capitalize(), a, b)
            # rel(d, 0), as the export's sides' difference
            return self._emit(op, self._int(args[0]), self._constant(0))
        if op == "and":
            return self._chain("and", [self._bool(a) for a in args])
        if op == "or":
            return self._any([self._bool(a) for a in args])
        if op == "not":
            return self._not(self._bool(args[0]))
        raise declined(f"no integer lowering for the IR node {op} in {render(n)}")

    def _relation(self, cls: str, a: Node, b: Node) -> int:
        if a.is_float or b.is_float:
            return self._float_relation(_FLOAT_COMPARISON_ROWS[cls], self._float(a), self._float(b))
        return self._emit(_COMPARISON_ROWS[cls], self._int(a), self._int(b))

    def _float(self, n: Node) -> int:
        row = self._float_memo.get(n)  # type: ignore[call-overload]
        if row is None:
            row = self._float_memo[n] = self._float_uncached(n)  # type: ignore[index]
        return row

    def _float_uncached(self, n: Node) -> int:
        """Lowering._lower_float of n's export: a double constant, ToFloat,
        FloatPow(x, 0.5), FloatTrueDiv, and the constants sympy folds
        FloatPow, FloatTrueDiv and IntTrueDiv of numbers to."""
        op, args = n.op, n.args
        value = None
        if op == "const":
            value = float(args[0])
            if value != args[0]:
                raise declined(f"the constant {args[0]} is not a double")
        elif op == "fconst":
            value = n.hint
        elif op == "ffromint":
            return self._emit("tofloat", self._int(args[0]))
        elif op in ("fpow", "fdiv") and all(a.op == "fconst" for a in args):
            x, y = (a.hint for a in args)
            ratio = self.ctx.int_ratio(n) if op == "fdiv" else None
            value = x**y if op == "fpow" else int(x) / int(y) if ratio is not None else x / y
        elif op == "fpow" and args[1].op == "fconst" and args[1].hint == 0.5:
            return self._emit("fsqrt", self._float(args[0]))
        elif op == "fdiv" and self.ctx.int_ratio(n) is None:
            return self._emit("fdiv", self._float(args[0]), self._float(args[1]))
        if value is None:
            raise declined(f"no exact float lowering for the IR node {op} in {render(n)}")
        if not math.isfinite(value):
            raise declined(f"the constant {value} is not a double")
        return self._constant(f64_bits(value))

    def _power_of(self, base: Node, exponent: int) -> int:
        if exponent < 0:
            raise declined(f"{render(base)} ** {exponent} is not a natural power")
        return self._natural_power(self._int(base), exponent)

    def _divide_nodes(self, num: Node, den: Node, *, floor: bool) -> int:
        n, d = self._int(num), self._int(den)
        nlo, nhi = self.ctx.bounds(num)
        dlo, dhi = self.ctx.bounds(den)
        n_sign = 1 if nlo is not None and nlo >= 0 else -1 if nhi is not None and nhi <= 0 else 0
        d_sign = -1 if dhi is not None and dhi < 0 else 1 if dlo is not None and dlo > 0 else 0
        return self._divide_rows(n, d, n_sign, d_sign, floor)


def _contains(n: Node, op: str) -> bool:
    todo, seen = [n], set()
    while todo:
        m = todo.pop()
        if m.op == op:
            return True
        if m.id not in seen:
            seen.add(m.id)
            todo.extend(children(m))
    return False
