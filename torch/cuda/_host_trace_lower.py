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

from typing import TYPE_CHECKING

import sympy
from sympy.logic.boolalg import BooleanFalse, BooleanTrue

from torch.cuda._host_trace import BitLength, declined
from torch.cuda._host_trace_program import MAX_I64, MIN_I64, OutOfDomain
from torch.utils._sympy.functions import (
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    CeilToInt,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    PowByNatural,
    PythonMod,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.cuda._host_trace_program import IntegerProgram


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
        guards = tuple(guards)
        conditions = []
        for symbol in self.sources:
            if symbol.is_positive or symbol.is_nonnegative:
                bound = self._constant(1 if symbol.is_positive else 0)
                conditions.append(self._emit("ge", self.lower(symbol), bound))
        rows = tuple(self.lower(g) for g in guards)
        for g, row in zip(guards, rows):
            if self.program.values[row] != 1:
                if g.free_symbols & addresses:
                    raise declined(f"the host read a traced address as an int (the guard {g})")
                raise AssertionError(f"host_trace: the guard {g} is false at the hints")
        result = self._constant(1)
        for row in (*conditions, *rows):
            result = self._emit("and", result, row)
        return result

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
        if type(e) in _COMPARISONS:
            lhs, rhs = (self._lower_integer(x) for x in e.args)
            return self._emit(_COMPARISONS[type(e)], lhs, rhs)
        if isinstance(e, sympy.And):
            rows = [self._lower_boolean(x) for x in e.args]
            result = rows[0]
            for row in rows[1:]:
                result = self._emit("and", result, row)
            return result
        if isinstance(e, sympy.Or):
            rows = [self._lower_boolean(x) for x in e.args]
            result = rows[0]
            for row in rows[1:]:
                result = self._emit("select", result, self._constant(1), row)
            return result
        if isinstance(e, sympy.Not):
            row = self._lower_boolean(e.args[0])
            return self._emit("select", row, self._constant(0), self._constant(1))
        if isinstance(e, Identity):
            return self._lower_integer(e.args[0])
        if isinstance(e, (sympy.Add, sympy.Mul)):
            op = "add" if isinstance(e, sympy.Add) else "multiply"
            rows = [self._lower_integer(x) for x in e.args]
            result = rows[0]
            for row in rows[1:]:
                result = self._emit(op, result, row)
            return result
        if isinstance(e, (sympy.Pow, PowByNatural)):
            return self._power(e)
        if isinstance(e, FloorDiv):
            return self._divide(*e.args, floor=True)
        if isinstance(e, (CeilToInt, FloorToInt)) and isinstance(e.args[0], IntTrueDiv):
            # the host rounds a float quotient, exactly so below 2**53
            return self._divide(*e.args[0].args, floor=isinstance(e, FloorToInt))
        if isinstance(e, (PythonMod, Mod)):
            # n - d * (n // d); torch's Mod is the same on its nonnegative domain
            n, d = e.args
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
        raise declined(f"no integer lowering for {type(e).__name__} in {e}")

    def _lower_integer(self, e: sympy.Basic) -> int:
        if e.is_Boolean or isinstance(e, sympy.Rel):
            raise declined(f"the boolean {e} is used as an integer")
        return self._lower(e)

    def _lower_boolean(self, e: sympy.Basic) -> int:
        if not (e.is_Boolean or isinstance(e, sympy.Rel)):
            raise declined(f"the integer {e} is used as a boolean")
        return self._lower(e)

    def _power(self, e: sympy.Basic) -> int:
        base, exponent = e.args
        if base == 2 and not isinstance(exponent, sympy.Integer):
            return self._emit("lshift", self._constant(1), self._lower_integer(exponent))
        if not (isinstance(exponent, sympy.Integer) and exponent >= 0):
            raise declined(f"{e} is not a natural power")
        result = self._constant(1)
        square = self._lower_integer(base)
        n = int(exponent)
        while n:
            if n & 1:
                result = self._emit("multiply", result, square)
            n >>= 1
            if n:
                square = self._emit("multiply", square, square)
        return result

    def _divide(self, n_expr: sympy.Basic, d_expr: sympy.Basic, *, floor: bool) -> int:
        """Python's floor (or ceiling) division from the evaluator's divisions,
        which admit a nonnegative numerator and a positive divisor:
        n / d = -n / -d, and for d > 0 and n < 0 floor(n / d) = -ceil(-n / d).
        Signs sympy proves need no select."""
        n, d = self._lower_integer(n_expr), self._lower_integer(d_expr)
        n_sign = 1 if n_expr.is_nonnegative else -1 if n_expr.is_nonpositive else 0
        if d_expr.is_negative:
            n, d, n_sign = self._negate(n), self._negate(d), -n_sign
        elif not d_expr.is_positive:
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
