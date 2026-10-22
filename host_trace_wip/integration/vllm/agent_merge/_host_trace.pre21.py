"""Host tracing (private): run a CUDA program's host code once over
symbolic sizes, strides and addresses and record what it launched and
every condition it branched on, so that a CUDA graph captured once can be
replayed for other inputs by patching its kernel parameters.

This module holds the trace's ShapeEnv, which records the guards.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from typing import Any, overload, TYPE_CHECKING

import sympy

from torch._guards import GuardSource, ShapeGuard, SLoc, Source
from torch.cuda import _host_trace_ir as _ir
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.cuda._host_trace_program import f32_bits
from torch.utils._sympy.functions import Mod, Where
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges


if TYPE_CHECKING:
    from collections.abc import Callable

    import torch
    from torch.types import FloatLikeType, IntLikeType


# the test suites raise an unexpected exception inside a trace, a capture or
# a harvest instead of declining the call
raise_unexpected = False

# an untrusted trace's symbolic backend: "ir" (torch.cuda._host_trace_ir) or
# "sympy" (_TraceShapeEnv); a trusted trace is always sympy's
symbolic = "ir"


class Declined(RuntimeError):
    """The host did something the tracer does not describe; the call runs
    eagerly instead."""

    # whether trace() had run its warm-up call when it declined, and that
    # call's return value: the warm-up was the call, so a consumer returns it
    warm_up_ran: bool = False
    warm_up_result: Any = None
    # whether a later call of the same class may trace: the fallback changed
    # what the trace depends on (it filled an autotune cache)
    retry: bool = False
    # the operator whose fake kernel's metadata disagreed with its real
    # output at the warm-up: a later trace of a call to it declines too
    meta_op: Any = None
    # nothing to capture: every operation runs eagerly, so a replay would
    # too; not a decline to report
    uncaptured: bool = False
    # the trace's operator calls are not the warm-up's (_check_witness)
    witness: bool = False


def declined(msg: str) -> Declined:
    """The decline of `msg`: every decline's message is built here."""
    return Declined(f"host_trace: {msg} (declined)")


class ProcessHold:
    """A process-wide change held while any thread's trace is inside: the
    first to enter calls `hold`, the last to leave `release`. Python has no
    per-thread switch for what is held (a class's methods, the collector)."""

    def __init__(self, hold: Callable[[], None], release: Callable[[], None]) -> None:
        self._hold, self._release = hold, release
        self._lock = threading.Lock()
        self._depth = 0

    def __enter__(self) -> None:
        with self._lock:
            if self._depth == 0:
                self._hold()
            self._depth += 1

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._depth -= 1
            if self._depth == 0:
                self._release()


@dataclass(frozen=True)
class _Src(Source):
    nm: str

    @property
    def _name_template(self) -> str:
        return self.nm

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        return GuardSource.LOCAL


_BOOL_ATOMS = (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)
_NO_SLOC = SLoc(None, None)
# the binary SymNode methods defined on a subset of their operands, by the
# names sym_node.py memoizes them under
_DIVISIONS = frozenset({"int_floordiv", "mod", "int_truediv", "float_truediv"})


class _SymOpMemo(dict):
    """ShapeEnv._symop_cache (Note [symbolic op memo]): every new binary
    SymNode operation is stored here once, so the trace sees a division the
    moment it is created, before any guard or value is built on it."""

    def __init__(self, on_division: Callable[[Any, Any, Any], None]) -> None:
        super().__init__()
        self.on_division = on_division

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        method, lhs, rhs, _version = key
        if method in _DIVISIONS:
            self.on_division(lhs, rhs, value[0])


class _TraceShapeEnv(ShapeEnv):
    """The trace's ShapeEnv: a guard is the expression the host evaluated,
    decided by its hint.

    A tape's guards are re-evaluated at every replay, so none of the ordinary
    evaluate_expr's static reasoning is needed: every non-constant expression
    the host branches on is recorded once, as written, with its value under
    the hints. Symbols are never replaced and value ranges never refined, so a
    specialization stays an explicit guard (`Eq(s, 1)`) and a fact the host
    depended on is never left unrecorded; the price is redundant guards. A
    partial operation (a division, a modulo) records its domain when it is
    created (`domain`), so the ordered guard list checks `Ne(divisor, 0)`
    before anything built on the operation.

    With `trusted` inputs the symbols' ranges are the caller's facts: an
    expression they decide is no guard.
    """

    # a key built from sizes (a compile cache's) is a specialization, as any value the host reads
    hash_symints_by_value = True
    # eager's host reads a concrete value where the trace's statically_known_true
    # would not decide (sdp_utils' mask shape check picking cuDNN attention)
    static_reads_guard = True

    def __init__(self, trusted: bool = False) -> None:
        super().__init__(duck_shape=False, specialize_zero_one=False)
        self.trusted = trusted
        self._index: dict[sympy.Basic, int] = {}
        # per guard, the top-level op (_Trace.ops) that recorded it, or None: a
        # graph-level guard, recorded outside an op or evaluated again outside it
        self.owners: list[int | None] = []
        # the op a guard recorded now belongs to
        self.op: int | None = None
        self._symop_cache = _SymOpMemo(self.domain)

    @overload
    def symbol(
        self, value: int, name: str, *, positive: bool = False
    ) -> torch.SymInt: ...

    @overload
    def symbol(
        self, value: float, name: str, *, positive: bool = False
    ) -> torch.SymFloat: ...

    def symbol(
        self, value: int | float, name: str, *, positive: bool = False
    ) -> IntLikeType | FloatLikeType:
        src = _Src(name)
        if positive:
            sym = self.create_symbol(
                value,
                src,
                DimDynamic.DYNAMIC,
                positive=True,
                do_not_specialize_zero_one=True,
            )
            # create_symbol's range admits 0; a division's domain is decided
            # against the declared one
            self.var_to_range[sym] = ValueRanges(1, int_oo)
        else:
            sym = self.create_unspecified_symbol(value, src, DimDynamic.DYNAMIC)
        if isinstance(value, float):
            return self.create_symfloatnode(sym, hint=value, source=src)
        return self.create_symintnode(sym, hint=value, source=src)

    def evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: int | bool | float | None = None,
        fx_node: Any = None,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
        *,
        forcing_spec: bool = False,
    ) -> sympy.Basic:
        if isinstance(orig_expr, _BOOL_ATOMS) or orig_expr.is_number:
            return orig_expr
        if self.trusted:
            decided = self._maybe_evaluate_static(orig_expr)
            if decided is not None:
                return decided
        if hint is None:
            hint = self.guarding_hint_or_throw(orig_expr)
        concrete = sympy.sympify(hint)
        if concrete is sympy.true:
            g = orig_expr
        elif concrete is sympy.false:
            g = sympy.Not(orig_expr)
        else:
            g = sympy.Eq(orig_expr, concrete)
        # sympy decides a relation from the symbols' declared properties at
        # construction: a true one is no guard, a false one contradicts the hint
        if g is sympy.false:
            raise AssertionError(f"host_trace: {orig_expr} is not {hint}")
        self._record(g, size_oblivious)
        return concrete

    def _record(self, g: sympy.Basic, size_oblivious: bool = False) -> None:
        if g is sympy.true:
            return
        i = self._index.get(g)
        if i is None:
            self._index[g] = len(self.guards)
            self.guards.append(ShapeGuard(g, _NO_SLOC, size_oblivious))
            self.owners.append(self.op)
        elif self.owners[i] != self.op:
            self.owners[i] = None

    def forget(self, n: int) -> None:
        """Drops the guards recorded after the first n."""
        for g in self.guards[n:]:
            del self._index[g.expr]
        del self.guards[n:], self.owners[n:]

    def domain(self, lhs: Any, rhs: Any, out: Any) -> None:
        """A partial operation's domain: the divisor of a floor division, a
        modulo or a true division is nonzero, and torch's Mod (which
        sym_node.py builds only when it knows both operands nonnegative) has
        both operands nonnegative. A condition the declared ranges decide (a
        size is positive, a literal is nonzero) is no guard."""
        self._record_undecided(sympy.Ne(rhs, 0), rhs, lambda r: 0 not in r)
        if isinstance(out, Mod):
            self._record_undecided(sympy.Ge(lhs, 0), lhs, lambda r: r.lower >= 0)
            self._record_undecided(sympy.Ge(rhs, 0), rhs, lambda r: r.lower >= 0)

    def _record_undecided(
        self, g: Any, e: Any, decided: Callable[[ValueRanges], bool]
    ) -> None:
        if g is sympy.true:
            return
        try:
            r = bound_sympy(e, self.var_to_range)
        except (KeyError, NotImplementedError):  # no rule for a function in e
            r = ValueRanges.unknown()
        if not decided(r):
            self._record(g)

    def _set_replacement(self, a: sympy.Symbol, tgt: sympy.Expr, msg: str) -> None:
        raise AssertionError(f"host_trace: {a} is never replaced ({msg})")


class BitLength(sympy.Function):
    """int.bit_length (of the magnitude): the program's "bitlength" row."""

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, a: sympy.Expr) -> sympy.Integer | None:
        if isinstance(a, sympy.Integer):
            return sympy.Integer(int(a).bit_length())
        return None


def bit_length(x: IntLikeType) -> IntLikeType:
    """x.bit_length() for a traced host: a size's is an expression."""
    if isinstance(x, int):
        return x.bit_length()
    node = x.node
    if isinstance(node, _ir.IRSymNode):
        return node.bit_length()
    env, hint = node.shape_env, node.hint
    if hint is None:
        raise NotImplementedError("host_trace: the bit length of an unbacked size")
    # pyrefly: ignore [missing-attribute]
    return env.create_symintnode(BitLength(node.expr), hint=hint.bit_length())


def select(c: torch.SymBool, a: IntLikeType, b: IntLikeType) -> IntLikeType:
    """`c ? a : b` for a traced host: a Where, which the program lowers to a select
    row (sym_ite's Piecewise makes sympy fold every expression it enters)."""
    node = c.node
    if isinstance(node, _ir.IRSymNode):
        return node.select(a, b)
    exprs = [sympy.Integer(x) if isinstance(x, int) else x.node.expr for x in (a, b)]
    hints = [x if isinstance(x, int) else x.node.hint for x in (a, b)]
    if node.hint is None or None in hints:
        raise NotImplementedError("host_trace: a select over an unbacked size")
    # pyrefly: ignore [missing-attribute]
    return node.shape_env.create_symintnode(Where(node.expr, *exprs), hint=hints[0] if node.hint else hints[1])


class F32Div(sympy.Function):
    """static_cast<float>(a) / b's bits as an int32: the program's "f32div" row."""

    is_integer = True

    @classmethod
    def eval(cls, a: sympy.Expr, b: sympy.Expr) -> sympy.Integer | None:
        if isinstance(a, sympy.Integer) and isinstance(b, sympy.Integer) and b > 0:
            return sympy.Integer(f32_bits(int(a), int(b)))
        return None


def f32_div(a: IntLikeType, b: IntLikeType) -> IntLikeType:
    """For a traced host: the bits of float(a) / float(b) in float32."""
    if isinstance(a, int) and isinstance(b, int):
        return f32_bits(a, b)
    node = (b if isinstance(a, int) else a).node
    if isinstance(node, _ir.IRSymNode):
        return node.f32_div(a, b)
    env = node.shape_env
    exprs = [sympy.Integer(x) if isinstance(x, int) else x.node.expr for x in (a, b)]
    hints = [x if isinstance(x, int) else x.node.hint for x in (a, b)]
    if None in hints:
        raise NotImplementedError("host_trace: a float division of an unbacked size")
    # pyrefly: ignore [missing-attribute]
    return env.create_symintnode(F32Div(*exprs), hint=f32_bits(*hints))
