"""Host tracing (private): run a CUDA program's host code once over
symbolic sizes, strides and addresses and record what it launched and
every condition it branched on, so that a CUDA graph captured once can be
replayed for other inputs by patching its kernel parameters.

This module holds the trace's ShapeEnv, which records the guards.
"""

from __future__ import annotations

import functools
import sys
import threading
from dataclasses import dataclass
from typing import Any, overload, TYPE_CHECKING

import sympy

from torch._guards import GuardSource, ShapeGuard, Source
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

# a trace runs an op's Python CUDA kernel registered with Library.impl (vLLM's
# and SGLang's direct_register_custom_op) as a custom op's; False leaves the
# call an eager step whose reason names the kernel
library_impls = True

# with library_impls, a Library.impl kernel is an eager step where its body
# calls a host-trace entry (a replay or a bound call, which the trace would
# inline at the Python state of the trace) or where the trace itself runs
# inside such a body (a nested call, whose body may read state the enclosing
# call set); False traces both
library_impls_reentry = True

# the same for a torch.library.custom_op kernel, descended whatever
# library_impls is
custom_op_reentry = True

# an untrusted trace's symbolic backend: "ir" (torch.cuda._host_trace_ir) or
# "sympy" (_TraceShapeEnv); a trusted trace is always sympy's
symbolic = "ir"

# programmatic (PDL) Triton and CuTe launches are traced as launches; False
# declines them (each runs as an eager call)
trace_pdl = True

# Triton launches with host TensorDescriptor arguments are traced, each
# descriptor encoded at replay from the trace's sizes; False declines them
# (each runs as an eager call)
trace_triton_tma = True

# with trace_triton_tma, a descriptor is encoded as Triton's launcher encodes
# it (its padding as the out-of-bounds fill, no CuTe DSL bit, its bit-21
# clear); where the native encode takes it (torch._C._host_trace_tma_flavors).
# False, or a native replay without it, runs each Triton launch with
# descriptors as an eager call: no other encode is Triton's launch
triton_tma_encode = True

# a TMA descriptor launches with the bits its library's encode adds to the
# driver's cuTensorMapEncodeTiled output, eager's bytes: the CuTe DSL's byte-8
# bit 0x02, and Triton's (as CUTLASS's) clear of bit 21 on drivers <= 13010,
# their workaround of a driver bug. False launches the driver's output as is:
# a Triton descriptor where Triton clears bit 21 is an eager call (a guard),
# and the DSL's bit only enters CuTe's stand-in compare; where the native
# replay takes it (torch._C._host_trace_tma_raw), else TMA launches decline
tma_library_bits = True

# with traced_impls on, a trace runs a registered op's launcher (register_traced_impl)
# in place of its other routes
traced_impls = True
_TRACED_IMPLS: dict[Any, Callable[..., Any]] = {}

# a traced size's SymInt.bit_length() (SGLang's next_power_of_2 is
# 1 << (n - 1).bit_length()) is an expression; otherwise it specializes the size
symint_bit_length = True


def register_traced_impl(op: Any, launcher: Callable[..., Any]) -> None:
    """Under a trace with traced_impls on, a call to the OpOverload `op` runs
    launcher(*args, **kwargs) on the traced tensors, as a Python CUDA kernel
    does: it records the op's launches (current_trace().record_launch) and
    returns its output. Where it declines (current_trace().decline), the call
    takes the trace's other routes: an opaque provider, else an eager call."""
    _TRACED_IMPLS[op] = launcher


# a function marked dispatch_unit is one op under a trace, its guards its own
# (OpRec.guards): a Python kernel choice ahead of a launch (SGLang's `if m <=
# 32:` ahead of its TGV GEMM) dispatches again where it flips, as a custom
# op's does, rather than guarding the graph, which traces again. False runs it
# as plain Python
dispatch_units = True


def dispatch_unit(fn: Callable[..., Any]) -> Callable[..., Any]:
    """fn(*args, **kwargs), under a trace with dispatch_units on, as one op
    that runs again alone from its arguments (a redispatch). The contract a
    custom op declares: what fn does is its launches, its allocations and its
    outputs; Python state it writes is read by no later code, and it reads
    traced tensors through its arguments only. A guard outside such a unit
    stays the graph's: the Python ahead of an op may decide more than it."""

    @functools.wraps(fn)
    def unit(*args: Any, **kwargs: Any) -> Any:
        tape = sys.modules.get("torch.cuda._host_trace_tape")
        tr = tape.current_trace() if tape is not None and dispatch_units else None
        if tr is None:
            return fn(*args, **kwargs)
        return tr.op(fn, args, kwargs, lambda: fn(*args, **kwargs), host=True, redo=lambda a, k: unit(*a, **k))

    return unit


# a view's symbolic size-1 dim may take another stride than eager's (it
# addresses nothing), so the trace does not branch on that size being 1
free_size_one_strides = True

# a tape keeps, per guard, every op that read it (Tape.owner_sets), and a guard
# only ops read (two or more) is each one's own (OpRec.guards, their selectors)
# rather than graph-level: a call where it fails dispatches those ops again
# (_host_trace_redispatch), no trace. False keeps no owner sets
shared_op_guards = True

# a bound opaque call's operand address is its root plus its offset, as a
# KeyedSite's row of any other key is (lower_tape's keyed_site): its key's
# exact sizes decide whether it is empty, so the trace does not branch on
# numel() == 0 (data_ptr) for it
bound_addresses_unguarded = True

# eager's validity checks of a traced call's values (as_strided's bounds, a
# negative size) are conditions a variant's calls must meet (its valid row),
# not guards: a call that fails one misses, and its trace's warm-up raises
# eager's error with the call's values. One the declared domains or a recorded
# guard decide adds nothing. False guards each, as a branch
validity_checks = True

# with free_size_one_strides, a view's -1 size is the numel over its other sizes,
# so the view takes the free strides rather than the fake kernel's guards
free_inferred_views = True

# a trace with no traced launch keeps its guards, and a call they hold for runs
# eagerly with no trace; off, each such call of a new class traces again
keep_eager_regions = True

# a trace (trace(), a replay's miss) and a CuTe intercept run their Python on a
# fresh, oversized frame-stack chunk (on_a_fresh_stack_chunk)
fresh_stack_chunk = True

# an entry made with this off is served by HostTraceReplay's Python dispatch
# (_call_slow), not the C++ entry; the variants' commits are native either way
cpp_entry = True


@functools.cache
def _ballast(slots: int) -> Callable[..., Any]:
    names = " = ".join(f"_{i}" for i in range(slots))
    ns: dict[str, Any] = {}
    exec(f"def ballast(f, *a, **k):\n    if f is None:\n        {names} = None\n    return f(*a, **k)\n", ns)
    return ns["ballast"]


def on_a_fresh_stack_chunk(slots: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorates a function to run on a fresh frame-stack chunk with room for
    about 8 * slots bytes of frames above it.

    CPython 3.11 and 3.12 keep frames on a per-thread stack of 16 KiB chunks,
    and _PyThreadState_PopFrame unmaps a chunk as soon as its first frame
    returns: a trace, which recurses through the host ops and the SymInt
    dunders, pays an mmap, a page fault and a munmap (about 50 us) each time its
    depth crosses a chunk boundary. A frame of `slots` never-bound locals takes
    a chunk of its size rounded up to a power of two, at least as large again.
    3.13+ caches a spare chunk instead: there this returns fn unchanged.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if sys.version_info >= (3, 13):
            return fn
        ballast = _ballast(slots)

        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> Any:
            return ballast(fn, *args, **kwargs) if fresh_stack_chunk else fn(*args, **kwargs)

        return run

    return decorate


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
    # work on a stream other than the trace's: the decline holds for the
    # argument contract whatever the shapes, and warns (_Trace.check_stream)
    side_stream: bool = False


class EagerFallback(RuntimeError):
    """Under HostTraceReplay(fullgraph=True), a call or an operation that
    would run eagerly: a decline, a whole-call fallback, or an eager step;
    under forbid_learners=True, an opaque call that does not learn at its
    trace."""


def declined(msg: str) -> Declined:
    """The decline of `msg`: every decline's message is built here."""
    return Declined(f"host_trace: {msg} (declined)")


def _drop_tracebacks(e: BaseException | None) -> None:
    """Drops the tracebacks of `e` and its causes, keeping the chain: an exception
    kept past its handler holds every frame up to its raise, the caller's too."""
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        e.__traceback__ = None
        e = e.__cause__ or e.__context__


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

    @property
    def symbolic_bit_length(self) -> bool:  # type: ignore[override]
        return symint_bit_length

    def __init__(self, trusted: bool = False) -> None:
        super().__init__(duck_shape=False, specialize_zero_one=False)
        self.trusted = trusted
        self._index: dict[sympy.Basic, int] = {}
        # per guard, the top-level op (_Trace.ops) that recorded it, or None: a
        # graph-level guard, recorded outside an op or evaluated again outside it
        self.owners: list[int | None] = []
        # with shared_op_guards on (_Trace sets a list), per guard every op that
        # recorded or evaluated it, None for outside an op
        self.owner_sets: list[set[int | None]] | None = None
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
        active = getattr(_ir.ACTIVE, "trace", None)
        if active is not None and active.shape_env is not self:
            raise _ir._foreign("a guard on a symbol of another trace")
        i = self._index.get(g)
        if i is None:
            i = self._index[g] = len(self.guards)
            self.guards.append(ShapeGuard(g, _ir.guard_sloc(), size_oblivious))
            self.owners.append(self.op)
            if self.owner_sets is not None:
                self.owner_sets.append(set())
        elif self.owners[i] != self.op:
            self.owners[i] = None
        if self.owner_sets is not None:
            self.owner_sets[i].add(self.op)

    def forget(self, n: int) -> None:
        """Drops the guards recorded after the first n."""
        for g in self.guards[n:]:
            del self._index[g.expr]
        del self.guards[n:], self.owners[n:]
        if self.owner_sets is not None:
            del self.owner_sets[n:]

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
