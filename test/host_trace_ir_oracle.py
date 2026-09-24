# Owner(s): ["module: cuda"]
"""Backend equivalence of the host-tracing recorder, as a test.

`trace_both` traces a call as `ht.trace` does under the sympy backend, then
once more under the IR backend (`_host_trace.symbolic = "ir"`, no warm-up),
compares the two tapes and returns the IR tape, so a test that runs under it
builds and replays the IR tape. The first difference raises
`BackendDifference`. `assert_family` runs it over every test of one family's
class; `run_family` returns the report (what agreed, what differed, the IR's
census).

Compared: the symbols (names and sources, in creation order), every section
of the tape's structure but the guards by value at the trace's hints and at
the alternate assignments of test/host_trace_two_hint.py (the two tapes name
their symbols alike, so a value is also compared as the IR node the two
expressions canonicalize to), and the guards as IR nodes: a guard of one tape
that the other tape lacks must be implied by the other tape's guards under
the dedupe pass's rules (`Env.implied`), else the tapes differ. The IR's
census (operations it could not express) is empty by construction when the
trace succeeded; an IR decline where sympy traced is a difference carrying
the census."""

from __future__ import annotations

import contextlib
import functools
import gc
import itertools
import unittest
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from unittest import mock

import sympy

from host_trace_two_hint import _env_values, _is_symbolic, _structure, alternate_hints

from torch.cuda import _host_trace as ht, _host_trace_ir as _ir
from torch.utils._sympy import functions as tf


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


_trace = ht.trace  # the first run, whatever a family run patches ht.trace to


def _expr(v: Any) -> Any:
    return v.node.expr if isinstance(v, ht._SYM_TYPES) else v


class BackendDifference(AssertionError):
    """The two backends traced different programs; the message names the
    first difference."""


@dataclass
class Comparison:
    guards_sympy: int
    guards_ir: int
    implied_by_ir: int  # sympy-tape guards the IR tape's guards imply
    implied_by_sympy: int  # IR-tape guards the sympy tape's guards imply
    values: int  # symbolic use-site values compared
    same_node: int  # of those, the same IR node once canonicalized
    folded: int  # of those, a number on one tape and symbolic on the other
    export_ms: float
    nodes: int
    census: list = field(default_factory=list)


@contextlib.contextmanager
def backend(name: str) -> Iterator[None]:
    prev = ht.symbolic
    ht.symbolic = name
    try:
        yield
    finally:
        ht.symbolic = prev


class _ToIR:
    """sympy (a tape's exported form) -> the IR node of `env`'s context: the
    symbols by name, every other form through the IR's own constructors."""

    def __init__(self, env: _ir.Env) -> None:
        self.ctx = env.ctx
        self.memo: dict = {}

    def __call__(self, e: Any) -> _ir.Node:
        r = self.memo.get(e)
        if r is None:
            r = self._to_ir(e)
            self.memo[e] = r
        return r

    def _to_ir(self, e: Any) -> _ir.Node:
        c = self.ctx
        if isinstance(e, sympy.Symbol):
            node = c.symbols.get(e.name)
            if node is None:
                raise BackendDifference(
                    f"symbol {e} of the sympy tape has no IR symbol"
                )
            return node
        if isinstance(e, sympy.Integer):
            return c.const(int(e))
        if e is sympy.true:
            return c.true()
        if e is sympy.false:
            return c.false()
        if isinstance(e, (sympy.Float, sympy.Rational)):
            return c.fconst(float(e))
        if isinstance(e, tf.Identity):
            return self(e.args[0])
        if isinstance(e, sympy.Add):
            r = c.const(0)
            for a in e.args:
                x = self(a)
                r = c.fbin("fadd", r, x) if (x.is_float or r.is_float) else c.add(r, x)
            return r
        if isinstance(e, sympy.Mul):
            r = c.const(1)
            for a in e.args:
                x = self(a)
                r = c.fbin("fmul", r, x) if (x.is_float or r.is_float) else c.mul(r, x)
            return r
        if isinstance(e, sympy.Pow):
            b, x = e.args
            if x.is_Integer and x >= 0:
                bi = self(b)
                if bi.is_float:
                    return c.fbin("fpow", bi, c.fconst(float(x)))
                return c.pow(bi, int(x))
            return c.fbin("fpow", self(b), self(x))
        if isinstance(e, tf.FloorDiv):
            return c.floordiv(self(e.args[0]), self(e.args[1]))
        if isinstance(e, tf.CeilDiv):
            return c.neg(c.floordiv(c.neg(self(e.args[0])), self(e.args[1])))
        if isinstance(e, (tf.Mod, tf.PythonMod, sympy.Mod)):
            return c.mod(self(e.args[0]), self(e.args[1]))
        if isinstance(e, (tf.Max, sympy.Max)):
            return c.max(*(self(a) for a in e.args))
        if isinstance(e, (tf.Min, sympy.Min)):
            return c.min(*(self(a) for a in e.args))
        if isinstance(e, tf.IsNonOverlappingAndDenseIndicator):
            half = len(e.args) // 2
            xs = [self(a) for a in e.args]
            return c.nod(xs[:half], xs[half:])
        if isinstance(e, tf.ToFloat):
            return c.to_float(self(e.args[0]))
        if isinstance(e, ht.Float32):
            return c.fun("fround32", self(e.args[0]))
        if isinstance(e, (tf.FloatTrueDiv, tf.IntTrueDiv)):
            return c.fbin("fdiv", self(e.args[0]), self(e.args[1]))
        if isinstance(e, tf.FloatPow):
            return c.fbin("fpow", self(e.args[0]), self(e.args[1]))
        if isinstance(e, tf.PowByNatural):
            return c.pow(self(e.args[0]), int(e.args[1]))
        if isinstance(e, tf.TruncToInt):
            return c.ftrunc(self(e.args[0]))
        if isinstance(e, (tf.CeilToInt, tf.FloorToInt)):
            ceil = isinstance(e, tf.CeilToInt)
            x = self(e.args[0])
            ratio = c.int_ratio(x)
            if ratio is not None:
                return (c.ceildiv if ceil else c.floordiv)(*ratio)
            return c.fint("fceil" if ceil else "ffloor", x)
        if isinstance(e, tf.OpaqueUnaryFn_sqrt):
            return c.fun("fsqrt", self(e.args[0]))
        rel = {
            sympy.Eq: c.eq,
            sympy.Ne: c.ne,
            sympy.Lt: c.lt,
            sympy.Le: c.le,
            sympy.Gt: c.gt,
            sympy.Ge: c.ge,
        }.get(type(e))
        if rel is not None:
            return rel(self(e.args[0]), self(e.args[1]))
        if isinstance(e, sympy.And):
            return c.junction("and", [self(a) for a in e.args])
        if isinstance(e, sympy.Or):
            return c.junction("or", [self(a) for a in e.args])
        if isinstance(e, sympy.Not):
            return c.not_(self(e.args[0]))
        raise BackendDifference(
            f"no IR form for the sympy node {type(e).__name__}: {e}"
        )


def compare_backends(
    first: ht.Tape, second: ht.Tape, extra_points: list[dict]
) -> tuple[str | None, Comparison]:
    """The first difference between a sympy tape and an IR tape of one call,
    or None, with the comparison's counts."""
    env = second.ir_env
    if env is None:
        raise BackendDifference("the second tape was not traced by the IR backend")
    seq1 = [
        (str(k), v[0].name if v else "")
        for k, v in first.shape_env.var_to_sources.items()
    ]
    seq2 = [
        (str(k), v[0].name if v else "")
        for k, v in second.shape_env.var_to_sources.items()
    ]
    for k, (a, b) in enumerate(itertools.zip_longest(seq1, seq2)):
        if a != b:
            return (
                f"symbol {k} is {a} under sympy and {b} under the IR: the host created "
                "its values in a different order",
                Comparison(0, 0, 0, 0, 0, 0, 0, second.export_ms, len(env.ctx.nodes)),
            )
    points = [_env_values(first.shape_env), *extra_points]
    prog = ht._Evaluator()
    to_ir = _ToIR(env)
    counts = Comparison(
        len(first.guards),
        len(second.guards),
        0,
        0,
        0,
        0,
        0,
        second.export_ms,
        len(env.ctx.nodes),
    )

    def text(v: Any) -> str:
        if isinstance(v, ht._SYM_TYPES):
            return f"`{prog.guard_text(v.node.expr)}`"
        if isinstance(v, sympy.Basic):
            return f"`{prog.guard_text(v)}`"
        return repr(v)

    def same_expr(a: Any, b: Any) -> bool:
        counts.values += 1
        ea, eb = _expr(a), _expr(b)
        try:
            if to_ir(ea) is to_ir(eb):
                counts.same_node += 1
                return True
        except BackendDifference:
            pass
        return all(prog.ev(a, pt) == prog.ev(b, pt) for pt in points)

    def same_leaf(a: Any, b: Any) -> bool:
        if _is_symbolic(a) != _is_symbolic(b):
            # a value the IR folded to a number where sympy kept a symbol
            # (or the reverse) is the same value when equal at every point
            if _is_symbolic(a) or _is_symbolic(b):
                counts.values += 1
                counts.folded += 1
                return all(prog.ev(a, pt) == prog.ev(b, pt) for pt in points)
            return False
        if not _is_symbolic(a):
            return ht._constant(a) == ht._constant(b)
        return same_expr(a, b)

    def walk(a: Any, b: Any, path: str) -> str | None:
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return f"{path}: keys {sorted(a.keys() ^ b.keys())} differ"
            for k in a:
                # `const` says whether a value is symbolic: a value one backend
                # folded to a number and the other kept symbolic is compared
                # by value under its `expr` key
                if k in ("expected", "block", "symbolic", "const"):
                    continue
                r = walk(a[k], b[k], f"{path}.{k}" if path else k)
                if r is not None:
                    return r
            return None
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return f"{path}: {len(a)} under sympy, {len(b)} under the IR"
            for k, (x, y) in enumerate(zip(a, b)):
                r = walk(x, y, f"{path}[{k}]")
                if r is not None:
                    return r
            return None
        if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        if not same_leaf(a, b):
            return f"{path}: {text(a)} vs {text(b)}"
        return None

    d1, d2 = _structure(first), _structure(second)
    for section in d1:
        if section == "guards":
            continue
        r = walk(d1[section], d2[section], section)
        if r is not None:
            if section == "launches":
                j = int(r[len("launches[") : r.index("]")])
                r = f"launch {j} (`{first.launches[j]['kernel']}`) {r}"
            return r, counts
    # the guards, as IR nodes: equal as multisets, or each odd one implied
    gs = [to_ir(g) for g in first.guards]
    gi = [to_ir(g) for g in second.guards]
    ids_i = {g.id for g in gi}
    ids_s = {g.id for g in gs}
    for g, e in zip(gs, first.guards):
        if g.id in ids_i or g.op == "true":
            continue
        if not env.implied(gi, g):
            return (
                f"guard `{prog.guard_text(e)}` of the sympy tape is neither on the IR tape "
                f"nor implied by its guards (IR form {_ir.render(g)})",
                counts,
            )
        counts.implied_by_ir += 1
    for g, e in zip(gi, second.guards):
        if g.id in ids_s:
            continue
        if not env.implied(gs, g):
            return (
                f"guard `{prog.guard_text(e)}` of the IR tape is neither on the sympy tape "
                f"nor implied by its guards (IR form {_ir.render(g)})",
                counts,
            )
        counts.implied_by_sympy += 1
    return None, counts


def trace_both(
    fn: Callable[..., Any],
    args: tuple,
    device: int | None = None,
    *,
    warm_up: bool = True,
    log: list[Comparison] | None = None,
) -> ht.Tape:
    """`ht.trace` under the sympy backend, then the same call under the IR
    backend (no warm-up: the first run's did it). Raises BackendDifference at
    the first difference between the two tapes; returns the IR tape and
    appends the comparison's counts to `log`."""
    with backend("sympy"):
        tape = _trace(fn, args, device, warm_up=warm_up)
    _hints, points, _report = alternate_hints(tape)
    positions = ht._tensor_positions(args)
    gc_enabled = gc.isenabled()  # as trace(): no collection while a capture is open
    gc.disable()
    try:
        with backend("ir"):
            try:
                tr, records, outputs = ht._trace_once(
                    fn, args, positions, tape.device.index, None
                )
            except ht.Declined as e:
                raise BackendDifference(
                    f"the IR backend declined where sympy traced: {e} (census {list(e.census)})"
                ) from e
            second = ht.Tape(tr, records, outputs, args)
            # the same call's warm-up: what the test seam checks the output
            # records against (the IR tape stands in for the sympy one)
            second.warm_up_outputs = tape.warm_up_outputs
    finally:
        if gc_enabled:
            gc.enable()
    diff, counts = compare_backends(tape, second, points)
    if diff is not None:
        raise BackendDifference(f"the two backends traced different programs at {diff}")
    if log is not None:
        log.append(counts)
    return second


@dataclass
class FamilyReport:
    verified: list[tuple[str, Comparison]]  # one entry per agreeing trace_both
    differences: list[tuple[str, str]]  # (test, the BackendDifference message)
    census: list[tuple[str, list]]  # (test, the IR census of a decline)
    failures: list[tuple[str, str]]  # (test, another error under the wrapper)
    skipped: list[str]
    untraced: list[str]  # ran without tracing in this process


class _Result(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.exceptions: list[BaseException] = []

    def addError(self, test: Any, err: Any) -> None:
        super().addError(test, err)
        self.exceptions.append(err[1])

    def addFailure(self, test: Any, err: Any) -> None:
        super().addFailure(test, err)
        self.exceptions.append(err[1])

    def addSubTest(self, test: Any, subtest: Any, err: Any) -> None:
        # a subTest's failure lands here, not in addFailure / addError
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.exceptions.append(err[1])


def run_family(cls: type[unittest.TestCase], skip: Any = ()) -> FamilyReport:
    """Every test of `cls` but `skip`, run with ht.trace replaced by
    trace_both: each trace the family makes in this process is made under
    both backends and the test goes on with the IR tape."""
    report = FamilyReport([], [], [], [], [], [])
    log: list[Comparison] = []
    names = sorted(
        n
        for n in dir(cls)
        if n.startswith("test_") and n not in skip and callable(getattr(cls, n))
    )
    with mock.patch.object(ht, "trace", functools.partial(trace_both, log=log)):
        for name in names:
            start = len(log)
            result = _Result()
            # through a suite, so the class fixtures (setUpClass) run
            unittest.TestSuite([cls(name)]).run(result)
            report.verified.extend((name, v) for v in log[start:])
            diffs = [e for e in result.exceptions if isinstance(e, BackendDifference)]
            declined = [
                e for e in result.exceptions if isinstance(e, ht.Declined) and e.census
            ]
            if diffs:
                report.differences.append((name, str(diffs[0])))
                cause = diffs[0].__cause__
                if isinstance(cause, ht.Declined) and cause.census:
                    report.census.append((name, list(cause.census)))
            elif declined:
                report.census.append((name, list(declined[0].census)))
            elif result.exceptions:
                e = result.exceptions[0]
                first_line = str(e).splitlines()[0] if str(e) else ""
                report.failures.append((name, f"{type(e).__name__}: {first_line}"))
            elif result.skipped:
                report.skipped.append(name)
            elif len(log) == start:
                report.untraced.append(name)
    return report


def assert_family(
    case: unittest.TestCase, exclude: dict[str, str] | None = None
) -> FamilyReport:
    """The property over the case's class: no test traced a different program
    under the IR backend, the IR's census is empty, none failed under the
    wrapper (but `exclude`, test name to why), and some traced."""
    skip = {case._testMethodName, *(exclude or {})}
    report = run_family(type(case), skip)
    if report.differences:
        case.fail(
            "the IR backend traced a different program:\n"
            + "\n".join(f"{name}: {message}" for name, message in report.differences)
        )
    if report.census:
        case.fail(
            "the IR backend could not express:\n"
            + "\n".join(f"{name}: {census}" for name, census in report.census)
        )
    if report.failures:
        case.fail(
            "failed under trace_both:\n"
            + "\n".join(f"{name}: {message}" for name, message in report.failures)
        )
    case.assertGreater(len(report.verified), 0, "no test of this class traced")
    return report
