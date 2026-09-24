# Owner(s): ["module: cuda"]
"""Hint independence of the host-tracing recorder, as a test.

`trace_twice` traces a call as `ht.trace` does, then once more under a second
hint assignment disjoint from the first: every CUDA input's base under another
placeholder top, every allocation at another base, and every input size
the recorded guards allow at another value (sizes an equality ties together
move as one, a dense input's strides are recomputed, opaque results are
re-evaluated through the host's own function). The two tapes must be one
program in everything but the hints; the first difference raises
`HintDependence` naming the launch and the field. A recorder that is sound by
construction (every mapping declared, driver-read or structural) never reads a
hint, so the tapes agree: this is a regression test of the recorder, not a
contract of `trace`. `assert_family` runs it over every test of one family's
class.

Compared, by value at four assignments (the two traces may solve an equality
for different symbols, so never as text): every section of the tape's structure
(inputs, allocations, launches with kernel, function handle, parameter layout,
grid, block expressions, smem and every field's offset / width / kind / name /
access / expression, opaque records, guards as relations, outputs) and the bytes
of an argument image outside its symbolic fields. Not compared: the hints
themselves, an opaque record's traced value and the block dims at the trace."""

from __future__ import annotations

import collections
import functools
import itertools
import unittest
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from unittest import mock

import sympy

from torch.cuda import _host_trace as ht


if TYPE_CHECKING:
    from collections.abc import Callable


# the second assignment: another non-canonical top over the same low 52 bits
# for an input (alignment and pointer differences unchanged, the absolute value
# not), a distinct 256 MiB-aligned base per allocation under the same top
_SECOND_TAG = 0x5B6 << 52
_ALLOC_SHIFT = 28
# the increments a size hint is tried at, in order, until every recorded guard
# keeps its value; then the hint itself and its multiples
_DELTAS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256)

_trace = ht.trace  # the first run, whatever a family run patches ht.trace to


class HintDependence(AssertionError):
    """The two hint assignments traced different programs; the message names
    the first difference."""


@dataclass
class Verification:
    moved: list[tuple[str, int, int]]  # (symbols, first hint, second hint)
    held: list[tuple[str, int, str | None]]  # (symbols, hint, blocking guard)
    pinned: int  # input size classes a guard pins to a number


def _env_values(env: Any) -> dict[str, Any]:
    # every backed symbol's hint, by name
    return {
        str(k): int(v) if v.is_integer else float(v)
        for k, v in env.backed_var_to_val.items()
    }


def _symbol_names(values: list) -> list[str] | None:
    # the names when every value is a plain symbol (an input's sizes and
    # strides are), else None
    names: list[str] = []
    for v in values:
        name = ht._symbol_name(v)
        if name is None:
            return None
        names.append(name)
    return names


def _dense_order(sizes: list, strides: list) -> list | None:
    # the dims from outermost to innermost when the strides are contiguous in
    # some permutation (a size-1 dim keeps its place by stride), else None
    order = sorted(range(len(sizes)), key=lambda d: (-strides[d], d))
    expected = 1
    for d in reversed(order):
        if strides[d] != expected:
            return None
        expected *= sizes[d]
    return order


def _strides_in_order(order: list, sizes: list) -> list:
    strides = [0] * len(sizes)
    expected = 1
    for d in reversed(order):
        strides[d] = expected
        expected *= sizes[d]
    return strides


def _symbol_pair(g: Any) -> tuple[str, str] | None:
    # the two symbols an equality guard ties together (`Eq(a, b)`, or
    # `Eq(a + 1, b + 1)` as a cat's lengths are recorded), else None
    if not isinstance(g, sympy.Eq):
        return None
    d = sympy.expand(g.lhs - g.rhs)
    if not (d.is_Add and len(d.args) == 2):
        return None
    pos = [t for t in d.args if isinstance(t, sympy.Symbol)]
    neg = [-t for t in d.args if isinstance(-t, sympy.Symbol)]
    if len(pos) == 1 and len(neg) == 1:
        return str(pos[0]), str(neg[0])
    return None


def alternate_hints(tape: ht.Tape) -> tuple[dict[str, int], list[dict], Verification]:
    """The second hint assignment for `tape`'s call. A guard's value never
    changes: the assignment is searched against the recorded guards (opaque
    results re-evaluated through the host's own function), so a host whose
    branches are all guards takes the same path under it, and a difference
    between the two tapes is a value the host took from a hint. Returns the
    hints by source name for the second run, the extra assignments the
    comparison evaluates expressions at, and what moved and what could not."""
    env = tape.shape_env
    prog = ht._Evaluator()
    base = _env_values(env)
    source = {
        str(sym): srcs[0].name for sym, srcs in env.var_to_sources.items() if srcs
    }
    guards = [(g, {str(x) for x in g.free_symbols}) for g in tape.guards]
    by_sym: dict[str, list[int]] = collections.defaultdict(list)
    for k, (_g, syms) in enumerate(guards):
        for name in syms:
            by_sym[name].append(k)
    # an opaque result is re-evaluated when its arguments move, through the
    # host's function, when that function reproduces the traced value on the
    # traced hints (the lookup form of opaque() hands the trace a value its
    # function cannot compute from hints: it keeps the traced value)
    opaques = []
    for o in tape.opaque:
        name = ht._symbol_name(o["sym"])
        if name is None:
            continue
        syms = (
            set().union(*(ht._free_symbols(a) for a in o["args"]))
            if o["args"]
            else set()
        )
        try:
            trusted = (
                o["call"]([int(prog.ev(a, base)) for a in o["args"]]) == o["expected"]
            )
        except Exception:
            trusted = False
        opaques.append((name, o["args"], o["call"], syms, trusted))

    def settle(trial: dict, changed: set) -> set:
        # host order: an opaque's arguments mention only earlier symbols
        for name, args, call, syms, trusted in opaques:
            if trusted and syms & changed:
                trial[name] = int(call([int(prog.ev(a, trial)) for a in args]))
                changed.add(name)
        return changed

    def failing(trial: dict, changed: set) -> Any:
        idx: set[int] = set()
        for name in changed:
            idx.update(by_sym.get(name, ()))
        for k in sorted(idx):
            if not prog.ev(guards[k][0], trial):
                return guards[k][0]
        return None

    def describe(names: Any) -> str:
        return ", ".join(
            f"{n} ({source.get(n, '?')} = {base[n]})" for n in sorted(names)
        )

    # the addresses
    current = dict(base)
    changed: set[str] = set()
    for i in tape.inputs:
        name = ht._symbol_name(i.root.sym)
        # a pinned input keeps its address: its copy reads it at the trace
        if getattr(i, "pinned", False) or name is None:
            continue
        current[name] = _SECOND_TAG | (base[name] & ht._PLACEHOLDER_LOW)
        changed.add(name)
    for k, a in enumerate(tape.allocs):
        name = ht._symbol_name(a.q)
        if name is not None:
            current[name] = (_SECOND_TAG | ((k + 1) << _ALLOC_SHIFT)) >> 8
            changed.add(name)
    changed = settle(current, changed)
    g = failing(current, changed)
    if g is not None:
        raise HintDependence(
            f"guard `{prog.guard_text(g)}` changes its value when the placeholder "
            f"addresses change (over {describe({str(x) for x in g.free_symbols})}): the "
            "host branched on an address hint rather than on an alignment or offset of it"
        )
    addresses_only = dict(current)

    # the sizes: symbols a recorded equality ties together move as one class,
    # a symbol a guard pins to a number does not move
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    pinned: set[str] = set()
    for g, _syms in guards:
        if not isinstance(g, sympy.Eq):
            continue
        pair = _symbol_pair(g)
        lhs, rhs = g.lhs, g.rhs
        if pair is not None:
            parent[find(pair[0])] = find(pair[1])
        elif isinstance(lhs, sympy.Symbol) and rhs.is_number:
            pinned.add(str(lhs))
        elif isinstance(rhs, sympy.Symbol) and lhs.is_number:
            pinned.add(str(rhs))
    members: dict[str, list[str]] = collections.defaultdict(list)
    for name in base:
        members[find(name)].append(name)
    frozen = {find(n) for n in pinned}
    for i in tape.inputs:
        if getattr(i, "pinned", False):  # its copy reads its extent at the trace: kept
            for v in (*i.sizes, *i.strides, i.offset, i.root.sym):
                name = ht._symbol_name(v)
                if name is not None:
                    frozen.add(find(name))
    dense: list[tuple[list[str], list[str], list[int]]] = []
    for i in tape.inputs:
        sizes, strides = _symbol_names(i.sizes), _symbol_names(i.strides)
        if getattr(i, "pinned", False) or not sizes or not strides:
            continue
        order = _dense_order([base[n] for n in sizes], [base[n] for n in strides])
        if order is not None:
            dense.append((sizes, strides, order))

    def search(start: dict, deltas: tuple) -> tuple[dict, list, list]:
        current = dict(start)
        moved: list = []
        held: list = []
        seen: set[str] = set()
        for i in tape.inputs:
            for v in i.sizes:
                name = ht._symbol_name(v)
                if name is None:
                    continue
                root = find(name)
                if root in seen:
                    continue
                seen.add(root)
                if root in frozen:
                    continue
                group = members[root]
                h = base[name]
                first_failure = None
                for delta in (*deltas, h, 2 * h, 3 * h):
                    if delta <= 0:
                        continue
                    trial = dict(current)
                    tchanged = set(group)
                    for m in group:
                        trial[m] = h + delta
                    for sizes, strides, order in dense:
                        if not tchanged.intersection(sizes):
                            continue
                        for n, st in zip(
                            strides, _strides_in_order(order, [trial[x] for x in sizes])
                        ):
                            if trial[n] != st:
                                trial[n] = st
                                tchanged.add(n)
                    tchanged = settle(trial, tchanged)
                    g = failing(trial, tchanged)
                    if g is None:
                        current = trial
                        moved.append((describe(group), h, h + delta))
                        break
                    if first_failure is None:
                        first_failure = prog.guard_text(g)
                else:
                    held.append((describe(group), h, first_failure))
        return current, moved, held

    current, moved, held = search(addresses_only, _DELTAS)
    other, _m, _h = search(addresses_only, tuple(reversed(_DELTAS)))
    hints = {
        source[name]: int(value)
        for name, value in current.items()
        if value != base[name] and source.get(name, "").startswith(("arg", "alloc"))
    }
    pinned_inputs = sum(
        1
        for r in frozen
        if any(source.get(n, "").startswith("arg") for n in members[r])
    )
    return hints, [addresses_only, other], Verification(moved, held, pinned_inputs)


def _is_symbolic(v: Any) -> bool:
    return isinstance(v, (*ht._SYM_TYPES, sympy.Basic))


def _structure(tape: ht.Tape) -> dict:
    # the tape's to_json structure with the values themselves, less the hints,
    # plus the launch handles and layouts to_json leaves out
    d = tape._structure(lambda v: v)
    del d["hints"]
    for L, s in zip(tape.launches, d["launches"]):
        s["func"] = L["func"]
        s["param_layout"] = [list(x) for x in L["param_layout"]]
    return d


def compare_tapes(
    first: ht.Tape, second: ht.Tape, extra_points: list[dict]
) -> str | None:
    """The first difference between two traces of one call under two hint
    assignments, or None (see the module docstring for what is compared)."""
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
                f"symbol {k} is {a} in the first trace and {b} in the second: the host "
                "created its values in a different order"
            )
    points = [
        _env_values(first.shape_env),
        _env_values(second.shape_env),
        *extra_points,
    ]
    prog = ht._Evaluator()

    def text(v: Any) -> str:
        if isinstance(v, ht._SYM_TYPES):
            return f"`{prog.guard_text(v.node.expr)}`"
        if isinstance(v, sympy.Basic):
            return f"`{prog.guard_text(v)}`"
        return repr(v)

    def same_expr(a: Any, b: Any) -> bool:
        return all(prog.ev(a, pt) == prog.ev(b, pt) for pt in points)

    def same_guard(a: Any, b: Any) -> bool:
        # the same relation between the same quantities, whatever symbols
        # each trace wrote it over
        if type(a) is not type(b):
            return False
        if isinstance(a, sympy.Rel):
            return same_expr(a.lhs - a.rhs, b.lhs - b.rhs)
        if isinstance(a, (sympy.And, sympy.Or)):
            rest = list(b.args)
            for x in a.args:
                k = next((j for j, y in enumerate(rest) if same_guard(x, y)), None)
                if k is None:
                    return False
                del rest[k]
            return not rest
        return str(a) == str(b) if isinstance(a, sympy.Basic) else a == b

    def same_leaf(a: Any, b: Any, path: str) -> bool:
        if _is_symbolic(a) != _is_symbolic(b):
            return False
        if not _is_symbolic(a):
            return ht._constant(a) == ht._constant(b)
        if path.startswith("guards["):
            return same_guard(a, b)
        return same_expr(a, b)

    def walk(a: Any, b: Any, path: str) -> str | None:
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return f"{path}: keys {sorted(a.keys() ^ b.keys())} differ"
            for k in a:
                if k in ("expected", "block"):
                    continue
                r = walk(a[k], b[k], f"{path}.{k}" if path else k)
                if r is not None:
                    return r
            return None
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return f"{path}: {len(a)} in the first trace, {len(b)} in the second"
            for k, (x, y) in enumerate(zip(a, b)):
                r = walk(x, y, f"{path}[{k}]")
                if r is not None:
                    return r
            return None
        if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        if not same_leaf(a, b, path):
            return f"{path}: {text(a)} vs {text(b)}"
        return None

    d1, d2 = _structure(first), _structure(second)
    for section in d1:
        r = walk(d1[section], d2[section], section)
        if r is None:
            continue
        if section == "launches":
            # name the launch, and the field when the difference is in one
            j = int(r[len("launches[") : r.index("]")])
            r = f"launch {j} (`{first.launches[j]['kernel']}`) {r}"
            if ".params[" in r:
                k = int(r[r.index(".params[") + 8 : r.index("]", r.index(".params["))])
                q = first.launches[j]["params"][k]
                r += f" (field {k}: offset {q['offset']}, {q['size']} bytes, {q['kind']})"
        return r
    for j, (a, b) in enumerate(zip(first.launches, second.launches)):
        img1, img2 = a["hint_image"], b["hint_image"]
        if len(img1) != len(img2):
            return f"launch {j} (`{a['kernel']}`): image of {len(img1)} bytes vs {len(img2)}"
        exempt = bytearray(len(img1))
        for p in a["params"]:
            if p["kind"] == "rng" or isinstance(p["value"], ht._SYM_TYPES):
                exempt[p["offset"] : p["offset"] + p["size"]] = b"\x01" * p["size"]
        for off, (x, y) in enumerate(zip(img1, img2)):
            if x == y or exempt[off]:
                continue
            owner = next(
                (
                    p
                    for p in a["params"]
                    if p["offset"] <= off < p["offset"] + p["size"]
                ),
                None,
            )
            what = (
                f"constant parameter {owner['name'] or 'at offset ' + str(owner['offset'])}"
                if owner is not None
                else "bytes no parameter covers"
            )
            return (
                f"launch {j} (`{a['kernel']}`): byte {off} of the argument image differs "
                f"({what}: {x:#04x} vs {y:#04x}): a value the host took from a hint without recording it"
            )
    return None


def trace_twice(
    fn: Callable[..., Any],
    args: tuple,
    device: int | None = None,
    *,
    warm_up: bool = True,
    log: list[Verification] | None = None,
) -> ht.Tape:
    """`ht.trace`, then the same call under the second hint assignment (no
    warm-up: the first run's did it). Raises HintDependence at the first
    difference between the two tapes; returns the first tape and appends what
    the second assignment moved, and its cost, to `log`."""
    tape = _trace(fn, args, device, warm_up=warm_up)
    if tape.warm_up_outputs is not None:
        # the output records at the traced call's own values against what
        # the warm-up (eager) returned for it
        from host_trace_testing import assert_output_metadata

        assert_output_metadata(tape, tape.warm_up_outputs)
    hints, points, report = alternate_hints(tape)
    positions = ht._tensor_positions(args)
    # as trace(): no collection while the capture is open, under the hold every
    # trace shares (a private flag re-enabled the collector under another
    # thread's capture, A398)
    with ht._gc_hold:
        try:
            tr, records, outputs = ht._trace_once(
                fn, args, positions, tape.device.index, hints
            )
        except ht.Declined as e:
            raise HintDependence(
                f"the second hint assignment declined where the first traced: {e}"
            ) from e
        second = ht.Tape(tr, records, outputs, args)
    diff = compare_tapes(tape, second, points)
    if diff is not None:
        raise HintDependence(
            f"the two hint assignments traced different programs at {diff}"
        )
    if log is not None:
        log.append(report)
    return tape


@dataclass
class FamilyReport:
    verified: list[tuple[str, Verification]]  # one entry per agreeing trace_twice
    disagreements: list[tuple[str, str]]  # (test, the HintDependence message)
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


def run_family(cls: type[unittest.TestCase], skip: Any = ()) -> FamilyReport:
    """Every test of `cls` but `skip`, run with ht.trace replaced by
    trace_twice: each trace the family makes in this process is made twice."""
    report = FamilyReport([], [], [], [], [])
    log: list[Verification] = []
    names = sorted(n for n in dir(cls) if n.startswith("test_") and n not in skip)
    with mock.patch.object(ht, "trace", functools.partial(trace_twice, log=log)):
        for name in names:
            start = len(log)
            result = _Result()
            cls(name).run(result)
            report.verified.extend((name, v) for v in log[start:])
            dependence = [e for e in result.exceptions if isinstance(e, HintDependence)]
            if dependence:
                report.disagreements.append((name, str(dependence[0])))
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
    under the second hints, none failed under the wrapper (but `exclude`, test
    name to why), and some traced."""
    skip = {case._testMethodName, *(exclude or {})}
    report = run_family(type(case), skip)
    if report.disagreements:
        case.fail(
            "traced a different program under the second hints:\n"
            + "\n".join(f"{name}: {message}" for name, message in report.disagreements)
        )
    if report.failures:
        case.fail(
            "failed under trace_twice:\n"
            + "\n".join(f"{name}: {message}" for name, message in report.failures)
        )
    case.assertGreater(len(report.verified), 0, "no test of this class traced")
    return report
