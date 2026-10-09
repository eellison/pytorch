# Owner(s): ["module: cuda"]
"""Backend equivalence of the host tracer (sympy vs the IR backend), as a harness.

`install()` rebinds every module attribute bound to `_host_trace_tape.trace`
or `_host_trace_tape._symbolic_run` (the suites' and the replay's). Inside a
trace() call the symbolic run happens twice, under sympy and then under the IR
backend (`_host_trace.symbolic = "ir"`) on a second _Trace, after the same
warm-up and in the same capture contexts, and the warm-up's witness check
applies to both; trace() then returns the IR tape, so the rest of the test runs
against it. A direct `_symbolic_run` (the suites' CPU helpers) runs twice the
same way and returns the sympy run's results (its test holds the `_Trace`).

Compared: the symbols (names and sources in creation order, and hints),
every record of the tape (inputs, int inputs, allocations, launches, sites,
outputs), a symbolic value by its sympy form, else as the IR node both forms
canonicalize to, else by value at the hints and at random assignments; and
the guards. A guard is matched by its sympy form, else by IR node; a sympy
guard with no IR match must be one the IR decides true (a tautology over the
declared domains, which sympy's range analysis did not decide); an IR guard
with no sympy match is a difference, as is a matched pair out of order. A
decline under one backend where the other traced is a difference.

`install()` also rebinds `lower_tape`: an IR tape is lowered twice, from its
nodes (IRLowering, the default) and from its sympy export (`direct=False`).
The two results' rows are paired structurally, and both programs are run on
the hints and on 12 perturbed leaf assignments. At every assignment they must
agree on whether the call misses, on every paired row's value when both
accept, and on each guard's value where neither program fails. A decline
under one lowering where the other accepted is a difference. The IR result is
returned when there is no difference.

`install_meta()` (part of `install()`) checks each call the trace routes
around its fake kernel (a traced ATen host, a view on a bare meta tensor, an
opaque call's meta function) against the fake kernel on the same operands
under the IR backend (META_LOG).
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import math
import random
import struct
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

import sympy

import torch
from torch.cuda import _host_trace as ht, _host_trace_ir as _ir, _host_trace_tape as tape_mod
from torch.cuda._host_trace_launch import KernelLaunch
from torch.fx.experimental.sym_node import SymNode
from torch.utils import _pytree as pytree
from torch.utils._sympy import functions as tf

from torch.cuda import _host_trace_lower_tape as lower_mod
from torch.cuda._host_trace_program import _step, LEAVES, Status


_TRACE = tape_mod.trace
_LOWER_TAPE = lower_mod.lower_tape
_SYMBOLIC_RUN = tape_mod._symbolic_run
_SYM = (torch.SymInt, torch.SymFloat, torch.SymBool)


@dataclass
class Entry:
    test: str
    kind: str  # "trace" or "run"
    diff: str | None = None
    guards_sympy: int = 0
    guards_ir: int = 0
    decided_by_ir: int = 0  # sympy guards the IR decides true
    matched_by_node: int = 0  # guards matched as IR nodes, not by sympy form
    values: int = 0
    same_str: int = 0
    same_node: int = 0
    by_value: int = 0
    folded_slots: int = 0  # sympy's symbolic launch slots the IR folds into the image
    both_declined: bool = False
    notes: list = field(default_factory=list)


LOG: list[Entry] = []
CURRENT = {"test": "?"}


class BackendDifference(AssertionError):
    pass


@contextlib.contextmanager
def backend(name: str):
    prev = ht.symbolic
    ht.symbolic = name
    try:
        yield
    finally:
        ht.symbolic = prev


class ToIR:
    """sympy -> the IR node of `ctx`: symbols by name, every other form
    through the IR's own constructors."""

    def __init__(self, ctx: _ir.Ctx) -> None:
        self.ctx = ctx
        self.memo: dict = {}

    def __call__(self, e: Any) -> _ir.Node:
        r = self.memo.get(e)
        if r is None:
            r = self._to_ir(e)
            self.memo[e] = r
        return r

    def _to_ir(self, e: Any) -> _ir.Node:
        c = self.ctx
        if isinstance(e, bool):
            return c.boolean(e)
        if isinstance(e, int):
            return c.const(e)
        if isinstance(e, sympy.Symbol):
            node = c.symbols.get(e.name)
            if node is None:
                raise BackendDifference(f"symbol {e} has no IR symbol")
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
        if isinstance(e, (tf.FloatTrueDiv, tf.IntTrueDiv)):
            return c.fbin("fdiv", self(e.args[0]), self(e.args[1]))
        if isinstance(e, tf.FloatPow):
            return c.fbin("fpow", self(e.args[0]), self(e.args[1]))
        if isinstance(e, tf.PowByNatural):
            b, x = self(e.args[0]), self(e.args[1])
            if x.op == "const":
                return c.pow(b, x.args[0])
            if b.op == "const":
                return c.pbn(b.args[0], x)
            raise BackendDifference(f"no IR form for {e}")
        if isinstance(e, tf.LShift):
            return c.mul(self(e.args[0]), c.pbn(2, self(e.args[1])))
        if isinstance(e, tf.RShift):
            return c.floordiv(self(e.args[0]), c.pbn(2, self(e.args[1])))
        if isinstance(e, ht.BitLength):
            return c.bitlen(self(e.args[0]))
        if isinstance(e, ht.F32Div):
            return c.f32div(self(e.args[0]), self(e.args[1]))
        for op, (_f, cls) in _ir._BITWISE.items():
            if isinstance(e, cls):
                return c.bitwise(op, self(e.args[0]), self(e.args[1]))
        if isinstance(e, tf.Where):
            return c.where(*(self(a) for a in e.args))
        if isinstance(e, tf.TruncToInt):
            return c.fint("ftrunc", self(e.args[0]))
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
        raise BackendDifference(f"no IR form for the sympy node {type(e).__name__}: {e}")


def _expr(v: Any) -> Any:
    if isinstance(v, _SYM):
        return v.node.expr
    return v


def _is_sym(v: Any) -> bool:
    return isinstance(v, _SYM) or (isinstance(v, sympy.Basic) and bool(v.free_symbols))


class _Compare:
    def __init__(self, env_s: Any, env_i: _ir.Env, entry: Entry) -> None:
        self.entry = entry
        self.to_ir = ToIR(env_i.ctx)
        self.seen: set = set()
        vals = env_s.backed_var_to_val
        srcs = env_s.var_to_sources
        rng = random.Random(0)
        self.points = [dict(vals)]
        for _ in range(3):
            pt = {}
            for s, v in vals.items():
                name = srcs[s][0].name if srcs.get(s) else ""
                if isinstance(v, sympy.Float):
                    pt[s] = sympy.Float(rng.uniform(-4, 4))
                elif ".size(" in name:
                    pt[s] = sympy.Integer(rng.randint(1, 97))
                elif ".base" in name:
                    pt[s] = v  # addresses: keep, a shift is not a valid call
                else:
                    pt[s] = sympy.Integer(rng.randint(-97, 97))
            self.points.append(pt)

    def _ev(self, e: Any, pt: dict) -> Any:
        try:
            r = sympy.sympify(e).xreplace(pt)
            if isinstance(r, sympy.Basic) and not r.is_number and r not in (sympy.true, sympy.false):
                r = r.doit()
            return ("ok", str(r))
        except Exception as ex:  # a division by zero at a random point
            return ("raised", type(ex).__name__)

    def same_expr(self, a: Any, b: Any) -> bool:
        en = self.entry
        en.values += 1
        ea, eb = _expr(a), _expr(b)
        if ea == eb or str(ea) == str(eb):
            en.same_str += 1
            return True
        try:
            if self.to_ir(ea) is self.to_ir(eb):
                en.same_node += 1
                return True
        except BackendDifference:
            pass
        if all(self._ev(ea, pt) == self._ev(eb, pt) for pt in self.points):
            en.by_value += 1
            return True
        return False

    def walk(self, a: Any, b: Any, path: str) -> str | None:
        if _is_sym(a) or _is_sym(b):
            if not self.same_expr(a, b):
                return f"{path}: {_expr(a)} vs {_expr(b)}"
            return None
        if isinstance(a, sympy.Basic) or isinstance(b, sympy.Basic):
            return None if str(_expr(a)) == str(_expr(b)) else f"{path}: {a} vs {b}"
        if type(a) is not type(b):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        key = (id(a), id(b))
        if key in self.seen:
            return None
        if isinstance(a, tape_mod._TracedTensor):
            self.seen.add(key)
            return (
                self.walk(a._root, b._root, f"{path}._root")
                or self.walk(list(a.shape), list(b.shape), f"{path}.shape")
                or self.walk(a._sym_strides, b._sym_strides, f"{path}.strides")
                or self.walk(a._sym_offset, b._sym_offset, f"{path}.offset")
                or self.walk(a.dtype, b.dtype, f"{path}.dtype")
            )
        if isinstance(a, torch.Tensor):
            if (a.shape, a.stride(), a.dtype) != (b.shape, b.stride(), b.dtype):
                return f"{path}: tensor {a.shape} vs {b.shape}"
            return None
        if isinstance(a, KernelLaunch) and a.fields is not None:
            self.seen.add(key)
            return self.launch(a, b, path)
        if dataclasses.is_dataclass(a) and not isinstance(a, type):
            self.seen.add(key)
            for f in dataclasses.fields(a):
                r = self.walk(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
                if r is not None:
                    return r
            return None
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return f"{path}: {len(a)} vs {len(b)} elements"
            for k, (x, y) in enumerate(zip(a, b)):
                r = self.walk(x, y, f"{path}[{k}]")
                if r is not None:
                    return r
            return None
        if isinstance(a, (set, frozenset)):
            return None if sorted(a) == sorted(b) else f"{path}: {sorted(a)} vs {sorted(b)}"
        if isinstance(a, dict):
            ka, kb = [str(k) for k in a], [str(k) for k in b]
            if ka != kb:
                return f"{path}: keys {ka} vs {kb}"
            for (k, x), y in zip(a.items(), b.values()):
                r = self.walk(x, y, f"{path}[{k!r}]")
                if r is not None:
                    return r
            return None
        if isinstance(a, float):
            return None if struct.pack("<d", a) == struct.pack("<d", b) else f"{path}: {a} vs {b}"
        if isinstance(a, (int, str, bytes, type(None), torch.dtype, torch.device)):
            return None if a == b else f"{path}: {a!r} vs {b!r}"
        return None  # an opaque object of one type (a kernel, a function)

    def launch(self, a: KernelLaunch, b: KernelLaunch, path: str) -> str | None:
        """A launch whose parameters are images with its symbolic values placed
        at `fields`: the IR may fold a value sympy keeps symbolic into the image
        (a constant over the declared domains); the image must then hold it."""
        for f in dataclasses.fields(a):
            if f.name not in ("slots", "fields", "pointers", "images"):
                r = self.walk(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
                if r is not None:
                    return r
        pa = {p: (v, i in a.pointers) for i, (p, v) in enumerate(zip(a.fields, a.slots))}
        pb = {p: (v, i in b.pointers) for i, (p, v) in enumerate(zip(b.fields, b.slots))}
        if extra := [p for p in pb if p not in pa]:
            return f"{path}: IR slots at {extra} are constants under sympy"
        if len(a.images) != len(b.images) or any(len(x) != len(y) for x, y in zip(a.images, b.images)):
            return f"{path}: parameter sizes differ"
        r = self.walk(a.slots[len(a.fields) :], b.slots[len(b.fields) :], f"{path}.trailing_slots")
        if r is not None:
            return r
        ia, ib = [bytearray(x) for x in a.images], [bytearray(x) for x in b.images]
        for p, (v, ptr) in pa.items():
            param, offset, width = p
            if p in pb:
                w, ptr2 = pb[p]
                if ptr != ptr2:
                    return f"{path}: slot at {p} is a pointer under one backend only"
                r = self.walk(v, w, f"{path}.slot{p}")
                if r is not None:
                    return r
            else:
                n = self.to_ir(_expr(v))
                if param >= len(ib):
                    return f"{path}: sympy slot at {p} ({_ir.render(n)[:200]}) has no IR slot and no image"
                got = int.from_bytes(ib[param][offset : offset + width], "little", signed=True)
                if n.op != "const" or ptr or (n.args[0] - got) % (1 << 8 * width):
                    return f"{path}: sympy slot at {p} ({_ir.render(n)[:200]}) is not the IR image's constant {got}"
                self.entry.folded_slots += 1
            if param < len(ia):
                ia[param][offset : offset + width] = ib[param][offset : offset + width] = bytes(width)
        if ia != ib:
            return f"{path}: parameter images differ outside the slots"
        return None

    def guards(self, gs: list, gi: list) -> str | None:
        en = self.entry
        en.guards_sympy, en.guards_ir = len(gs), len(gi)
        by_str: dict[str, list[int]] = {}
        for k, g in enumerate(gs):
            by_str.setdefault(str(g), []).append(k)
        node_of_s = {}
        for k, g in enumerate(gs):
            try:
                node_of_s[k] = self.to_ir(g)
            except BackendDifference as e:
                return f"sympy guard {g}: {e}"
        used: set[int] = set()
        order: list[int] = []
        for g in gi:
            k = next((k for k in by_str.get(str(g), ()) if k not in used), None)
            if k is None:
                n = self.to_ir(g)
                k = next((k for k, m in node_of_s.items() if m is n and k not in used), None)
                if k is None:
                    return f"IR guard {g} is not on the sympy tape"
                en.matched_by_node += 1
            used.add(k)
            order.append(k)
        if order != sorted(order):
            return f"guard order differs: sympy indices {order}"
        for k, g in enumerate(gs):
            if k in used:
                continue
            if node_of_s[k].op != "true":
                return f"sympy guard {g} is not on the IR tape (IR form {_ir.render(node_of_s[k])})"
            en.decided_by_ir += 1
        return None


def _symbols(env: Any) -> list:
    return [
        (str(s), v[0].name if v else "", str(env.backed_var_to_val.get(s)))
        for s, v in env.var_to_sources.items()
    ]


_TAPE_FIELDS = ("inputs", "int_inputs", "allocs", "launches", "sites", "outputs", "result_kind", "argument_pairs", "contract")
_TRACE_FIELDS = ("inputs", "int_inputs", "allocs", "launches", "sites", "eager_outputs", "argument_pairs")


def compare(first: Any, second: Any, entry: Entry, fields: tuple) -> str | None:
    env_s, env_i = first.shape_env, second.shape_env
    if not isinstance(env_i, _ir.Env):
        return "the second run was not traced by the IR backend"
    s1, s2 = _symbols(env_s), _symbols(env_i)
    if s1 != s2:
        k = next(k for k in range(max(len(s1), len(s2))) if k >= min(len(s1), len(s2)) or s1[k] != s2[k])
        return f"symbol {k}: {s1[k] if k < len(s1) else None} vs {s2[k] if k < len(s2) else None}"
    c = _Compare(env_s, env_i, entry)
    for f in fields:
        r = c.walk(getattr(first, f), getattr(second, f), f)
        if r is not None:
            return r
    return c.guards([g.expr for g in env_s.guards], [g.expr for g in env_i.guards])


_CHECK_WITNESS = tape_mod._check_witness
_PAIR = threading.local()
_SYMPY_INT = SymNode.int_


def _sympy_int(self: SymNode) -> int:
    # IRSymNode.int_'s len() peephole on the sympy reference: the comparison
    # len(t)'s result feeds is the guard
    if self.has_hint() and not self.is_constant() and (use := _ir._len_compare()) is not None:
        getattr(self, use[0])(self.wrap_int(use[1])).guard_bool("", 0)
        return int(self.hint)
    return _SYMPY_INT(self)


@contextlib.contextmanager
def _len_peephole() -> Any:
    SymNode.int_ = _sympy_int
    try:
        yield
    finally:
        SymNode.int_ = _SYMPY_INT


def _paired_run(tr: Any, fn: Any, args: tuple, positions: list, ints: list) -> Any:
    """trace()'s symbolic run under trace_both: sympy's, then the IR's on a
    second _Trace inside the same capture and interception contexts."""
    pair = getattr(_PAIR, "slot", None)
    if pair is None or tr.trusted is not None or "ran" in pair:
        return _SYMBOLIC_RUN(tr, fn, args, positions, ints)
    pair["ran"] = True
    raised = None
    try:
        with _len_peephole():
            result = _SYMBOLIC_RUN(tr, fn, args, positions, ints)
        pair["sympy_ran"] = True
    except BaseException as e:  # noqa: B036
        raised, result = e, None
    with backend("ir"):
        tr2 = tape_mod._Trace(tr.device, tr.trusted, tr.opaque)
    try:
        out2, traced2 = _SYMBOLIC_RUN(tr2, fn, args, positions, ints)
        pair["ir"] = (tr2, *tape_mod._output_records(out2, traced2, positions))
    except ht.Declined as e:
        pair["ir_declined"] = e
    except Exception as e:
        pair["ir_raised"] = e
    if raised is not None:
        raise raised
    return result


def _paired_witness(tr: Any, witness: Any) -> None:
    pair = getattr(_PAIR, "slot", None)
    if pair is not None and "ir" in pair and tr is not pair["ir"][0]:
        try:
            _CHECK_WITNESS(pair["ir"][0], witness)
        except ht.Declined as e:
            pair["ir_witness"] = e
    try:
        _CHECK_WITNESS(tr, witness)
    except ht.Declined as e:
        if pair is not None:
            pair["sympy_witness"] = e
        raise


def trace_both(fn: Any, args: tuple, **kwargs: Any) -> Any:
    if kwargs.get("trusted") is not None or ht.symbolic != "sympy" or getattr(_PAIR, "slot", None) is not None:
        return _TRACE(fn, args, **kwargs)
    entry = Entry(CURRENT["test"], "trace")
    LOG.append(entry)
    pair = _PAIR.slot = {}
    try:
        first = _TRACE(fn, args, **kwargs)
    except ht.Declined as e:
        ir_decline = pair.get("ir_declined") or pair.get("ir_witness")
        if "ran" not in pair:
            pass  # declined before the symbolic run: one decline for both
        elif ir_decline is not None:
            entry.both_declined = True
            if str(ir_decline) != str(e):
                entry.notes.append(f"declined differently: sympy {e} / IR {ir_decline}")
        elif "sympy_witness" in pair:
            entry.diff = f"the IR tape passed the witness check sympy's declined: {e}"
        elif "sympy_ran" in pair and "ir" in pair:
            # the output records, the capture's end or the witness declined
            # after both runs traced: a check on the call, not on a backend
            entry.both_declined = True
            entry.notes.append(f"declined after both symbolic runs: {e}")
        elif "ir_raised" in pair:
            entry.notes.append(f"sympy declined, IR raised {type(pair['ir_raised']).__name__}: {pair['ir_raised']}")
        else:
            entry.diff = f"IR traced where sympy declined: {e}"
        raise
    finally:
        _PAIR.slot = None
    if "ir" not in pair or "ir_witness" in pair:
        why = pair.get("ir_declined") or pair.get("ir_witness") or pair.get("ir_raised")
        entry.diff = f"IR declined where sympy traced: {why!r}"
        return first
    tr2, result_kind, outputs = pair["ir"]
    second = tape_mod.Tape(tr2, args, outputs, result_kind, first.warm_up_result)
    entry.diff = compare(first, second, entry, _TAPE_FIELDS)
    return first if entry.diff is not None else second


def symbolic_run_both(tr: Any, fn: Any, args: tuple, positions: list, ints: list) -> Any:
    if tr.trusted is not None or isinstance(tr.shape_env, _ir.Env):
        return _SYMBOLIC_RUN(tr, fn, args, positions, ints)
    entry = Entry(CURRENT["test"], "run")
    LOG.append(entry)
    with backend("ir"):
        tr2 = tape_mod._Trace(tr.device, tr.trusted, tr.opaque)
    try:
        result = _SYMBOLIC_RUN(tr, fn, args, positions, ints)
    except ht.Declined as e:
        try:
            _SYMBOLIC_RUN(tr2, fn, args, positions, ints)
        except ht.Declined as e2:
            entry.both_declined = True
            if str(e2) != str(e):
                entry.notes.append(f"declined differently: sympy {e} / IR {e2}")
            raise e from None
        except Exception as e2:
            entry.notes.append(f"sympy declined, IR raised {type(e2).__name__}: {e2}")
            raise e from None
        entry.diff = f"IR traced where sympy declined: {e}"
        raise
    try:
        _SYMBOLIC_RUN(tr2, fn, args, positions, ints)
    except ht.Declined as e:
        entry.diff = f"IR declined where sympy traced: {e}"
        return result
    entry.diff = compare(tr, tr2, entry, _TRACE_FIELDS)
    return result


# ---- the lowering: an IR tape lowered from its nodes vs from its sympy export


@dataclass
class LowerEntry:
    test: str
    diff: str | None = None
    both_declined: bool = False
    rows_sympy: int = 0
    rows_ir: int = 0
    guards: int = 0
    compared: int = 0  # paired result rows
    samples: int = 0
    valid_samples: int = 0  # samples where both programs accept
    notes: list = field(default_factory=list)


LOWER_LOG: list[LowerEntry] = []
# the fields of the lowered records that are rows
_ROW_FIELDS = {
    "ScalarSlot": ("row",),
    "PointerSlot": ("displacement", "address"),
    "LoweredLaunch": ("grid", "block", "smem"),
    "LoweredMemset": ("width", "height", "pitch"),
    "LoweredMemcpy": ("nbytes",),
    "LoweredAllocation": ("sizes", "strides", "nbytes"),
    "LoweredView": ("sizes", "strides", "offset"),
    "PredictedOutput": ("sizes", "strides", "offset"),
    "LoweredEagerCall": ("grid",),
    "LoweredOpaqueCall": ("rows",),
    "LoweredKeyedSite": ("rows",),
    "LoweredSelector": ("predicate",),
    "LoweredTape": ("valid",),
}


def _row_pairs(a: Any, b: Any, path: str, out: list) -> str | None:
    if a is None or b is None:
        return None if a is b else f"{path}: {a} vs {b}"
    if isinstance(a, int) and isinstance(b, int):
        out.append((a, b, path))
        return None
    if len(a) != len(b):
        return f"{path}: {len(a)} vs {len(b)} rows"
    for i, (x, y) in enumerate(zip(a, b)):
        if (r := _row_pairs(x, y, f"{path}[{i}]", out)) is not None:
            return r
    return None


def _pair(a: Any, b: Any, path: str, out: list) -> str | None:
    """The paired rows of two lowerings of one tape, or where they differ in
    anything but rows."""
    if a is b:
        return None
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    name = type(a).__name__
    if name in _ROW_FIELDS:
        for f in dataclasses.fields(a):
            if f.name in ("tape", "program", "compiled", "lowering"):
                continue
            x, y, p = getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}"
            if f.name in _ROW_FIELDS[name]:
                r = _row_pairs(x, y, p, out)
            elif name == "LoweredKeyedSite" and f.name in ("operands", "scratch"):
                items_a = list(x.items() if isinstance(x, dict) else enumerate(x))
                items_b = list(y.items() if isinstance(y, dict) else enumerate(y))
                if [(k, v[1]) for k, v in items_a] != [(k, v[1]) for k, v in items_b]:
                    return f"{p}: {x} vs {y}"
                r = _row_pairs([v[0] for _, v in items_a], [v[0] for _, v in items_b], p, out)
            else:
                r = _pair(x, y, p, out)
            if r is not None:
                return r
        return None
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return f"{path}: {len(a)} vs {len(b)} items"
        for i, (x, y) in enumerate(zip(a, b)):
            if (r := _pair(x, y, f"{path}[{i}]", out)) is not None:
                return r
        return None
    if isinstance(a, dict):
        if list(a) != list(b):
            return f"{path}: keys {list(a)} vs {list(b)}"
        for k in a:
            if (r := _pair(a[k], b[k], f"{path}[{k!r}]", out)) is not None:
                return r
        return None
    try:
        same = bool(a == b)
    except Exception:
        same = False
    return None if same else f"{path}: {a!r} vs {b!r}"


def _run(program: Any, leaves: dict) -> list | None:
    """Every row's value at `leaves` under the evaluator's rules, None when a
    row fails (the call misses)."""
    vals: list = []
    for row in program.instructions:
        op = row[0]
        if op == "constant":
            v = row[1]
        elif op in LEAVES:
            v = leaves[row]
        else:
            v = _step(op, [vals[i] for i in row[1:]])
            if isinstance(v, Status):
                return None
        vals.append(v)
    return vals


def _leaf_samples(programs: list, count: int) -> list[dict]:
    hints: dict = {}
    for prog in programs:
        for row, v in zip(prog.instructions, prog.values):
            if row[0] in LEAVES:
                hints[row] = v
    rng = random.Random(len(hints))
    samples = [dict(hints)]
    for k in range(count):
        # half keep equal hints equal (a size shared by two tensors), half
        # move each leaf on its own
        shared = k % 2 == 0
        remap: dict = {}
        pt = {}
        for row, v in hints.items():
            if row[0] == "pointer":
                pt[row] = v + rng.choice((0, 0, 16, 256 * rng.randint(1, 64), 1))
                continue
            if shared and v in remap:
                pt[row] = remap[v]
                continue
            if v >= 2:
                w = rng.choice((v, v + 1, max(v - 1, 1), 2 * v, 3 * v + 1, 1, 2, 64, v + 17))
            elif v in (0, 1):
                w = rng.choice((v, v, v, 1 - v, 2))
            else:
                w = rng.choice((v, v - 1, -v, 2 * v))
            remap[v] = w
            pt[row] = w
        samples.append(pt)
    return samples


def compare_lowered(a: Any, b: Any, rows_a: tuple, rows_b: tuple, entry: LowerEntry) -> str | None:
    pairs: list = []
    if (r := _pair(a, b, "lowered", pairs)) is not None:
        return r
    if len(rows_a) != len(rows_b):
        return f"{len(rows_a)} vs {len(rows_b)} guards"
    guard_pairs = [(x, y, f"guard {k}") for k, (x, y) in enumerate(zip(rows_a, rows_b))]
    entry.compared, entry.guards = len(pairs), len(rows_a)
    entry.rows_sympy, entry.rows_ir = len(a.program.instructions), len(b.program.instructions)
    for k, pt in enumerate(_leaf_samples([a.program, b.program], 12)):
        va, vb = _run(a.program, pt), _run(b.program, pt)
        if k == 0 and (va != a.program.values or vb != b.program.values):
            return "the reference evaluator disagrees with the hints"
        entry.samples += 1
        miss_a = va is None or va[a.valid] != 1
        miss_b = vb is None or vb[b.valid] != 1
        if miss_a != miss_b:
            return f"sample {k}: the sympy lowering {'misses' if miss_a else 'accepts'}, the IR's {'misses' if miss_b else 'accepts'}"
        if va is None or vb is None:
            continue
        # a guard's value, accepted or not; every result row where both accept
        for x, y, path in guard_pairs + (pairs if not miss_a else []):
            if va[x] != vb[y]:
                return f"sample {k}: {path} is {va[x]} (sympy) vs {vb[y]} (IR)"
        entry.valid_samples += not miss_a
    return None


class _Lowered:
    """lower_tape's result or exception, and its guard rows."""

    def __init__(self, tape: Any, direct: bool) -> None:
        self.result, self.error, self.guard_rows = None, None, ()
        real = lower_mod._TapeLowering

        captured: list = []

        class Capture(real):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                captured.append(self)

        lower_mod._TapeLowering = Capture
        try:
            self.result = _LOWER_TAPE(tape, direct=direct)
        except (ht.Declined, AssertionError) as e:
            self.error = e
        finally:
            lower_mod._TapeLowering = real
        if captured:
            self.guard_rows = getattr(captured[0].lowering, "guard_rows", ())


def lower_both(tape: Any, **kwargs: Any) -> Any:
    if not isinstance(getattr(tape, "shape_env", None), _ir.Env) or kwargs:
        return _LOWER_TAPE(tape, **kwargs)
    entry = LowerEntry(CURRENT["test"])
    LOWER_LOG.append(entry)
    ref, ir = _Lowered(tape, False), _Lowered(tape, True)
    if ref.error is not None and ir.error is not None:
        entry.both_declined = True
        if type(ref.error) is not type(ir.error):
            entry.diff = f"sympy raised {type(ref.error).__name__}: {ref.error} / IR {type(ir.error).__name__}: {ir.error}"
        elif str(ref.error) != str(ir.error):
            entry.notes.append(f"declined differently: sympy {ref.error} / IR {ir.error}")
        raise ir.error
    if ref.error is not None:
        entry.diff = f"the IR lowering accepted what the sympy one declined: {ref.error}"
        raise ref.error
    if ir.error is not None:
        entry.diff = f"the IR lowering declined what the sympy one accepted: {type(ir.error).__name__}: {ir.error}"
        return ref.result
    entry.diff = compare_lowered(ref.result, ir.result, ref.guard_rows, ir.guard_rows, entry)
    return ir.result if entry.diff is None else ref.result


# ---- the metadata mode: a call the trace routes around its fake kernel (a
# traced ATen host, a view on a bare meta tensor, an opaque metadata formula)
# against the fake kernel on the same operands, under the IR backend


@dataclass
class MetaEntry:
    test: str
    route: str  # "aten", "view" or "opaque"
    op: str
    diff: str | None = None
    notes: list = field(default_factory=list)


META_LOG: list[MetaEntry] = []


def _meta_value(v: Any) -> Any:
    # an int, or a SymInt's IR node (a constant node as its int)
    if isinstance(v, torch.SymInt):
        n = v.node.node
        return n.args[0] if n.op == "const" else n
    return int(v)


def _meta_layout(t: torch.Tensor) -> tuple:
    strides = t._sym_strides if isinstance(t, tape_mod._TracedTensor) else t.stride()
    offset = t._sym_offset if isinstance(t, tape_mod._TracedTensor) else t.storage_offset()
    return (
        [_meta_value(v) for v in t.shape],
        [_meta_value(v) for v in strides],
        _meta_value(offset),
        t.dtype,
    )


def _pins(records: list) -> set:
    # symbol == constant (c + k*s == 0), as (symbol, value)
    pins = set()
    for g, _ in records:
        d = g.args[0] if g.op == "eq" else None
        if d is not None and d.op == "sym":
            pins.add((d, 0))
        elif d is not None and d.op == "add" and len(d.args[1]) == 1 and d.args[1][0][0].op == "sym":
            (s, k), = d.args[1]
            pins.add((s, -d.args[0] / k))
    return pins


def _terms(d: Any) -> tuple[int, list] | None:
    # c + sum k * prod f**e, each product's common factor divided out
    if d.op == "sym":
        return 0, [(1, ((d, 1),))]
    if d.op != "add":
        return None
    terms = []
    for t, k in d.args[1]:
        if t.op == "mul":
            terms.append((k * t.args[0], tuple(t.args[1])))
        else:
            terms.append((k, ((t, 1),)))
    if d.args[0] == 0:
        common = collections.Counter(dict(terms[0][1]))
        for _, fs in terms[1:]:
            common &= collections.Counter(dict(fs))
        if common:
            terms = [(k, tuple((f, e - common[f]) for f, e in fs if e > common[f])) for k, fs in terms]
    return d.args[0], terms


def _solve_order(records: list, movable: set) -> list:
    """(symbol, (c, other terms, coefficient)) in evaluation order: each
    equality record solved for one movable symbol it holds linearly, no
    symbol's definition depending on itself."""
    deps: dict = {}
    defs: dict = {}
    for g in records:
        lin = _terms(g.args[0]) if g.op == "eq" else None
        if lin is None:
            continue
        c, terms = lin
        for i, (k, fs) in enumerate(terms):
            if k not in (1, -1) or len(fs) != 1 or fs[0][1] != 1 or fs[0][0].op != "sym":
                continue
            var = fs[0][0].args[0]
            if var not in movable or var in defs:
                continue
            others = terms[:i] + terms[i + 1 :]
            used = set().union(*(f.free_symbols for _, fs2 in others for f, _ in fs2)) if others else set()
            todo, seen = list(used), set()
            while todo:
                u = todo.pop()
                if u not in seen:
                    seen.add(u)
                    todo.extend(deps.get(u, ()))
            if var in seen:
                continue
            defs[var], deps[var] = (c, others, k), used
            break
    order: list = []
    placed: set = set()

    def place(v: str) -> None:
        if v in placed or v not in defs:
            return
        placed.add(v)
        for u in deps[v]:
            place(u)
        order.append((v, defs[v]))

    for v in defs:
        place(v)
    return order


class _Undecided(Exception):
    pass


def _ev(n: Any, pt: dict, memo: dict) -> Any:
    # an integer or boolean node's value with the symbols at pt
    if not isinstance(n, _ir.Node):
        return n
    r = memo.get(n.id)
    if r is not None:
        return r
    op, a = n.op, n.args
    ev = lambda x: _ev(x, pt, memo)  # noqa: E731
    if op == "const":
        v = a[0]
    elif op == "sym":
        v = pt.get(a[0], n.hint)
    elif op in ("true", "false"):
        v = op == "true"
    elif op == "add":
        v = a[0] + sum(cf * ev(t) for t, cf in a[1])
    elif op == "mul":
        v = a[0]
        for f, e in a[1]:
            v *= ev(f) ** e
    elif op in ("floordiv", "ceildiv", "mod"):
        x, d = ev(a[0]), ev(a[1])
        if d == 0:
            raise _Undecided("division by zero")
        v = x // d if op == "floordiv" else -(-x // d) if op == "ceildiv" else x % d
    elif op == "pbn":
        e = ev(a[1])
        if e < 0:
            raise _Undecided("negative exponent")
        v = ev(a[0]) ** e
    elif op in ("eq", "ne", "lt", "le"):
        x = ev(a[0])
        v = {"eq": x == 0, "ne": x != 0, "lt": x < 0, "le": x <= 0}[op]
    elif op == "not":
        v = not ev(a[0])
    elif op in ("and", "or"):
        vals = [ev(x) for x in a]
        v = all(vals) if op == "and" else any(vals)
    elif op == "where":
        v = ev(a[1]) if ev(a[0]) else ev(a[2])
    elif op in ("min", "max"):
        v = (min if op == "min" else max)(ev(x) for x in a)
    elif op == "nod":
        half = len(a) // 2
        v = int(_ir._dense([ev(x) for x in a[:half]], [ev(x) for x in a[half:]]))
    elif op == "bitlen":
        v = ev(a[0]).bit_length()
    else:
        raise _Undecided(op)
    memo[n.id] = v
    return v


def _equal_under_records(env: Any, got: list, want: list, records: list, ndim: int) -> tuple[str | None, int]:
    """(None, points checked) if got and want agree at every perturbed point
    that satisfies the records sharing their symbols (closed over shared
    symbols); else the first disagreeing point."""
    syms: set = set()
    for v in (*got, *want):
        if isinstance(v, _ir.Node):
            syms |= v.free_symbols
    rel = [g for g, _ in records]
    keep: list = []
    grew = True
    while grew:
        grew = False
        for g in rel:
            fs = g.free_symbols
            if fs & syms and g not in keep:
                keep.append(g)
                if not fs <= syms:
                    syms |= fs
                    grew = True
    hints = {s: env.ctx.symbols[s].hint for s in syms if s in env.ctx.symbols}
    # a symbol the records pin (c + k*s == 0, a common factor divided out)
    # takes that value
    pinned: dict = {}
    lins = [lin for g in keep if g.op == "eq" and (lin := _terms(g.args[0])) is not None]
    grew = True
    while grew:
        grew = False
        for c, terms in lins:
            const, var = c, []
            for k, fs in terms:
                if all(f.free_symbols <= pinned.keys() for f, _ in fs):
                    const += k * math.prod(_ev(f, pinned, {}) ** e for f, e in fs)
                else:
                    var.append((k, fs))
            if len(var) == 1 and len(var[0][1]) == 1 and var[0][1][0][1] == 1 and var[0][1][0][0].op == "sym":
                k, ((sym, _),) = var[0]
                if const % k == 0:
                    pinned[sym.args[0]] = -const // k
                    grew = True
    movable = [s for s in sorted(hints) if s not in pinned and ".base" not in env.names.get(s, "")]
    rng = random.Random(len(hints))
    defs = _solve_order(sorted(keep, key=lambda g: len(g.free_symbols)), set(hints) - set(pinned))
    valid = 0
    blocked: collections.Counter = collections.Counter()
    for k in range(48):
        pt = {**hints, **pinned}
        remap: dict = {}
        for s in movable:
            h = hints[s]
            if k % 2 == 0 and h in remap:
                pt[s] = remap[h]
                continue
            if h >= 2:
                # scaling keeps a hint's divisibility and its distance from 0 and 1
                w = rng.choice((h, 2 * h, 3 * h, 5 * h) if k % 3 else (h + 1, max(h - 1, 1), 1, 2, h + 17))
            elif h in (0, 1):
                w = rng.choice((h, h, 1 - h, 2))
            else:
                w = rng.choice((h, 2 * h, h - 1, -h))
            remap[h] = pt[s] = w
        if k == 0:
            pt = {**hints, **pinned}
        for var, (c, terms, cf) in defs:
            memo = {}
            try:
                rest = c + sum(k2 * math.prod(_ev(f, pt, memo) ** e for f, e in fs) for k2, fs in terms)
            except _Undecided:
                continue
            pt[var] = -rest * cf
        memo: dict = {}
        try:
            if not all(_ev(g, pt, memo) for g in keep):
                if k == 0:
                    bad = [_ir.render(g) for g in keep if not _ev(g, pt, memo)]
                    raise AssertionError(f"records false at their hints: {bad[:4]}")
                blocked[next(_ir.render(g) for g in keep if not _ev(g, pt, memo))] += 1
                continue
            a, b = [_ev(v, pt, memo) for v in got], [_ev(v, pt, memo) for v in want]
        except _Undecided:
            continue
        valid += 1
        # a size-1 dim's stride addresses nothing
        for d in range(ndim):
            if a[d] == 1:
                a[ndim + d] = b[ndim + d] = None
        if a != b:
            return f"at {pt}: routed {a}, fake {b}", valid
    return (None if valid > 1 else f"blocked by {blocked.most_common(3)}, movable {movable[:8]}, defs {[v for v, _ in defs]}, pinned {sorted(pinned)[:10]}"), valid


def _meta_compare(tr: Any, route: str, func: Any, outs: list, before: tuple, fake_fn: Any) -> None:
    """Run fake_fn (the fake kernel on the call's operands) from the guard
    record as it was before the routed call, compare its tensors with outs,
    and put the routed call's record back."""
    env = tr.shape_env
    entry = MetaEntry(CURRENT["test"], route, str(func))
    META_LOG.append(entry)
    after = env.records, env._index, env.owners
    env.records, env._index, env.owners = list(before[0]), dict(before[1]), list(before[2])
    try:
        fakes = [o for o in pytree.tree_leaves(fake_fn()) if isinstance(o, torch.Tensor)]
    except Exception as e:
        entry.diff = f"the fake kernel raised {type(e).__name__}: {str(e).splitlines()[0]}"
        return
    finally:
        fake_records = env.records[len(before[0]) :]
        env.records, env._index, env.owners = after
    routed_records = after[0][len(before[0]) :]
    if len(fakes) != len(outs):
        entry.diff = f"{len(outs)} outputs, the fake kernel {len(fakes)}"
        return
    for i, (o, f) in enumerate(zip(outs, fakes)):
        got, want = _meta_layout(o), _meta_layout(f)
        if got == want:
            continue
        flat_got, flat_want = [*got[0], *got[1], got[2]], [*want[0], *want[1], want[2]]
        if got[3] == want[3] and len(got[0]) == len(want[0]) and len(flat_got) == len(flat_want):
            # the fake is eager's stand-in only where its own records hold too
            why, points = _equal_under_records(env, flat_got, flat_want, before[0] + routed_records + fake_records, len(got[0]))
            if why is None:
                entry.notes.append(f"output {i}: equal under the records, not as nodes, {'10+' if points >= 10 else '2-9'} points")
                continue
            if points <= 1:
                entry.diff = f"output {i}: undecided, {points} points satisfy the records, {why}"
                return
        sizes = got[0]
        # a size-1 dim's stride addresses nothing, and the fake's may differ
        # from eager's there
        free = [s == 1 for s in sizes]
        if (got[0], got[2], got[3]) == (want[0], want[2], want[3]) and len(got[1]) == len(want[1]):
            if all(a == b or f1 for a, b, f1 in zip(got[1], want[1], free)):
                entry.notes.append(f"output {i}: a size-1 dim's stride differs")
                continue
        show = lambda lay: ([_ir.render(v) if isinstance(v, _ir.Node) else v for v in (*lay[0], *lay[1], lay[2])], lay[3])
        entry.diff = f"output {i}: routed {show(got)}, fake {show(want)}"
        return
    # the recorder (route "aten") also ran after the fake kernel before it was
    # routed, so its pins were already recorded
    new_pins = set() if route == "aten" else _pins(routed_records) - _pins(fake_records) - _pins(before[0])
    if new_pins:
        entry.diff = f"the routed call pins {sorted(f'{_ir.render(s)} == {v}' for s, v in new_pins)}, the fake kernel does not"
    elif {g for g, _ in routed_records} != {g for g, _ in fake_records}:
        entry.notes.append(
            f"guards: routed {len(routed_records)} new records, fake {len(fake_records)}"
        )


_VIEW = tape_mod._Trace.view
_TRACED_ATEN_CALL = tape_mod._Trace._traced_aten


def _meta_view(tr: Any, func: Any, args: tuple, kwargs: dict) -> Any:
    env = tr.shape_env
    src = args[0]
    if not isinstance(env, _ir.Env) or func not in tape_mod._META_VIEWS or not isinstance(src, tape_mod._TracedTensor):
        return _VIEW(tr, func, args, kwargs)
    before = list(env.records), dict(env._index), list(env.owners)
    out = _VIEW(tr, func, args, kwargs)

    def fake() -> Any:
        # wrapped as both routes wrap it (_TracedTensor's construction guards)
        with tr.fake_mode:
            out = func(tr._twin(src), *args[1:], **kwargs)
        # view() takes eager's computeStride strides over the fake kernel's
        eager = func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default)
        strides = lambda o: tape_mod._compute_stride(src.shape, src._sym_strides, o.shape) if eager else list(o.stride())
        wrap = lambda o: tape_mod._TracedTensor(src._root, list(o.shape), strides(o), o.storage_offset(), o.dtype, src.device)
        return pytree.tree_map_only(torch.Tensor, wrap, out)

    _meta_compare(tr, "view", func, [o for o in pytree.tree_leaves(out) if isinstance(o, torch.Tensor)], before, fake)
    return out


def _meta_traced_aten(tr: Any, func: Any, args: tuple, kwargs: dict, witnessed: bool = True) -> Any:
    env = tr.shape_env
    if not isinstance(env, _ir.Env):
        return _TRACED_ATEN_CALL(tr, func, args, kwargs, witnessed)
    before = list(env.records), dict(env._index), list(env.owners)
    out = _TRACED_ATEN_CALL(tr, func, args, kwargs, witnessed)
    if out is None or isinstance(out, str):
        return out

    def fake() -> Any:
        with tr.fake_mode:
            twin = lambda a: tr._twin(a) if isinstance(a, tape_mod._TracedTensor) else a
            return func(*pytree.tree_map(twin, args), **pytree.tree_map(twin, kwargs))

    outs = list(out) if isinstance(out, tuple) else [out]
    _meta_compare(tr, "aten", func, outs, before, fake)
    return out


_OPAQUE_META = tape_mod._Trace._opaque_meta


def _meta_opaque(tr: Any, func: Any, leaves: list, spec: Any) -> Any:
    env = tr.shape_env
    if not isinstance(env, _ir.Env):
        return _OPAQUE_META(tr, func, leaves, spec)
    before = list(env.records), dict(env._index), list(env.owners)
    meta = _OPAQUE_META(tr, func, leaves, spec)
    if meta is None:
        return meta

    def fake() -> Any:
        with tr.fake_mode:
            twins = [tr._twin(a) if isinstance(a, tape_mod._TracedTensor) else a for a in leaves]
            args, kwargs = pytree.tree_unflatten(twins, spec)
            return func(*args, **kwargs)

    _meta_compare(tr, "opaque", func, [o for o in pytree.tree_leaves(meta[0]) if isinstance(o, torch.Tensor)], before, fake)
    return meta


def install_meta() -> None:
    """The metadata mode alone: every routed call under the IR backend checked
    against its fake kernel (META_LOG)."""
    tape_mod._Trace.view = _meta_view
    tape_mod._Trace._traced_aten = _meta_traced_aten
    tape_mod._Trace._opaque_meta = _meta_opaque


def install() -> None:
    """Rebind trace / _symbolic_run / lower_tape in every loaded module that
    holds them."""
    # the pairing's first run is sympy's
    ht.symbolic = "sympy"
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None)
        if not d or mod is tape_mod or d.get("__name__") == __name__:
            continue
        if d.get("trace") is _TRACE:
            mod.trace = trace_both
        if d.get("_symbolic_run") is _SYMBOLIC_RUN:
            mod._symbolic_run = symbolic_run_both
        if d.get("lower_tape") is _LOWER_TAPE and mod is not lower_mod:
            mod.lower_tape = lower_both
    tape_mod.trace = trace_both
    tape_mod._symbolic_run = _paired_run
    tape_mod._check_witness = _paired_witness
    install_meta()
