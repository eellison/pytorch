"""Host tracing (private): the hint audit, a check that a trace never uses a
symbolic value's hint (its value at the traced call) without a guard.

The soundness rule: a trace either keeps a value symbolic or records a guard
on what it read; specializing a symbolic value to its hint is a guard. Only a
closed library's harvest (cuBLAS, cuDNN, closed attention cubins) may be
empirical, and it keys and checks what it measured.

Every way a trace turns a symbolic value into a concrete one without a guard
reaches a hint read audited here:

  - IR backend: IRSymNode.hint / _hint / require_hint (tape._hint, torch's
    hint_int, optimization_hint, guarding_hint_or_throw, SymInt.hint all read
    it), Env.backed_var_to_val (the exported hints);
  - sympy backend (_TraceShapeEnv): SymNode.hint / _hint / require_hint, and
    the ShapeEnv's size_hint / optimization_hint / guarding_hint_or_throw
    outside evaluate_expr.

int() / bool() / __index__ / guard_* of a SymInt or SymBool and the static
reads (statically_known_true, guard_or_false, guard_size_oblivious) record
their guard where they read (Env.guard_bool / guard_value,
_TraceShapeEnv.evaluate_expr): they are not audited. Neither are the IR's own
hint arithmetic (a new node's hint) nor a constant's value.

A read is sound when, among the guards of the trace that froze (the tape's
first guard_count, a dispatch-again's records), either
  - a guard pins the value: Eq(value, c), or Eq(s, c) for each of its
    symbols; or
  - a guard over the value was recorded while the reading frame was live (the
    reader branched on it and recorded the condition: `if _hint(n) == 0 and
    bool(n == 0)`); or
  - its call site is in ALLOWLIST: a backend's own bookkeeping ("safe"), or
    "guarded" (a guard over the value is recorded by another frame, checked
    to exist).
A read on a symbol of another trace than the one running is never sound, nor
is a value of the warm-up call (a real tensor's layout) written into the
trace as a constant. The target is no hint read at all: OFFENDERS lists the
sites still reading one, each with its symbolic, guard or decline
replacement; strict mode raises on any site in neither table.

Reads while a trace builds its guards are resolved when it freezes (freeze);
reads after (lowering, capture, launchers) at once, against the frozen guards.
A guard recorded after its trace froze is checked by nothing: also a finding.

mode: None (production: nothing is installed, no cost), "survey" (log every
finding, raise nothing) or "strict" (raise HintAuditError). The test suites
run strict. TORCH_HOST_TRACE_HINT_AUDIT=survey|strict enables it at import;
TORCH_HOST_TRACE_HINT_AUDIT_LOG=<path> appends the per-site counts (JSON
lines) at exit.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import traceback
import weakref
from dataclasses import dataclass, field
from typing import Any


mode: str | None = None


class HintAuditError(BaseException):
    """A hint used without a guard, in strict mode. A BaseException: the
    trace's own handlers (a decline, a SegmentFailed) do not turn it into a
    fallback."""


@dataclass(frozen=True)
class Allowed:
    # "safe", "guarded" or "harvest"
    kind: str
    why: str


_TAPE = "torch/cuda/_host_trace_tape.py"
_CUTE = "torch/cuda/_host_trace_cute.py"
_AT_CALL = "evaluate it at the traced call's inputs (the input record's values through the program's rows), not the node's hint"
_WITNESS = f"the witness's stand-ins and comparisons: {_AT_CALL}"

# (path under the torch package's parent, co_qualname) -> why the hint read
# there needs no guard, each site approved by name (user, 10-08)
ALLOWLIST: dict[tuple[str, str], Allowed] = {
    # a new node's trace-time value computed from its operands' is the sympy backend's bookkeeping, as the IR's Node.hint is
    ("torch/cuda/_host_trace.py", "bit_length"): Allowed("safe", "the sympy backend's value of a new node, from its operand's"),
    ("torch/cuda/_host_trace.py", "select"): Allowed("safe", "the sympy backend's value of a new node, from its operands'"),
    ("torch/cuda/_host_trace.py", "f32_div"): Allowed("safe", "the sympy backend's value of a new node, from its operands'"),
}

# every other site that reads a hint, with its replacement (symbolic, a guard,
# or a decline): strict mode notes these and raises on any new site. A fix
# removes its entry; the target is none. Owners (lanes, 10-08): symti
# (land/core/symti: the symbolic TensorIterator retires the pointwise
# witness), CuTe (land/core/cute70).
OFFENDERS: dict[tuple[str, str], str] = {
    # owner: the symti lane (pointwise / elementwise)
    (_TAPE, "_elementwise"): "guards: decide the broadcast from the fake kernel's symbolic output shape, each size comparison a guard (bool(n == m) / bool(n == 1)), not meta at the hints",
    (_TAPE, "_Trace._pointwise_host"): _WITNESS,
    (_TAPE, "_Trace._pointwise_host.<locals>.stand_in"): _WITNESS,
    (_TAPE, "_Trace._pointwise_host.<locals>.check"): _WITNESS,
    (_TAPE, "_Trace._pointwise_host.<locals>.check.<locals>.address"): _WITNESS,
    # owner: the CuTe lane
    (_CUTE, "_describe"): _WITNESS,
    (_CUTE, "_intercept"): _WITNESS,
    (_CUTE, "_intercept_foreign"): _WITNESS,
    (_CUTE, "_pin"): "int(v) (a guard that returns the value) in place of v == _hint(v)",
}


@dataclass
class _Site:
    key: tuple[str, str]
    line: int
    via: str  # the innermost host-trace frame below a site outside it


@dataclass
class _Read:
    site: _Site
    what: str
    node: Any  # an IR Node, or a sympy expression
    frame: int  # id of the reading frame
    code: Any
    at: int  # the env's guard count at the read: a forget below it rolls the read back
    stack: str | None
    by: Any = None  # the guard recorded while the reading frame was live


@dataclass
class _EnvAudit:
    env: weakref.ref
    pending: list[_Read] = field(default_factory=list)
    # reading frame id -> its pending reads, for the frame rule
    frames: dict[int, list[_Read]] = field(default_factory=dict)
    frozen: int | None = None
    reopened: int = 0


@dataclass
class _Stats:
    count: int = 0
    lines: set = field(default_factory=set)
    sample: str = ""
    via: str = ""
    stack: str = ""


_lock = threading.RLock()
_ENVS: dict[int, _EnvAudit] = {}
# (site key, what, status) -> counts
STATS: dict[tuple[tuple[str, str], str, str], _Stats] = {}
violations: list[str] = []
_tls = threading.local()  # guarding: depth inside a guard-recording read (sympy)
_SITE_CACHE: dict[Any, tuple[str, str]] = {}
_STACKED: set[tuple[tuple[str, str], str]] = set()
_installed: list[tuple[Any, str, Any]] = []  # (owner, attribute, original)
_TRANSPARENT: set[Any] = set()
_TRANSPARENT_FILES: tuple[str, ...] = ()
# directories whose reads and guards are not the trace's: the test suites' own
# checks of a tape (set by enable)
exempt: tuple[str, ...] = ()
_ROOT = ""
_HOST_TRACE_DIR = ""
_IR: Any = None


def _rel(path: str) -> str:
    return os.path.relpath(path, _ROOT) if path.startswith(_ROOT) else path


def _key(code: Any) -> tuple[str, str]:
    k = _SITE_CACHE.get(code)
    if k is None:
        k = _SITE_CACHE[code] = (_rel(code.co_filename), code.co_qualname)
    return k


def _reading_frame(skip: tuple[str, ...] = ()) -> Any:
    # the innermost frame that is not the audit's, a hint helper's (tape._hint,
    # hint_int, SymInt.hint), pytree's or sympy's, nor in the files `skip`
    f = sys._getframe(1)
    while f is not None and (
        f.f_code in _TRANSPARENT or f.f_code.co_filename.startswith(_TRANSPARENT_FILES) or f.f_code.co_filename in skip
    ):
        f = f.f_back
    return f


def _site(f: Any) -> _Site:
    via, g = "", f
    if not f.f_code.co_filename.startswith(_HOST_TRACE_DIR):
        while g is not None and not g.f_code.co_filename.startswith(_HOST_TRACE_DIR):
            g = g.f_back
        if g is not None:
            via = f"{_rel(g.f_code.co_filename)}:{g.f_lineno} ({g.f_code.co_qualname})"
    return _Site(_key(f.f_code), f.f_lineno, via)


def _render(node: Any) -> str:
    if isinstance(node, str):
        return node
    if _IR is not None and isinstance(node, _IR.Node):
        return _IR.render(node)[:160]
    return str(node)[:160]


def _note(site: _Site, what: str, status: str, node: Any, stack: str | None) -> None:
    s = STATS.setdefault((site.key, what, status), _Stats())
    s.count += 1
    s.lines.add(site.line)
    if not s.sample:
        s.sample, s.via = _render(node), site.via
    if stack and not s.stack:
        s.stack = stack


def _violations(bad: list[tuple[_Read, str]]) -> None:
    msgs = []
    for read, status in bad:
        site = read.site
        if site.key in OFFENDERS:
            continue
        msg = f"host_trace hint audit: {read.what} of {_render(read.node)} at {site.key[0]}:{site.line} ({site.key[1]}): {status}"
        if site.via:
            msg += f", via {site.via}"
        violations.append(msg)
        msgs.append(f"{msg}\n{read.stack or ''}")
    if msgs and mode == "strict":
        raise HintAuditError("\n".join(msgs))


# ---- the reads


_late = [True]


def _late_helpers() -> None:
    # the hint helpers of modules that import _host_trace_tape (enable runs inside its import)
    from torch.cuda import _host_trace_cute as cute

    for f in (cute._stand_in, cute._stand_ins):
        _TRANSPARENT.update(_codes(f.__code__))
    _late.clear()


def _audit(env: Any, node: Any, what: str) -> None:
    if _late:
        _late_helpers()
    f = _reading_frame()
    if f is None or f.f_code.co_filename.startswith(exempt):
        return
    site = _site(f)
    allowed = ALLOWLIST.get(site.key)
    active = getattr(_IR.ACTIVE, "trace", None)
    first = (site.key, what) not in _STACKED
    stack = None
    if first:
        _STACKED.add((site.key, what))
        stack = "".join(traceback.format_stack(f, limit=30))
    read = _Read(site, what, node, id(f), f.f_code, _guard_count(env), stack)
    if active is not None and active.shape_env is not env:
        read.stack = read.stack or "".join(traceback.format_stack(f, limit=30))
        _note(site, what, "foreign", node, read.stack)
        _violations([(read, "a symbol of another trace than the running one")])
        return
    if allowed is not None and allowed.kind != "guarded":
        _note(site, what, f"allowed:{allowed.kind}", node, stack)
        return
    with _lock:
        ea = _env_audit(env)
        if ea.frozen is not None and not ea.reopened:
            _resolve(ea, [read], ea.frozen, env)
            return
        ea.pending.append(read)
        ea.frames.setdefault(read.frame, []).append(read)


def _env_audit(env: Any) -> _EnvAudit:
    ea = _ENVS.get(id(env))
    if ea is None or ea.env() is not env:
        ea = _ENVS[id(env)] = _EnvAudit(weakref.ref(env, lambda _, k=id(env): _dropped(k)))
    return ea


def _dropped(k: int) -> None:
    # an env that died: a trace that never froze declined, so did its reads
    with _lock:
        ea = _ENVS.pop(k, None)
        for r in ea.pending if ea is not None else ():
            _note(r.site, r.what, "unfrozen" if ea.frozen is None else "pending at death", r.node, r.stack)


def _guard_count(env: Any) -> int:
    return len(env.records) if isinstance(env, _IR.Env) else len(env.guards)


def _ir_hint(self: Any) -> Any:
    n = self.node
    if n.op not in ("const", "fconst"):
        _audit(self.env, n, "hint")
    return n.hint


def _ir_require_hint(self: Any, fallback: Any = None) -> Any:
    n = self.node
    if n.op not in ("const", "fconst"):
        _audit(self.env, n, "require_hint")
    return n.hint


class _AuditedValues(dict):
    """Env.backed_var_to_val: each symbol's hint, a read when looked up."""

    def __init__(self, env: Any, values: dict) -> None:
        super().__init__(values)
        self.env = env

    def __getitem__(self, s: Any) -> Any:
        _audit(self.env, self.env.ctx.symbols[s.name], "backed_var_to_val")
        return super().__getitem__(s)

    def get(self, s: Any, default: Any = None) -> Any:
        return self[s] if s in self else default


def _ir_values(self: Any) -> dict:
    return _AuditedValues(self, self.export.symbol_table()[1])


def _sympy_env(node: Any) -> Any:
    from torch.cuda._host_trace import _TraceShapeEnv

    env = node.shape_env
    return env if isinstance(env, _TraceShapeEnv) else None


def _sympy_read(node: Any, what: str) -> None:
    if getattr(_tls, "guarding", 0):
        return
    env = _sympy_env(node)
    if env is None:
        return
    expr = node.expr
    if not expr.is_number:
        _audit(env, expr, what)


def _symnode_hint_get(self: Any) -> Any:
    hint = self.__dict__.get("_hint")
    if hint is not None and not sys._getframe(1).f_code.co_filename.startswith(_SYM_NODE_FILE):
        _sympy_read(self, "SymNode._hint")
    return hint


def _symnode_hint_set(self: Any, value: Any) -> None:
    self.__dict__["_hint"] = value


def _symnode_hint(self: Any) -> Any:
    hint = self.__dict__.get("_hint")
    if hint is not None and not sys._getframe(1).f_code.co_filename.startswith(_SYM_NODE_FILE):
        _sympy_read(self, "SymNode.hint")
    return hint


_SYM_NODE_FILE = ""
# where a guard is recorded on the host's behalf: a late guard's site is the frame outside them
_GUARD_FILES: tuple[str, ...] = ()


def _guarding(fn: Any) -> Any:
    def run(*args: Any, **kwargs: Any) -> Any:
        _tls.guarding = getattr(_tls, "guarding", 0) + 1
        try:
            return fn(*args, **kwargs)
        finally:
            _tls.guarding -= 1

    return run


def _env_hint_api(name: str, fn: Any) -> Any:
    def run(self: Any, expr: Any, *args: Any, **kwargs: Any) -> Any:
        import sympy

        if not getattr(_tls, "guarding", 0) and isinstance(expr, sympy.Basic) and not expr.is_number:
            _audit(self, expr, name)
        return fn(self, expr, *args, **kwargs)

    return run


# ---- the warm-up's values


class _WarmUpInt(int):
    """An int of a real tensor's layout (the warm-up's, tape._real_layout):
    a value of one call, which may enter the trace only through a guard."""


def _warm_up_layout(original: Any) -> Any:
    def real_layout(t: Any) -> tuple:
        layout = original(t)
        if len(layout) == 1:
            return layout
        sizes, strides, offset, *rest = layout
        return (tuple(map(_WarmUpInt, sizes)), tuple(map(_WarmUpInt, strides)), _WarmUpInt(offset), *rest)

    return real_layout


def _warm_up_value(v: Any, what: str) -> None:
    # a warm-up value entering the trace (a tape layout, an IR constant): a constant of one call, unguarded
    f = _reading_frame(_GUARD_FILES[:2])
    if f is None or f.f_code.co_filename.startswith(exempt):
        return
    site = _site(f)
    stack = "".join(traceback.format_stack(f, limit=30))
    read = _Read(site, what, v, id(f), f.f_code, 0, stack)
    status = "warm-up constant" if site.key not in OFFENDERS else "known offender, warm-up constant"
    _note(site, what, status, f"{int(v)} (the warm-up's)", stack)
    _violations([(read, "a value of the warm-up call written into the trace as a constant")])


def _checked_layout(name: str) -> property:
    def get(self: Any) -> Any:
        return self.__dict__[name]

    def set(self: Any, value: Any) -> None:
        for v in value if isinstance(value, (list, tuple)) else (value,):
            if type(v) is _WarmUpInt:
                _warm_up_value(v, f"TracedTensor.{name}")
        self.__dict__[name] = value

    return property(get, set)


def _checked_const(original: Any) -> Any:
    def const(self: Any, v: int) -> Any:
        if type(v) is _WarmUpInt:
            _warm_up_value(v, "IR constant")
        return original(self, v)

    return const


# ---- the guards


# what a relation the declared domains decide is "recorded" by: a fact at
# every call (the entry checks the domains), so never dropped
_DOMAIN = object()


def _on_record(env: Any, g: Any, by: Any = None) -> None:
    ea = _ENVS.get(id(env))
    if ea is None or ea.env() is not env:
        return
    if by is None and ea.frozen is not None and not ea.reopened and _guard_count(env) > ea.frozen:
        f = _reading_frame(_GUARD_FILES)
        if f is not None and not f.f_code.co_filename.startswith(exempt):
            site = _site(f)
            read = _Read(site, "late guard", g, id(f), f.f_code, 0, "".join(traceback.format_stack(f, limit=30)))
            _note(site, "late guard", "unchecked", g, read.stack)
            _violations([(read, "a guard recorded after its trace froze, which nothing checks")])
        return
    if not ea.frames:
        return
    inside = None
    f = sys._getframe(1)
    while f is not None:
        reads = ea.frames.get(id(f))
        if reads:
            if inside is None:
                inside = _over(g)
            for r in reads:
                if r.by is None and r.code is f.f_code and _covers(inside, r.node):
                    r.by = g if by is None else by
        f = f.f_back


def _over(g: Any) -> Any:
    # what a guard is over: its IR nodes' ids (of a node or a tuple of them), or the sympy expression
    if _IR is not None and isinstance(g, (_IR.Node, tuple)):
        seen, todo = set(), list(g) if isinstance(g, tuple) else [g]
        while todo:
            n = todo.pop()
            if n.id not in seen:
                seen.add(n.id)
                todo.extend(_IR.children(n))
        return seen
    return g


def _covers(over: Any, node: Any) -> bool:
    if isinstance(over, set):
        return node.id in over
    return over.has(node)


def _ir_record(original: Any) -> Any:
    def record(self: Any, g: Any, written: Any) -> None:
        original(self, g, written)
        _on_record(self, g)

    return record


def _ir_rel(original: Any) -> Any:
    def rel(self: Any, cls: str, other: Any) -> Any:
        out = original(self, cls, other)
        if out.node.op in ("true", "false"):
            sides = tuple(n for n in (self.node, other.node) if n.op != "const")
            if sides:
                _on_record(self.env, sides, _DOMAIN)
        return out

    return rel


def _ir_guard_bool(original: Any) -> Any:
    # a SymBool the host branched on is covered by the guard recorded for it,
    # its negation where it was false
    def guard_bool(self: Any, g: Any, written: Any = None) -> bool:
        out = original(self, g, written)
        if g.op not in ("true", "false"):
            _on_record(self, g, g if out else self.ctx.not_(g))
        return out

    return guard_bool


def _sympy_record(original: Any) -> Any:
    def record(self: Any, g: Any, size_oblivious: bool = False) -> None:
        original(self, g, size_oblivious)
        _on_record(self, g)

    return record


def _forgetting(original: Any) -> Any:
    def forget(self: Any, n: int) -> None:
        original(self, n)
        with _lock:
            ea = _ENVS.get(id(self))
            if ea is not None and ea.env() is self:
                # a rollback: the reads after its mark are the rolled-back attempt's
                for r in ea.pending:
                    if r.at >= n:
                        _note(r.site, r.what, "rolled back", r.node, r.stack)
                ea.pending = [r for r in ea.pending if r.at < n]
                ea.frames = {}
                for r in ea.pending:
                    ea.frames.setdefault(r.frame, []).append(r)

    return forget


# ---- resolution


def freeze(env: Any, count: int) -> None:
    """env's first `count` guards are a trace's, from now on fixed: its
    pending reads are resolved against them."""
    if mode is None:
        return
    with _lock:
        ea = _env_audit(env)
        ea.frozen = count if ea.frozen is None else max(ea.frozen, count)
        pending, ea.pending, ea.frames = ea.pending, [], {}
        _resolve(ea, pending, count, env)


class reopened:
    """A scope that adds guards to a frozen env (bind_opaque's keep_guards):
    its reads are pending until it freezes again or leaves."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def __enter__(self) -> None:
        if mode is not None:
            with _lock:
                _env_audit(self.env).reopened += 1

    def __exit__(self, *exc: object) -> None:
        if mode is None:
            return
        with _lock:
            ea = _env_audit(self.env)
            ea.reopened -= 1
            if not ea.reopened and ea.frozen is not None:
                pending, ea.pending, ea.frames = ea.pending, [], {}
                _resolve(ea, pending, ea.frozen, self.env)


def _guards(env: Any, count: int) -> list[Any]:
    # IR: (node, written) records; sympy: expressions
    if isinstance(env, _IR.Env):
        return list(env.records[:count])
    return [g.expr for g in env.guards[:count]]


def _pins(env: Any, guards: list[Any]) -> tuple[set, set]:
    """The values the guards pin to a constant (Eq(value, c)), and the symbols among them."""
    nodes: set = set()
    if isinstance(env, _IR.Env):
        for g, written in guards:
            if written is not None and written[0] == "Eq":
                a, b = written[1:]
                if b.op == "const":
                    nodes.add(a.id)
                elif a.op == "const":
                    nodes.add(b.id)
            nodes.add(g.id)
            if g.op in ("eq", "ne"):
                nodes.add(("eq" if g.op == "ne" else "ne", g.args))
            elif g.op == "not":
                nodes.add(g.args[0].id)
            if g.op != "eq":
                continue
            # eq(d, 0), d = value - c: an add of one term, or the value itself
            d = g.args[0]
            if d.op == "add":
                (t, coeff), *more = d.args[1]
                if not more and abs(coeff) == 1:
                    nodes.add(t.id)
            elif d.op != "const":
                nodes.add(d.id)
        syms = {n.args[0] for n in env.ctx.symbols.values() if n.id in nodes}
        return nodes, syms
    import sympy

    for g in guards:
        if isinstance(g, sympy.Eq):
            a, b = g.args
            if b.is_number:
                nodes.add(a)
            elif a.is_number:
                nodes.add(b)
    return nodes, {s for s in nodes if isinstance(s, sympy.Symbol)}


def _symbols_of(node: Any) -> set:
    if _IR is not None and isinstance(node, _IR.Node):
        out, seen, todo = set(), set(), [node]
        while todo:
            n = todo.pop()
            if n.id not in seen:
                seen.add(n.id)
                if n.op in ("sym", "fsym"):
                    out.add(n.args[0])
                todo.extend(_IR.children(n))
        return out
    return set(node.free_symbols)


def _resolve(ea: _EnvAudit, reads: list[_Read], count: int, env: Any) -> None:
    if not reads:
        return
    guards = _guards(env, count)
    pinned, pinned_syms = _pins(env, guards)
    if isinstance(env, _IR.Env):
        guards = [g for g, _ in guards]
    kept = set(map(id, guards))
    over = None
    bad = []
    for r in reads:
        ident = r.node.id if isinstance(env, _IR.Env) else r.node
        twin = (r.node.op, r.node.args) if isinstance(env, _IR.Env) and r.node.op in ("eq", "ne") else None
        if ident in pinned or twin in pinned or _symbols_of(r.node) <= pinned_syms:
            status = "pinned"
        elif r.by is _DOMAIN:
            status = "decided by the domains in frame"
        elif r.by is not None and id(r.by) in kept:
            status = "guarded in frame"
        else:
            allowed = ALLOWLIST.get(r.site.key)
            if over is None:
                over = [_over(g) for g in guards]
            mentioned = any(_covers(o, r.node) for o in over)
            if allowed is not None and allowed.kind == "guarded" and mentioned:
                status = "allowed:guarded"
            else:
                status = "unguarded (a guard mentions it)" if mentioned else "unguarded"
                bad.append((r, status))
                if r.site.key in OFFENDERS:
                    status = f"known offender, {status}"
        _note(r.site, r.what, status, r.node, r.stack)
    _violations(bad)


# ---- install


def _codes(code: Any) -> list[Any]:
    # a function's code and its nested functions', lambdas' and generator expressions'
    out = [code]
    for c in code.co_consts:
        if hasattr(c, "co_code"):
            out += _codes(c)
    return out


def _patch(owner: Any, name: str, value: Any) -> None:
    _installed.append((owner, name, owner.__dict__.get(name, _MISSING)))
    setattr(owner, name, value)


_MISSING = object()


def enable(new_mode: str = "strict") -> None:
    """Installs the audit ("survey" or "strict"); idempotent."""
    global mode, _ROOT, _HOST_TRACE_DIR, _IR, _TRANSPARENT_FILES, _SYM_NODE_FILE, _GUARD_FILES, exempt
    if new_mode not in ("survey", "strict"):
        raise ValueError(f"hint audit mode {new_mode!r}")
    if mode is not None:
        mode = new_mode
        return
    import sympy

    import torch
    import torch.utils._pytree as pytree
    from torch.cuda import _host_trace, _host_trace_ir as ir, _host_trace_tape as tape
    from torch.fx.experimental import sym_node, symbolic_shapes

    _IR = ir
    _ROOT = os.path.dirname(os.path.dirname(torch.__file__)) + os.sep
    _HOST_TRACE_DIR = os.path.join(os.path.dirname(torch.__file__), "cuda", "_host_trace")
    _SYM_NODE_FILE = sym_node.__file__
    exempt = (os.path.join(_ROOT, "test") + os.sep,)
    _GUARD_FILES = (ir.__file__, _host_trace.__file__, sym_node.__file__, symbolic_shapes.__file__, torch.__file__, tape.__file__)
    import unittest

    # unittest's and torch.testing's assertEqual of a SymBool is the test's own: its site is the test's frame
    testing = os.path.join(os.path.dirname(torch.__file__), "testing") + os.sep
    _TRANSPARENT_FILES = (__file__, pytree.__file__, os.path.dirname(sympy.__file__) + os.sep, os.path.dirname(unittest.__file__) + os.sep, testing)
    helpers = (
        tape._hint,
        tape._traced_layout,
        tape._meta_at_hints,
        symbolic_shapes.guarding_hint_or_throw,
        symbolic_shapes.optimization_hint,
        torch.SymInt.hint.fget,
        torch.SymFloat.hint.fget,
        torch.SymBool.hint.fget,
    )
    for f in helpers:
        _TRANSPARENT.update(_codes(f.__code__))
    for name in ("hint_int", "size_hint", "has_hint"):
        if callable(f := getattr(symbolic_shapes, name, None)) and hasattr(f, "__code__"):
            _TRANSPARENT.add(f.__code__)
    # the hint helpers: functions that only hand hints on (as values, layouts,
    # meta tensors or stand-in tensors), so a read's site is the frame using them
    _patch(ir.IRSymNode, "hint", property(_ir_hint))
    _patch(ir.IRSymNode, "_hint", property(_ir_hint))
    _patch(ir.IRSymNode, "require_hint", _ir_require_hint)
    _patch(ir.Env, "backed_var_to_val", property(_ir_values))
    _patch(ir.Env, "_record", _ir_record(ir.Env._record))
    _patch(ir.Env, "guard_bool", _ir_guard_bool(ir.Env.guard_bool))
    _patch(ir.IRSymNode, "_rel", _ir_rel(ir.IRSymNode._rel))
    _patch(ir.Env, "forget", _forgetting(ir.Env.forget))
    env_cls = _host_trace._TraceShapeEnv
    _patch(env_cls, "_record", _sympy_record(env_cls._record))
    _patch(env_cls, "forget", _forgetting(env_cls.forget))
    _patch(env_cls, "evaluate_expr", _guarding(env_cls.evaluate_expr))
    _patch(env_cls, "evaluate_sym_node", _guarding(symbolic_shapes.ShapeEnv.evaluate_sym_node))
    for name in ("size_hint", "optimization_hint", "guarding_hint_or_throw"):
        _patch(env_cls, name, _env_hint_api(name, getattr(symbolic_shapes.ShapeEnv, name)))
    _patch(tape, "_real_layout", _warm_up_layout(tape._real_layout))
    _patch(tape._TracedTensor, "_sym_strides", _checked_layout("_sym_strides"))
    _patch(tape._TracedTensor, "_sym_offset", _checked_layout("_sym_offset"))
    _patch(ir.Ctx, "const", _checked_const(ir.Ctx.const))
    _patch(sym_node.SymNode, "hint", property(_symnode_hint))
    _patch(sym_node.SymNode, "_hint", property(_symnode_hint_get, _symnode_hint_set))
    mode = new_mode
    if os.environ.get("TORCH_HOST_TRACE_HINT_AUDIT_LOG"):
        atexit.register(write_log, os.environ["TORCH_HOST_TRACE_HINT_AUDIT_LOG"])


def enable_for_tests() -> None:
    """The test suites' gate: strict unless TORCH_HOST_TRACE_HINT_AUDIT chose a mode."""
    if mode is None:
        enable("strict")


def disable() -> None:
    global mode
    while _installed:
        owner, name, original = _installed.pop()
        if original is _MISSING:
            delattr(owner, name)
        else:
            setattr(owner, name, original)
    mode = None


def summary() -> list[dict]:
    """Per (site, what, status): its count, lines, a sample value and the host-trace frame below it."""
    rows = []
    for (key, what, status), s in sorted(STATS.items(), key=lambda kv: -kv[1].count):
        allowed = ALLOWLIST.get(key)
        rows.append(
            {
                "offender": OFFENDERS.get(key),
                "file": key[0],
                "function": key[1],
                "lines": sorted(s.lines),
                "what": what,
                "status": status,
                "count": s.count,
                "sample": s.sample,
                "via": s.via,
                "allowed": None if allowed is None else [allowed.kind, allowed.why],
                "stack": s.stack,
            }
        )
    return rows


def write_log(path: str) -> None:
    """Appends summary() as JSON lines, each with the process's argv."""
    rows = summary()
    with open(path, "a") as f:
        for row in rows:
            row["argv"] = sys.argv
            f.write(json.dumps(row) + "\n")


# the mode TORCH_HOST_TRACE_HINT_AUDIT asks for; _host_trace_tape enables it once its modules are loaded
requested = os.environ.get("TORCH_HOST_TRACE_HINT_AUDIT")
