# Owner(s): ["module: cuda"]
"""The guard census of a host-tracing tape, as a test-time tool.

A tape records per raw guard the op that raised it, the kernel-choice depth
the host had declared where it was raised (Recorder.h KernelChoice: depth > 0
picks a kernel or a launch configuration, depth 0 decides metadata) and its
origin by the op's route (`Tape.guard_rows`, `Tape.op_table`,
`Tape.kept_raw`). This module is the oracle for that tagging: it decides
each raw guard's effect class by flipping it and running the op's meta at
concrete metadata on the other side (`classify_effect`), then reports both
directions: kernel-tagged guards whose flip changes the op's output metadata
(`Census.danger`, none allowed: `check`), and metadata-tagged guards the
oracle calls kernel-only (`Census.depth0_kernel`: hosts still raising
kernel-only guards at depth 0, candidates for a KernelChoice entry). The
census tables (by origin, by effect, origin x effect, phase x effect, top
ops) come from the tape's own fields; the Python site of a guard (a frame
walk) is the debug form, recorded only under `record_sites`.

Effect classes (the oracle's vocabulary, per raw guard):
  K         a flip exists and the op's meta at the flipped inputs gives the
            tape's output expressions evaluated there: kernel-only
  K-opaque  reads an opaque (occupancy) result and no input metadata
  K-range   an int32 index-width test no candidate reaches
  K-addr    address symbols only (an alignment or identity class)
  K-foreign no input-metadata symbol of the op
  M         a flip exists and the meta's sizes, strides or dtype differ
  M-alias   the op conditionally aliases an input (no meta oracle)
  V         every flip is an error in the op's meta (validity)
  P         no op on the stack: the Python between ops decides the program
  D         an A243 domain guard the declared domains make true
  R         no flip exists under the op's earlier guards (implied)
  U         unclassified (a constant, a non-sympy guard)
The classifier is an instrument, not a contract: it perturbs at most two
symbols, reasons per op, and trusts the meta's fidelity (a meta that lacks a
check eager makes reports a validity guard as K).

  python test/host_trace_guard_census.py --case ln [--backend ir] [--out DIR]
  python test/host_trace_guard_census.py --setup harness.py:setup_gpt2 --case gpt2
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import importlib.util
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.cuda import _host_trace as ht, _host_trace_ir as _ir


if TYPE_CHECKING:
    from collections.abc import Iterator


SELECTOR = ("K", "K-opaque", "K-addr", "K-range", "K-foreign")  # the per-op selector's
PERSISTED = ("M", "M-alias", "V", "P")  # the persisted set under the design
_RECORDER_FILES = ("torch/cuda/_host_trace.py", "torch/cuda/_host_trace_ir.py")
_RECORDER_MACHINERY = {
    "evaluate_expr", "_record", "domain", "_domain", "_range", "_root_identity",
    "__setitem__", "note_pin", "guard_bool", "guard_int", "guard_value", "_bool",
    "evaluate_sym_node", "bool_", "int_", "evaluate", "_attribution",
}  # fmt: skip
_MACHINERY = (
    "/sympy/",
    "torch/fx/experimental/sym_node.py",
    "torch/fx/experimental/symbolic_shapes.py",
    "torch/fx/experimental/recording.py",
    "torch/utils/_sympy/",
    "torch/__init__.py",
    "torch/utils/_python_dispatch.py",
    "torch/_ops.py",
    "torch/_C/",
    "torch/utils/_stats.py",
    "torch/utils/_pytree.py",
    "torch/_subclasses/functional_tensor.py",
    "host_trace_guard_census.py",
)


@dataclass
class OpView:
    """An op of the tape's op table with its argument descriptors as sympy."""

    index: int
    func: Any
    name: str
    route: str | None
    parent: int
    depth: int
    seq: list
    args: tuple
    kwargs: dict
    inputs: list  # tensor descriptors among the arguments
    outputs: list  # tensor descriptors among the outputs
    aliases: bool  # an output's root is an input's
    declined: bool
    guards: list  # raw indices of the guards the op raised itself


@dataclass
class GuardRow:
    raw: int
    text: str
    kept: int | None  # the position in Tape.guards, None when the pass dropped it
    kept_text: str | None
    op: int  # -1 between ops
    func: str | None
    route: str | None
    origin: str
    depth: int
    phase: str  # "kernel" (depth > 0) or "meta"
    effect: str | None = None
    detail: str = ""
    site: str | None = None  # the Python site, under record_sites only
    syms: tuple = ()


@dataclass
class Census:
    case: str
    backend: str
    trace_s: float
    effect_s: float
    n_ops: int
    n_top_ops: int
    n_raw: int
    n_kept: int
    launches: int
    regions: int
    allocs: int
    rows: list = field(default_factory=list)
    ops: list = field(default_factory=list)

    # ---- the two directions of the oracle, over the kept guards
    @property
    def danger(self) -> list:
        """Kernel-tagged kept guards whose flip changes the op's output
        metadata (or that conditionally alias): a mis-declared context."""
        return [
            r
            for r in self.rows
            if r.kept is not None
            and r.phase == "kernel"
            and r.effect in ("M", "M-alias")
        ]

    @property
    def mis_tags(self) -> list:
        """Kernel-tagged kept guards the oracle calls validity or Python."""
        return [
            r
            for r in self.rows
            if r.kept is not None and r.phase == "kernel" and r.effect in ("V", "P")
        ]

    @property
    def depth0_kernel(self) -> list:
        """Metadata-tagged kept guards the oracle calls kernel-only: hosts
        still raising kernel-only guards outside a KernelChoice context."""
        return [
            r
            for r in self.rows
            if r.kept is not None and r.phase == "meta" and r.effect in SELECTOR
        ]


# ---------------------------------------------------------------- the tape's fields


def _tensors_in(x: Any, out: list) -> list:
    if isinstance(x, dict) and "root" in x:
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _tensors_in(y, out)
    return out


def _as_sympy(a: Any, tape: ht.Tape) -> Any:
    # an op-table argument with its symbolic values as sympy expressions: a
    # tensor descriptor keeps its dict form, a symbolic scalar becomes
    # ("sym", expr), a list / tuple maps elementwise
    if isinstance(a, dict) and "root" in a:
        d = dict(a)
        d["sizes"] = [tape.sym_expr(v) for v in a["sizes"]]
        d["strides"] = [tape.sym_expr(v) for v in a["strides"]]
        d["offset"] = tape.sym_expr(a["offset"])
        d["kind"] = "traced" if a["root"] is not None else "plain"
        return d
    if isinstance(a, ht._SYM_TYPES):
        return ("sym", tape.sym_expr(a))
    if isinstance(a, (list, tuple)):
        return type(a)(_as_sympy(x, tape) for x in a)
    return a


def op_views(tape: ht.Tape) -> list[OpView]:
    """The op table with sympy descriptors (`Tape.op_table` with `sym_expr`)."""
    out = []
    for rec, row in zip(tape.ops, tape.op_table()):
        args = _as_sympy(row["args"], tape)
        kwargs = {k: _as_sympy(v, tape) for k, v in row["kwargs"].items()}
        outputs = _as_sympy(row["outputs"], tape)
        inputs = _tensors_in(args, []) + _tensors_in(list(kwargs.values()), [])
        outs = _tensors_in(outputs, [])
        roots = {d["root"] for d in inputs if d["root"]}
        out.append(
            OpView(
                rec.index,
                rec.func,
                str(rec.func),
                rec.route,
                rec.parent,
                rec.depth,
                row["seq"],
                args,
                kwargs,
                inputs,
                outs,
                any(d["root"] in roots for d in outs),
                rec.declined,
                row["guards"],
            )
        )
    return out


def guard_rows(tape: ht.Tape, sites: list | None = None) -> list[GuardRow]:
    """Per raw guard of the tape: its text, its kept position, and the tape's
    attribution (op, origin, depth, phase); `sites` from `record_sites`."""
    raw = tape.raw_guards()
    kept_of = {r: k for k, r in enumerate(tape.kept_raw)}
    rows = []
    for i, (g, (op, depth, origin)) in enumerate(zip(raw, tape.guard_rows)):
        rec = tape.ops[op] if op >= 0 else None
        k = kept_of.get(i)
        rows.append(
            GuardRow(
                i,
                str(g),
                k,
                str(tape.guards[k]) if k is not None else None,
                op,
                str(rec.func) if rec is not None else None,
                rec.route if rec is not None else None,
                origin,
                depth,
                "kernel" if depth > 0 else "meta",
                site=sites[i] if sites is not None and i < len(sites) else None,
                syms=tuple(sorted(str(s) for s in getattr(g, "free_symbols", ()))),
            )
        )
    return rows


# ---------------------------------------------------------------- the debug form: sites


def _site() -> str:
    """The innermost Python frame outside the symbolic machinery: where the
    guard was raised from the recorder's point of view (a frame walk, the
    debug form)."""
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        name = f.f_code.co_name
        if any(r in fn for r in _RECORDER_FILES) and name in _RECORDER_MACHINERY:
            f = f.f_back
            continue
        if not any(m in fn for m in _MACHINERY):
            return f"{os.path.basename(fn)}:{f.f_lineno} {name}"
        f = f.f_back
    return "?"


_tls = threading.local()


@contextlib.contextmanager
def record_sites() -> Iterator[dict]:
    """Record the site of every raw guard while tracing: the yielded dict maps
    an env (by id) to its list of sites, aligned with its raw record. A guard
    a C++ host evaluated names the host's file and line (what c10's
    guard_bool / guard_int hand the SymNode); one raised from Python names
    the innermost Python frame outside the symbolic machinery. Test-time
    only: a frame walk per guard and a wrapper on every guard read."""
    sites: dict[int, list] = {}

    def wrap_record(cls: Any) -> Any:
        orig = cls._record

        @functools.wraps(orig)
        def record(self: Any, g: Any, *args: Any, **kwargs: Any) -> None:
            n = len(self.guards)
            orig(self, g, *args, **kwargs)
            if len(self.guards) != n:
                cpp = getattr(_tls, "cpp_site", None)
                sites.setdefault(id(self), []).append(cpp or _site())

        cls._record = record
        return orig

    def wrap_read(cls: Any, name: str) -> Any:
        orig = getattr(cls, name)

        @functools.wraps(orig)
        def read(self: Any, file: Any = "", line: Any = 0, *args: Any) -> Any:
            # a C++ evaluation passes its __FILE__ / __LINE__; Python passes ""
            prev = getattr(_tls, "cpp_site", None)
            _tls.cpp_site = f"{os.path.basename(str(file))}:{line}" if file else prev
            try:
                return orig(self, file, line, *args)
            finally:
                _tls.cpp_site = prev

        setattr(cls, name, read)
        return orig

    from torch.fx.experimental.sym_node import SymNode

    reads = (
        "guard_bool",
        "guard_int",
        "guard_float",
        "expect_true",
        "guard_size_oblivious",
    )
    saved = []
    for cls in (SymNode, _ir.IRSymNode):
        for name in reads:
            if name in cls.__dict__:
                saved.append((cls, name, wrap_read(cls, name)))
    o1 = wrap_record(ht._TraceShapeEnv)
    o2 = wrap_record(_ir.Env)
    try:
        yield sites
    finally:
        ht._TraceShapeEnv._record = o1
        _ir.Env._record = o2
        for cls, name, orig in saved:
            setattr(cls, name, orig)


def sites_of(tape: ht.Tape, sites: dict) -> list | None:
    env = tape.ir_env if tape.ir_env is not None else tape.shape_env
    return sites.get(id(env))


# ---------------------------------------------------------------- the effect oracle

_SIZE_CANDIDATES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65,
                    127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047,
                    2048, 2049, 4095, 4096, 4097, 8192, 16384, 2147483648, 4294967297]  # fmt: skip


def _candidates(hint: Any) -> list:
    h = int(hint)
    extra = [h + 1, h - 1, 2 * h, h // 2, h + 7, h + 16, h * 3]
    seen: list = []
    for v in extra + _SIZE_CANDIDATES:
        if v >= 0 and v != h and v not in seen:
            seen.append(v)
    return seen


def _truth(expr: Any, assign: dict) -> bool | None:
    try:
        v = expr.xreplace(assign)
    except Exception:
        return None
    if v is sympy.true or v is True:
        return True
    if v is sympy.false or v is False:
        return False
    try:
        v = v.doit()
    except Exception:
        return None
    if v is sympy.true:
        return True
    if v is sympy.false:
        return False
    return None


def _val(expr: Any, assign: dict) -> Any:
    if isinstance(expr, (int, bool, float)) or expr is None:
        return expr
    if isinstance(expr, sympy.Basic):
        v = expr.xreplace(assign)
        if v.is_Integer:
            return int(v)
        if v.is_Float or v.is_Rational:
            return float(v)
        if v is sympy.true:
            return True
        if v is sympy.false:
            return False
        try:
            return int(sympy.simplify(v))
        except Exception:
            raise ValueError(f"unresolved {expr} -> {v}") from None
    return expr


def _sym_free(a: Any) -> set:
    if isinstance(a, tuple) and len(a) == 2 and a[0] == "sym":
        return set(a[1].free_symbols)
    if isinstance(a, (list, tuple)):
        s: set = set()
        for x in a:
            if not isinstance(x, dict):
                s |= _sym_free(x)
        return s
    return set()


def _size_syms(op: OpView) -> set:
    out: set = set()
    for d in op.inputs:
        if d["kind"] == "traced":
            for v in d["sizes"]:
                if isinstance(v, sympy.Basic):
                    out |= v.free_symbols
    return out


def _meta_syms(op: OpView) -> set:
    out: set = set()
    for d in op.inputs:
        if d["kind"] != "traced":
            continue
        for v in [*d["sizes"], *d["strides"], d["offset"]]:
            if isinstance(v, sympy.Basic):
                out |= v.free_symbols
    for a in list(op.args) + list(op.kwargs.values()):
        out |= _sym_free(a)
    return out


def _mk(d: dict, assign: dict, fm: Any) -> torch.Tensor:
    sizes = [int(_val(v, assign)) for v in d["sizes"]]
    strides = [int(_val(v, assign)) for v in d["strides"]]
    if any(s < 0 for s in sizes) or any(s < 0 for s in strides):
        raise ValueError("negative size or stride")
    with fm:
        return torch.empty_strided(sizes, strides, dtype=d["dtype"], device=d["device"])


def _arg_at(a: Any, assign: dict, fm: Any) -> Any:
    if isinstance(a, dict) and "kind" in a:
        return _mk(a, assign, fm)
    if isinstance(a, tuple) and len(a) == 2 and a[0] == "sym":
        return _val(a[1], assign)
    if isinstance(a, list):
        return [_arg_at(x, assign, fm) for x in a]
    if isinstance(a, tuple):
        return tuple(_arg_at(x, assign, fm) for x in a)
    return a


def _same_meta(desc: dict, t: torch.Tensor, assign: dict) -> tuple[bool, str]:
    """The tape's recorded output metadata evaluated at `assign` against a meta
    output `t`: sizes, dtype, strides where the size is > 1."""
    if desc is None or desc["kind"] != "traced":
        return True, ""
    sizes = [int(_val(v, assign)) for v in desc["sizes"]]
    if list(t.shape) != sizes:
        return False, f"sizes tape {sizes} meta {list(t.shape)}"
    if desc["dtype"] != t.dtype:
        return False, f"dtype tape {desc['dtype']} meta {t.dtype}"
    strides = [int(_val(v, assign)) for v in desc["strides"]]
    for i, s in enumerate(sizes):
        if s > 1 and strides[i] != t.stride(i):
            return False, f"stride dim {i} tape {strides} meta {list(t.stride())}"
    return True, ""


def _tensor_outputs(out: Any) -> list:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        r: list = []
        for x in out:
            r += _tensor_outputs(x)
        return r
    return []


def classify_effect(
    g: Any,
    op: OpView | None,
    earlier: list,
    hints: dict,
    fm: Any,
    addr_syms: set,
    opaque_syms: set,
    max_flips: int = 2,
    stride_first: bool = False,
) -> tuple[str, str]:
    """The effect class of raw guard `g` (a sympy relation) raised by `op`
    after its `earlier` guards, decided by flipping it: the guard's symbols
    that are input metadata of the op are perturbed (one symbol, then pairs)
    over candidates around the hints; an assignment that makes the guard
    false with the op's earlier guards still true is a flip; the op's meta
    (FakeTensorMode `fm`) runs at the flip and its outputs are compared with
    the tape's output expressions evaluated there. With `stride_first` the
    symbols that are not sizes (strides, offsets) are tried first: a size
    flip an op rejects (cat's shape agreement) then does not exhaust the
    budget before the stride flip a contiguity test turns on."""
    if op is None:
        return "P", "no op on the stack"
    if not isinstance(g, sympy.Basic):
        return "U", "non-sympy guard"
    syms = g.free_symbols
    names = [str(s) for s in syms]
    if not syms:
        return "U", "constant"
    meta = _meta_syms(op) - addr_syms
    if syms <= addr_syms:
        return "K-addr", ""
    if syms & opaque_syms and not (syms & meta):
        return "K-opaque", ""
    if not (syms & meta):
        return "K-foreign", f"symbols {names}"
    if op.aliases and op.route not in ("view", "routed"):
        # an in-place op aliases unconditionally (out is self); a composite
        # that may return self or a copy is the conditional case
        ins = {d["root"] for d in op.inputs if d["root"]}
        outs = [d for d in op.outputs if d]
        if op.func.is_view or op.func._schema.name.endswith("_"):
            pass
        elif any(o["root"] in ins for o in outs) and op.route == "composite":
            return "M-alias", f"{op.name} returns an input's root"
    size_syms = _size_syms(op)
    flip_syms = sorted(syms & meta, key=str)
    if stride_first:
        flip_syms.sort(key=lambda s: s in size_syms)
    tried = 0
    evals = 0
    results: list = []
    big = (
        len(op.inputs) > 64
    )  # foreach / fused optimizer ops: hundreds of fake tensors per try
    max_tries = 3 if big else 12
    for k in (1, 2):
        # the pair search runs when no single flip gave a valid op (a size the
        # op unifies with another must move together)
        if k == 2 and (
            any(r[0] != "invalid" for r in results) or len(flip_syms) > 6 or big
        ):
            break
        for combo in itertools.combinations(flip_syms, k):
            cand_lists: list | None = []
            for s in combo:
                h = hints.get(s)
                if h is None:
                    cand_lists = None
                    break
                c = _candidates(h)
                if s in size_syms:
                    c = [v for v in c if v >= 1]
                cand_lists.append(c if k == 1 else c[:8])
            if cand_lists is None:
                continue
            for values in itertools.product(*cand_lists):
                evals += 1
                if evals > 800:
                    break
                assign = dict(hints)
                for s, v in zip(combo, values):
                    assign[s] = sympy.Integer(v)
                if _truth(g, assign) is not False:
                    continue
                if any(_truth(e, assign) is False for e in earlier):
                    continue
                tried += 1
                try:
                    args = _arg_at(op.args, assign, fm)
                    kwargs = _arg_at(op.kwargs, assign, fm)
                    with fm:
                        out = op.func(*args, **kwargs)
                except Exception as e:  # the op's meta rejects the flipped inputs
                    results.append(
                        (
                            "invalid",
                            f"{combo}={values}: {type(e).__name__}: {str(e).splitlines()[0][:120]}",
                        )
                    )
                    if tried >= max_tries:
                        break
                    continue
                outs = _tensor_outputs(out)
                if len(outs) != len(op.outputs):
                    results.append(
                        (
                            "M",
                            f"{combo}={values}: {len(outs)} outputs vs {len(op.outputs)} recorded",
                        )
                    )
                else:
                    ok, why = True, ""
                    for d, t in zip(op.outputs, outs):
                        ok, why = _same_meta(d, t, assign)
                        if not ok:
                            break
                    results.append(("K" if ok else "M", f"{combo}={values}: {why}"))
                valid = [r for r in results if r[0] != "invalid"]
                if len(valid) >= max_flips:
                    break
            valid = [r for r in results if r[0] != "invalid"]
            if len(valid) >= max_flips or tried >= max_tries:
                break
        valid = [r for r in results if r[0] != "invalid"]
        if len(valid) >= max_flips or (tried >= max_tries and (valid or k == 2)):
            break
        if k == 1 and not valid:
            tried = 0  # the pair search gets its own budget
    valid = [r for r in results if r[0] != "invalid"]
    if not results:
        text = str(g)
        if "2147483647" in text:
            return "K-range", "an int32 index-width guard; no flip within the search"
        if (isinstance(g, sympy.Ne) and g.rhs == 0) or (
            isinstance(g, sympy.Ge) and g.rhs == 0
        ):
            return "D", "a domain guard (A243) the declared domains make true"
        return (
            "R",
            "no flip falsifies the guard under the op's earlier guards (implied or domain-true)",
        )
    if not valid:
        return "V", results[0][1]
    if any(r[0] == "M" for r in valid):
        return "M", next(r[1] for r in valid if r[0] == "M")
    return "K", valid[0][1]


def _address_symbols(tape: ht.Tape) -> set:
    syms = set(tape.shape_env.roots.keys())
    for hb in getattr(
        tape, "host_buffers", ()
    ):  # host tables exist from the h2d commit on
        v = hb.get("root")
        if isinstance(v, ht._SYM_TYPES):
            syms |= set(v.node.expr.free_symbols)
    return syms


def _opaque_symbols(tape: ht.Tape) -> set:
    syms: set = set()
    for o in tape.opaque:
        e = tape.sym_expr(o.get("sym"))
        if isinstance(e, sympy.Basic):
            syms |= set(e.free_symbols)
    return syms


def census(
    tape: ht.Tape,
    *,
    case: str = "",
    trace_s: float = 0.0,
    effects: bool = True,
    max_guards: int = 0,
    sites: list | None = None,
    log: Any = None,
    stride_first: bool = False,
    own_guards_only: bool = False,
) -> Census:
    """The census of `tape`: every raw guard with the tape's attribution and,
    with `effects`, the oracle's class (the meta at concrete metadata on both
    sides of the guard). A flip must keep the op's earlier guards true: every
    guard raised while the op ran before this one, its nested ops' included
    (an allocation's stride rule constrains the same symbols); with
    `own_guards_only` only the op's own rows count, as the scoping census
    (runtime_review/local_miss_scoping/lm_hooks.py) did."""
    ops = op_views(tape)
    rows = guard_rows(tape, sites)
    raw = tape.raw_guards()
    c = Census(
        case,
        tape.symbolic,
        trace_s,
        0.0,
        len(ops),
        sum(1 for o in ops if o.parent == -1),
        len(rows),
        len(tape.guards),
        len(tape.launches),
        len(
            getattr(tape, "regions", ())
        ),  # closed regions exist from the gemm commit on
        len(tape.allocs),
        rows,
        ops,
    )
    if not effects:
        return c
    from torch._subclasses.fake_tensor import FakeTensorMode

    logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
    fm = FakeTensorMode()
    fm.cache_enabled = False
    hints = dict(tape.shape_env.backed_var_to_val)
    addr, opaque = _address_symbols(tape), _opaque_symbols(tape)
    t0 = time.perf_counter()
    for n, r in enumerate(rows):
        if max_guards and n >= max_guards:
            break
        op = ops[r.op] if r.op >= 0 else None
        if op is None:
            earlier = []
        elif own_guards_only:
            earlier = [raw[j] for j in op.guards if j < r.raw]
        else:
            earlier = [raw[j] for j in range(tape.ops[op.index].guard_range[0], r.raw)]
        try:
            r.effect, r.detail = classify_effect(
                raw[r.raw],
                op,
                earlier,
                hints,
                fm,
                addr,
                opaque,
                stride_first=stride_first,
            )
        except Exception as e:  # keep the census going
            r.effect, r.detail = "ERR", f"{type(e).__name__}: {str(e)[:160]}"
        if log is not None and (n + 1) % 250 == 0:
            log(f"  classified {n + 1} in {time.perf_counter() - t0:.0f} s")
    c.effect_s = time.perf_counter() - t0
    return c


# ---------------------------------------------------------------- the report


def _short(name: str | None) -> str:
    return (name or "(python, no op)").replace("aten.", "")


def _pattern(text: str | None) -> str:
    return re.sub(r"s\d+", "s", text or "")


def check(c: Census) -> list[str]:
    """What the oracle refuses: kernel-tagged kept guards whose flip changes
    the op's output metadata (a KernelChoice context that reaches a
    metadata decision). Empty when the tagging is sound on this tape."""
    out = []
    for r in c.danger:
        out.append(
            f"kernel-tagged {r.effect}: `{r.kept_text}` raised by {_short(r.func)} ({r.origin}) :: {r.detail[:120]}"
        )
    return out


def report(c: Census) -> str:
    """The census as markdown: the tables of LOCAL_MISS.md section 1.2 from
    the tape's fields, the phase against the oracle, both directions."""
    rows = c.rows
    kept = [r for r in rows if r.kept is not None]
    L = []
    L.append(f"# Guard census: {c.case} ({c.backend} backend)\n")
    L.append(
        f"trace {c.trace_s:.2f} s; ops dispatched {c.n_ops} (top-level {c.n_top_ops}); raw guards {c.n_raw}; "
        f"kept {c.n_kept} ({len(kept)} mapped); launches {c.launches}; regions {c.regions}; allocs {c.allocs}; "
        f"effect classification {c.effect_s:.1f} s\n"
    )

    def table(key: Any, title: str) -> None:
        ca: collections.Counter = collections.Counter(key(r) for r in rows)
        ck: collections.Counter = collections.Counter(key(r) for r in kept)
        L.append(f"\n## {title} (raw / kept)\n")
        L.append("| class | raw | kept |\n| --- | --- | --- |")
        for k, v in ca.most_common():
            L.append(f"| {k} | {v} | {ck.get(k, 0)} |")

    table(lambda r: r.origin, "By origin")
    effs = [
        k for k, _ in collections.Counter(r.effect or "-" for r in rows).most_common()
    ]
    table(lambda r: r.effect or "-", "By effect")
    L.append("\n## Origin x effect (kept guards)\n")
    L.append("| origin | " + " | ".join(effs) + " | total |")
    L.append("| --- |" + " --- |" * (len(effs) + 1))
    cross: collections.Counter = collections.Counter(
        (r.origin, r.effect or "-") for r in kept
    )
    for o in [k for k, _ in collections.Counter(r.origin for r in kept).most_common()]:
        L.append(
            f"| {o} | "
            + " | ".join(str(cross.get((o, e), 0)) for e in effs)
            + f" | {sum(v for (oo, _), v in cross.items() if oo == o)} |"
        )
    L.append(
        "\n## Phase (the KernelChoice depth at the record) x effect (kept guards)\n"
    )
    L.append("| phase | " + " | ".join(effs) + " | total |")
    L.append("| --- |" + " --- |" * (len(effs) + 1))
    ph: collections.Counter = collections.Counter(
        (r.phase, r.effect or "-") for r in kept
    )
    for p in ("kernel", "meta"):
        L.append(
            f"| {p} | "
            + " | ".join(str(ph.get((p, e), 0)) for e in effs)
            + f" | {sum(v for (pp, _), v in ph.items() if pp == p)} |"
        )
    agree = sum(
        v
        for (p, e), v in ph.items()
        if (p == "kernel" and e in SELECTOR) or (p == "meta" and e in PERSISTED)
    )
    L.append(
        f"\nagree {agree}; kernel-tagged but metadata by the oracle (danger, must be 0) {len(c.danger)}; "
        f"kernel-tagged but validity / Python by the oracle {len(c.mis_tags)}; meta-tagged but kernel-only by the "
        f"oracle (hosts still raising kernel-only guards at depth 0) {len(c.depth0_kernel)}; "
        f"implied / domain rows not compared {sum(v for (_p, e), v in ph.items() if e not in SELECTOR and e not in PERSISTED)}\n"
    )
    L.append("\n## Top ops by kept guards (innermost op)\n")
    L.append("| op | route | kept | " + " | ".join(effs) + " | kernel-phase |")
    L.append("| --- | --- | --- |" + " --- |" * (len(effs) + 1))
    by_op: dict = collections.defaultdict(collections.Counter)
    kphase: collections.Counter = collections.Counter()
    routes: dict = {}
    for r in kept:
        by_op[_short(r.func)][r.effect or "-"] += 1
        routes.setdefault(_short(r.func), r.route or "-")
        if r.phase == "kernel":
            kphase[_short(r.func)] += 1
    for name, cnt in sorted(by_op.items(), key=lambda kv: -sum(kv[1].values()))[:30]:
        L.append(
            f"| {name} | {routes[name]} | {sum(cnt.values())} | "
            + " | ".join(str(cnt.get(e, 0)) for e in effs)
            + f" | {kphase[name]} |"
        )

    def rows_table(title: str, sel: list, with_detail: bool) -> None:
        L.append(f"\n### {title}\n")
        head = "| op | route | origin | pattern | count | site |" + (
            " oracle detail |" if with_detail else ""
        )
        L.append(head)
        L.append(
            "| --- | --- | --- | --- | --- | --- |" + (" --- |" if with_detail else "")
        )
        cnt: collections.Counter = collections.Counter()
        ex: dict = {}
        for r in sel:
            key = (
                _short(r.func),
                r.route or "-",
                r.origin,
                _pattern(r.kept_text),
                r.site or "-",
            )
            cnt[key] += 1
            ex.setdefault(key, r.detail)
        for key, v in cnt.most_common(40):
            line = f"| {key[0]} | {key[1]} | {key[2]} | `{key[3]}` | {v} | {key[4]} |"
            if with_detail:
                line += f" {ex[key][:110]} |"
            L.append(line)

    rows_table(
        "Danger rows: kernel-tagged, the oracle says metadata (M / M-alias)",
        c.danger,
        True,
    )
    rows_table(
        "Kernel-tagged, the oracle says validity or Python (V / P)", c.mis_tags, True
    )
    rows_table(
        "Meta-tagged, the oracle says kernel-only: hosts still raising kernel-only guards at depth 0",
        c.depth0_kernel,
        False,
    )
    L.append("\n## Sample kept guards per (origin, effect)\n")
    seen: collections.Counter = collections.Counter()
    for r in kept:
        key = (r.origin, r.effect)
        if seen[key] >= 3:
            continue
        seen[key] += 1
        L.append(
            f"- [{r.origin} / {r.effect} / {r.phase}] `{r.kept_text}` op {_short(r.func)} route {r.route or '-'} site {r.site or '-'} :: {r.detail[:140]}"
        )
    return "\n".join(L) + "\n"


def to_json(c: Census) -> dict:
    d = asdict(c)
    d["ops"] = [
        {
            "index": o.index,
            "name": o.name,
            "route": o.route,
            "parent": o.parent,
            "depth": o.depth,
            "seq": o.seq,
            "aliases": o.aliases,
            "declined": o.declined,
            "guards": o.guards,
            "inputs": [{k: str(v) for k, v in dd.items()} for dd in o.inputs],
            "outputs": [{k: str(v) for k, v in dd.items()} for dd in o.outputs],
        }
        for o in c.ops
    ]
    d["danger"] = len(c.danger)
    d["mis_tags"] = len(c.mis_tags)
    d["depth0_kernel"] = len(c.depth0_kernel)
    return d


# ---------------------------------------------------------------- the command line


def setup_ln() -> tuple:
    x = torch.randn(32, 1024, device="cuda")
    w = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    fn = lambda x, w, b: torch.nn.functional.layer_norm(x, (1024,), w, b, 1e-5)  # noqa: E731
    return fn, (x, w, b), ()


def _load_setup(spec: str) -> Any:
    path, name = spec.rsplit(":", 1)
    module_spec = importlib.util.spec_from_file_location(
        "host_trace_census_setup", path
    )
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--case", default="ln", help="a name for the report; 'ln' is built in"
    )
    ap.add_argument(
        "--setup",
        default=None,
        help="path/to/module.py:function returning (fn, args, keep)",
    )
    ap.add_argument("--backend", default="sympy", choices=("sympy", "ir"))
    ap.add_argument(
        "--out", default=None, help="directory for <case>_<backend>.md / .json"
    )
    ap.add_argument(
        "--no-effect",
        action="store_true",
        help="the tape's attribution only, no oracle",
    )
    ap.add_argument(
        "--sites",
        action="store_true",
        help="record each guard's Python site (a frame walk)",
    )
    ap.add_argument("--max-guards", type=int, default=0)
    ap.add_argument(
        "--stride-first",
        action="store_true",
        help="try stride / offset symbols before sizes",
    )
    ap.add_argument(
        "--own-guards-only",
        action="store_true",
        help="a flip keeps only the op's own earlier guards (the scoping census's rule)",
    )
    ap.add_argument("--tag", default="", help="a suffix for the output files")
    a = ap.parse_args(argv)
    setup = _load_setup(a.setup) if a.setup else setup_ln
    print(
        "loadavg", os.getloadavg(), "torch", torch.__version__, ht.__file__, flush=True
    )
    fn, args, _keep = setup()
    torch.cuda.synchronize()
    prev = ht.symbolic
    ht.symbolic = a.backend
    try:
        t0 = time.perf_counter()
        with contextlib.ExitStack() as stack:
            sites = stack.enter_context(record_sites()) if a.sites else None
            with torch.no_grad():
                tape = ht.trace(fn, args)
        trace_s = time.perf_counter() - t0
    finally:
        ht.symbolic = prev
    print(
        f"trace {trace_s:.2f} s: ops {len(tape.ops)} raw guards {len(tape.guard_rows)} kept {len(tape.guards)} "
        f"launches {len(tape.launches)} regions {len(getattr(tape, 'regions', ()))} allocs {len(tape.allocs)}",
        flush=True,
    )
    c = census(
        tape,
        case=a.case,
        trace_s=trace_s,
        effects=not a.no_effect,
        max_guards=a.max_guards,
        sites=sites_of(tape, sites) if sites is not None else None,
        log=lambda s: print(s, flush=True),
        stride_first=a.stride_first,
        own_guards_only=a.own_guards_only,
    )
    md = report(c)
    print(md[:6000], flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        base = os.path.join(a.out, f"{a.case}_{c.backend}{a.tag}")
        with open(base + ".md", "w") as f:
            f.write(md)
        with open(base + ".json", "w") as f:
            json.dump(to_json(c), f)
        print("wrote", base + ".md", flush=True)
    failures = check(c) if not a.no_effect else []
    for line in failures:
        print("FAIL", line, flush=True)
    print(
        f"danger {len(c.danger)} mis_tags {len(c.mis_tags)} depth0_kernel {len(c.depth0_kernel)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
