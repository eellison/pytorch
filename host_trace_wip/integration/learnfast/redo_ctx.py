"""RETIRED on step_ms34+ (core agent 10-20; memset/notes/INTEGRATION_RETIREMENT.md): the core covers it. Kept for older cores.

PROTOTYPE (learnfast, 2026-10-14; DESIGN_section_sizing.md (A)): a redispatch's redo runs inside the adapter's own
step-function state swap, so an op body that reads per-step state (vLLM's forward context: attention metadata, slot
mapping; module attributes; parameters) sees the redo's symbolic stand-ins, as a trace does, instead of the current
call's concrete values. Diagnostic / measurement use only -- not for timed or reported runs until reviewed.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import redo_ctx

Mechanism (existing APIs only):
  - torch.cuda._host_trace.redo_resolves_arguments = True, so _redo installs its resolver hook; this module's resolver
    builds the redo's stand-in for every argument of the step with _redo's own closures (a tensor argument: its root at
    its input record's layout, fresh symbols bound to the variant's; an int argument: a fresh SymInt bound to the
    variant's int input; anything else: the traced value), and wraps that one redo: the entry's step function (the
    adapter's `step(*flat)`) is called on the stand-ins with its proxy's `orig_forward` replaced by the op's redo, so
    every swap the step makes before its forward is live during the redo and undone by its own finally / with blocks.
    The resolver itself resolves nothing (identity).
  - no_lib_redo sees a declared context (REDO_CONTEXT set); no_sections' refusal is lifted (check_section is core's).
  - _host_trace_cute._roots_of also searches a redo's outer roots (rule 3 of DESIGN_redo_resolve.md).
  - OPSPAN (opt-in, REDO_CTX_OPSPAN=1): plan_memory keeps the allocations an op-guarded op reaches live through its
    last launch, so check_section accepts sections (memory cost: +42% plan peak at 27B T=8192; the design replaces it
    with an exact overlap check).
STATS counts contextual redos and failures (a failure leaves the redo without a context: no_lib_redo's refusal does
not apply then, so a failure is logged loudly)."""
import os
import sys

import torch
import torch.cuda._host_trace  # noqa: F401
import torch.cuda._host_trace_cute as _cu
import torch.cuda._host_trace_redispatch as _rd
import torch.cuda._host_trace_replay as _hr
import torch.cuda._host_trace_tape as _tt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_lib_redo as _nlr  # noqa: E402
import no_sections as _ns  # noqa: E402

STATS = {"contexts": 0, "failures": 0, "last_failure": None}
_ENTRY = {"fn": None}

torch.cuda._host_trace.redo_resolves_arguments = True
_nlr.REDO_CONTEXT = "learnfast redo_ctx prototype"
_hr.check_section = _ns._orig

_orig_redispatch = _hr.HostTraceReplay._redispatch


def _redispatch(self, *a, **k):
    _ENTRY["fn"] = self.fn
    return _orig_redispatch(self, *a, **k)


_hr.HostTraceReplay._redispatch = _redispatch


class _RedoDone(Exception):
    pass


def _resolver(old, lo, tr, root, made, at_call, inputs, resolved):
    step = _ENTRY["fn"]
    frame = sys._getframe(1)
    while frame is not None and "op" not in frame.f_locals:  # _redo's frame (wrappers in between)
        frame = frame.f_back
    op = frame.f_locals.get("op")
    try:
        cells = {n: c.cell_contents for n, c in zip(root.__code__.co_freevars, root.__closure__ or ())}
        value = cells["value"]
        proxy = {n: c.cell_contents for n, c in zip(step.__code__.co_freevars, step.__closure__ or ())}["proxy"]
        n0 = len(made)
        flat = list(old.args)
        with lo.symbolic.total():
            for rec in old.inputs:
                r = root(rec.root)
                rec2 = next(i for i in inputs if i.root is r)
                tr.arguments[id(r)] = rec2
                x = old.args[rec.position]
                dtype = x.dtype if isinstance(x, torch.Tensor) else x[3]
                flat[rec.position] = _tt._TracedTensor(r, list(rec2.sizes), list(rec2.strides), rec2.offset, dtype, old.device)
            for irec in old.int_inputs:
                flat[irec.position] = value(irec.sym)
        for m, x in zip(made[n0:], at_call[n0:]):
            tr.values.bind(m, x)
        stand_ins = tuple(flat)
        orig_redo = op.redo

        def redo_in_ctx(*a, **k):
            op.redo = orig_redo  # this redo's stand-ins only
            held, orig_fwd = {}, proxy.orig_forward

            def forward_is_the_redo(*_x, **_kw):
                held["r"] = orig_redo(*a, **k)
                raise _RedoDone

            proxy.orig_forward = forward_is_the_redo
            try:
                step(*stand_ins)
            except _RedoDone:
                pass
            finally:
                proxy.orig_forward = orig_fwd
            if "r" not in held:
                raise RuntimeError("redo_ctx: the step function returned without calling its forward")
            return held["r"]

        op.redo = redo_in_ctx
        STATS["contexts"] += 1
    except Exception as e:  # loud: this redo runs without a context
        STATS["failures"] += 1
        STATS["last_failure"] = repr(e)[:300]
        print("redo_ctx: no context for this redo:", repr(e)[:300], flush=True)
    return lambda t: t


_rd._resolver = _resolver


def _roots_of(tr, v):
    key = _tt._sym_key(v)
    held = set() if isinstance(key, int) else key.free_symbols
    roots = (*(i.root for i in tr.inputs), *(a.root for a in tr.allocs), *(t._root for t in tr.eager_outputs),
             *(t._root for t in getattr(tr, "outer", {}).values()))
    seen, out = set(), []
    for r in roots:
        if id(r) not in seen and _tt._sym_key(r.sym).free_symbols & held:
            seen.add(id(r))
            out.append(r)
    return out


_cu._roots_of = _roots_of

if os.environ.get("REDO_CTX_OPSPAN") == "1":
    import inspect
    import textwrap

    import torch.cuda._host_trace_memory as _mem

    _src = textwrap.dedent(inspect.getsource(_mem.plan_memory))
    _anchor = "    # a keyed site's operands reach its nodes through its table\n"
    if _anchor not in _src:
        raise RuntimeError("redo_ctx OPSPAN: plan_memory anchor not found")
    _ins = """    for sel in lowered.selectors:
        if sel.nodes:
            end = lowered.launches[sel.nodes[-1]].seq
            for j in sel.nodes:
                lo = lowered.launches[j]
                refs = [s.root for s in lo.slots if isinstance(s, PointerSlot)] + [lowered.lowering.roots[id(r)] for r in lo.launch.roots]
                for kind, k in refs:
                    if kind == "allocation" and k < n_alloc:
                        last_seq[k] = max(last_seq[k], end)
"""
    _ns_ = {}
    exec(compile("from __future__ import annotations\n" + _src.replace(_anchor, _ins + _anchor), _mem.__file__ + ":opspan", "exec"), _mem.__dict__, _ns_)
    _mem.plan_memory = _ns_["plan_memory"]
    _hr.plan_memory = _ns_["plan_memory"]
