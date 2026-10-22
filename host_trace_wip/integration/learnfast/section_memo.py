"""RETIRED on step_ms34+ (core agent 10-20; memset/notes/INTEGRATION_RETIREMENT.md): the core covers it. Kept for older cores.

PROTOTYPE (learnfast, 10-14): memoize check_section refusals per selector. A selector whose section check_section
refused once (same lowered variant, same replaced nodes) fails fast at later redispatches, before any redo: refusing
more, never accepting more (a pure optimization if the refusal set is unchanged; equality checked by a memo-off run).

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import section_memo
"""
import collections
import weakref

import torch.cuda._host_trace_redispatch as _rd
import torch.cuda._host_trace_replay as _hr
from torch.cuda._host_trace_lower_tape import FoldRefused

STATS: collections.Counter = collections.Counter()
_REFUSED: dict = {}  # id(lowered) -> (weakref(lowered), set of replaced-node tuples)

def _wrap(inner):
    def check_section(lowered, plan, nodes, launches, temporaries, drops):
        why = inner(lowered, plan, nodes, launches, temporaries, drops)
        if why is not None:
            ent = _REFUSED.get(id(lowered))
            if ent is None or ent[0]() is not lowered:
                ent = _REFUSED[id(lowered)] = (weakref.ref(lowered), set())
            ent[1].add(tuple(nodes))
            STATS["recorded"] += 1
        return why

    check_section._section_memo = True
    return check_section


def _ensure_outermost():
    # an integration may wrap check_section after this import (the MiMo adapter's section audit / refusal): stay outermost
    if not getattr(_hr.check_section, "_section_memo", False):
        _hr.check_section = _wrap(_hr.check_section)


_ensure_outermost()

if not getattr(_rd._full_redo, "_section_memo", False):
    _inner = _rd._full_redo

    def _full_redo(lowered, i, selector, dispatches, opaque=(), *rest, **kwargs):  # rest: a later core's arguments (ms34's redo)
        _ensure_outermost()
        ent = _REFUSED.get(id(lowered))
        if ent is not None and ent[0]() is lowered and tuple(selector.nodes) in ent[1]:
            STATS["fast_refused"] += 1
            e = FoldRefused(f"{lowered.tape.ops[selector.op].func}'s section: refused before (section memo)")
            e.op = selector.op
            raise e
        return _inner(lowered, i, selector, dispatches, opaque, *rest, **kwargs)

    _full_redo._section_memo = True
    _rd._full_redo = _full_redo
