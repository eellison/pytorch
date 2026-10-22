"""PROTOTYPE (learnfast, 10-15; coordinator): fast refusal of a redispatch that would rebuild a section check_section refused.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import refusal_memo

_add_entries runs the whole redispatch (every failing selector redone or instantiated), then checks each section entry; a
refused section refuses the attempt. Each refused entry is recorded as (site, its predicate row, the refusal) per lowered
variant. The predicate is the conjunction of the op's own guards from that redo (the rows stay in the variant's program: the
refused program was set before the check). A later redispatch of that variant whose call fails the site's selector and holds
the predicate would redo the op down the same branch, rebuild the same section and refuse it again: it is refused at once,
with the same message. Exact by construction (an earlier variant memo keyed on (variant, replaced nodes) failed equality);
checked by trace count and refusal set with the memo on / off. STATS counts recorded / fast refusals."""
import collections
import sys
import weakref

import torch.cuda._host_trace_replay as _hr
from torch.cuda._host_trace_lower_tape import FoldRefused

STATS: collections.Counter = collections.Counter()
_MEMO: dict = {}  # id(lowered) -> (weakref(lowered), {(site, predicate row): message})


def _wrap(inner):
    def check_section(lowered, plan, nodes, launches, temporaries, drops):
        why = inner(lowered, plan, nodes, launches, temporaries, drops)
        if why is not None:
            f = sys._getframe(1)
            if f.f_code.co_name == "_add_entries":
                loc = f.f_locals
                site, predicate, variant = loc["site"], loc["_"], loc["variant"]
                op = lowered.selectors[site - len(lowered.sites)].op
                ent = _MEMO.get(id(lowered))
                if ent is None or ent[0]() is not lowered:
                    ent = _MEMO[id(lowered)] = (weakref.ref(lowered), {})
                ent[1][(site, predicate)] = f"{variant.tape.ops[op].func}'s section: {why}"
                STATS["recorded"] += 1
        return why

    check_section._refusal_memo = True
    return check_section


def _ensure_outermost():
    # integrations wrap check_section after import (no_sections, diag27's resolve): see the final verdict
    if not getattr(_hr.check_section, "_refusal_memo", False):
        _hr.check_section = _wrap(_hr.check_section)


if not getattr(_hr.HostTraceReplay._redispatch, "_refusal_memo", False):
    _inner = _hr.HostTraceReplay._redispatch

    def _redispatch(self, variant, args, values, unselected, dispatches=None):
        _ensure_outermost()
        lowered = variant.captured.lowered
        ent = _MEMO.get(id(lowered))
        if ent is not None and ent[0]() is lowered:
            failing = set(unselected)
            for (site, predicate), msg in ent[1].items():
                if site in failing and predicate < len(values) and values[predicate] == 1:
                    STATS["fast_refused"] += 1
                    e = FoldRefused(msg)
                    e.op = lowered.selectors[site - len(lowered.sites)].op
                    self.redispatch_refusals[msg] = self.redispatch_refusals.get(msg, 0) + 1
                    self._last_refusal[id(variant.native)] = e.op, msg
                    return e
        return _inner(self, variant, args, values, unselected, dispatches)

    _redispatch._refusal_memo = True
    _hr.HostTraceReplay._redispatch = _redispatch
