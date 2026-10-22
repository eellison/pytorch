"""Shared helper (learnfast, 2026-10-13; the trtllm_moe_cpp lane's mechanism, packaged): refuse RD2 section entries where
they are created, until the core's section-sizing bug is fixed (a section made at one size and replayed at a larger one
reads past its temporaries; memcheck: FLA chunk_fwd_kernel_o on Qwen3.5-9B GDN).

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import no_sections

_host_trace_replay._add_entries checks every section entry with check_section before adding it; a returned reason is
the core's own refusal path (the entry is not added, the redispatch refuses, the call takes its other routes: another
variant, or a trace, counted in retrace_causes). This wraps that check to refuse every section the core would accept.
It leaves torch._C._host_trace_section_entries as the build sets it (True): flipping that attribute desynchronizes
_form's site list from the native variant's arms and crashes at the first new form. Same-chain entries (no section) are
unaffected. REFUSED["sections"] counts refusals; SECTIONS keeps each refused section's replaced launches, its own and
its temporaries' byte rows at creation (for the RD2 investigation)."""
import torch.cuda._host_trace_replay as _hr

REFUSED = {"sections": 0}
SECTIONS: list = []
WHY = "section entries disabled (RD2 sizing bug)"

if not getattr(_hr.check_section, "_no_sections", False):
    _orig = _hr.check_section

    def check_section(lowered, plan, nodes, launches, temporaries, drops):
        why = _orig(lowered, plan, nodes, launches, temporaries, drops)
        if why is None:
            why = WHY
            REFUSED["sections"] += 1
            values = lowered.lowering.program.values
            name = lambda lo: getattr(getattr(lo, "launch", None), "name", type(lo).__name__)[:100]
            SECTIONS.append({"replaced": [name(lowered.launches[n]) for n in nodes][:8], "launches": [name(lo) for lo in launches][:8],
                             "temporaries": [(b, r, values[r] if r < len(values) else None) for b, r in temporaries][:16]})
        return why

    check_section._no_sections = True
    _hr.check_section = check_section
