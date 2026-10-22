"""Shared helper (learnfast, 2026-10-13): refuse redispatch of library-body ops until the core has a redo_context.

    import sys; sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast"); import no_lib_redo

A redo runs the op's Python body outside the integration's state swap: a body that reads per-step Python ints from
module or forward-context state (vLLM's attention metadata) bakes the current call's values into the entry with no
guard (DESIGN_section_sizing.md: GDN section temporaries sized with the literal token count; a same-chain entry would
bake launch values the same way, silently). Until HostTraceReplay takes an integration's redo_context, this refuses,
before the redo runs, the redispatch of:
  - an operator outside the aten namespace (vllm.*, flashinfer.*, _C.*, any custom op: its Python body runs in the redo);
  - a dispatch unit (a Python function run as one op: its body runs in the redo).
ATen operators, Triton / CuTe DSL launches and jiterator ops redo from their recorded arguments and stay allowed. The
refusal is redispatch's own FoldRefused path: the call takes its other routes (another variant, or a trace, counted in
retrace_causes). REFUSED counts refusals by op. REDO_CONTEXT is the hook a future core API fills (None: refuse)."""
import collections

import torch
import torch.cuda._host_trace_redispatch as _rd
from torch._ops import OpOverload
from torch.cuda._host_trace_lower_tape import FoldRefused

REFUSED: collections.Counter = collections.Counter()
REDO_CONTEXT = None
_WHY = "redo of a library-body op refused (no redo_context: per-step state would be baked)"


def library_body(func) -> bool:
    if isinstance(func, OpOverload):
        return func.namespace != "aten"
    if isinstance(func, str):  # a jiterator op's label
        return False
    mod = type(func).__module__ or ""
    if mod.startswith(("triton", "cutlass", "torch.cuda._host_trace")):  # a Triton / CuTe DSL launch
        return False
    return callable(func)  # a dispatch unit: a Python function


if not getattr(_rd._full_redo, "_no_lib_redo", False):
    _orig = _rd._full_redo

    def _full_redo(lowered, i, selector, dispatches, opaque=(), *rest, **kwargs):  # rest: a later core's arguments (ms34's redo)
        func = lowered.tape.ops[selector.op].func
        if REDO_CONTEXT is None and library_body(func):
            REFUSED[str(func)] += 1
            e = FoldRefused(f"{func} dispatches otherwise; {_WHY}")
            e.op = selector.op
            raise e
        return _orig(lowered, i, selector, dispatches, opaque, *rest, **kwargs)

    _full_redo._no_lib_redo = True
    _rd._full_redo = _full_redo
