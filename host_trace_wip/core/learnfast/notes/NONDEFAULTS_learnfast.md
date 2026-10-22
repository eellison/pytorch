# Ours-only deviations in learnfast cold-start / sizing runs (labelled; none in any timed perf set)
- fla_symlen.py (overlay): flash_linear_attention chunked-prefill ops use .shape[0] for len(); same values, removes the
  prefill-length concretization; ours only; not used in any timed set.
- gdn_maskfill.py (overlay, coordinator-approved 10-15): Qwen GDN's prefill initial-state reset as masked_fill_ instead of
  boolean-mask setitem; same values, removes the sequence-count concretization; ours only; not used in any timed set.
  Dropped once the core records boolean-mask setitem with a scalar value symbolically.
- ARMV_PIN_UNPINNED=1 (r27 adapter knob): GDN nums_dict "nums" passed as a pinned host copy (~5.5 us/step host copy stock
  vLLM does not make) so its shape no longer keys the entry form; sizing runs only, not in any timed set; replaced by the core
  argument kind for an unread unpinned CPU tensor (QUESTIONS X1) if approved.
- ARMV_MEMORY=eager in every sections-on run (startup27.sh default with SECTIONS=1): the adapter default run_buffer holds every
  section's temporaries for the whole run (+107 GiB at 16K tokens after a decode phase; notes/MEMORY_DECODE_THEN_PREFILL.md).
  Eager order makes and frees them at their op (live peak 92.45 vs vLLM eager 92.54 GiB). Until the core places section
  temporaries in the run buffer by lifetime.
- xvr.py (PROTOTYPE, 10-17, diag27 DIAG27_XVR=1): cross-variant reuse of redo results. A textually patched copy of step_ms16's
  _redo (exact-match asserts) whose dispatch cache (core's, per contract) keys arguments by what the trace reads of their extent
  (dtype, length, empty, cow + the inputs' overlap relation) instead of the absolute trace-time bytes, and also stores redos whose
  body reached state outside the call (the prototypes' resolver stand-ins), recorded as variant-independent events (argument
  position; the same object at a static position) and replayed before the guard check. Ours only, cold-start sizing runs only,
  bitwise-checked (n27wf --check a / b) before any number; not in any timed perf set.
- tpl_layer TPL_DECLVAL=1 (PROTOTYPE, 10-18): a template key item mixing static-prefix layout symbols with per-call symbols
  (GDN prefill allocations 128*s26*T, s26 a layer's own weight dimension) is keyed with the static-prefix symbols at their
  values, so the layers of a KV group share one GDN prefill template; each (key, layer)'s first instance is confirmed by a full
  redo (tpl_layer's confirmation). Ours only, sizing runs only, bitwise-checked before any number; core fix proposed to the core agent.
