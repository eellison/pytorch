# Integration helpers after step_ms34 (core agent, 10-20)

The adapters (core/r27/armV_r27/adapter.py, scratch/integration/vllm/armV_stock_fast_m4/adapter.py) now pass
HostTraceReplay(redo_policy="refuse") where the core has the option (step_ms34+) and import learnfast's no_lib_redo only on older
cores (CORE_REDO_POLICY; backups *.bak_redopolicy). res["lib_redo_refused"] counts per op either way. ported.py (the
cold-start measurement overlay) switches a replay of the r27 hybrid step to "inputs_only" with lifted / redo_context /
close_over at its first slow call.

## Retired on step_ms34+ (core covers them)
- no_lib_redo.py: core redo_policy="refuse" (kept for older cores; its wrapper forwards extra arguments).
- prov_redo.py (DIAG27_PROV, PROV_LIFT, PROV_RELAX_GUARDS, PROV_CLOSURE): core redo_policy="inputs_only" (static scalar-read
  check + confirmation, sound per traced op, shared within a variant) with lifted / redo_context / close_over. prov_redo
  confirmed per operator name once per process, which is not sound (the ms33 template hole).
- redo_ctx.py / DIAG27_REDOCTX / DIAG27_LIFT: core redo_context (RedoCall(args, func, rank), rank tracked).
- DIAG27_CLOSURE / DIAG27_CLOSURE_STORAGE: core close_over (declared objects only; storage mode, which took any view of a
  static-prefix storage at a constant layout unguarded, is not ported).
- DIAG27_FNRESOLVE / DIAG27_CUTE_RESOLVE: core RD5 identity at the torch-function layer, Triton, record_launch, CuTe.
- section_memo.py: core _Variant.refused_sections (refuses again at its site and predicate; the memo refused per selector).
- xvr.py: landed as step_ms21 (dispatch key by extent class).

## Remaining, and why
- no_sections.py: the adapters' interim section refusal (DIAG27_SECTIONS_ON lifts it for the cold-start arms); a separate
  decision (sections in the timed arms), not a redo matter.
- tpl_layer.py / tpl_closure.py (DIAG27_TPL, cross-layer key-out): held for the user (vtn vs vt: ~45 s of 27B warm-up).
- fla_symlen.py: labelled integration overlay (flash-linear-attention's len() on a symbolic size), per the sheet.
- gdn_maskfill.py: labelled overlay, diagnostic; it does not change the 16 GDN refusals.
- ARMV_PIN_UNPINNED (X1): held for the user.
- coldwrap.py / diag27.py: measurement harnesses (stamps, instrumentation), not core behavior.
