# Redispatch overhead profile (learnfast, 10-15)

Run: 27B Qwen3.8-NVFP4, ours V (eager forward), step_ms10 + Delta 3 (learnfast/sec_snap), lifted state + fla overlay +
cross-layer templates (tpl_layer, per-(key, layer) confirmation) + refusal memo + sections on, vt RANGE=1 warm-up + 50
stream steps. GPU 3, untimed. cProfile restricted to HostTraceReplay._redispatch (diag27 DIAG27_PROFILE_REDISPATCH; the
memo's fast refusals are outside it). pstats: learnfast/out/prof_redispatch.pstats; log: logs/rw_vtprof1.log.

Total profiled: 370 s over 55 successful redispatches and ~70 attempts (cProfile inflates Python-heavy code; the
uninstrumented phases' redispatch time on the same workload was 332 s). Call counts are exact.

## Where the time goes (cumulative seconds; categories overlap where noted)

| category | cum s | calls | notes |
|---|---|---|---|
| op redo bodies (eager_call -> _traced_host) | 180 | 1,794 redos | includes the confirmations' full redos |
|   GDN body (qwen_gdn_attention_core_fused_norm_packed) | 88 | 487 | 180 ms per redo |
|   mm_fp4 (flashinfer, CuTe DSL intercept) | 60 | 1,201 | CuTe _intercept 37.7 s, host_trace.run 46.6 s |
|   Triton launch intercepts | 14 | 4,149 | |
|   prototype torch-function resolve (diag27 FNRESOLVE: tree_map_only per torch call) | ~47 | 383k calls | prototype only; core's resolver would not pay this |
| core _for_instance (template specialization per instance) | 87 | 10,037 | tottime 13.9 s |
|   `addresses = frozenset(lo.key(rec.root.sym) for rec in old.inputs)` (redispatch.py:349) | 18.3 | 19.1M genexpr steps | recomputed per instance over every input (~1,900): cache per lowering |
|   IR transfer (guard renaming) | 17.9 | 3.27M | |
|   lowering conjunction / lowering.total | 17.2 / 11.4 | 11.6k / 33.6k | |
| redo stand-ins resolver (diag27 _resolver) | 30 | 1,794 | prototype; tottime 7.6 s |
| lower_entry | 22.4 | 1,790 | |
| opdeps.classify (prov_redo, once per variant) | 18.3 | 5 | prototype; 3.7 s per new variant |
| check_section (Delta 3) | 9.7 | 6,560 | every section entry of every attempt |
| guard_sloc | 11.2 | 172k | guards recorded in redos |
| _template / _reads | 6.0 / 3.2 | 725 / 1,572 | |
| compile_program | 3.0 | 56 | |
| _native_mutate | 0.3 | 10.3k | |

Hot primitives (tottime): builtins.id 13.2 s (31.8M), dict.get 12.1 s (30.6M), _sym_key 11.8 s (26.1M), isinstance 9.3 s
(106M), IR free_symbols 9.0 s (2.9M), pytree unflatten / helper 12.6 s (prototype resolve mode).

## Reading
- Core-side, generic (MiMo / 8B pay it too): _for_instance (87 s) is the largest non-redo cost. Its frozenset of every
  input's root key per instance is pure waste (18 s: cache it in lo per lowering); transfer + conjunction per instance
  (35 s) could be done once per (template, renamed-symbol map) shape. lower_entry 22 s, guard_sloc 11 s.
- Redo bodies (180 s) are the work itself; GDN's 180 ms per redo and mm_fp4's 50 ms (CuTe intercept) are the per-op costs.
  Fewer redos come from template sharing (done: GDN keys 48 -> 3 per class) and fewer confirmations (lever i).
- Prototype-only: FNRESOLVE's torch-function tree_map (~47 s), diag27 resolver (30 s), opdeps classify (18 s): these go
  away or shrink in a core port (resolution at the root index; classification once per tape).

## After the core agent's cuts (wip14 = step_ms12 + core memo + symbolic masked setitem + per-attempt cuts), build_cpp_ms9
Same harness, GPU 0 untimed. pstats: out/prof_redispatch_wip14.pstats (before: out/prof_redispatch_step_ms12.pstats).

| function | step_ms12 cum s | wip14 cum s | calls (wip14) |
|---|---|---|---|
| core _for_instance | 90.9 | 35.0 | 10,037 |
| _host_trace_lower total() | 14.0 | 0.1 | |
| lowering conjunction | 17.1 | 17.0 | 11,575 (_guard_rows 12.2 s of it; 742k genexpr steps) |
| lower_entry | 24.4 | 15.5 | 1,790 |
| check_section (incl. the memo wrapper) | 10.0 | 10.9-12.8 | 6,560 |
| guard_sloc | 11.2 | 11.1 | 170,690 (tottime 10.3 s) |
| IR _compare / _compare_new | | 21.9 / 20.3 | 1.6M / 382k |
| IR free_symbols | 20.4 | 17.4 | 2.9M |
| IR transfer | 17.9 | 15.6 | 3.27M |
| redo bodies (eager_call) | 162.3 | 215.6 | 10,240 (GDN 487 redos: 160 -> 269 ms each; see the GC 3-way) |

Remainder for the core agent (per attempt, after ms14):
- conjunction / _guard_rows: 17 s over 11.6k calls (1.5 ms each), the generator over every guard row per conjunction.
- lower_entry: 15.5 s over 1,790 entries (8.7 ms each).
- guard_sloc: 11 s for 171k guards recorded in redos (65 us each, all tottime: the frame walk per guard). Could be skipped
  inside redos whose guards only feed the entry predicate.
- IR comparison / relation building during redos (_compare, _rel, sub, add): about 20 s.
- Confirmations (learnfast prototype, one full redo per (key, layer)): 22 s in the rs5 warm-up; template-port question.
