---
name: nested-reduction-temporal-battery
description: "2026-09-01 temporal/ordering battery for nested R-grouped reduction fusion (worktree pr191974_simplified_wt @ 61b333901fa); harness location, protocol, and the two loud compile crashes it found"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8b2c3728-fc6d-4a7a-a0a5-405d8b3ed8ba
  modified: 2026-09-01T23:49:17.557Z
---

Battery lives in `agent_space/pr191974_failure_investigation/tmp_temporal_{common,cases,worker,driver,table}.py`
(git-ignored). Driver spawns one process per (case, mode) round-robin over 8 GPUs; modes P (D=512),
L (D=16384 looped), LF (D=512 + triton.persistent_reductions=False). Protocol: nested compile first in a
fresh process, two calls (second inputs x3 scale), eager oracle + unnested-compile byte/numeric oracle.
Results JSON: `tmp_temporal_results*.json`; render with `tmp_temporal_table.py`.

Findings as of 2026-09-01 (commit 61b333901fa, nested_reduction default ON):
- Zero silent wrong results across ~600 (case, mode) runs; every "second outer reduction reads a
  displaced LOCAL_REDUCTION_INPUT node" topology falls back (ancestor check incl. WeakDeps works).
- LOUD: dynamic batch (mark_dynamic or automatic dynamic after a 2nd batch size) + block-reduce input
  materialized (graph output) -> "nested reduction plan was lost before codegen". Mechanism: codegen
  re-plan after merge_loops sees one merged iteration range; `_r_grouped_stage_accesses_match` gets
  `ModularIndexing(v, 1, 32*s77)` that only simplifies for static extents. Needs
  loop_ordering_after_fusion=True (default). Repro: `tmp_dyn_basic_repro.py rms_y_out 32 1024 auto defaults`.
- LOUD: looped parent (D>=2048 fp32 or persistent_reductions=False) + any materialized prologue before
  rmsnorm + sub-parent epilogue -> "lost required sub-parent source". Persistent parent fine.
  Repro: `tmp_subparent_lost_repro.py prologue_out_norealize 4096 1 defaults`.
- Env: eager inline_asm e2m1x2 JIT fails on sm_100 (needs sm_100a) -> use unnested compile as oracle.
- bf16/fp16 graphs: nested vs unnested differ (1 bf16 ulp; ~0.7% of nvfp4 nibbles) unless
  emulate_precision_casts=True, under which they are byte-identical. Cast-pair folding, not ordering.
  Always compare low-precision nested results against the unnested compile, not eager.

**Why:** the earlier 94-case battery had no multi-reduction topologies; this one is built around them.
**How to apply:** rerun `tmp_temporal_driver.py --filter <prefix>` after fixes; add cases to
`tmp_temporal_cases.py` with the `@case` decorator. See [[nested-reduction-stack]].
