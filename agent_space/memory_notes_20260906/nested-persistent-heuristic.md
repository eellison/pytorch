---
name: nested-persistent-heuristic
description: PARKED 2026-09-06 (user: "heuristics later"): nested kernels should be persistent up to 16384 wide with 1-4 warps; edits sit uncommitted in agent_space/persist_pr; must land after the lane fold or NVFP4 regresses
metadata:
  type: project
---

Default-tuning sweeps 2026-09-06 (B200, RMSNorm->quant, ~64MB inputs):

- Persistent beats looped at every width 2048-16384 for MXFP8 and NVFP4
  (e.g. 4096: 30.3 -> 21.5 / 24.6 -> 21.5; 2048: 38.9 -> 28.7 / 31.5 -> 25.4).
  Today's threshold is 1024 (INNER), so users get looped.
- Warp count: best is ~64-128 elements per thread -> 1-2 warps at 2048/4096,
  4 at 8192; the `r // 128` formula gives 8-16. Generated-kernel sweep:
  K=2048 nw1 18.8 vs default 29.4; K=4096 nw1 18.9 / nw2 20.1 vs 22.1;
  K=8192 nw4 19.4 vs 21.7 (nw1 spills at 8192).
- Persistent INNER reductions > 1024 wide get exactly ONE config candidate
  (`configs = configs[:1]` in get_persistent_configs); only cd explores warps.

Implemented but UNCOMMITTED in worktree `agent_space/persist_pr` (branch
`nested-reduction-persistent-heuristic`, origin/main 071dd4d98ee):
`SIMDKernelFeatures(nested_reduction=...)` flag set at both nested codegen
sites; `should_use_persistent_reduction` threshold = max(threshold, 16384)
when nested; `inductor_meta["nested_reduction"]`; `get_persistent_configs`
offers warp candidates {r//4096, r//2048, r//1024} clamped [1,16] with
min_num_warps=1 for nested INNER > 1024. Measured default (overlay, no fold):
MXFP8 8192x4096 32.6 -> 18.6us (beats cd 21.2), 16384x2048 -> 18.7,
4096x8192 -> 22.0. NVFP4 on that tree gets WORSE (28.1 vs 24.6) because the
persistent NVFP4 form is slow without the lane fold -- **land after
[[sub-parent-lane-fold]]**, then re-measure. No suites run yet; no test yet.

**Why parked:** user said "we can do heuristics later" mid-implementation.
**How to apply:** rebase persist_pr onto the lane-fold branch, add a test
(nested rmsnorm->quant at K=4096 emits `persistent_reduction`), run nested +
loop_ordering + torchinductor, then peak matrix default mode.
