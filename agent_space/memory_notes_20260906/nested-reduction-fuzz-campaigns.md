---
name: nested-reduction-fuzz-campaigns
description: "Randomized topology fuzzer for nested reductions (PR 191974) — harness location, oracle design, and the three benign divergence classes it keeps rediscovering"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8b2c3728-fc6d-4a7a-a0a5-405d8b3ed8ba
  modified: 2026-09-02T05:38:47.318Z
---

2026-09-01. Randomized fuzz harness for nested reductions lives in
`agent_space/pr191974_failure_investigation/`: `tmp_fuzz_gen.py` (v1 topologies)
and `tmp_fuzz_gen2.py` (adds transposed consumers, lane reversal, a second
grouped stage with a different G, chained parents, buffer mutation, backward),
`tmp_fuzz_run2.py` (comparator), `tmp_fuzz_worker2.py`, `tmp_fuzz_driver.py`
(`--seeds 0-199 --modes P,L,LF --jobs 16 --worker <abs path>`).

**Oracle design that made results trustworthy:** the *unnested* compile is the
oracle, since that is the only thing the flip changes. Eager is a third opinion,
and recompiling the same config is the self-noise control. Without the noise
control, autotune variation reads as a correctness bug.

**Three benign divergence classes** — check these before believing a numeric
flag is a nested bug:
1. Autotune picks a different launch config, so an identical `tl.sum` returns a
   different last ulp. Per-compile, not per-invocation; the compiled artifact is
   bit-stable.
2. Folded cast pairs forward fp32 inside one kernel. `emulate_precision_casts`
   removes this one.
3. Fusing across a *realized* low-precision buffer keeps the pre-cast value live
   instead of re-reading the store. This one **survives**
   `emulate_precision_casts`. Signature: fp32 reduction outputs bit-identical,
   1-2 of ~131072 low-precision elements differ by a few ulps.

Classes 2 and 3 both reproduce with `triton.nested_reduction` off by comparing a
fused against an unfused schedule, so neither is nested-specific, and in both the
fused kernel carries *more* precision.

Beware: a hand-written repro of a fuzz case usually does not reproduce, because
returning an extra intermediate changes the fusion. Reproduce by importing
`build(seed, B, D)` from the generator instead.

See [[nested-reduction-stack]] and [[nested-reduction-temporal-battery]].
