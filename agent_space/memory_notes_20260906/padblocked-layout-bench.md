---
name: padblocked-layout-bench
description: agent_space/padblocked/bench.py -- workload x impl matrix for padded-blocked-scale layouts; settles the #195930 prim-vs-ATen question
metadata:
  type: project
---

`agent_space/padblocked/bench.py` is the swap-in bench for padded-blocked-scale
layout implementations (PR #195930's `to_padded_blocked`). Two registries:
WORKLOADS (real quantize tails: `dcn_mxfp6`, `dcn_mxfp6_native`,
`rmsnorm_mxfp8`, `rmsnorm_nvfp4`, `scale_only`) x IMPLS (`prim`, `aten`,
`scatter`, `xdl_zeros`). Scores correctness, fused kernel count, CUDA-graph
us, and guards under a dynamic compile. One subprocess per cell.

Measured 2026-09-03 on `dcn_mxfp6_native` (2048x3072, the production kernel):

| impl | kernels | us | dynamic compile |
|---|---|---|---|
| prim | 2 | 16.32 | 1 compile, 0 guards |
| aten (pad/reshape/permute) | 2 | 18.38 | 5 guards, ConstraintViolationError |
| scatter (full + index_put) | 3 | 18.57 | 0 guards |
| xdl_zeros (zeros + _unsafe_index_put) | 4 | 32.09 | 0 guards |

2026-09-03 follow-on, RESOLVED: the prim's win was an artifact of a bail, now
fixed -- see [[sub-parent-mutation-hoisting-fix]]. Post-fix on
`dcn_mxfp6_native` the scatter forms *tie* the prim (prim 16.27, scatter 16.34,
xdl_zeros 16.58, all 2 kernels), and on `scale_only` the scatter wins
(12.27 vs 14.31). So the fix removes the prim's fusion advantage; it does not
make the best DCN number faster than it already was. Do not claim the scatter
"dominates on every axis" -- on the production tail it is a tie, and the prim
still wins the dynamic-shape axis outright (1 compile / 0 guards vs ATen's 5
guards + ConstraintViolationError).

Structural point that still holds: the prim iterates the *destination* (padded
extent 270336) and gathers, so its iteration space can never match a producer's
logical extent (196608) -- unfusable by construction, always a second pass. The
scatter iterates the *source* and emits an indexed store, which fuses into the
reduction once the bail is gone.

**Why:** two traps burned a lot of time before this existed. (1) A producer
that returns the scale alone cannot see what a realizing layout costs -- on
`scale_only` the scatter is the *fastest* impl. The payload epilogue is the
whole point. (2) "The scatter" is not one thing: `zeros` +
`_unsafe_index_put` is 4 kernels/32us but `full` + `index_put` is 3
kernels/18.6us. Naming which scatter you mean is mandatory.

**How to apply:** run `python agent_space/padblocked/bench.py` before making
any claim about this layout. `dcn_mxfp6_native` reproduces the known 16.38us
production number, so it is the calibration check that the harness is
measuring the right thing. Adding an impl or workload is one function.

Related: [[to-padded-blocked-prim]], [[bench-variant-isolation]],
[[nested-reduction-stack]].
