---
name: sub-parent-mutation-hoisting-fix
description: Replaces the blanket has_aliasing_or_mutation bail in sub-parent epilogue planning; unblocks scatter-form padding fusion (-8 to -22% on MXFP6/XDL)
metadata: 
  node_type: memory
  type: project
  originSessionId: 8b2c3728-fc6d-4a7a-a0a5-405d8b3ed8ba
  modified: 2026-09-04T00:46:12.716Z
---

2026-09-03, approved by the user and implemented (uncommitted at time of
writing): `NestedReduction._mutations_survive_hoisting(nodes, group=None)` in
`torch/_inductor/scheduler.py` replaces the two blanket
`has_aliasing_or_mutation()` bails -- one in `sub_parent_epilogue_plan`, one in
`_plan_nested_sub_parent_stage`. Both carried the same "TODO: No fundamental
limitation" comment and both came from the user's own #190594/#190595.

The predicate: reject aliasing outright; reject a second node touching the
mutated storage; otherwise allow. Two details that are easy to get wrong and
were both found empirically, not by reading:

- The mutating node carries a **StarDep on its own mutation target** -- the
  synthetic self-edge, not a value read. A naive "target must not be read in
  the group" rule rejects everything.
- Deps are renamed through `Scheduler.mutation_renames` at scheduler.py:6497,
  and `update_mutated_names` runs *before* the node's own mutation registers.
  So the mutator keeps the pre-mutation name (`buf1`) while every later node
  sees the post-mutation name (`buf2`). The predicate must claim **both** names.
- The second bail site only saw `sub_parent_nodes`; a reader elsewhere in the
  fused kernel would be missed. It passes `(outer_node, *grouped_nodes)` as
  `group`.

Reachability: functionalization collapses almost everything before the
scheduler sees it. `out.copy_(v)` plus `out + 1.0` reads the *value*, not the
buffer; two `copy_`s to one destination fuse into one. Only a **pre-mutation
MemoryDep read by another node in the hoist set** actually rejects today
(reachable via two slice-`copy_`s into one destination). The post-mutation-name,
write, and two-mutator branches are unexercised but kept so the predicate is
conservative rather than wrong if that changes.

Measured (user's `agent_space/padding_matrix_20260903` matrix, 208 cells, fix
isolated in a worktree at the baseline commit f8546ee64c3): **32 cells dropped
kernel count, 0 rose**, mostly 4 kernels -> 2. Best-of-all-variants improves on
every XDL/MXFP6 shape and the winner flips from `fpad` to `fill_scatter`:
dcn_mxfp6 2048x3072 9.52 -> 8.58us, dcn_mxfp6 95-97x3072 -15 to -17%,
mxfp6_quant 95x3072 -21.6%, rmsnorm_mxfp6 2048x3072 -7.0%. The 128x4 family
(mxfp4/mxfp8/nvfp4) is unchanged -- flashinfer/fpad already won.

Tests: the four failures were the two `*_rejects_*mutation` tests written to
assert the bail. Their **numerics assertions passed**; only the fusion-count
checks failed, so they asserted policy, not corruption. Both flipped positive
(`test_producer_consumer_sub_parent_mutation`,
`test_standalone_sub_parent_mutation`, each `check_fusion()` = 1 kernel), plus
new `test_standalone_sub_parent_rejects_shared_mutation_target`. Full suite 423
tests OK (skipped=8).

**Why:** the bail, not the layout, was why every scatter-form padding impl lost.
It is the single highest-leverage thing found in the padded-blocked work, and it
reframes #195930 -- the prim's fusion advantage disappears, leaving guards as its
only durable justification.

**How to apply:** cross-run matrix deltas under ~5% with identical kernel counts
are coordinate-descent autotuner drift, not real (rmsnorm_mxfp8 fill_scatter
129x4128 moved 3.35 -> 4.31us at 2 kernels both runs, tight percentiles, same
GPU; its `default`-tuning twin was unchanged). Trust the kernel-count drops.
The matrix now has a `prim` variant (`xdl_prim` in XDL_VARIANTS), added
2026-09-03 to settle the #195930 question.

Related: [[padblocked-layout-bench]], [[to-padded-blocked-prim]],
[[nested-reduction-stack]], [[worktree-overlay-harness]].

## 2026-09-03: PR prepared, validated standing alone on main

Commit `e5196889c55` on branch `nested-reduction-mutation-hoist`, worktree
`agent_space/mutfix_pr`, based on `origin/main`. **Not pushed** -- awaiting the
user's explicit approval of exact content per AI_POLICY. The patch applies
cleanly to bare main; neither the crash-fix commit nor the prim is a dependency
(both hook functions are already in origin/main).

Validated on the branch through `agent_space/run_wt.py` (overlay confirmed
resolving to the worktree, not the main checkout):

- test_nested_reduction.py: **419 OK** (skipped=8). NOT 423 -- that count came
  from the main tree, which carries the crash-fix commit's four extra tests.
  The commit message was amended to say 419. Always re-count on the PR's actual
  base.
- test_loop_ordering.py: 125 OK.
- test_torchinductor.py: 3098 ran, 104 errors -- **byte-identical failure set on
  a clean origin/main baseline arm**, so all pre-existing. They are
  `CppCompileError` on the CPU/AOTI path from stale `torch/include` headers in
  this env (the overlay swaps Python only); nothing a GPU-side Triton scheduling
  change can reach. Do not re-investigate these.

Method worth reusing: run the suspect suite on both arms in parallel on separate
GPUs, capture **full** logs (piping through `tail` loses the failure list), then
`diff` the sorted `^(ERROR|FAIL): ` sets. Identical sets settle attribution in
one pass.
