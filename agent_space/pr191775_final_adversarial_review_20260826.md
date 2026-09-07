# PR #191775 final adversarial review

Worktree:
`/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt`

## Verdict

Three independent read-only reviews found no correctness, codegen, design, or
test blocker in the current worktree.

## Highest-value review points

1. Exact standalone append:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:9858)
   permits a parent-shaped append only after a `FusedStagedReduction` has
   already been formed. The complete plan is rebuilt and exact dependency
   proof still runs. A shifted scale read rejects.
2. Looped parent placement:
   [scheduler.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/scheduler.py:785)
   and
   [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:2797)
   keep the internal source chain and derived epilogue in the final reduction
   pass. `OrderedParentNodes` names the reordered nodes and boundary.
3. Codegen replay:
   [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3417)
   and
   [simd.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/torch/_inductor/codegen/simd.py:3717)
   preserve masks, lane order, and CSE liveness for persistent and looped
   kernels.
4. Regression:
   [test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr191775_layered_split_wt/test/inductor/test_nested_reduction.py:2012)
   pins one-kernel exact scale swizzle and fail-closed shifted scale access in
   both persistent and looped modes. Restoring the old append guard makes both
   exact cases fail with two kernels.

## Independent validation

- Full nested-reduction file: 395 passed, 8 skipped.
- Full scheduler file: 108 passed, 6 skipped.
- Focused MXFP6 tests: 30 passed, 2 skipped.
- `spin quicklint`, Python compilation, and `git diff --check` passed.
- Four-shape native E2M3 comparison: exact, one nested kernel, one asm site,
  and faster than AITER for every shape.
- DCN 2048x3072 with aligned scale preshuffle: exact versus the software graph,
  one kernel, 32 registers, zero spills. Final rerun measured 8.499 us native
  and 9.114 us with zero canonicalization versus 24.211 us for the corrected
  reference.
- Odd tail 7x96: exact, one kernel, and all four stores masked.

## Residual risks

- Grouped-stage reads retain the scoped legacy equivalence proof until the
  indexed-forwarding follow-up records those relations directly.
- Dynamic padded scale preshuffle remains separate work. It declines safely in
  this stack rather than broadening fusion legality.
- Native SM100 E2M3 composition is validated in scratch benchmarks rather than
  a checked-in architecture-specific test. The checked-in generic test pins
  the scheduler relation independently of GPU instruction availability.
- Fusion and codegen rebuild the plan independently. Future planner drift fails
  loudly with a compiler assertion, but end-to-end coverage must remain in
  place.

No production-code change was requested by this review round.
