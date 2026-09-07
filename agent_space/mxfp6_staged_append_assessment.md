# Adversarial review: MXFP6 scale-only staged append

Date: 2026-08-26

## Verdict

The scale-only source formulation is sound and the scheduler problem is narrower than the original seven-line workaround. Keep the reviewed `(4,3)` lowering and change only the existing initial-formation guard:

```diff
-            elif not all(
+            elif type(node1) is not FusedStagedReduction and not all(
```

This is preferable to duplicating reduced-domain classification in `Scheduler.can_fuse`, and preferable to deleting the guard entirely.

## Why the change is needed

For production standalone/DCN scale-only preshuffle, pairwise fusion proceeds in this order:

1. The row-major code/pack branch fuses with the group reduction.
2. That valid plan gives the group `FusedStagedReduction` identity.
3. The preshuffled scale output is considered as a later append.

The scale cannot reliably fuse first. Before a sub-parent candidate exists, `reduction + swizzled scale` has no sub-parent plan, and ordinary fusion gives zero shared-data score because the scale node iterates in its permuted output frame. Fusion logs report `no shared data`. Output ordering does not change that dependency/indexing fact.

The old scheduler check then requires the later scale node itself to belong to `plan.sub_parent_stages[0].epilogue_nodes`. It is instead a valid reduced-domain `plan.parent_nodes` member, so the check rejects it even though the complete plan is valid.

## Why the one-line guard is sound

The guard is necessary only when creating staged identity. It ensures the incoming consumer actually introduces the sub-parent stage. Once `node1` is exactly `FusedStagedReduction`, that invariant has already been established.

Every append still passes all of these checks:

1. `NestedReduction.sub_parent_epilogue_plan` is rebuilt over every leaf node.
2. Every leaf must classify as the parent reduction, reduced output, full-parent pointwise, or supported sub-parent rate.
3. Source, broadcast, and internal forwarding projections must be proven.
4. Parent nodes are topologically ordered around the completed reduction.
5. Sub-parent outputs must remain unread by later nodes.
6. `_prove_staged_fusion_dependencies` validates the current producer/consumer edge.
7. Ordinary vertical/horizontal legality and backend legality still run.
8. Triton rebuilds the full plan, checks 2D tiling, and rebuilds it again at codegen after loop merging.

This also matches existing backend intent. `_sub_parent_epilogue_decision` already returns `FUSE` for an exact `FusedStagedReduction`, while requiring a non-staged consumer to be wholly in the sub-parent epilogue. The scheduler was redundantly applying only the stricter initial rule.

Use exact type comparison to mirror the backend. `FusedNestedReductions` is a subclass with its own `_plan_fusion_with` path and is handled earlier; `isinstance` works today but would also exempt future subclasses with potentially different append contracts.

## Negative cases

The allowance does not authorize a plan by itself. A scale shifted by one element still produces a structural plan but fails `_prove_staged_fusion_dependencies`, remains at two kernels, and logs `staged fusion dependency proof failed`. A consumer of a sub-parent output also remains unfused.

Probe results:

```text
mxfp6_exact_reduced   baseline 2 kernels / patched 1 kernel
mxfp6_shifted_reduced patched 2 kernels / nested 1
reads_epilogue         patched 2 kernels / nested 0
```

## Alternatives

- Reordering fusion is not sufficient because the reduced scale pair is not an ordinary candidate before staged identity exists.
- Adding `_plan_fusion_with` to generic `FusedStagedReduction` could encode the same rule, but that marker lacks the structured fields used by `FusedNestedReductions`; it would duplicate plan reconstruction already performed by scheduler and backend.
- Deleting the scheduler gate entirely also passes the tested matrix, but unnecessarily relies on the backend to preserve initial-formation ownership.
- General outer-frame reindexing preserves the original pre-permuted source and reaches one kernel, but needs a substantially larger proof/rollback mechanism and is slower here.

## Validation

- Clean scratch worktree: `agent_space/mxfp6_43_append_guard_fix`
- Production DCN `(2048,3072)`: 1 kernel, nested=1, 4 stores, 16.277 us; corrected D112902015: 24.056 us.
- Focused MXFP6 nested tests: 26 passed, 2 skipped.
- Scheduler unit tests: 24 `nested` and 12 `sub_parent` tests passed.
- Full nested-reduction file with the equivalent already-staged guard: 391 passed, 8 skipped.
- Direct-store dump: `agent_space/dumps/dcn_prod_scale_only_43_append_guard/source_0.py`.
