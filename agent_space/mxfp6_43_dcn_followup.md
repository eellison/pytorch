# MXFP6 `(4,3)` DCN preshuffle follow-up

Date: 2026-08-26. B200 measurements. No reviewed worktree or GitHub state was changed.

## Result

Keep the reviewed `(4,3)` packing design. Change D112902012 so packed data stays row-major and only the scale is reshaped/permuted, then let an already formed staged reduction append any consumer covered by the recomputed complete plan.

This emits one nested kernel with four direct stores and no temporary scale copy or `ir.Scatter`:

| Case | Shape | Scale-only, no planner fix | `(4,3)` + fix | Three slices + fix | Corrected D112902015 |
|---|---:|---:|---:|---:|---:|
| Standalone preshuffle | 2048x3072 | 17.611 us / 2 | 15.456 us / 1 | 15.050 us / 1 | 23.859 us / 1 |
| DCN preshuffle | 2048x3072 | 17.611 us / 2 | 16.280 us / 1 | 16.275 us / 1 | 24.054 us / 1 |
| RMSNorm preshuffle | 128x384 | 1.944 us / 1 | 1.939 us / 1 | 1.941 us / 1 | n/a |

The slice form has no meaningful performance advantage. The `(4,3)` result has one output pointer for packed data and three direct lane stores; the slice result has three aliased output pointers. Retaining `(4,3)` is the smaller design.

## Exactness

The scale-only formulation is bitwise equal to the original D112902012-style pre-permuted formulation for standalone and DCN at `(2048,3072)`.

Standalone compiled output is bitwise equal to eager and the corrected D112902015 kernel. Fused DCN differs from eager/corrected D by 67 of 196,608 scale bytes and 16,359 of 4,718,592 packed bytes because Inductor removes the fp16 `addcmul` materialization. The `(4,3)` and slice outputs are bitwise equal. Explicitly realizing the fp16 producer restores exactness but requires another kernel.

The supplied D112902015 kernel itself needs the previously reported per-program packed-output tail mask at this shape (`GROUPS_PER_THREAD=79`, `GROUP_LOAD=64`). The timings above use the corrected kernel.

## Original rejection

The original pre-permuted DCN graph takes 28.461 us, emits two kernels, and has `codegen_nested_reduction=0`. Its `(4,3)` candidate is recognized, but `_try_get_sub_parent_source_projections` rejects the internal code buffer.

Producer access:

```text
parent_r
+ 393216*(parent_x//12288)
+ 128*ModularIndexing(parent_x,1,3)
+ 24576*ModularIndexing(parent_x,3,16)
+ 384*ModularIndexing(parent_x,48,64)
+ 32*ModularIndexing(parent_x,3072,4)
```

First pack-lane read:

```text
4*child_r + 32*parent_x
```

The existing proof may replace only the reduction coordinate. The producer also permutes the outer coordinate, so relaxing this check would be incorrect. Production and small DCN both decline; standalone/SILU fuse at both sizes. A test-only row-major code realization produces one kernel at 20.890 us and confirms that the mismatch is the layout, not size.

A second scratch experiment reindexed the pack and final output nodes into the producer frame before fusion. It preserved the original full-preshuffle source, produced byte-identical output, and reduced the persistent case from 28.67 to 18.22 us and the looped case from 28.46 to 21.09 us. The proof of concept uses shape-specific loop orders in `agent_space/diagnose_dcn_preshuffle_43_subagent.py`; making it production-safe requires a general dense-permutation proof, coordinated closure reindex, and rollback. That is materially broader than the scale-only source rewrite and slower in this case.

## Narrow fix

With packed data row-major, the remaining failure is later: the pack has already formed a `FusedStagedReduction`, and the reduced scale consumer is in `plan.parent_nodes` rather than the sub-parent epilogue. The scheduler incorrectly reapplies its initial-formation rule while appending to an existing staged group.

The Triton backend already expresses the correct invariant: an initial fusion must introduce the sub-parent epilogue, but an already formed `FusedStagedReduction` may append whenever complete plan reconstruction succeeds. The smallest scheduler change is one condition:

```python
existing_standalone_epilogue = isinstance(node1, FusedStagedReduction)
if plan is not None and not existing_standalone_epilogue:
    new_nodes_are_epilogue = all(
        node in plan.sub_parent_stages[0].epilogue_nodes
        for node in node2.get_nodes()
    )
    if not new_nodes_are_epilogue:
        return False
```

This avoids duplicating domain classification in `can_fuse`. The complete planner still classifies every leaf as reduction, reduced, full-parent, or sub-parent; validates source, broadcast, and internal projections; orders post-reduction work; and rejects later reads of sub-parent outputs. `_prove_staged_fusion_dependencies`, ordinary vertical legality, backend plan/tiling checks, and codegen-time plan reconstruction remain unchanged.

Scheduler ordering does not remove the need. Before the pack forms the staged group, the swizzled scale and reduction have zero ordinary shared-data score because their iteration frames differ. The pack therefore forms the staged group first and the scale arrives as a later append. `FusedNestedReductions` already has a dedicated append planner for this situation; standalone `FusedStagedReduction` does not.

Adversarial probes:

- Exact scale append: 2 kernels before, 1 kernel after.
- `roll(scale, 1)` append: remains 2 kernels; rejected by `staged fusion dependency proof failed`.
- Consumer reading a sub-parent output: remains 2 kernels and does not create a staged group.

The generated `(4,3)` kernel directly stores scale with the R128c4-equivalent index and directly stores all three packed lanes. See:

- Clean one-condition worktree: `agent_space/mxfp6_43_append_guard_fix`
- Kernel: `agent_space/dumps/dcn_prod_scale_only_43_append_guard/source_0.py`
- Full survey: `agent_space/mxfp6_swizzle_survey.md`

## Verification

```bash
MXFP6_SCALE_ONLY_PRESHUFFLE=1 conda run -n pytorch-3.12 \
  python agent_space/run_internal_mxfp6_43_append_guard_fix.py \
  --case standalone_prod_preshuffle --case dcn_prod_preshuffle \
  --case rmsnorm_preshuffle

conda run -n pytorch-3.12 python agent_space/run_mxfp6_43_append_guard_tests.py -k mxfp6

conda run -n pytorch-3.12 python \
  agent_space/run_mxfp6_43_append_guard_script.py \
  agent_space/probe_staged_parent_append_gate.py mxfp6_shifted_reduced
```

The focused nested-reduction MXFP6 suite passed 26 tests with 2 skipped. Scheduler unit subsets passed 24 `nested` and 12 `sub_parent` tests. The full nested-reduction file passed 391 tests with 8 skipped for the equivalent already-staged guard.
