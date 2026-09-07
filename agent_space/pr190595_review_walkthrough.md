# #190595 review walkthrough

PR: https://github.com/pytorch/pytorch/pull/190595  
Local commit: `359b11d5aa1`  
Parent: `86038d6e1cd` (#190594, landed)

The review worktree is stopped at `359b11d5aa1` with current review cleanups
left uncommitted. These cleanups retain the direct eager-broadcast design; do
not read the submitted GitHub diff as the final local review state.

This checklist assumes #190594 has already been reviewed. It does not repeat
the standalone lane proof, staged node identity, derived iteration family, or
basic source-load resolver. Review only how #190595 attaches that machinery to
the existing nested-reduction pipeline.

To inspect the isolated change:

```bash
cd /data/users/eellison/pytorch/agent_space/pr190594_ci_fix
git diff 86038d6e1cd
```

Using `359b11d5aa1` as the upper endpoint would omit the current uncommitted
review/removal changes.

Current isolated verification is 296 passed / 1 skipped in
`test_nested_reduction.py` and 60 passed / 6 skipped in
`test_inductor_scheduler.py`; `git diff --check`, `python -m py_compile`, and
`with-proxy spin quicklint` are clean.

The pipeline being added is:

```text
outer reduction -> grouped reduction -> reduced/full pointwise
                                  \----> factor-2 SUB_PARENT epilogue
```

The main review question is whether the new append path preserves every
ordinary fusion gate while producing a final staged plan whose domains and
source values agree with codegen after loop merging.

## Design questions

Resolve these before reviewing implementation details:

1. **Is `SUB_PARENT` another nested pointwise domain?** It should extend the
   existing nested-domain classifier, not introduce another standalone
   planner or kernel identity.
2. **Is the initial capability boundary appropriate?** The append path accepts
   factor-2 INTERLEAVED sources on a grouped R axis. Wider factors,
   CONTIGUOUS layouts, and grouped X are deferred rather than described as
   fundamental limitations.
3. **Does specialized discovery preserve ordinary fusion legality?** The path
   must re-enter the normal scheduler checks and normalize reverse discovery
   to producer-first order.
4. **What crosses the fusion/codegen phase boundary?** Stable nested topology
   crosses it; mutable domains and the final `StagedReductionPlan` are rebuilt
   after loop merging. The fusion-time approval is not a codegen plan cache.
5. **How are source values reused safely?** Parent values are split, grouped or
   reduced values are eagerly broadcast, and unrelated inputs remain indexed
   loads. Reuse is permitted only while ordinary kernel CSE says the value is
   live.
6. **What is deliberately deferred?** Optimizing the eager group-to-lane
   broadcast is outside this two-PR landing scope.

## 1. Domain model

- [ ] Read `NestedReduction.PointwiseDomain` and
  `PointwiseDomainContext.create()` in `scheduler.py:582-652`.
- [ ] Confirm `SUB_PARENT` is a fourth pointwise resolution, not a second
  standalone planner.
- [ ] Confirm the domain retains grouped coordinates
  `(outer, groups, local / 2)`. A flattened `(parent_numel, parent_rnumel / 2)`
  would incorrectly admit reshapes that cross group boundaries.
- [ ] Confirm the domain is available only when the grouped axis is R and its
  local group is divisible by two.

Then read
`test_nested_reduction_sub_parent_domain_preserves_group_axis` in
`test_inductor_scheduler.py:396`. It pins both the retained R-axis boundary and
the absence of a sub-parent domain when grouping is in X.

## 2. Pointwise classification

- [ ] Read `_classify_grouped_pointwise_nodes` in `scheduler.py:1081`.
- [ ] Follow the priority among `REDUCED`, `LOCAL_REDUCTION_INPUT`,
  `PARENT_FULL`, and `SUB_PARENT`.
- [ ] Pay attention to the G=2 ambiguity: reduced and sub-parent nodes can have
  the same numel. Reading the reduction source is what selects `SUB_PARENT`
  when both shapes are compatible.
- [ ] Confirm source names come from grouped-reduction reads, not
  `used_buffer_names()` (which also includes the reduction output).
- [ ] Read `_pointwise_domain_is_compatible` in `scheduler.py:1178` and confirm
  each classification is checked against the exact ranges codegen will use.

The baseline numeric test is
`test_producer_consumer_rmsnorm_interleaved_pair_epilogue` in
`test_nested_reduction.py:804`. Its values make a reduced/sub-parent
misclassification numerically visible rather than merely changing fusion.

## 3. Nested sub-parent stage planning

- [ ] Read `_plan_nested_sub_parent_stage` in `scheduler.py:1208`.
- [ ] Separate its sources into:
  - parent-resolution values that need the #190594 lane proof;
  - grouped or reduced values that are constant over a lane and broadcast;
  - unrelated external inputs that remain ordinary indexed loads.
- [ ] Confirm `allow_contiguous=False` and factor 2 are explicit append-path
  limitations, not accidental consequences of index matching.
- [ ] Confirm temporal dependency names come from each node's `read_writes`.
  Applying scheduler-global `mutation_renames` here could replace an earlier
  read with a later in-place version.
- [ ] Confirm epilogue outputs still satisfy the leaf constraint before their
  stage is accepted.

Continue through `NestedReduction.plan()` at `scheduler.py:1357` and
`plan_from_topology()` at `scheduler.py:1457`. Fusion builds an approval plan;
codegen rebuilds mutable domains after loop merging from stable nested
topology. This is recomputation for phase correctness, not a cache.

## 4. Append fusion path

- [ ] Read `FusedNestedReductions.can_fuse_with()` in `scheduler.py:3737`.
- [ ] Confirm the candidate consumes the grouped stage and is not a
  local-reduction-input producer, which would require insertion before the
  grouped reduction.
- [ ] Follow `_can_fuse_nested_reduction_append` at `scheduler.py:8681` back
  into the ordinary `_can_fuse` path.
- [ ] Confirm device, stream, mempool, `no_fuse_buffer_names`, resource,
  peak-memory, backend, and vertical legality checks are not bypassed.
- [ ] Check `check_all_pairs` at `scheduler.py:7578`: reverse discovery must be
  normalized to producer-first before fusion bookkeeping.
- [ ] Confirm append dependency relaxation includes proven reads from both the
  grouped and outer stages, so output ordering cannot change fusion.

Next read `_plan_append()` and `fuse_with()` at `scheduler.py:3779-3804`.

- [ ] `can_fuse_with()` and `fuse_with()` call the same complete append planner,
  so admission and construction cannot validate different predicates.
- [ ] The internally created grouped node inherits its mempool explicitly,
  because this operation bypasses `Scheduler.fuse_two_nodes()`.

The focused bookkeeping test is
`test_nested_reduction_fuse_with_propagates_mempool` in
`test_inductor_scheduler.py:172`.

## 5. Final codegen plan

- [ ] Start at `SIMDScheduling.codegen_staged_reduction` in `simd.py:3089`.
- [ ] Follow the nested branch into `_codegen_nested_reduction` at
  `simd.py:3116`.
- [ ] Confirm emission order is outer reduction, grouped reduction, ordinary
  reduced/full pointwise, then the sub-parent epilogue.
- [ ] Confirm tiling still sees the grid-owning outer schedule, while
  index-width analysis sees all emitted stages.
- [ ] Confirm kernel-local buffers are removed only after every stage has
  emitted.

Read `_codegen_nested_grouped_schedule` at `simd.py:3354` only for the normal
nested stages. The `SUB_PARENT` nodes are emitted by the post-schedule
`_codegen_remapped_pointwise` path at `simd.py:3570`; they should not appear as
a second branch inside the grouped schedule.

## 6. On-demand source materialization and CSE lifetime

- [ ] Revisit `_SubParentSourceLoadResolver` at `simd.py:2333` for the new
  `broadcast_source_names` behavior.
- [ ] Confirm the first actual epilogue load triggers materialization; there is
  no pre-epilogue sweep over every planned name.
- [ ] Confirm parent-resolution values are split into factor-2 lanes before
  that load is resolved.
- [ ] Confirm grouped/reduced values are eagerly reshaped and broadcast to
  sub-parent resolution rather than being split as parent values.
- [ ] Confirm an in-kernel source is identified by `kernel.store_buffer_names`
  and must forward, while an unavailable external source falls back to an
  ordinary derived-index load.
- [ ] Confirm unrelated external lane-resolution inputs are not forced through
  the resolver.
- [ ] Confirm split, broadcast, and direct-width values receive the active
  sub-parent family's shape-compatible `mask_vars` when materialized. Ordinary
  scalar CSE propagation then carries those masks to indirect accesses.
- [ ] Confirm the mask is not repaired in `indirect_indexing`; it belongs to
  the value as soon as the value enters the derived domain.

There must be no persistent-only materializer. Persistent and looped kernels
share this path; their difference follows ordinary CSE lifetime and reduction
loop invalidation.

## 7. Optimization boundary

- [ ] Confirm the on-demand materializer uses the direct eager-broadcast path.
- [ ] Treat repeated lane-width scale work as a known performance limitation,
  not a fusion-legality defect in this PR.
- [ ] Confirm no additional forwarding or projection mechanism is required to
  understand or approve this PR.

## 8. Named constants

- [ ] Read `_grouped_axis_named_constants` in `simd.py:1759`.
- [ ] Read `TritonKernel._codegen_named_constant` in `triton.py:7927` and where
  `_named_constant_defs` is spliced at `triton.py:7501`.
- [ ] Confirm the reduced-output and sub-parent families can request the same
  constant expression without duplicate definitions.
- [ ] Confirm definitions live at kernel-function scope, independently of the
  loop-local indexing buffer that first activates a derived family.
- [ ] Confirm conflicting expressions for the same symbol fail loudly.

## 9. Behavior tests

Review tests in this order:

1. `test_producer_consumer_rmsnorm_interleaved_pair_epilogue`: basic nested
   append and numerics.
2. `test_dynamic_sub_parent_epilogue`: multiple runtime batch and reduction
   sizes through one compiled graph, including `D=4608`.
3. `test_sub_parent_append_respects_fusion_gate`: ordinary resource and
   `no_fuse_buffer_names` gates remain effective.
4. `test_producer_consumer_sub_parent_intermediate`,
   `test_producer_consumer_inlined_parent_full_source`, and
   `test_producer_consumer_independent_sub_parent_source`: the three source
   categories from stage planning.
5. The `rejects_*` tests at `test_nested_reduction.py:1005-1267`: shifted,
   conflicting, mutated, wrong-axis, and non-leaf graphs decline fusion rather
   than reaching invalid codegen.
6. `test_nvfp4_inline_asm_kernel_form` and
   `test_mxfp4_inline_asm_kernel_form` at `test_nested_reduction.py:2772-2809`:
   exact nested kernel form, load/store counts, and packing instructions.
7. `test_rmsnorm_block_scale_swizzle_kernel_form`: the ordinary reduced-domain
   scale-layout store remains covered independently of the sub-parent
   epilogue.

The base tests are inherited by forced-persistent and forced-looped classes.
Together with the dynamic tests, this covers static/dynamic by
persistent/looped combinations.

## 10. Final review checklist

- [ ] The append path supports factor-2 INTERLEAVED sources only; broader
  support is described as unimplemented, not impossible.
- [ ] No specialized path bypasses ordinary scheduler fusion gates.
- [ ] Producer-first ordering is preserved for stream and mempool bookkeeping.
- [ ] Fusion-time approval and post-merge codegen planning share the same
  domain and source-layout helpers.
- [ ] Parent, broadcast, and external source values take distinct, justified
  codegen paths.
- [ ] Group/reduced values are eagerly broadcast in #190595.
- [ ] CSE/store-load forwarding remains valid across loop boundaries.
- [ ] Dynamic R does not require a power-of-two extent; factor divisibility is
  the required proof.
- [ ] Both persistent and looped generated forms have source-level checks.
- [ ] `triton.multi_kernel` does not silently replace a staged persistent form
  with a discarded looped alternative.

At the stack tip #191775 replaces the stored factor-2 domain with per-candidate
rates and supports interleaved factors through 4, including `(4, 3)`. Keep that
later widening separate from this commit's factor-2 review boundary.
