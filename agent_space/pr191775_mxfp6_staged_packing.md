# #191775 - Fuse staged MXFP6 packing epilogues

> Historical monolithic overview. The active uncommitted proposal is described
> in [pr191775_split_review_20260824.md](/data/users/eellison/pytorch/agent_space/pr191775_split_review_20260824.md),
> with review order in
> [pr191775_split_step_by_step.md](/data/users/eellison/pytorch/agent_space/pr191775_split_step_by_step.md).

Current ghstack orig commit `b225e3b078b` - #191775, based on the submitted
#190595 orig commit `70d9ccee025`. The exact dependency-plan redesign is local
and uncommitted for review.

Read #190594 first. This commit extends the common sub-parent stage from a
one-output lane group to MXFP6's four-input, three-output packing rate.

## Review order

1. **Highest scrutiny:** rate inference and grouped-coordinate validation in
   [`NestedReduction._sub_parent_epilogue_rate`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:892>)
   and
   [`NestedReduction._nested_sub_parent_rate`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:920>).
2. **Highest scrutiny:** looped internal-source ordering in
   [`NestedReduction._order_sub_parent_parent_nodes`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:746>).
3. Review the multi-output plan representation in
   [`ProjectedSourceAccess`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1904>)
   and
   [`SubParentEpilogueStage`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/scheduler.py:1927>).
4. Review source capture/materialization in
   [`_SubParentSourceLoadResolver`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:2358>)
   and the standalone emission sequence in
   [`_codegen_reduction_with_sub_parent_epilogue`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/simd.py:3682>).
5. Review the recursive factor-4 split in
   [`TritonKernel._emit_recursive_split`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/torch/_inductor/codegen/triton.py:6367>).
6. Use the MXFP6 functional and kernel-form tests starting at
   [`test_producer_consumer_mxfp6_four_to_three_pack`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:1764>)
   and
   [`test_mxfp6_four_to_three_pack_kernel_form`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/wt/test/inductor/test_nested_reduction.py:3385>)
   as the executable specification.

## The format shape

MXFP6 packs four 6-bit values into three bytes:

```text
input lanes:   v0 v1 v2 v3
output lanes:  low mid high
rate:          factor 4, output_lanes 3
```

The scale reduction still owns the full parent tile. The packing consumers run
over one quarter of it and emit three values for every derived child index.

## Planner representation

`_sub_parent_epilogue_rate()` accepts exactly two families:

- power-of-two factors 2..4 with one output lane
- factor 4 with three output lanes

This is deliberately not an arbitrary rational-rate interface. The shape ratio
derives 4:3 symbolically, but the planner then allowlists `(4, 3)`.

`SubParentEpilogueStage.output_groups` stores the output-lane count with the
nodes emitted at that rate. Construction rejects empty, duplicate, or
out-of-order groups, and codegen consumes the groups directly.

`MAX_INTERLEAVED_SUB_PARENT_FACTOR` is 4 here, and the general factor bound is
temporarily the same. #190596 later widens one-output factors to 16 when it
adds CONTIGUOUS layouts; the interleaved bound stays at 4.

This generalization also removes #190595's stored factor-2
`PointwiseDomainContext.sub_parent_domain`. The context retains grouped-axis
geometry, while `_nested_sub_parent_rate()` derives `(factor, output_lanes)`
for each candidate and checks its exact grouped-coordinate domain. The
scheduler regression migrates accordingly: the #190595 field assertion becomes
a direct `(2, 1)` rate assertion for R grouping, rejection for X grouping, and
rejection of a same-numel consumer whose ranges cross the grouped-axis boundary.

## Looped codegen

MXFP6's full-resolution source can be produced inside the fused kernel. A
looped reduction cannot keep that source live across reduction iterations, so
the source chain is moved to the end of the parent schedule. It is then emitted
in the final loop with the derived epilogue instead of forcing the entire
reduction persistent.

Codegen moves the source chain only when doing so preserves dependencies;
otherwise planning declines the fusion. A full-resolution sibling that also
reads the internal source is valid and remains in the final loop. The positive
fork test verifies one correct staged kernel in both looped and persistent
modes.

Internal and external sources use the common lazy
`_SubParentSourceLoadResolver`. An external value invalidated by the parent
reduction loop reloads at the derived index. A planned internal source is
deferred until its value is live, then projected through the same resolver;
codegen requires it rather than silently falling back to an unavailable load.
Internal values that must survive the parent-body flush are materialized before
that flush. The derived-family mask helper activates its family while assigning
the target-domain masks, so this pre-flush path and ordinary on-demand loads use
the same mask contract.

In a looped kernel, the deferred source chain begins a new reduction pass
after the leading schedule closes. Reductions are never deferred
(`_order_sub_parent_parent_nodes` only defers non-reduction chains); the
three-pass form arises when the leading schedule itself already needs two
passes because a second reduction consumes the first's result, and the
deferred boundary adds the third. The boundary insertion is inline in
`_codegen_reduction_with_sub_parent_epilogue` and keys on the actual
generated leading schedule, not merely on the number of reduction nodes. The
three-pass regression catches an otherwise undefined accumulator in the final
epilogue.

The stage does not impose `min_xblock`. The casts and bit packing may increase
register pressure, but that is a reason to permit smaller blocks, not a
correctness floor. The autotuner chooses XBLOCK.

## Scheduler relaxation

Preshuffled packing can expose producer/consumer deps with equivalent indices
under different loop orders. Normal strict `MemoryDep` matching remains the
first path. The staged planner records each proved source projection as:

```text
source MemoryDep(s) -> consumer MemoryDep + lane -> projected layout
```

Fusion then examines only residual `consumer.unmet_dependencies` that read a
producer output. Every such `MemoryDep` must either pass the existing normalized
reshape/broadcast proof or match an exact source/consumer pair in the staged
plan. A partial match rejects the whole candidate. `StarDep` and `WeakDep` stay
on ordinary vertical-fusion legality because a lane projection says nothing
about indirect indexing or mutation ordering.

The actual producer write must still be unique, dense, injective,
synchronization-free, and free of TMP symbols. This preserves the guarantees of
the original scheduler index-equivalence change (#183432) while replacing its
name-level permission with exact read/write relations. MXFP6's four lane reads
therefore become four exact matches to one producer write, and scoring counts
that write once rather than four times.

No fusion plan is cached. The planner may perform its local consumer reindex,
then builds exact records from the resulting dependencies. A nonempty exact
record set suppresses later generic expand/reorder/inversion passes that would
invalidate those records. Nested append replans after legality and `fuse_with`
replans again, while codegen independently rebuilds the final post-merge plan.

`ProjectedSourceAccess` is deliberately the record shape intended for the
indexed-forwarding follow-up: source accesses, consumer access plus lane, and
layout. This PR's existing codegen still reads the compatibility
`(buffer_name, layout)` view; exact access-keyed codegen consumption remains the
separate indexed-forwarding change.

This commit also replaces the landed `allow_index_equivalence`-specific test
with direct coverage of the exact-index API introduced here. Keeping the
replacement in this commit avoids a one-commit coverage hole; #191974 no
longer owns scheduler coverage.

Speculative planner reindexing is protected by loop-state snapshots and rollback
when the full fusion decision fails.

## Codegen

`emit_split_via_reshape` recursively applies binary `tl.split`, producing four
interleaved source lanes. The pointwise body is emitted three times with:

```text
child_index * 3 + 0
child_index * 3 + 1
child_index * 3 + 2
```

Constant folding selects low, middle, and high byte expressions. Kernel CSE
reuses shared loads and conversions across the three emissions.

## Tests

The commit includes:

- fused 4:3 packing against eager
- an exact integer-valued reference case (`atol=0`, `rtol=0`)
- forced looped large-reduction coverage
- preshuffled and shifted-preshuffled layouts
- nested `(4, 3)` packing and a non-trailing output-lane fallback
- simple and preshuffled dynamic batch with multiple runtime values
- shifted-intermediate and source-read-by-reduction rejection
- shifted parent/reduction-output rejection for name-level dependency matching
- a rolled grouped scale used by the pair epilogue, which must remain a
  separate correct kernel
- a safe internal-source full-resolution fork in both kernel modes
- a looped internal-source chain containing another reduction, requiring three
  correctly closed passes
- a reduced sibling reading an unplanned source without blocking fusion
- persistent, looped, and default kernel-form checks; the internal-source case
  pins looped/persistent input loads (2/1), five emitted stores and matching
  metadata across three output pointers, and three `tl.split` operations

Store metadata tracks the number of emitted stores per kernel-local buffer, so
removing a multi-lane intermediate subtracts every store rather than one.

The software `_float_to_mxfp6_e2m3` conversion is intentionally not folded into
a shorter expression. Its graph shape affects realization and therefore
whether the staged fusion forms; #191974 records that dependency.

## Changed later

#190596 later adds CONTIGUOUS layouts, optional split permutation, and factors
8/16 without changing MXFP6's factor-4 interleaved path. #191974 only
deduplicates test helpers. The former #191975 source-dependency lookup
simplification now lands in this commit, alongside the internal-source support
whose names it handles.

## Rebase resolution

The append path now derives exact producer-write/consumer-read matches from the
staged plan. A matching buffer name is insufficient, and every residual
producer-output `MemoryDep` must be accounted for. The 2026-08-12 replay also
fixed the multi-reduction boundary above after adversarial review exposed an
undefined accumulator in the third stage. Both behaviors are pinned before
#190596 is applied.

## Verification

At submitted orig commit `b225e3b078b`:

```text
python test/inductor/test_nested_reduction.py -k mxfp6
  30 passed, 3 skipped

python test/inductor/test_nested_reduction.py -k indirect_index_mask
  6 passed

with-proxy spin quicklint
  clean
```

The full rebased stack additionally ran the complete nested file: 441 passed,
12 skipped, with the two known narrow-runner `kernel_num_gb` failures.

This review state deliberately does not include the later indexed-forwarding or
lazy domain-projection work. #191775 is therefore checked against #190595's
ordinary eager projection behavior; those follow-ups remain separate PRs.

The exact review diff, including the visible integration fix, is
[`pr191775.rebased.patch`](</data/users/eellison/pytorch/agent_space/stack_rebase_20260824/pr191775.rebased.patch>)
(SHA256 `eaf3110a5c9cec37e74edb954fcdf85c63ec3ba04fefc07a3100fd5c7662187a`).
The current uncommitted dependency-plan redesign is
[`pr191775_exact_dependency_plan.patch`](</data/users/eellison/pytorch/agent_space/pr191775_exact_dependency_plan.patch>)
(SHA256 `6e08c00459e55efdaf286fa64ea53311e960c32dc484aca7323d0942602922e9`).
The small rebase integration delta alone is
[`pr191775_rebase_integration_fix.patch`](</data/users/eellison/pytorch/agent_space/pr191775_rebase_integration_fix.patch>)
(SHA256 `273fc4ab0a5aa655b0efa67ca1448936ad0ce953f7c7d8c5896d44606f4ca749`).

## Remaining reservations

The repeating four-value flagship input distinguishes lanes within a group but
is weaker against group off-by-one errors than random or group-asymmetric data.
The exact and preshuffled tests mitigate this but do not replace stronger input
coverage. The conversion in this test workload is also software E2M3, not a
production native MXFP6 instruction.

The planner still rebuilds legality after loop merging, and enabling the feature
adds symbolic planning work during fusion. Projected candidates also retain the
ordinary final `score_fusion_key` ordering behavior from #183432. Both remain
acceptable while `nested_reduction` is default-off; profile before changing that
default.

## Metadata

The existing explanatory paragraph that followed the ghstack trailers was
moved before them. Their values are unchanged, and Git now recognizes #191775's
`Pull-Request` and `ghstack-source-id`.
