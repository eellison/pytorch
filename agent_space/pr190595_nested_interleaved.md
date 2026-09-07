# #190595 - Fuse interleaved epilogues in nested reductions

Local commit `359b11d5aa1` - PR #190595. Restacked on landed #190594 and
resubmitted on 2026-08-21. The review worktree is stopped at this commit with
uncommitted review cleanups that retain the direct eager-broadcast design.

For an ordered file-by-file review, use
[pr190595_review_walkthrough.md](/data/users/eellison/pytorch/agent_space/pr190595_review_walkthrough.md).
For an execution-order check-off with checkpoint questions, use
[pr190595_step_by_step_checklist.md](/data/users/eellison/pytorch/agent_space/pr190595_step_by_step_checklist.md).

If #190594 is unfamiliar, read
[pr190594_interleaved_subparent.md](/data/users/eellison/pytorch/agent_space/pr190594_interleaved_subparent.md)
first for the common staged identity, plan types, lane proof, and standalone
emitter. This commit uses them for an existing nested-reduction pipeline such
as RMSNorm followed by block amax and NVFP4/MXFP4 packing.

## What this commit adds

The nested kernel already has an outer reduction and a dependent grouped
reduction. Packing needs a fourth pointwise domain:

```text
REDUCED                 grouped reduction output
LOCAL_REDUCTION_INPUT   grouped reduction input
PARENT_FULL             outer reduction tile
SUB_PARENT              factor-2 lane view of the parent tile
```

[PointwiseDomainContext.create()](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:623)
derives these domains once. The append path classifies pointwise consumers,
proves their interleaved source reads with #190594's normalized `MemoryDep`
domain mapper, and records a
[SubParentEpilogueStage](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1671)
on the common
[StagedReductionPlan](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1682).

The sub-parent domain stays in grouped coordinates
`(outer, groups, local/factor)`, rather than flattening to
`(parent_numel, parent_rnumel/factor)`. That is intentionally stricter: the
emitter preserves the grouped-axis boundary, so legality must reject reshapes
that cross groups merely because their total numel matches. The factory test
pins `(3, 6, 8)` for `B=3, D=96, G=16` and verifies that grouped-axis X has no
sub-parent domain; flattening the R case would produce `(3, 48)`.

The nested append path is deliberately narrower than standalone planning:

- factor 2 only
- INTERLEAVED sources only
- grouped axis R only

CONTIGUOUS and wider factors are not fundamental codegen limitations; they are
unimplemented nested-append cases.

## Fusion legality

[FusedNestedReductions.can_fuse_with()](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:3737)
accepts only consumers of the grouped stage. It classifies the candidate,
rejects local-reduction-input producers, then calls
[Scheduler._can_fuse_nested_reduction_append()](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:8681).

The scheduler helper re-enters the ordinary `_can_fuse` path with the producer
and consumer in topological order. Device, stream, mempool, `no_fuse`, resource,
peak-memory, backend, and vertical legality checks therefore still run. There
is no specialized combined-backend hook: CUDA and XPU use their ordinary
backend ladders, which reach SIMD after the common checks. Reverse discovery is
normalized producer-first before stream/mempool bookkeeping.

Nested source planning uses the temporal names already stored in each node's
`read_writes`. It must not apply the scheduler-global `mutation_renames` map:
that map describes the final version of a buffer and can collapse a source
read onto an in-place mutation that occurs later in the graph. The stage keeps
the temporal name so codegen materializes the value that the node actually
read.

`can_fuse_with()` and
[fuse_with()](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:3795)
both use `_plan_append()`. The admission and construction phases therefore
validate the same complete prospective topology without retaining mutable
approval state on the fused node.

The retained grouped node is created outside `Scheduler.fuse_two_nodes()`, so
`fuse_with()` explicitly propagates its mempool assignment before the next
append query. The scheduler unit test covers this bookkeeping boundary.

Generic benchmark fusion is skipped for this topology because generic node
scheduling cannot represent its derived domains and would benchmark a different
kernel.

## Codegen

At codegen,
[NestedReduction.plan_from_topology()](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/scheduler.py:1457)
builds the final `StagedReductionPlan` after loop merging.
[_codegen_nested_reduction()](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:3116)
emits:

```text
outer reduction loop
grouped reduction stage
reduced/full-resolution pointwise stages
SUB_PARENT epilogue stage
```

Both derived families may request the same grouped-axis named constants.
Triton records constants at kernel-function scope and emits each expression
once.

The stage plan carries parent-resolution source layouts and reduced values by
their node-local temporal buffer names. Codegen materializes those names once
after their producers emit, then uses the ordinary index-dependent pointwise
handler. Parent-stage pointwise sources are therefore supported; an unrelated
lane-resolution input remains a normal indexed load.

The outer, grouped, and sub-parent stages share one kernel context, so
kernel-local buffer removal happens only after all stages emit. Tiling still
uses the grid-owning outer schedule; index-width analysis uses a second
schedule containing every emitted stage.

Parent values are made available through the same
[_SubParentSourceLoadResolver](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:2333)
used by the standalone path. The first epilogue load materializes a live value
at sub-parent resolution. A value written by this kernel must forward or
codegen fails; an unavailable external input falls back to its ordinary
derived-index load. There is no persistent-only materializer.

## Current eager projection

This isolated commit intentionally keeps projection simple. When a sub-parent
load first requests a planned source,
[_SubParentSourceLoadResolver.materialize()](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/torch/_inductor/codegen/simd.py:2369)
materializes that source if it is still CSE-live:

- parent-resolution values are split into factor-2 lane values;
- group/reduced values are reshaped and broadcast to the sub-parent lane
  width; and
- unrelated external inputs remain ordinary derived-index loads.

Each spatial value receives the active sub-parent family's shape-compatible
`mask_vars` at this boundary. Thus a later indirect access inherits the
derived tail predicate through ordinary CSE propagation, just as it would from
a normal Triton load. This covers split, broadcast, and the `G=2` path where a
value is already at child width.

Consequently an inlined scale chain may be recomputed after its group value is
broadcast. That is a performance limitation, not a legality requirement. The
current landing scope accepts that cost and stops after #191775.

## Tests

Coverage is in
[test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py)
and
[test_inductor_scheduler.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_inductor_scheduler.py).
This commit covers:

- RMSNorm -> NVFP4 and MXFP4 end to end
- dynamic batch and dynamic reduction shapes with multiple runtime values
- forced looped and persistent kernel forms
- full fusion-gate rejection (`max_fusion_size`, `no_fuse_buffer_names`)
- mempool propagation for repeated grouped-stage appends
- shifted, ambiguous, conflicting, and non-leaf source/output rejection
- inlined parent-full sources and unrelated indexed sub-parent inputs
- reduced-only G=2 classification and append-order independence
- stable staged-kernel selection under `triton.multi_kernel`
- a planned source followed by an observable in-place mutation
- producer-first reverse discovery

The dynamic-reduction test runs one compiled graph at `D=4096` and `D=4608`
and requires one fused kernel. Because the base suite is inherited by both
forced-persistent and forced-looped classes, dynamic/static and
persistent/looped combinations are all exercised. Divisibility of the derived
factor-2 extent is proved symbolically; the parent reduction extent itself does
not need to be a compile-time constant or a power of two.

## Rebase resolution

The commit retains node-local temporal dependency names rather than applying
the scheduler-global final mutation map. Parent-stage pointwise sources now use
the same normalized source-layout proof as standalone planning. The earlier
temporary `TODO(#190595)` restriction on those sources is resolved here, not
left as an upper-stack assumption.

Current isolated verification with the uncommitted review/removal diff is
296 passed / 1 skipped in
[test_nested_reduction.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_nested_reduction.py)
and 60 passed / 6 skipped in
[test_inductor_scheduler.py](/data/users/eellison/pytorch/agent_space/pr190594_ci_fix/test/inductor/test_inductor_scheduler.py).
`git diff --check`, `python -m py_compile`, and `with-proxy spin quicklint`
are also clean.

## Changed later

#191775 removes the stored factor-2 `sub_parent_domain` from
`PointwiseDomainContext`. It keeps the grouped-axis geometry and derives each
candidate's `(factor, output_lanes)` with `_nested_sub_parent_rate()` instead.
The isolated #190595 test correctly pins `(3, 6, 8)` here; #191775 migrates it
to assert rate `(2, 1)` for the same R-grouped geometry and rejection for an
X-grouped geometry.

It also widens nested INTERLEAVED rates through factor 4, including the tested
MXFP6 `(4, 3)` rate. CONTIGUOUS nested append remains unsupported. The current
landing scope stops after #191775.

## Remaining reservation

At this isolated commit nested append is factor-2 INTERLEAVED-only. At the stack
tip it supports interleaved factors through 4 and the `(4, 3)` MXFP6 rate;
CONTIGUOUS append remains unsupported. A realized parent-full source still
declines, while the tests here deliberately cover the inlined case.
