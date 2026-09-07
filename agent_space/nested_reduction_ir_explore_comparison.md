# Nested Reduction IR Explore Comparison

Worktrees:

- Full feature branch:
  `/data/users/eellison/pytorch_nested_reduction_single_commit_wip`
  `03baa04b98436b1400dd5d3593eb93dd9cb5c6c3`
- Reduced landing branch:
  `/data/users/eellison/pytorch_nested_reduction_core_landable`
  `81f3811ba73`
- IR/design exploration branch:
  `/data/users/eellison/pytorch_nested_reduction_ir_explore`

## Goal

Explore a cleaner layering for nested reduction without trying to add a true
IR node in the same branch.

The idea is to move grouped-reduction semantics upward into scheduler-owned
metadata, while keeping derived iteration families in codegen for remapped
consumer execution.

## What Changed In The Exploration Branch

Added in `torch/_inductor/scheduler.py`:

- `BlockLocalReductionSpec`
- `NestedReductionPlan`
- `FusedNestedReductions.plan`

The scheduler now computes and stores:

- the grouped reduction subnode
- grouped reduction output name
- group size
- `small_dim_in_r`
- shared reads
- whether the pattern requires persistent reduction
- which node2 pointwise epilogues run at reduced-output resolution
- which node2 pointwise epilogues run at full resolution
- which downstream half-resolution consumers are fused
- which node1 outputs stay purely internal to the fused kernel
- which grouped-reduction outputs stay purely internal after fused consumers

This means `simd.py` no longer has to:

- split `node2` into reduction vs epilogues
- re-derive `shared_reads`
- re-derive `group_size`
- re-classify pointwise epilogues as reduced-output vs full-resolution
- re-predict persistent reduction for half-resolution legality
- run the early/late half-resolution consumer discovery passes
- rediscover which node1 buffers become internal-only
- rediscover whether the grouped reduction output remains internal

## What Stayed In Codegen

Still in `torch/_inductor/codegen/simd.py`:

- `_DerivedIterationFamily`
- `DerivedIterationRangesRoot`
- `_GroupReductionLayout`
- `_GroupedReductionOpsHandler`
- `_PointwiseRemapHandler`
- `codegen_nested_reduction`
- half-resolution/NVFP4 codegen

One deliberate compromise remains:

- the scheduler-owned plan stores half-resolution consumer *names*, and
  codegen resolves them through `scheduler.name_to_fused_node` at emission
  time. This keeps the plan stable if downstream pointwise nodes fuse again
  after the plan is built.

This is intentional.

The exploration branch does **not** try to:

- add a true `BlockLocalReduction` IR node
- remove `codegen_nested_reduction`
- make reduction codegen itself family-aware
- move kernel-feature sizing / profitability into the scheduler

## Why This Direction Is Useful

It separates two concerns more clearly:

1. Grouped reduction semantics
   This is closer to IR/scheduler work.

2. Consumer-side remapped execution
   This is codegen work, and `_DerivedIterationFamily` is still the right
   abstraction for it.

The branch is therefore a concrete test of the claim:

> Nested reduction feels heavy in `simd.py` because `simd.py` is carrying both
> grouped-reduction semantics and remapped-consumer execution.

This branch moves only the first part upward.

## What It Buys

Qualitatively:

- `codegen_nested_reduction` reads more like “execute the plan” than
  “recognize the pattern again.”
- `FusedNestedReductions` becomes a better semantic owner for the fused pattern.
- The grouped second reduction starts to look more like an operation with a
  plan, not just a special codegen path.

Quantitatively, relative to `03baa04...`:

- `scheduler.py`: `+237` lines
- `simd.py`: `-289` lines

Net: `+274 / -289` across those two files, with the main orchestration path
shrinking in `simd.py` and the semantic plan moving upward into `scheduler.py`.

The biggest concrete reductions in `simd.py` are:

- `codegen_nested_reduction`: `229` lines → `177`
- codegen-side half-resolution discovery helpers removed entirely
- internal-buffer ownership checks removed from codegen

## What It Does Not Solve

This branch does **not** eliminate the need for consumer-side derived families.

Even with a future IR-level `BlockLocalReduction`, codegen would still need:

- reduced-output consumer execution
- full-resolution consumer execution with broadcast lift
- half-resolution consumer execution with split/broadcast behavior

So the likely long-term architecture is:

- IR/scheduler owns grouped reduction semantics
- codegen owns remapped iteration families

not:

- IR replaces `_DerivedIterationFamily`

## Validation

On `/data/users/eellison/pytorch_nested_reduction_ir_explore`:

- `python -m py_compile torch/_inductor/scheduler.py torch/_inductor/codegen/simd.py`
- `python test/inductor/test_nested_reduction.py`

Both passed.

## What This Branch Now Validates

This branch is now a stronger architectural comparison point than the original
spike:

- grouped-reduction semantics are scheduler-owned
- half-resolution consumer membership is scheduler-owned
- internal-buffer ownership decisions are scheduler-owned
- persistent-reduction requirement is scheduler-owned
- codegen mostly consumes the plan and executes families

That still stops short of a true IR node, but it is enough to answer the
layering question with code instead of only with design prose.

## Likely Next Step If We Keep Pushing This Branch

The next logical extension would be to replace the plan dataclasses with a
real IR-level grouped/block-local reduction node, then let the scheduler tag
fused pointwise consumers with the family they should activate.

That is a separate architecture project. This branch is already far enough to
compare with the current implementation and decide whether the layering shift
is worth pursuing.

## Current Recommendation

Use this branch as the comparison point for the architectural question:

- “What if grouped-reduction semantics were scheduler-owned?”

Do **not** confuse it with the reduced landing branch:

- the reduced landing branch is about reviewability and scope
- this branch is about architecture and layer placement
