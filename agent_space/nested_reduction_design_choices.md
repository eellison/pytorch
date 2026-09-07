# Nested Reduction Design Choices

This note is meant to help a reader who is **not already familiar** with the
nested-reduction work understand:

- what problem the feature is solving
- why the current implementation feels large
- what concrete implementation directions we already have in code
- what each design choice buys, and what it does not buy
- which direction is best for a first landing branch versus later architecture
  work

This is not a purely hypothetical architecture note. It compares **actual
implementation directions** we already explored in separate worktrees.

This note intentionally excludes affine masked subregions. The focus here is
the current nested-reduction family of workloads:

- dependent cross-axis reductions
- reduced-output consumers
- full-resolution consumers
- half-resolution / NVFP4-style consumers


## 1. Short Version

There are really two separate problems in this feature:

1. **Grouped / block-local reduction semantics**
   Example: "take the output tile of one reduction, reinterpret one axis in
   groups of 16, and reduce within each group."

2. **Consumer-side iteration-space remapping**
   Example: "this epilogue runs at grouped-output resolution," or "this
   consumer runs at full resolution and must broadcast grouped values," or
   "this half-resolution consumer needs split lanes."

The current full implementation solves both problems mostly in `simd.py`.
That is why it feels large.

The most important architectural lesson from the work so far is:

- **derived iteration families are the right codegen abstraction**
- the open design question is **where grouped/block-local reduction semantics
  should live**

That gives us a menu of design choices:

- keep everything codegen-centric
- move semantics into a scheduler-owned plan
- make the plan more explicit/staged
- eventually make block-local reduction a first-class IR/scheduler concept
- much later, eliminate the special nested-reduction codegen entry point


## 2. Problem Statement

Nested reduction is the pattern where one reduction is followed by another,
smaller grouped reduction that depends on values produced by the first stage,
and then optional pointwise consumers run at one of several resolutions.

The canonical shape is:

1. run an **outer reduction** over a large tile
2. reinterpret one axis of that tile as grouped structure
3. run a **grouped / block-local reduction**
4. optionally run pointwise consumers at:
   - grouped output resolution
   - full original resolution
   - half resolution of the grouped axis

Examples:

- `rms_norm -> grouped amax`
- `rms_norm -> grouped fp8 quantization`
- `rms_norm -> grouped amax -> NVFP4 pass-3 packing`

The performance goal is simple:

- read the shared large input once
- keep as much as possible in registers
- avoid splitting the computation into several kernels


## 3. A Concrete Worked Example

One representative shape is:

```python
x = rms_norm(x, weight)                 # outer reduction
amax = grouped_amax(x, group_size=16)  # grouped reduction
scale = amax_to_fp8_scale(amax)         # reduced-output consumer
q = quantize_fullres(x, scale)          # full-resolution consumer
packed = pack_pairs(q)                  # half-resolution consumer (NVFP4)
```

What makes this hard is that each stage wants a different logical iteration
space:

- `rms_norm` wants the outer tile space
- `grouped_amax` wants one value per group
- `quantize_fullres` wants the original tile resolution again
- `pack_pairs` wants a smaller derived space (for example, pairwise lanes)

So this is not just "fuse another pointwise op." It is:

- recognize a multi-stage reduction pattern
- keep the right values internal
- run several logical stages inside one kernel
- switch iteration space between stages without losing correctness


## 4. Why This Feels "Bonkers"

The feature is difficult because the compiler has several layers where this
logic could live:

1. **IR / operation semantics**
   What the program *is*.

2. **Scheduler / fusion planning**
   Which nodes fuse, in what groups, and with what shared constraints.

3. **Backend-shared codegen (`simd.py`)**
   Range trees, indexing, kernel staging, generic codegen decisions.

4. **Backend-specific emission (`triton.py`)**
   Actual Triton load/store/reshape/broadcast/split emission.

Today, the current full implementation puts a lot of layer-1 and layer-2 work
into layer-3:

- `_GroupReductionLayout` is partly describing the operation itself
- codegen re-classifies consumers by resolution
- codegen decides which buffers are internal
- codegen predicts persistent-reduction requirements
- codegen orchestrates stage order explicitly

That is why the branch can be correct and still feel too large.


## 5. Useful Terms

These terms appear in the code and in the options below.

### Outer reduction

The first, larger reduction, called `node1` in the implementation.

### Grouped / block-local reduction

The second reduction, called `node2`'s reduction subnode in the current code.
This is the stage that reduces over `group_size`.

This note uses **block-local reduction** and **grouped reduction**
interchangeably.

### Reduced-output consumer

A pointwise consumer that runs at one value per group.

### Full-resolution consumer

A pointwise consumer that runs at the original tile resolution and reads grouped
values via broadcast/lift.

### Half-resolution consumer

A pointwise consumer that runs at a smaller derived resolution, currently used
for NVFP4-style pairwise pack paths.

### Derived iteration family

A codegen abstraction representing "run this body in a derived iteration
space," including:

- active range trees
- index substitutions
- flat-index remaps
- optional register-backed value sources

This is the `_DerivedIterationFamily` concept in the current code.


## 6. What Exists Today

We already have three concrete branches/worktrees that correspond to different
points in the design space.

### Full saved feature branch

- worktree: `/data/users/eellison/pytorch_nested_reduction_single_commit_wip`
- commit: `03baa04b98436b1400dd5d3593eb93dd9cb5c6c3`

This is the full all-in implementation, including half-resolution / NVFP4 work.

### Reduced-scope landable branch

- worktree: `/data/users/eellison/pytorch_nested_reduction_core_landable`
- commit: `81f3811ba733d08494f84a2ad35704a1432c32b4`

This trims the half-resolution / NVFP4 path so that the first landing branch is
smaller and easier to review.

### Architecture exploration branch

- worktree: `/data/users/eellison/pytorch_nested_reduction_ir_explore`
- branch: `nested_reduction_path_b_plus_candidate`

This is the branch where grouped-reduction semantics have been pushed upward
into a scheduler-owned plan while derived iteration families remain codegen
objects.


## 7. Evaluation Criteria

When comparing designs, these are the questions that matter most.

### A. Correctness

- Are legality decisions made exactly once, or rediscovered in several places?
- Can dynamic shapes and degenerate shapes like `B=1` still work?
- Are internal buffers and external outputs handled consistently?

### B. Layering

- Does scheduler own semantic decisions?
- Does codegen only do emission/materialization work?
- How much operation semantics leak into `simd.py`?

### C. Reviewability

- Can the design be explained without reading 400 lines of `simd.py` first?
- Are the staged pieces visible in data, or only implicit in control flow?

### D. Locality of change

- Does the design require a new IR node or compiler-wide abstraction?
- Or can it be contained to nested reduction plus its scheduler/codegen path?

### E. Future extensibility

- Does the design make reduced/full/half-resolution consumers cleaner?
- Does it help future grouped/blockwise kernels?
- Does it reduce the chance that every similar feature adds another custom
  path?


## 8. Choice A: Codegen-Centric Nested Reduction

This is the original full implementation shape.

### Semantic center

- `SIMDScheduling.codegen_nested_reduction()` in `simd.py`

### Ownership

- scheduler:
  - recognizes that nested reduction is legal
  - builds `FusedNestedReductions`
- codegen:
  - reclassifies grouped reduction vs epilogues
  - reclassifies reduced-output vs full-resolution consumers
  - discovers half-resolution consumers
  - decides which buffers are internal
  - decides persistent-reduction requirement
  - materializes derived iteration families
  - orchestrates stage order

### What it looks like in practice

This is the version where `simd.py` carries most of the complexity:

- `_GroupReductionLayout` is large because it mixes:
  - operation semantics
  - shape math
  - family materialization
  - split/broadcast helpers
- `codegen_nested_reduction()` is large because it does:
  - planning
  - orchestration
  - codegen

### What is good about it

- minimal disturbance to the rest of Inductor
- the feature can be shipped without inventing a new compiler abstraction
- it lets the implementation teach us what a better abstraction would need

### What is bad about it

- too much semantic work lives in `simd.py`
- codegen has to rediscover decisions that logically belong to scheduler
- the special entry point grows large because it is doing pattern recognition,
  planning, and execution
- half-resolution / NVFP4 makes the branch substantially harder to review

### When to choose it

- when the primary goal is shipping the feature quickly inside current
  architecture


## 9. Choice A-Prime: Reduced-Scope Landable Branch

This is not a different architecture. It is Choice A with less surface area.

### Semantic center

- still `codegen_nested_reduction()` in `simd.py`

### Scope

- drop half-resolution / NVFP4 from the first landing branch
- keep:
  - dependent grouped reduction
  - reduced-output consumers
  - full-resolution consumers

### Why this is useful

This option is about **review strategy**, not architecture. It accepts that the
core design is still codegen-centric, but reduces the amount of specialized
logic that must be reviewed in the first PR.

### Pros

- materially easier to review
- preserves the good family abstraction without forcing the most specialized
  path through the first PR

### Cons

- does not solve the layering issue
- mostly a reviewability strategy, not an architecture improvement

### When to choose it

- when the main concern is landability rather than architecture


## 10. Choice B: Scheduler-Owned Semantic Plan, Codegen-Owned Families

This is the current architecture exploration branch.

### Semantic center

- `NestedReductionPlan` on `FusedNestedReductions` in `scheduler.py`

### Ownership

- scheduler owns:
  - grouped reduction semantics
  - reduced-output vs full-resolution consumer partitioning
  - half-resolution consumer membership
  - internal `node1` outputs
  - internal grouped-reduction outputs
  - persistent-reduction requirement
- codegen owns:
  - `DerivedIterationRangesRoot`
  - `_DerivedIterationFamily`
  - family materialization from the scheduler plan
  - staged execution
  - split / broadcast / reshape emission

### What changed relative to Choice A

- `simd.py` no longer re-discovers half-resolution consumers
- `simd.py` no longer re-discovers internal buffer ownership
- `simd.py` no longer re-predicts persistent reduction
- `codegen_nested_reduction()` is reduced to "consume plan, materialize
  families, execute stages"

### One deliberate compromise

The scheduler-owned plan stores half-resolution consumer **names**, not
concrete node objects. Codegen resolves them through
`scheduler.name_to_fused_node` at emission time.

That is intentional. It keeps the plan stable even if downstream pointwise
nodes fuse again after the plan is built.

### What this does **not** change

This option does **not** introduce a new IR node. It also does **not** remove
the special nested-reduction codegen entry point. It is a layering cleanup
inside the current architecture, not a full rewrite.

### Pros

- better layering without a full IR rewrite
- makes scheduler the source of truth for semantics
- keeps family mechanics where they belong: codegen
- gives a real, testable alternative to the codegen-centric design

### Cons

- still has a dedicated `codegen_nested_reduction()` entry point
- still leaves `_GroupReductionLayout` fairly large in `simd.py`
- the plan is a scheduler/codegen convention, not a true IR object

### When to choose it

- when the goal is to improve architecture meaningfully without rewriting
  Inductor around a new IR node


## 11. Choice C: Nested-Reduction-Only Staged Plan

This is a plausible next step from Choice B if we want a cleaner shape with
minimal churn.

### Idea

- keep the change local to `FusedNestedReductions`
- make the scheduler plan more explicit about stages
- avoid introducing a generic staged-kernel abstraction for all of Inductor

### Sketch

```python
@dataclass(frozen=True)
class NestedReductionConsumerSpec:
    kind: Literal["reduced_output", "full_resolution", "half_resolution"]
    node_names: tuple[str, ...]
    factor: int | None = None


@dataclass(frozen=True)
class NestedReductionPlan:
    block_local_reduction: BlockLocalReductionSpec
    consumers: tuple[NestedReductionConsumerSpec, ...]
    internal_node1_outputs: tuple[str, ...]
    internal_late_outputs: tuple[str, ...]
```

### What improves relative to Choice B

- the plan stops being a bag of parallel fields
- reduced/full/half-resolution become explicit stage specs
- codegen can iterate plan stages rather than juggling several lists

### What stays the same

- no true IR node
- no generic staged-kernel framework
- families are still materialized in codegen

### Why someone might dislike this option

If the complaint is "I do not like codegen carrying around plan-like data
structures that still feel ad hoc," this option may not be satisfying enough.
It cleans the shape up, but it still leaves nested reduction as a
nested-reduction-specific plan format rather than a real compiler-wide concept.

### Pros

- probably the best minimal-change cleanup beyond Choice B
- makes the stage structure explicit without rocking the rest of the compiler

### Cons

- still a nested-reduction-specific abstraction
- still not enough to eliminate the special codegen entry point

### When to choose it

- when we want a cleaner nested-reduction-only design without committing to a
  larger architecture project


## 12. Choice D: Explicit BlockLocalReduction Above Codegen

This is the next real architectural step if the layering win is worth paying
for.

### Semantic center

- explicit `BlockLocalReduction` concept above codegen
- this could begin as a stronger scheduler-owned object, or as a true IR node

### What moves up

- grouped reduction semantics
- grouped axis / group size
- legality for reduced/full/half-resolution consumers
- consumer-family assignment
- internal buffer ownership

### What stays down

- `DerivedIterationRangesRoot`
- `_DerivedIterationFamily`
- activation of families around bodies
- split / broadcast / reshape mechanics

### Why this matters

Right now a large part of `_GroupReductionLayout` is really describing the
operation, not the emission strategy. Making block-local reduction explicit
above codegen would reduce that semantic load in `simd.py`.

### Important nuance

This option still does **not** remove the need for derived iteration families.
Even if grouped reduction becomes explicit in IR or scheduler, consumers still
need to run in different logical spaces:

- reduced-output space
- full-resolution lifted space
- half-resolution derived space

So this option helps with the **producer-side** semantics. It does not remove
the **consumer-side** remapping problem.

### Pros

- makes grouped reduction an explicit operation
- reduces the semantic role of `_GroupReductionLayout`
- pushes nested-reduction logic toward normal scheduling instead of special
  codegen inference

### Cons

- larger change than Choice B or C
- still does not by itself remove the need for remapped iteration families

### When to choose it

- when we want the compiler to understand grouped reduction as a first-class
  operation, not just as codegen convention


## 13. Choice E: Composite IR Node / No Special `codegen_nested_reduction()`

This is the end state if the goal is to eliminate the separate
`codegen_nested_reduction()` path.

### Semantic center

- one explicit composite staged node

### Sketch

```python
NestedReductionNode(
    outer_reduction=...,
    block_local_reduction=...,
    consumer_stages=[
        ConsumerStage(kind="reduced_output", ...),
        ConsumerStage(kind="full_resolution", ...),
        ConsumerStage(kind="half_resolution", factor=2, ...),
    ],
)
```

Then standard staged codegen would do:

1. build kernel
2. run outer reduction stage
3. run block-local reduction stage
4. run consumer stages by activating the appropriate family
5. finalize outputs

### Why this is the real end state

The biggest long-term win is not just "less code in `simd.py`." It is removing
the need for a nested-reduction-specific codegen entry point entirely.

### Why it is not as different as it sounds

This option is **not** a radically different execution model. In a sense, the
current explore branch is already a shadow version of this design:

- it has a plan
- it has stage buckets
- it has consumer-family assignment
- it has a semantic center above codegen

What this option adds is:

- explicit ownership in one composite compiler object
- no re-resolution from existing separate nodes at codegen time
- no bespoke nested-reduction codegen entry point

### Pros

- no bespoke nested-reduction codegen entry point
- stage ordering becomes explicit compiler data
- best long-term architecture

### Cons

- this is a real Inductor architecture project
- it wants a staged-kernel abstraction, not just local cleanup
- much larger scope than the current feature work

### When to choose it

- when we are ready to invest in a multi-PR architecture effort, not just
  feature implementation


## 14. What Survives Across Almost Every Choice

One important insight from the exploration so far is that some of the current
work is probably "the right permanent shape," regardless of which higher-level
design we choose.

Most likely survivors:

- `DerivedIterationRangesRoot`
- `_DerivedIterationFamily`
- activation of families around consumer bodies
- split / broadcast / reshape mechanics for materializing values at different
  resolutions

In other words: even if we move grouped reduction semantics out of `simd.py`,
the codegen abstraction for "run this body in a remapped iteration family" is
still useful.


## 15. What Dissolves in the More Architectural Choices

If we move toward Choices D or E, these are the pieces that should shrink or
disappear:

- much of `_GroupReductionLayout`'s semantic role
- codegen-side re-recognition of consumer categories
- codegen-side internal-buffer ownership rediscovery
- codegen-side persistent-reduction prediction
- eventually the special `codegen_nested_reduction()` entry point itself


## 16. Recommendation by Goal

### If the goal is: "land something soon"

Choose **Choice A-prime**.

That means:

- land the reduced-scope core branch first
- keep half-resolution / NVFP4 for a follow-up PR

### If the goal is: "improve architecture without rewriting Inductor"

Choose **Choice B**, and possibly evolve toward **Choice C**.

That means:

- scheduler owns the nested-reduction semantic plan
- codegen owns family mechanics
- no new IR node yet

### If the goal is: "fix the layering issue at the root"

Move toward **Choice D**, and eventually **Choice E**.

That means:

- make block-local reduction explicit above codegen
- eventually represent the fused multi-stage structure explicitly enough that
  no separate `codegen_nested_reduction()` path is needed


## 17. My Current Read

For the current project, the most realistic interpretation is:

- **A-prime** is the best first landing strategy
- **B** is the best architectural exploration branch we have today
- **C** is a plausible cleanup if we want a more explicit plan shape without
  a new compiler abstraction
- **D/E** are real future directions, but they are architecture work, not
  "just one more cleanup pass"

The most important conclusion is:

- we should not throw away the full implementation, because it taught us what
  semantics and codegen machinery are actually needed
- but we also should not pretend that all of that complexity belongs in the
  first landable PR

That is why keeping:

- the full saved feature branch
- the reduced landable branch
- the architecture exploration branch

in parallel is the right strategy.


## 18. Follow-Up Design Questions

If we continue the architecture exploration later, the most useful questions
are:

1. Should `BlockLocalReduction` become a true IR node, or remain a
   scheduler-owned semantic object for a while?
2. Is Choice C actually worthwhile, or is it just an intermediate stop between
   B and D?
3. What is the smallest staged-kernel abstraction that would let nested
   reduction stop having a special codegen entry point?
4. How should producer-side grouped-reduction semantics and consumer-side
   family activation meet cleanly, without over-generalizing to a giant
   "iteration context" abstraction too early?

Those are good next-project questions. They are not required to land the
current feature work.
