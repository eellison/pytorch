# Nested Reduction: IR Uplift

## Goal

The non-local-optimum doc generalizes the **consumer-side** abstraction:
derived iteration families become a first-class scheduler/codegen concept
reused across patterns.

This doc covers an orthogonal axis: **lifting nested-reduction work out of
codegen into Inductor IR**. The two axes compose — together they describe
what a fully landed redesign looks like.

The codegen-only architecture works but accumulates feature-specific
machinery in `simd.py`. Each new fusion pattern pays the same tax. Lifting
structurally similar work to IR distributes the cost across all patterns
instead of repeating it per pattern.


## Core Premise

Inductor's stack has four layers where fusion-pattern work *can* live:

1. Inductor IR (`ir.py`)
2. Scheduler (fusion decisions, node grouping)
3. Backend-shared codegen (`simd.py`)
4. Backend-specific emission (`triton.py`)

Today's nested-reduction codebase puts most of its work in layer 3, with
slivers in layer 2. This doc proposes moving structurally appropriate
pieces up to layers 1 and 2.


## What this enables that codegen-only cannot

- Pattern recognition runs once, at graph-rewrite time, not at fusion time.
- Reductions go through the standard reduction codegen path, no special handler.
- Scheduler reasons over iteration families with full layer-1 context.
- Future fusion patterns (cat/scatter/MLA) reuse the IR-level scaffolding
  instead of reinventing it.


## Migration Paths

### Path A: `BlockLocalReduction` as an IR primitive

Add a new Inductor IR node expressing "reduce over an inner axis after a
reshape." Carries layout metadata: `group_size`, `parent_axis`,
`small_dim_in_r`.

Pattern recognition (today: `NestedReduction.can_fuse`) becomes a graph
rewrite: `Reduction(reshape(x, B, G, g))` → `BlockLocalReduction(x,
group_size=g, ...)`.

Lowering: standard reduction codegen handles the reshape internally. No
`_GroupedReductionOpsHandler`.

**What dissolves:**
- `_GroupReductionLayout` becomes IR metadata
- `_GroupedReductionOpsHandler.reduction()` becomes standard reduction emit
- `NestedReduction.can_fuse` graduates to graph rewrite

**What stays:**
- `_DerivedIterationFamily` (still needed for consumer-side remapping)
- Half-resolution discovery (still codegen-time)
- The `codegen_nested_reduction` orchestrator (now thinner — no special
  reduction handling, but still drives consumer activation)

**Hard parts:**
- Where in the lowering pipeline to recognize the pattern (FX rewrite vs.
  post-decomposition)
- Backend dispatch: every reduction codegen needs to know about
  `BlockLocalReduction`, or lower it to standard `Reduction` with metadata
- Dynamic shapes: layout fields may be symbolic; existing fusion-time
  recognition handles this naturally, IR-time may need additional care


### Path B: Scheduler-owned iteration families

This is path-3 from `nested_reduction_non_local_optimum.md`, but enabled by
Path A.

After Path A, the scheduler sees `BlockLocalReduction` nodes natively.
Standard fusion rules apply. The scheduler can additionally **annotate**
each fused consumer with the iteration family it activates: reduced-output,
full-resolution, or half-resolution.

Codegen reads the annotation, activates the family, runs the body. No
re-recognition.

**What dissolves:**
- Early/late half-resolution dual pass (collapses to one decision at fusion
  time)
- Re-recognition of internal node1 outputs in codegen
- The `RuntimeError` agreement check

**What stays:**
- Codegen still owns family activation (the `with family.activate(kernel)`
  machinery)
- `_DerivedIterationFamily` survives as the codegen primitive
- Half-resolution lane-validity check moves to scheduler but the predicate
  is the same

**Hard parts:**
- Family choice depends on layer-3 facts (block sizes, persistence vs
  loop). Layer 2 has to learn enough about layer 3 to make these decisions,
  or layer 3 has to provide them as inputs to layer 2.
- Scheduler-codegen interface needs a clean shape for "tagged fusion plan."


### Path C: Cross-stage register state in IR

This is the speculative one — and it might be wrong.

Today, producer→consumer register-resident value plumbing is implicit:
`cse.store_cache` for stored values, `_node1_cse_vars` snapshot,
`_stage_load_values` for loads. The plumbing works but it's stateful and
hard to follow.

Path C asks: should "this value flows from producer to consumer in
registers, possibly at a different shape" be expressible in IR?

Two sub-options:

**C1: Explicit "register-resident" edges in scheduler IR.**
Scheduler node carries a list of `(producer_buffer, consumer_buffer,
transformation)` tuples. Codegen consumes these declaratively.

**C2: Don't lift it.**
Cross-stage register state is genuinely codegen-shaped. Layers 1 and 2 are
the wrong home. Accept that codegen owns this; just clean up the
bookkeeping.

Lean toward **C2**. Register-resident value flow is a runtime/codegen
concept; pulling it into IR means inventing primitives that don't otherwise
exist. The current `store_cache` mechanism is fine; the only issue is
naming and discoverability, not architectural placement.


### Path D: Generalize beyond reductions

After A + B, the infrastructure (`_DerivedIterationFamily`, scheduler
family annotation, codegen activation) is generic enough to express any
"produce in shape A, consume in shape B within one kernel" pattern.

Cat fusion, scatter fusion, MLA-style heterogeneous fusion all use the same
scheduler-annotated-family machinery. The IR primitives are different
(`BlockLocalReduction` becomes one of N "shape-shifting producers"), but
the consumer-side and orchestration are shared.

This is path-5 from `nested_reduction_non_local_optimum.md`, achieved via
IR.

**Hard parts:**
- Each new producer pattern likely needs its own IR primitive.
- Consumer-side abstraction needs to grow to cover affine masked subregions
  (path-4 from the other doc).
- The "iteration family" abstraction itself may need to expand from
  "factorized split" to "arbitrary subregion."


## Combined Roadmap

The two docs interleave:

| Step | Source | What |
|------|--------|------|
| 0 | current branch | Local optimum, codegen-owned |
| 1 | non_local doc Path 2 | Unified `_DerivedIterationFamily` *(done in current branch)* |
| 2 | non_local doc Path 3 | Scheduler-owned families (= IR Path B) |
| 3 | IR doc Path A | `BlockLocalReduction` IR primitive |
| 4 | non_local doc Path 4 | Affine masked subregions in family model |
| 5 | IR doc Path D + non_local doc Path 5 | Generalize across fusion patterns |

Path B and Path A can land in either order. Doing B first
(scheduler-owned families) is incremental and lower-risk. Doing A first
(IR primitive) is higher-leverage but requires more cross-cutting Inductor
changes.


## Non-Goals

- Eliminate `_DerivedIterationFamily`. It's the right shape at layer 3,
  regardless of what moves to layers 1-2.
- Push register-state plumbing into IR (Path C1). Probably not worth it.
- Build all of this before shipping the current branch. The current branch
  is the substrate; uplift comes after.


## Open Questions

1. **Where does graph rewrite for `BlockLocalReduction` recognition live?**
   Pre-Inductor (FX) or post-decomposition (Inductor IR)?
2. **Does scheduler need a structured "fusion plan" output**, or can
   iteration-family annotation piggyback on existing scheduler-node
   attributes?
3. **What's the right answer for Path C?** Lean C2, but a concrete
   example where register-state plumbing causes review pain would be
   evidence for C1.
4. **Does dynamic-shape support need anything beyond the existing
   range-tree machinery** when families are scheduler-owned? The current
   branch's `is_loop=parent.is_loop` fix suggests there are subtleties.


## Summary

The IR-uplift axis says: *operation semantics belong in IR, fusion
decisions belong in the scheduler, body emission belongs in codegen*. The
current branch puts work in all three layers because layer 1 doesn't have
the primitives yet. Adding those primitives — starting with
`BlockLocalReduction` — distributes the work appropriately and makes future
fusion patterns cheaper.

This doc is a roadmap, not a plan-of-record. It's the architectural
retrospective on the current branch; whether to act on it is a separate
(and longer-horizon) decision.
