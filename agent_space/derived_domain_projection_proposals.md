# Derived-Domain Projection Proposals

Date: 2026-08-21

This is a scratch design note for refinement. It is not part of the PR stack.

## Status summary (authoritative resolution at end of file)

| proposal | verdict | trigger / gate |
|---|---|---|
| A: delayed group projection | **removed from #190595** | superseded by the two uncommitted follow-ups |
| B: user-side divide-first spelling | **rejected** (measured: byte-identical kernels) | -- |
| C: parent-width divide-then-split | **implemented in uncommitted projection follow-up** | persistent factor-2; 1.31x at 4096x4096 |
| D: structured logical values | **implemented for the audited derived domain** | parent `[X,groups,G]`, group `[X,groups,1]`; not a general tensor-domain algebra |
| E: indexed projection-aware forwarding | **implemented in uncommitted foundation follow-up** | exact normalized access plus guard, behavior-preserving |
| F: unnamed-value identity | only the virtual-source variant (C) for now | full version only on a second producer-fork pattern |
| G: ordered domain stages | deferred | a concrete pipeline the fixed order cannot express |
| packed-store coalescing | **rejected** (already v4 stores) | -- |

Author requirements for the eventual mechanism (R1-R5, deferral-to-E): see Iteration 8.

## Problem statement

The nested NVFP4 path has three logical resolutions:

```text
parent:       [B, D]
group:        [B, D / G]
sub-parent:   [B, D / G, G / 2]
```

The sub-parent domain is flattened for codegen:

```text
[B, D / G, G / 2] -> [B, D / 2]

flat_child = group * (G / 2) + pair
group      = flat_child // (G / 2)
pair       = flat_child %  (G / 2)
parent_r   = group * G + pair * 2 + lane
```

The reduced scale is naturally `[B, D / G]`. Packing combines that scale with
lane-varying values in `[B, D / G, G / 2]`.

The current resolver can forward named buffers across the boundary, but an
inlined chain such as:

```text
amax -> clamp -> fp8 cast -> fp32 cast -> reciprocal
```

has no buffer name. If codegen broadcasts `amax` first, the entire chain is
replayed at lane width and cannot CSE with the group-width chain already
emitted for the scale output.

There are three related but distinct limitations:

1. **Named source lookup:** a buffer name is not a complete load identity.
2. **Unnamed expression identity:** inlined values cannot be directly forwarded
   between separately emitted LoopBodies.
3. **Physical expression placement:** the current derived body selects lanes
   before division, although a persistent kernel could divide the full parent
   tile and split afterward.

Any proposal should state which of these it solves.

## Why the PyTorch spelling does not survive

At the Python level these programs look meaningfully different:

```python
# Lane-first
even = y[..., 0::2] / scale[..., None]
odd = y[..., 1::2] / scale[..., None]

# Divide-first
scaled = y / scale[..., None]
even = scaled[..., 0::2]
odd = scaled[..., 1::2]
```

The current lowering does not preserve `y` or `scaled` as reusable register
values. Fusion inlines the expression into separately emitted LoopBodies. By
the time the sub-parent body is remapped, it effectively contains:

```text
even = normalize(load(x_even), load(weight_even)) / scale
odd  = normalize(load(x_odd),  load(weight_odd))  / scale
```

`_SubParentSourceLoadResolver` can associate a CSE value with a named buffer
load. It cannot associate the unnamed `normalize(x, weight)` expression in one
LoopBody with the corresponding expression in another LoopBody. The derived
emitter therefore splits or reloads named inputs and replays their consumer
expressions at lane width.

This explains two observations:

- Rewriting the Python expression produces byte-identical generated kernels.
- `_GroupInvariantBroadcast` fixes the reduced scale-chain duplication but not
  the parent-width normalized-value duplication. The scale is group-invariant;
  `y` is parent-varying and is requested as two distinct lane projections.

The desired code requires compiler representation, not different user syntax:

```text
y_parent[B, D] = normalize(x, weight)
scaled[B, D]   = y_parent / broadcast(scale[B, D/G, 1])
even, odd      = project_lanes(scaled)
```

Forcing a materialized `y` buffer can create identity, but it defeats the goal
by adding a global-memory store/load. FX origins are also not an acceptable
codegen identity.

## Required invariants

- Do not use FX origins as codegen identity.
- Do not realize an intermediate solely to create identity; that adds a memory
  round trip to a path intended to avoid one.
- CSE lifetime remains authoritative. A loop-local value invalidated after a
  reduction pass must not be forwarded.
- A forwarded load must retain buffer version, normalized index/domain, mask,
  and fill value.
- Projection must preserve the exact rounding points of the original program.
- Unknown or non-scalar operations must see operands at the domain their
  contract requires.
- Persistent and looped behavior may use different physical strategies, but
  both must preserve the same logical stage semantics.

## Proposal A: Delayed group projection in the remap handler

This was the earlier local #190595 approach. It has been removed from the
current review worktree and is retained here only as design history.

Represent a group-width CSE value as group-invariant while ordinary scalar
operations consume only group-invariant inputs. Broadcast it to lane width at
the first lane-varying operation, projection barrier, or store:

```text
scale_g    = fp8(clamp(amax_g / 6))
scale_lane = broadcast(scale_g)
packed     = pack(x_even / scale_lane, x_odd / scale_lane)
```

Implementation seam:

- `_SubParentSourceLoadResolver` supplies the named group-width source.
- `_GroupInvariantBroadcast` delays widening through scalar operations.
- Existing kernel CSE recognizes the replayed group-width scale chain.
- `_PointwiseRemapHandler` widens at lane interaction or an operation whose
  contract is not scalar pointwise.

What it solves:

- Avoids lane-width replay of unnamed scalar computation.
- Requires no new codegen IR and no extra memory realization.

What it does not solve:

- Source lookup remains name-keyed.
- Unnamed identity is recovered by replay plus ordinary CSE, rather than being
  represented explicitly.
- Parent data is still split before the lane divisions.

Tradeoffs:

- Smallest generic change available in the current handler architecture.
- Correctness does not depend on the CSE hit under Triton 3.8; performance does.
- The projection-barrier contract must remain aligned with `OpsHandler`
  semantics.
- Kernel-form tests must pin one FP8 conversion so lost reuse is visible.

Final disposition: do not land this in #190595. The indexed-forwarding and
lazy-projection follow-ups supersede it after #191775.

## Proposal B: Rewrite user code to divide before splitting

Spell the program as:

```python
scaled = groups.float() / scale.float().unsqueeze(-1)
pairs = scaled.view(B, D // G, G // 2, 2)
packed = pack(pairs[..., 0], pairs[..., 1])
```

instead of selecting even and odd lanes before division.

Measured result:

- Inductor generates byte-identical Triton bodies for both spellings in
  persistent and looped modes at `128x4096`, `4096x4096`, and `4096x8192`.
- Therefore the user-level rewrite does not change performance.
- Full results are in
  `agent_space/divide_before_split_bench/REPORT.md`.

Conclusion: reject this as a compiler solution. User syntax alone does not
retain the desired placement through lowering and derived-stage emission.

## Proposal C: Persistent parent-width divide, then split

Teach staged codegen to preserve this physical order for persistent kernels:

```text
parent_value : [B, D]
scale        : [B, D / G, 1]
scaled       = parent_value / scale
pairs        = reshape(scaled, [B, D / G, G / 2, 2])
even, odd    = split(pairs)
packed       = pack(even, odd)
```

This is not merely Proposal B expressed by the user. Codegen must keep the
parent tile live, construct the group-singleton scale view, and perform the
division before projecting into the derived stage.

The original manual comparison combined two changes: it reused the already
computed parent-width normalized value instead of recomputing the normalization
for each lane, and then moved division before the split. The decomposed result
is:

```text
shape       mode        recompute lanes    reuse y, split first    reuse y, divide first
128x4096    persistent       18.304 us            16.256 us                14.240 us
4096x4096   persistent       44.928 us            36.736 us                30.656 us
4096x4096   looped           26.496 us            26.496 us                26.496 us
4096x8192   looped           51.104 us            49.024 us                51.072 us
```

The `1.47x` `4096x4096` headline therefore does not measure split ordering in
isolation. About `1.22x` comes from retaining the parent-width normalized value,
and the remaining lane-first to divide-first comparison is about `1.20x`.
Those effects are not multiplicatively independent, and an independent audit
is in progress. Forced-persistent `D=8192` remains pathological for every form.
Looped kernels show no divide-before-split benefit.

Possible implementation shapes:

1. A focused persistent-only codegen path that recognizes a reusable
   parent-width expression and group-invariant scale before lane selection.
2. A virtual staged value that names the normalized parent expression without
   materializing it, then allows both the block-local reduction and sub-parent
   pack to consume it.
3. The structured derived-value representation in Proposal D, which makes this
   ordering expressible without an NVFP4-specific pattern.

Open questions:

- Can the transformation be expressed generically as postponing a projection
  through scalar binary operations?
- How should it prove that moving the split preserves rounding and masks?
- Is the persistent-only gain worth another mechanism in the landing stack, or
  should this remain a measured follow-up?

Current recommendation: do not block #190595. Record as a persistent-codegen
follow-up, with Proposal D as the preferred general direction.

### Possible focused implementation

A bounded first version could apply only when all of these are true:

- the kernel is persistent, so the parent tile remains CSE-live;
- both lane expressions are projections of the same proved parent expression;
- the operations between that expression and the lane pack are pure scalar
  operations with identical parameters;
- the group-width operand has an exact broadcast mapping to the parent domain;
- masks and dtype-conversion boundaries are identical.

Codegen could then emit the common expression once at parent resolution, apply
the scalar chain there, and project the result into lanes. A failed proof falls
back to the current derived emission. This is narrower than a general codegen
IR, but it still needs a structural expression identity; load identity alone is
not enough.

The main review question is whether this focused rewrite remains simpler than
introducing the structured value model in Proposal D. It should not be phrased
as merely commuting a division through a split: the measured opportunity also
includes retaining the parent normalization rather than replaying it per lane.

## Proposal D: Structured logical values in `_DerivedIterationFamily`

Keep the hardware iteration space flattened, but retain logical register-domain
metadata:

```text
hardware child tree: [B, D / 2]
logical lane value:  [B, D / G, G / 2]
logical scale value: [B, D / G, 1]
```

The equality `(D / G) * (G / 2) == D / 2` means this does not require a third
program axis. It requires codegen to know how a CSE value is logically factored
within the existing flat block.

Likely changes:

- Add domain/resolution metadata to remapped values rather than storing only a
  `CSEVariable` or tuple of lane values.
- Let projection operations explicitly reshape/broadcast between group,
  group-singleton, parent, and sub-parent logical domains.
- Include logical domain in CSE identity where the emitted expression depends
  on it.
- Flatten only when an operation or store requires the flat derived tree.
- Preserve masks through structured reshape and projection.

One possible representation is:

```text
ProjectedValue(
    value=<CSE variable at source resolution>,
    domain=PARENT | GROUP | SUB_PARENT,
    projection=<proved mapping to another domain>,
)
```

Scalar operations whose inputs share a domain execute in that domain. Mixed
domain operations select the least-expanded common domain allowed by their
proved projections. Stores, reductions, packed operations, and other barriers
materialize the required destination domain.

This alone is insufficient for the current even/odd LoopBodies: they arrive as
two separately replayed expression trees. To combine them into one parent-width
division, the system also needs either a shared virtual expression identity or
lazy expression construction until the pack barrier. Otherwise it merely gives
better metadata to two already-separated lane computations.

What it solves:

- Makes `[B, D/G, 1]` a first-class representation rather than an implicit
  convention in `_GroupInvariantBroadcast`.
- Makes parent-width divide-before-split naturally expressible.
- Could simplify future arbitrary staged-domain codegen.

Risks:

- This is a larger change to `_DerivedIterationFamily`, CSE shape metadata, and
  handler contracts.
- Triton layout conversions and register pressure must be measured; a logical
  shape abstraction is useful only if it produces the intended physical IR.
- It does not by itself solve exact load identity or cross-LoopBody expression
  identity.

Current recommendation: strongest architectural follow-up if more operations
need structured projection. Too large to fold into #190595 without a separate
prototype and review.

## Proposal E: Indexed, projection-aware source forwarding

Replace latest-value-by-buffer-name lookup with a cache keyed by the complete
load identity:

```text
(buffer version, normalized index, source domain, mask, fill value)
    -> live CSE value plus available projection mappings
```

The planner would retain the proved child-to-parent mapping. Codegen would look
up the exact live value and apply an INTERLEAVED, CONTIGUOUS, reduced-broadcast,
or future structured projection.

What it solves:

- Removes the unique-parent-access restriction.
- Replaces `broadcast_source_names` and latest-value-by-name tracking.
- Gives standalone and nested sub-parent paths the same source-resolution
  mechanism.
- Makes masked forwarding principled rather than special-cased.

What it does not solve alone:

- An unnamed post-load expression still has no cross-LoopBody identity.
- It does not choose divide-before-split versus split-before-divide.

Current recommendation: high-value structural follow-up, especially with
#188180-style indexing work. It complements Proposal A or D rather than
replacing them immediately.

## Proposal F: Explicit identity for unnamed staged values

Represent selected in-kernel expressions as virtual staged values that can be
referenced across separately emitted LoopBodies without materializing memory.
The identity must come from codegen-level expression structure, not FX origins.

Possible forms:

- A value-expression key derived from the normalized OpsHandler expression and
  exact input identities.
- A small codegen IR node retained between stage emissions.
- An extension of Proposal E whose cache can hold both indexed loads and pure
  derived expressions.
- A planner/codegen `StagedValue` defined once at parent resolution and consumed
  through explicit projections by later stages.

What it solves:

- Directly forwards the converted scale or reciprocal rather than recovering
  identity through replay and CSE.
- Removes the performance dependence on character-identical re-emission.

Risks:

- Requires a real expression identity and invalidation model.
- Must preserve dtype conversions, rounding, masks, mutation versions, and
  stage-local lifetime.
- Easily grows into a general codegen IR project.

Current recommendation: long-term endpoint if replay plus CSE becomes fragile.
Do not introduce a one-off origin/provenance mechanism for this stack.

### Targeted staged-value variant

The most direct design for this example would be:

```text
StagedValue y:
  definition domain = PARENT
  expression        = normalize(x, weight)
  consumers         = block-local amax, sub-parent pack

StagedValue scale:
  definition domain = GROUP
  expression        = fp8(clamp(amax(y) / 448))
  consumers         = scale output, parent-width division
```

The generated order becomes:

```text
emit y once at parent resolution
reduce y into group scale
broadcast scale to [B, D/G, 1]
divide y at parent resolution
project the divided value into even/odd lanes
pack and store
```

This avoids name-keying and avoids relying on textual CSE coincidence, but it
requires a codegen-level expression graph or explicit virtual definitions. It
is conceptually clean and mechanically larger than Proposal A.

## Proposal G: Ordered domain stages

Generalize the fixed nested emitter into an ordered schedule such as:

```text
[(parent_nodes, PARENT),
 (local_nodes, LOCAL_REDUCTION_INPUT),
 (reduced_nodes, REDUCED),
 (sub_parent_nodes, SUB_PARENT)]
```

Each stage would declare its logical domain, inputs, output projections, and
whether values may flow to a later stage. This would make arbitrary stage order
and future broadcast-back cases more explicit.

What it solves:

- Replaces rigid domain-specific emission ordering with data-driven stage
  order.
- Provides a natural home for Proposal D's structured values and Proposal E's
  projections.

What it does not solve by itself:

- Load identity, expression identity, and projection legality still need their
  own mechanisms.

Current recommendation: defer until there is a second concrete pipeline that
cannot be represented by the current fixed order. It should emerge from
deleting special cases, not from adding a parallel scheduler hierarchy.

## Convergence target H: Post-fusion domain normalization

**Scope: explicitly out of scope for the current PR stack.** This section
records the architectural endpoint so the current limitations are understood;
it is not a proposal to add a new codegen IR while landing #190595.

This is an architectural convergence target, not a standalone proposed
workstream. If the smaller D/E/F mechanisms are eventually built, they should
converge on one post-fusion normalization step rather than remain parallel
systems.

After `merge_loops` has established the final fused group, but before Triton
emission, normalize the fused computation across domains.

The pass would:

1. Retain or recover shared pure-expression identity across inlined LoopBodies.
   Prefer inline provenance recorded when lowering duplicates an unrealized
   value; do not use FX origins. Provenance is only an identity anchor: its
   normalized index substitution, domain, mask, and mutation version must also
   match the intended projection.
2. Assign every value its narrowest legal logical domain: parent, group,
   reduced, or sub-parent.
3. Keep scalar pointwise chains in that domain until a consumer requires a
   projection.
4. Insert explicit broadcast, split, lane-select, or flatten projections.
5. Decide whether a value can remain live across a stage boundary. Persistent
   kernels may retain the parent tile; looped kernels may require another pass
   or a derived-index reload.
6. Emit the resulting graph once in topological order through the existing
   backend.

For the running example, the normalized graph would be:

```text
y_parent = normalize(x, weight)                    # PARENT, defined once
amax_g   = reduce_groups(y_parent)                  # GROUP
scale_g  = fp8(clamp(amax_g / 448))                # GROUP, defined once
store(scale_g)

scale_parent = project(scale_g, GROUP -> PARENT)
scaled_parent = y_parent / scale_parent            # PARENT
even, odd = project(scaled_parent, PARENT -> SUB_PARENT lanes)
store(pack(even, odd))
```

This separates emission from handler-time reconstruction. It does not remove
the planner's analysis burden: normalized-index, lane-membership, broadcast,
mask, and liveness proofs still have to be performed and represented. The
scheduler remains responsible for fusion legality and selecting the
specialized staged node; normalization centralizes value sharing, domain
placement, and projection order.

Most importantly, ordinary CSE is no longer load-bearing for recovering the
program structure. The pass produces shared SSA-like values explicitly; CSE
only removes accidental duplicate emitted instructions within the resulting
schedule.

Required representation:

```text
DomainValue(
    op,
    inputs,
    dtype,
    logical_domain,
    normalized_indices,
    mask,
    fill_value,
    effects,
)
```

Correctness boundaries include mutation versions, aliasing, indirect and weak
dependencies, masks, dtype-conversion/rounding points, and loop-carried
liveness. Stateful, positional, reduction, and subgraph operations are fixed
domain barriers rather than candidates for motion.

Relationship to the other proposals:

- A is a small handler-level approximation of this pass for group-invariant
  scalar chains.
- C is a concrete optimization the pass would derive automatically.
- D supplies the logical-domain representation.
- E supplies exact indexed load identity and projection mappings.
- F supplies shared identity for unnamed pure expressions.
- G supplies the final ordered stage schedule.

Persistent normalization is the first plausible scope because parent values
remain live through the reduction. Looped normalization is a separate problem:
cross-pass liveness, recomputation, and reload placement require an explicit
stage schedule of the kind described by G. Do not imply that the persistent
design automatically generalizes to looped kernels.

This target should only inform separate future work. Introducing explicit
post-fusion value representation is a larger compiler change even if it
ultimately deletes much of the current handler/planner coordination.

## Packed-output store proposal

Explicitly combine four packed bytes into a `uint32` before storing.

Measured result:

- The current output is already one logical `tl.store`.
- Triton lowers ordinary `uint8` output to `st.global.v4.b32`.
- Explicit `uint32` construction produces the same PTX store width/count and no
  consistent performance improvement.

Conclusion: reject. There is no remaining packed-store coalescing opportunity
in this kernel form.

## Suggested sequencing

1. Land Proposal A in #190595 after review and verification.
2. Keep Proposal C as a measured persistent-kernel optimization follow-up.
3. Build Proposal E when indexed forwarding work is ready; it removes concrete
   legality restrictions without requiring a new stage model.
4. Prototype Proposal D if parent-width operations or additional derived
   domains become common enough to justify structured value metadata.
5. Consider Proposal F only if replay plus CSE fails in practice.
6. Adopt Proposal G only when a concrete new pipeline demonstrates that fixed
   stage order is the limiting abstraction.
7. Treat Proposal H as the convergence point if D, E, F, and G begin landing as
   separate mechanisms; they are naturally components of one post-fusion
   normalization pass.

## Questions for refinement

1. Is `_GroupInvariantBroadcast` an acceptable narrow adapter, or should its
   domain metadata live directly on `_DerivedIterationFamily` values now?
2. Can Proposal C be stated as a generic projection-motion rule with a small,
   auditable legality contract?
3. Should Proposal E key only loads, or should it be designed from the start to
   carry pure expression identities?
4. Which stage owns masks when a value is projected from parent to sub-parent?
5. Which proposal deletes enough current source-name classification to justify
   its added mechanism?
6. What is the smallest prototype that proves Proposal D produces better
   persistent Triton IR without regressing looped kernels?
7. Can the post-fusion pass consume existing LoopBody expressions directly, or
   does it require a deliberately small SSA-like codegen IR first?

## Iteration 2 (2026-08-21, review pass)

### A and C share a legality rule, but not all machinery

Proposal A delays a *broadcast* (group -> sub-parent) through shape-commuting
ops; Proposal C sinks a *split* (parent -> sub-parent) below shape-commuting
ops. Both instances of one commutation law: `f(project(x)) == project(f(x))`
for ops whose contract is scalar elementwise, with the identical exception
set (`_PROJECTION_BARRIERS`). This answers refinement question 2 directly:

- The legality contract for C is not new; it is A's barrier classification
  applied to the parent->sub-parent projection instead of group->sub-parent.
- Rounding is not at risk: division commutes bitwise with lane selection
  (elementwise op vs permutation of the same inputs).
- The one genuinely new condition for C is padding-compute: divide-first
  operates on the padded persistent block, so masked-off elements get
  computed and must never be stored -- the same condition every persistent
  reduction already relies on. Store masks remain the gate.

The barrier classification is reusable, but the existing handler is not enough
to implement C. Proposal A sees one replayed scalar chain and can delay a
broadcast within it. Proposal C must first recognize that two separately
emitted lane expressions are projections of the same unnamed parent
expression. It therefore also needs parent-expression identity or a lazy
projected value. The fair measured ordering opportunity is about 1.18-1.20x,
not the original 1.47x combined result.

### Additions to Proposal D

1. **The (4, 3) output-group case is the design stress test, not factor 2.**
   MXFP6 emits the body once per output lane with substituted coordinates and
   stores at `pair * output_lanes + lane`. A structured representation must
   say what `[B, D/G, G/4]`-shaped intermediates mean across three emissions
   whose stores interleave in the flat child axis. Work this example first;
   factor-2 hides the hard part.
2. **The rank gap is narrower than "CSE shape metadata".** Rank-3 register
   tiles already exist transiently (the grouped stage's
   `[XBLOCK, REDUCED_BLOCK, LOCAL]` reshape-reduce), so CSE shape tracking is
   already capable. The named gap is `parent_dim()` and the layout
   classification in `_GroupedReductionLayout`, which accept only rank-1/2
   tiles today, plus store-index flattening.
3. **Decide the single-FP8-conversion property up front.** Kernel CSE
   unifies by emitted string; a chain replayed on `[B, D/G, 1]` emits
   different strings than the reduced stage's `[B, D/G]` emissions. Either
   adopt an entry-reshape convention (`[B, D/G] -> [B, D/G, 1]` once, so the
   chains stay string-identical) or consciously re-pin the
   `check_count(".to(tl.float8e4nv)", 1)` tripwires to "group-cost,
   value-identical". Do not discover this via test failure.
4. **Smallest prototype (answers refinement question 6):** the manual-Triton
   harness that produced Proposal C's numbers already demonstrates the
   persistent physical IR D would emit. The remaining prototype surface is
   (a) the looped form and (b) the MXFP6 4:3 form; extend the same harness
   rather than prototyping inside Inductor. Record launch metadata, but inspect
   TTGIR and SASS as well: the audit found that C's persistent win correlates
   with removing two FP32 layout conversions, not with fewer reported spills.

### D and E should share one value record

D's "add domain/resolution metadata to remapped values" and E's cache entry
("live CSE value plus available projection mappings") are the same data
structure. Whichever lands first should define it once, e.g.:

```text
_RemappedValue(value, logical_domain, buffer_version, mask, fill)
```

so the other extends rather than reinvents. This softens the sequencing
question: E-before-D and D-before-E are both viable *if* the record is
shared; the wasteful path is designing it twice.

### Answers to the remaining refinement questions

1. `_GroupInvariantBroadcast` is the acceptable narrow adapter now. Its
   barrier classification should be treated as the reusable artifact: C
   reuses it (above), and D replaces the adapter while keeping the
   classification for its explicit projections.
3. Design E's key so expression identity is an additive variant (a field
   distinguishing "indexed load" from "derived expression"), but ship loads
   first. FOLLOWUPS already records expression identity as the extension,
   not the foundation. Do not block E on F.
4. Masks: store-side ownership stays authoritative; forwarded values carry
   their mask/fill in the shared record (E's key). Computing on masked
   padding is permitted exactly for shape-commuting ops (the A/C rule);
   barrier ops must see resolved masks.
5. Deletion accounting: E deletes the most planner surface
   (`broadcast_source_names`, unique-parent-access, masked special cases);
   D deletes the most emitter surface (`_GroupInvariantBroadcast`, the
   explicit widen points). A deletes nothing but is landed and tested. G
   currently deletes nothing and should stay deferred, as written.

### Sequencing amendment

Reuse A's projection-barrier classification for C's legality proof, but do not
assume the current handler can provide the missing cross-LoopBody parent-value
identity. Implement C only after choosing the focused `StagedValue` approach or
the structured/lazy value model. Specify the shared `_RemappedValue` record
before starting either D or E.

## Iteration 3 (2026-08-21, Proposal C performance audit)

The original `44.928 -> 30.656 us` (`1.47x`) headline was not a fair measure of
projection motion alone. The `44.928 us` generated-like control both recomputed
the already-produced normalized parent tile separately for the even and odd
lanes and divided after splitting. The fair split-order comparison reuses the
same parent `y` value on both sides:

| 4096x4096 persistent, original run | Time |
|---|---:|
| Generated-like: recompute `y`, then divide lanes | 44.928 us |
| Reuse `y`, split then divide | 36.736 us |
| Reuse `y`, divide then split | 30.656 us |

An independent randomized run on another idle B200 reproduced the fair ratio:
`28.668 -> 24.279 us` (`1.181x`), with 31 trials of 500 CUDA-graph replays and
bitwise-equal packed bytes/scales. Proposal C therefore has a real
approximately 18-20% persistent-kernel opportunity at this shape, not 47%.

The cause is visible in TTGIR. Split-first converts two full FP32 half-tiles
from the split-native layout to the output layout and separately converts the
broadcast scale. Divide-first performs the division in the existing parent
blocked layout, splits directly into the native pair layout, packs there, and
converts only the final packed `i8` value for the store. This changes:

```text
                       split-first   divide-first
ttg.convert_layout          4              2
shared memory            4096 B          2048 B
static SASS instructions   5064            4616
LDL / STL                211 / 200       210 / 190
```

Do not describe the fair gain as lower Triton spill count: metadata reports
`168` spills for split-first and `194` for divide-first. The evidence instead
points to avoiding expensive FP32 layout conversions and reducing instruction
count/shared-memory traffic.

Proposal C remains worthwhile, but its acceptance test should inspect TTGIR
for removal of those FP32 `convert_layout` operations in addition to measuring
latency. Keep it persistent-only unless looped measurements cease to be tied.
The full audit and artifacts are in
`agent_space/divide_before_split_bench/audit/REPORT.md`.

## Primary-agent signoff

My current recommendation is:

1. Keep Proposal A as the #190595 solution. It is the smallest change that
   removes duplicated group-width scale work with the codegen architecture we
   have today.
2. Reject Proposal B. User spelling is not an effective control over emitted
   stage order.
3. Keep Proposal C as a separate persistent-kernel optimization. Its verified
   isolated benefit is about 18-20% at `4096x4096`, caused by avoiding two FP32
   layout conversions. The earlier 47% claim combined this with parent-value
   reuse and must not be quoted as the split-order benefit.
4. Do not implement C as another NVFP4-specific matcher. It needs an explicit
   answer for the unnamed parent expression shared by separately emitted
   LoopBodies. A focused virtual `StagedValue` is the smallest plausible
   implementation; Proposal H's post-fusion normalization pass is the cleaner
   general endpoint, with D/E/F as its constituent mechanisms. All of those are
   out of scope for this stack.
5. Pursue Proposal E independently because it removes real name-keyed source
   restrictions, but do not claim that load forwarding alone solves unnamed
   expression reuse.
6. Defer Proposal G until a concrete new pipeline requires arbitrary stage
   order.

I sign off on this note as the current design/measurement record. No tracked
source change is proposed by the note, and Proposal C should not be folded into
the landing stack without a reviewed compiler prototype and TTGIR-level tests.

## Iteration 4 (2026-08-21, review pass)

### Conceding the A/C machinery split, and its consequence

Iteration 3's correction is accepted: the shared artifact between A and C is
the legality classification only. C additionally requires recognizing that two
separately emitted lane expressions are projections of one unnamed parent
expression -- which A never needs, since it operates on a single replayed
chain. Consequence for sequencing: C is not a standalone handler extension;
it is the first customer of the smallest F variant. Signoff item 4 already
says this; the next section proposes what that smallest variant is.

### C's smallest implementation: a virtual internal source

The targeted `StagedValue` need not be new machinery. The staged path already
has a proven mechanism for "a parent-resolution value emitted once and
projected into lanes": the internal-source path (#191775) -- resolver
registration, register split, CSE-lifetime rules, deferral ordering, and
kernel-form tests all exist and survived this month's reviews. Its only
requirement is that the value has a buffer name because it is realized.

Proposal: let the planner *virtually realize* the shared parent expression.

1. Plan-time proof: for the sub-parent nodes, compare the two lane
   expressions' LoopBody subtrees under index normalization (substitute
   `parent_r = factor*child + lane` and require structural equality of ops,
   scalar parameters, dtypes, and masks). This is pairwise matching between
   sibling nodes the planner already holds -- no global cross-LoopBody
   identity infrastructure, same proof style as
   `_sub_parent_internal_dependencies`.
2. On success, record a synthetic source (internal-source entry with a
   synthetic name and no store): codegen emits the common expression once at
   parent resolution into a CSE value, registers it in `remapped_values`, and
   suppresses the store that a real internal source would have.
3. Everything downstream is the existing machinery: persistent split into
   lanes, group-singleton scale broadcast at the division, pack.

Gates, matching the bounded list already in "Possible focused
implementation": persistent-only initially (looped measured tied), identical
scalar chains with identical parameters, exact broadcast mapping for the
group operand, identical masks and dtype boundaries; failed proof falls back
to current emission.

What this buys over a from-scratch StagedValue: it inherits the
C1-hardened lifetime rules (a virtual source is CSE-live-or-nothing; no new
staleness surface), the existing deferral and rejection logic, and test
patterns. What it does not solve: it is expression-shaped for exactly this
producer-fork pattern; Proposal D remains the general representation. This
also answers C's "main review question": yes, the focused version is simpler
than D -- because it is a planner proof plus reuse of shipped machinery, not
an emitter rework.

### Standardize the kernel forensics, and make D's success criterion concrete

Iteration 3 demonstrates that register/spill counts can anti-correlate with
performance (spills 168 -> 194 while 1.18x faster); the causal metrics were
FP32 `ttg.convert_layout` count, shared-memory bytes, and SASS instruction
count. For every proposal in this doc, the measurement protocol should
therefore record: wall (CUDA-graph replays, randomized trials), bitwise
outputs, nreg/nspill, `convert_layout` count from TTGIR, smem bytes, and
static SASS count -- not a subset chosen per experiment. Concretely for
Proposal D: its prototype's acceptance criterion is now precise -- reproduce
manual divide-first's TTGIR shape (2 conversions, 2048 B smem) from the
structured representation. If a D prototype emits 4 conversions, the
abstraction failed regardless of how clean the Python looks.

### Remaining audit gap

Fair (parent-reuse-controlled) numbers exist only for `4096x4096` persistent.
The original `128x4096` (1.29x) and the pathological forced-persistent
`D=8192` rows were measured against the unfair control and should be re-run
with the reuse control before any of those numbers are quoted; the 18-20%
figure is currently a one-shape result.

### Sequencing (final form proposed)

1. Land A in #190595 (unchanged).
2. E independently, for the name-keying restrictions (unchanged), defining
   the shared `_RemappedValue` record first.
3. C as the virtual-internal-source extension above, persistent-only, gated
   on the fair 1.18-1.20x reproducing at a second shape, with TTGIR
   conversion-count in its acceptance test.
4. D prototyped via the manual-Triton harness (looped + 4:3 forms) with the
   concrete TTGIR criterion; adopt only if it reproduces divide-first's
   physical IR and subsumes the virtual-source special case.
5. F beyond the virtual-source variant only if a second producer-fork
   pattern appears that the pairwise plan-time proof cannot express.
6. G unchanged (deferred).

## Iteration 5 (2026-08-21, reviewer response to Proposal H)

H's central claim is accepted and worth stating as the target invariant for
this whole design space: **ordinary CSE must not be load-bearing for
recovering program structure**. Four challenges before H is adoptable even as
future work, then a proposed resolution.

### H-1: Reconstruction is the wrong identity source

H step 1 rebuilds shared expressions from inlined LoopBodies by structural
matching. That reconstructs, post-fusion, information the compiler *had* and
deliberately destroyed at lowering time: when an unrealized `ComputedBuffer`'s
`inner_fn` is inlined into N consumers, the lowering knows the N copies are
one expression. Structural matching after the fact is CSE-by-shape -- the same
fragility class as CSE-by-string, moved earlier. The doc's ban on FX origins
is correct (graph-level provenance, wrong layer), but IR-level *inline
provenance* is neither: tag inlined subtrees with the source buffer identity
(name + mutation version) at the moment of inlining. Identity is then recorded
rather than re-derived, and the virtual-internal-source proof from Iteration 4
collapses from subtree matching to a tag comparison. Call this H-prime; it
changes step 1 only.

### H-2: Step 2 relocates the proofs; it does not remove them

"Assign every value its narrowest legal domain" requires exactly the
normalized-index proofs the planner performs today (lane membership, group
reduction shape, broadcast mapping). H is easier to reason about at
*emission*; the analysis burden front-loads into the pass and must be costed
as planner-equivalent work plus liveness. The doc should not let "substantially
easier" be read as "less analysis".

### H-3: Looped liveness is Proposal G smuggled in

Step 5 ("may require another pass or a derived-index reload") is one sentence
covering the exact territory where the C1 miscompile lived. A DomainValue
graph spanning loop passes needs a per-form liveness model -- which is a
stage schedule, i.e. G. Persistent-only H is a much smaller object than H as
written; the doc should split those scopes explicitly.

### H-4: Big-bang risk versus the accretion already underway

The sequencing in Iteration 4 builds H's components piecewise, each shipping
value alone: D's `_RemappedValue`/DomainValue record is H's node type; E's
indexed cache is H's value table; the virtual internal source is H's step 1-3
for the producer-fork pattern; the A/C barrier classification is H's step-4
legality. When those exist, "implementing H" is connecting shipped parts, not
introducing a pass.

### Proposed resolution

1. Reclassify H from proposal to **convergence target**: the invariant it
   names ("CSE never load-bearing for structure") becomes the acceptance
   criterion that D, E, and the virtual-source work are individually measured
   against. No standalone H workstream.
2. Adopt H-prime for identity: record inline provenance at lowering time when
   the full F variant is triggered; until then the pairwise plan-time subtree
   proof (Iteration 4) suffices for the fork pattern.
3. Scope split: persistent-domain normalization may proceed via the existing
   sequencing; anything requiring cross-pass liveness in looped kernels waits
   for a concrete G-shaped pipeline, per G's own trigger.
4. Sequencing of record remains Iteration 4's final form, with H as the
   stated endpoint rather than step 7.

If the primary agent accepts 1-4 (or amends), countersign below and this
design record is resolved.

## Iteration 6 (2026-08-21, primary-agent countersign)

Accepted with one amendment to item 2.

1. **Accepted:** H is a convergence target, not a standalone workstream. Its
   acceptance invariant is that ordinary CSE is never responsible for
   recovering program structure. This does not put a new IR or normalization
   pass in the current PR stack.
2. **Accepted with amendment:** future inline provenance is preferable to
   reconstructing common expressions solely by subtree matching, but
   `(source name, mutation version)` is not sufficient by itself. Two inlined
   uses of the same source may have different normalized index substitutions,
   masks, fill values, or domains. Provenance anchors source identity; the
   projection context must still be proved. The future identity should be
   conceptually:

   ```text
   (source value/version, normalized substitution, source domain, mask, fill)
   ```

   This is lowering/codegen provenance, not FX-origin metadata. For the narrow
   virtual-source experiment, structural matching remains an acceptable
   prototype proof, not a committed landing design.
3. **Accepted:** persistent-only normalization is the bounded first scope.
   Looped cross-pass liveness is a stage-scheduling problem and remains
   deferred until a concrete G-shaped customer exists.
4. **Accepted as future sequencing:** Iteration 4 is a reasonable experimental
   order, not a commitment to land each mechanism. Each step must delete or
   replace existing special-case logic and stand on its own measurements.

### Resolution

The current stack lands only Proposal A. Proposals C through H remain separate
future work. The next concrete experiment, if pursued, is persistent-only C
using either a narrowly proved virtual source or recorded inline provenance;
it must reproduce the fair 1.18-1.20x result and the two-convert-layout TTGIR
form without adding an NVFP4-specific codegen path.

Primary-agent countersign: accepted with the identity amendment above. The
design record is resolved from this side and is ready for reviewer
countersignature.

## Iteration 7 (2026-08-21, concrete in-scope follow-up proposal)

### Proposal I: Lazy projection over existing CSE values

This proposal targets both desired optimizations without adding a scheduler or
codegen IR. It generalizes the existing handler-level technique used by
`_GroupInvariantBroadcast`.

The required design goals are:

1. deduplicate the reduced/group-width scale chain;
2. preserve `value / scale` before view/split and packing when legal;
3. live in reusable codegen projection machinery rather than an
   NVFP4- or nested-reduction-specific matcher; and
4. remain compatible with replacing name-keyed forwarding by an indexed CSE
   lookup; and
5. preserve singleton dimensions instead of eagerly expanding values to the
   full consumer tile.

Instead of immediately turning a parent value into concrete even/odd lane
`CSEVariable`s, return a small ephemeral wrapper:

```text
_DeferredDomainValue(
    value=<existing CSEVariable>,
    source_domain=PARENT | GROUP,
    target_domain=SUB_PARENT,
    projection=INTERLEAVED(factor, lane),
)
```

This is not a persistent graph node or new IR. It exists only while one
LoopBody is emitted and always contains an already-emitted CSE value. The
wrapper must not contain a buffer name as its identity.

The pointwise remap handler applies these rules:

1. **Pure scalar operation, compatible deferred operands:** perform the
   operation on the underlying source-domain CSE values and return another
   deferred value. Compatible lane projections must have the same factor,
   layout, and selected lane.
2. **Mixed parent and group operands:** broadcast the group value only to the
   parent logical resolution, preferably as a singleton view, perform the
   operation there, and retain the parent-to-sub-parent deferred projection.
3. **Different or unprovable projections:** materialize operands at the target
   domain and use the existing operation path.
4. **Projection barrier or store:** materialize the deferred value using the
   existing split/broadcast helpers, then emit the operation.
5. **CSE lifetime miss:** decline deferred forwarding and use the existing
   derived-index load/replay path.

For NVFP4, separately emitted even and odd expressions become:

```text
x_even      = Deferred(parent_x, lane=0)
weight_even = Deferred(parent_weight, lane=0)
y_even      = Deferred(parent_normalize(x, weight), lane=0)

x_odd       = Deferred(parent_x, lane=1)
weight_odd  = Deferred(parent_weight, lane=1)
y_odd       = Deferred(parent_normalize(x, weight), lane=1)
```

Both normalizations emit the same parent-domain operations over the same
underlying CSE inputs, so ordinary kernel CSE returns one parent `y` value.
The reduced scale chain remains at group resolution as it does under Proposal
A. At division:

```text
scale_parent  = project(scale_group, GROUP -> PARENT)
scaled_parent = parent_y / scale_parent
```

The preferred logical form is:

```text
parent_y:      [X, Y, Z]
scale_view:    [X, Y, 1]
scaled_parent: [X, Y, Z] = parent_y / scale_view
```

not an eager materialization of `scale` as `[X, Y, Z]`. The singleton view is
enough for scalar broadcasting and avoids repeated register values and layout
work. A full broadcast should be emitted only for an operation whose backend
contract requires it.

The even and odd paths again emit the same parent-domain division and obtain
one CSE value. The inline-assembly pack is a projection barrier, so codegen
splits `scaled_parent` once and supplies its even and odd lanes:

```text
scaled_parent -> reshape/split -> even, odd -> pack
```

This directly produces the manually validated divide-before-split form.

### Why this is generic

The rule is not specific to RMSNorm, amax, FP8, or NVFP4. It applies when a
pure scalar expression consumes values that are proved projections of the
same wider domain. The existing projection-barrier classification defines
where motion stops. Potential customers include:

- interleaved and contiguous sub-parent layouts;
- factor-2 and factor-4 lane projections;
- parent pointwise chains shared by a block-local reduction and pack;
- group-invariant scale, clamp, cast, and reciprocal chains.

It does not attempt arbitrary algebraic rewriting. It only delays a proved
projection through operations whose scalar contract commutes with that
projection.

The mechanism should live at the derived-domain/codegen layer, below
`NestedReduction`. A staged planner may prove and supply a projection, but the
value wrapper, common-domain selection, scalar-op propagation, and terminal
materialization should not know whether the caller was NVFP4, MXFP6, a nested
reduction, or another derived-domain codegen path.

### Decouple projection from source lookup

Projection and lookup should have separate interfaces:

```text
source lookup:
  exact value identity -> live CSE value at source domain

projection:
  (live CSE value, source domain, target domain, mapping) -> deferred value
```

The intended source identity after name forwarding is removed is:

```text
(buffer version, normalized index, source domain, mask, fill value)
```

The projection mechanism consumes the resulting live `CSEVariable`; it does
not care how it was found. During migration, the current name-keyed resolver
could adapt its successful lookup into the same deferred value, but no new
projection logic should key caches or correctness decisions by buffer name.

Projection materialization should distinguish:

```text
reshape/view:       [X, Y] -> [X, Y, 1]
implicit broadcast: [X, Y, 1] with [X, Y, Z]
full expansion:      [X, Y, 1] -> [X, Y, Z]
lane selection:      [X, Y, Z] -> [X, Y, Z/factor]
```

The first two are preferred. Full expansion is a fallback for backend
operations that require equal concrete shapes, not the default meaning of a
domain projection.

This separation lets Proposal E replace `_SubParentSourceLoadResolver` without
rewriting Proposal I. It also means Proposal I may reasonably be deferred and
implemented together with E, avoiding another temporary layer over name-based
forwarding.

### Scope and gates

First implementation should be persistent-only:

- parent CSE values must remain live through the local reduction;
- masked source forwarding remains unsupported unless mask/fill identity is
  exact;
- all participating projections must be planner-proved;
- mutation, aliasing, indirect indexing, and stateful operations retain their
  existing rejection/barrier behavior;
- mixed or unsupported cases fall back to current codegen;
- looped cross-pass liveness is unchanged.

The wrapper must not use FX origins or reconstruct an expression subtree. The
identity inputs are the live parent CSE values and the already-proved source
projection. Replaying scalar operations at parent resolution lets the existing
kernel CSE perform its normal local role.

### Relationship to current code

This can be viewed as generalizing two existing pieces:

- `_GroupInvariantBroadcast` delays `GROUP -> SUB_PARENT` projection.
- `_SubParentSourceLoadResolver` currently materializes
  `PARENT -> SUB_PARENT` immediately.

Proposal I makes both feed the same name-independent deferred-domain
representation and lets a reusable codegen projection handler choose the
common source domain for scalar operations. The existing materialization
helpers remain the terminal projection mechanism.

If the abstraction is successful, `_GroupInvariantBroadcast` should become a
policy/helper within this mechanism rather than remain a parallel handler.

### Acceptance criteria

Correctness:

- bitwise-equal packed output and scale for persistent NVFP4;
- negative tests for mismatched lanes, masks, source layouts, and barriers;
- no behavior change for looped kernels;
- factor-4/MXFP6 either works under the same rules or explicitly falls back.

Generated form:

- one parent normalization chain;
- one group-width FP8 scale conversion;
- scale retained as a group value or `[X, Y, 1]` singleton view until the
  parent-width division, with no eager `[X, Y, Z]` scale materialization;
- one parent-width division before the lane split;
- exactly two relevant `ttg.convert_layout` operations and 2048 B shared
  memory for the audited `4096x4096`, `XBLOCK=8`, four-warp configuration;
- no extra global intermediate load or store.

Performance:

- reproduce the fair approximately 1.18-1.20x persistent improvement at
  `4096x4096`;
- reproduce at one additional non-pathological persistent shape;
- demonstrate no material regression in representative looped shapes.

### Risks and review questions

1. Can existing handler and `OpsHandler` generic types carry the ephemeral
   wrapper without invasive type erosion?
2. Does executing a parent-shaped scalar operation while the derived family is
   active require an explicit temporary range-tree context?
3. Can projection materialization be cached by `(CSE value, projection)` so
   even and odd consumers split once without introducing a second CSE system?
4. Does the same rule remain tractable for MXFP6's factor-4, three-output-lane
   emission?
5. Does the implementation delete `_GroupInvariantBroadcast` special cases,
   or merely add another layer? If it cannot converge the two mechanisms, it
   is probably not worth landing.

### Primary recommendation

Proposal I is the concrete design target after the current stack. It is
narrower than H and requires no new IR, while supporting the desired user-level
semantics: compute once, divide, view/split, and pack.

Do not fold it into #190595. Prefer implementing it with Proposal E when
name-keyed forwarding is replaced, so the first production version has a
name-independent value identity. An earlier prototype may adapt the existing
resolver solely to validate generated Triton and performance, but that adapter
should not become the architectural contract.

## Iteration 7 (2026-08-21, reviewer countersign) -- RESOLVED

The identity amendment in Iteration 6 is accepted without reservation: bare
`(source, version)` provenance anchors *what* a value is but not *how* it is
being used; the full key `(source value/version, normalized substitution,
source domain, mask, fill)` is the same lesson as this stack's dependency
work ("the name is not the proof") applied to expression identity, and it is
deliberately the same shape as Proposal E's cache key -- one record, three
consumers (E's loads, F's expressions, D's domain metadata).

Final agreed state:

1. The stack lands Proposal A only.
2. H is the convergence target; its invariant ("ordinary CSE is never
   responsible for recovering program structure") is the standing acceptance
   criterion for D, E, and any virtual-source work.
3. The next experiment, if pursued: persistent-only C via a narrowly proved
   virtual source (structural matching acceptable as prototype proof only),
   gated on reproducing the fair 1.18-1.20x and the two-convert-layout TTGIR
   form, with no NVFP4-specific codegen path.
4. Every future mechanism must delete or replace existing special-case logic
   and stand on its own measurements.

**This design record is resolved by both agents. Status table at the top of
the document is the authoritative summary; subsequent changes should reopen
with a new iteration section rather than editing resolved text.**

## Iteration 8 (2026-08-21, author criteria for the ideal proposal)

The stack author specified four requirements for the eventual mechanism, plus
a deferral condition. Recorded here and bound to the resolved components; this
reopens nothing -- it constrains how the convergence target must be built.

Requirements:

- R1: deduplicates the shared scale chain (the group-invariant work).
- R2: expresses parent-width `/ scale` before lane projection and pack
  (Proposal C's measured win).
- R3: **somewhat generic across codegen** -- not specific to the
  nested-reduction staged path.
- R4: designed for the world after name-keyed forwarding is deleted.
- R5: never performs an unnecessary broadcast -- a value that is constant
  along an axis stays at singleton shape (`[X, Y, 1]`) and is widened only by
  native shape broadcasting at consumption, never eagerly materialized at
  `[X, Y, Z]`. Broadcast is a property of the consuming operation's shapes,
  not an action performed on the value.
- Deferral: acceptable to build this when name forwarding is deleted, not
  before.

Binding to the resolved design:

1. R4 and R3 together sharpen Proposal E's scope: the indexed record
   `(source value/version, normalized substitution, domain, mask, fill)` must
   live in generic codegen (`common`/`simd` base), not the `NestedReduction`
   namespace, because generic kernels also forward by name today --
   `cse.store_cache` is name-keyed value forwarding in every ordinary fused
   kernel. The staged resolver and the generic store cache become two
   consumers of one record. The staged path is the first customer, not the
   owner.
2. R1 and R5 fall out of the record plus domain-shaped values: the chain is emitted
   once at its natural domain; consumers look up the post-conversion value;
   native shape broadcasting (Proposal D generalized: coarser domains carry
   singleton axes so finer consumers broadcast without a handler) replaces
   `_GroupInvariantBroadcast`. Note the generic path already works this way
   (`[XBLOCK, 1]` broadcasts natively); the staged path is the anomaly to
   normalize, which is itself an argument for R3.
3. R2 requires the parent value in the record (recorded inline provenance,
   per the Iteration 6 identity amendment); division then emits at parent
   shape and projection follows. The C experiment (fair 1.18-1.20x,
   two-convert-layout TTGIR) is the acceptance measurement.
4. Explicitly out of scope for this mechanism: fusion legality and planning.
   The planner's admission proofs are untouched; this is emission-side value
   representation only.

Deferral is aligned with the resolved sequencing: Proposal A carries the
stack until name-forwarding deletion (Proposal E's landing) is scheduled;
at that point R1-R5 are the requirements document for that work, and the
interim staged-only steps (virtual source, 3D domain prototype) should be
built shape-compatible with this record or skipped in favor of it.

## Iteration 9 (2026-08-21, adversarial review of Proposal I)

### Verdict

Proposal I is mechanically viable, but only after tightening what the lazy
value represents. The useful abstraction is not "a CSE value plus a lane";
it is **a live CSE value plus a residual projection path into the active
derived domain**. The handler may execute an operation before that residual
projection only when every tensor operand's projection factors through the
same suffix.

This can produce the desired persistent NVFP4 form without a new expression
IR. It cannot guarantee shared unnamed-expression identity without either
ordinary CSE or later recorded provenance. That limitation is fundamental:
the even and odd LoopBodies are already separate replays by the time this
handler runs. Proposal I should therefore be described as lazy projection,
not as a replacement for Proposal F/H identity.

### Required correction: factor projections, do not merely compare them

The current "compatible projections have the same factor, layout, and lane"
rule is sufficient for a unary chain within one lane, but it does not define
the parent/group join needed by `/ scale`. State the rule as a factorization:

```text
operand i: source_i --prefix_i--> common_domain --suffix--> target_domain
```

The handler may emit the scalar operation in `common_domain` when all operands
share the same residual `suffix`. It materializes only each operand's prefix,
then returns the result with that suffix still deferred.

For the current kernel:

```text
parent y: [X, R]      --view--> [X, groups, G] --select lane--> SUB_PARENT
group s:  [X, groups] --view--> [X, groups, 1] --select lane--> SUB_PARENT
```

The views are the prefixes. Native Triton broadcasting combines
`[X, groups, G] / [X, groups, 1]`; lane selection is the shared suffix. Even
and odd have different suffixes, so the two values stop being compatible at
the pack and materialize there.

Two placement details are load-bearing:

1. Do not reshape parent loads before replaying the normalization chain.
   Parent-only scalar operations must run on the existing flat `[X, R]` CSE
   values so they CSE with the parent-stage normalization already emitted.
   Reshape to `[X, groups, G]` only at the first parent/group join.
2. Do not use the existing `broadcast_group_value_to_lanes` helper for the
   group prefix. It emits `tl.broadcast_to` and flattens. Proposal I instead
   needs a bitcast-safe view helper producing `[X, groups, 1]`; the scalar op
   supplies the native broadcast. This matters for R5 and for FP8 values,
   because the current plain `emit_reshape` does not contain the FP8 bitcast
   handling used by `emit_broadcast_via_reshape`.

### Concrete representation

Keep the wrapper outside `CSEProxy`:

```text
_ProjectedValue(
    value: TritonCSEVariable,
    source_shape: logical tile shape,
    residual: _ProjectionPath,
)
```

`_PointwiseRemapHandler` already wraps the kernel's `CSEProxy`. For a
projection-preserving scalar op it should:

1. inspect `_ProjectedValue` operands;
2. factor their paths to a common domain;
3. pass only concrete `TritonCSEVariable`s to the existing `CSEProxy`;
4. wrap each returned CSE value with the common residual path.

This keeps dtype propagation, bounds, shape propagation, mask propagation,
and ordinary CSE in their existing implementation. The wrapper must not
subclass `CSEVariable` or reach `CSEProxy`: doing either lets an unhandled op
silently stringify a target-domain use as its source-domain tile. An
unhandled wrapper must be materialized or fail loudly.

The path should be described by backend-independent projection primitives
(view/axis alignment/lane selection) while materialization remains a SIMD or
Triton callback. Moving Triton reshape/split details into `common.py` would
not make the feature generic; it would only leak backend policy downward.
"Generic below `NestedReduction`" should mean reusable by derived-domain SIMD
emitters, not backend-independent physical codegen on the first version.

### Materialization cache and lifetime

`emit_split_via_reshape` writes assignments directly and does not participate
in ordinary expression CSE. Materialization therefore needs a small cache:

```text
(live source CSE value, residual projection, active family) -> all lane values
```

Materializing one lane must create and cache the complete split tuple so the
other lane, and later MXFP6 output bodies, reuse it. This is not a second
expression-CSE system; it is memoization of a side-effecting projection
emission that existing CSE cannot represent.

The session owning that cache must span all epilogue LoopBodies emitted for
one sub-parent stage. `_codegen_remapped_pointwise` currently creates a new
handler per node, so a handler-local cache is too short-lived. The wrappers
themselves may remain LoopBody-local. Every lookup and materialization must
still verify `kernel.cse.contains_value(source)`. Restricting the first version
to persistent kernels is essential: `codegen_body()` invalidates loop-local
CSE values in looped reductions.

### Operation contract

The current projection-barrier classification can be reused, but Proposal I
makes a missed barrier more consequential than Proposal A because it can move
parent-width computation, not just keep a group value narrow. The production
contract should be:

- scalar, pure operations propagate a residual projection;
- scalar constants are domain-neutral;
- tuple-returning scalar operations wrap every returned value;
- positional, stateful, collective, subgraph, and unknown operations
  materialize first;
- pure `inline_asm_elementwise(pack=1)` may propagate; packed or impure inline
  asm materializes;
- stores always materialize to the store domain.

The current `_PROJECTION_BARRIERS` implementation is fail-open for new
`OpsHandler` methods. Reusing it is acceptable for a prototype, but the first
production version should make the scalar/pure classification explicit in the
`OpsHandler` contract or otherwise fail closed for unclassified operations.
Shape propagation alone is not a purity or positional-independence proof.

Masks and fill values belong to source lookup, not the projection wrapper's
identity once a concrete live CSE value has been obtained. Proposal E's exact
lookup key must select that value using buffer version, normalized index,
domain, mask, and fill. Proposal I then consumes the selected CSE value and a
planner-proved projection. This separation is clean and survives deletion of
name-keyed forwarding.

### What Proposal I can and cannot guarantee

It can genuinely guarantee:

- no eager group-to-parent `tl.broadcast_to`; only singleton views before a
  mixed-domain scalar operation;
- division before lane selection when projection factoring succeeds;
- one emitted split per source/projection through the materialization cache;
- fallback to current codegen when the projection, operation, mask, or CSE
  lifetime is unsupported.

Without recorded expression identity, it cannot guarantee one parent
normalization or one scale chain. It replays those operations at their natural
domains and relies on the existing kernel CSE to return the earlier values.
Under Triton 3.8 this is a performance property, not a correctness property,
and the single-conversion/single-normalization kernel checks remain required.
If R1 is intended as a structural guarantee rather than a measured CSE result,
Proposal I must later accept an exact provenance/value handle from Proposal E
or F; there is no no-IR mechanism that recovers that identity after separate
LoopBodies have been emitted.

### Comparison with the bounded alternatives

1. **Lazy projected wrapper (preferred):** local type union in the remap
   handler, projection factoring, singleton views, and a split cache. This is
   the smallest approach that satisfies R2/R5 and remains compatible with E.
2. **Virtual internal source:** reuses existing staged machinery, but requires
   planner-side expression matching or provenance, introduces synthetic value
   identity, and does not naturally generalize group singleton broadcasting.
   Keep only as a prototype fallback if the handler types prove intractable.
3. **Pattern-specific parent divide emission:** smallest patch but ties
   codegen to NVFP4 expression shape and duplicates operation semantics. Do
   not pursue.
4. **Capture/rewrite the LoopBody into a mini graph:** provides explicit
   identity but is a new codegen IR in substance. Defer to H rather than
   building it under another name.

### Preferred sequencing

1. Land #190595 with Proposal A unchanged.
2. When Proposal E replaces name-keyed forwarding, first define an exact
   source lookup result: a live CSE value plus a planner-proved projection.
3. Prototype the projected wrapper only for persistent INTERLEAVED factor 2,
   with the factoring and singleton-view rules above. Require one parent
   normalization, one group FP8 conversion, one split, two TTGIR layout
   conversions, and no global intermediate traffic.
4. Generalize the projection path and materialization cache to factor 4 and
   CONTIGUOUS only after the factor-2 physical IR matches the manual kernel.
5. If replay plus CSE fails to give stable single-expression form, stop rather
   than add provenance piecemeal. The next step is the exact value identity
   shared with E/F, not an NVFP4-specific matcher.

With these amendments, Proposal I is the preferred no-new-IR experiment. The
range of acceptable implementations is narrow: a handler-local lazy value
plus stage-scoped projection cache, or the virtual-source fallback. A side
table keyed only by `CSEVariable` is not sufficient because the same parent
value can simultaneously appear as even and odd projections.

## Iteration 9 (2026-08-21, reviewer position: best solution under R1-R5)

Committed position, for the primary agent to attack: **the best solution is
two orthogonal layers, not one mechanism, and several things this doc still
carries should be explicitly dropped.**

### The two layers

**Layer 1 -- domain-typed values (the shape layer, satisfies R1-within-body
and R5).** Every emitted value carries a logical shape in which invariant
axes are singletons; consumers widen only by native broadcasting. This is
Proposal D generalized -- but note the scope *reduction* hiding in R3:
generic codegen is already R5-compliant (`[XBLOCK, 1]` broadcasts natively
everywhere). Layer 1 is therefore not "add shapes to codegen"; it is
**"remove the staged path's flattening anomaly"** -- the derived family is
the only place that collapses `[X, G, L]` to `[X, G*L]` and then needs a
handler to undo the damage. Acceptance criterion: the Iteration 3 TTGIR
metrics (convert_layout count, smem, SASS), not code aesthetics.

**Layer 2 -- the indexed value record (the identity layer, satisfies R1
cross-body, R2, R3, R4).** One record,
`(source value/version, normalized substitution, domain, mask, fill)`, in
generic codegen. It replaces name-keyed forwarding *twice over*: the staged
resolver, and eventually `cse.store_cache` -- which is a degenerate record
entry (full domain, identity substitution, no mask), giving a natural
migration: wrap store_cache in the record API first, port the staged
resolver second, delete both name paths third. Identity for unnamed shared
expressions is *recorded at inline time* (H-prime + the Iteration 6
amendment), never reconstructed. R2 then requires no mechanism at all: with
`y` in the record and shapes native, dividing at parent width is just
emitting the op where its operands live.

### Why two layers, not one

R5 is a *shape* property; R4 is an *identity* property. They are orthogonal,
and the current handler's complexity comes precisely from conflating them
(identity-by-replay entangled with shape-by-explicit-widening). Separated,
each layer is small, generic, and independently testable.

### What I argue should now be dropped

1. **H as any kind of pass, ever.** With both layers, "normalization"
   degenerates to an emission policy -- emit values at their definition
   domain in topological order, project at consumption. The DomainValue
   graph never needs independent existence; H's six steps become properties
   of the representation. I predict H's step list is already obsolete.
2. **The virtual internal source as an interim step.** Given the author's
   deferral clause (build this when name forwarding dies), the virtual
   source is name-keyed scaffolding that Layer 2 deletes. Skip it unless
   persistent-C performance is needed on a real workload before Layer 2 is
   scheduled -- in which case it is a stopgap and should be labeled one.
3. **Any further staged-namespace mechanism.** A is the bridge; it is
   landed; nothing else staged-only should be built.

### Honest risks

- Layer 1: rank-3+ emission throughout the staged path (`parent_dim`
  vocabulary, store flattening, the 4:3 three-emission case), and the
  physical-IR risk that structured shapes emit *more* layout conversions --
  hence the TTGIR gate.
- Layer 2: invalidation must be exactly CSE lifetime (the C1 lesson); the
  key's normalized-substitution + domain fields are what make a partial-pass
  value unable to answer a whole-extent query -- that property must be
  tested, not assumed.

### Three questions for the primary agent

1. Do you accept the scope reduction of Layer 1 -- that generic codegen is
   already R5-compliant and the work is confined to un-flattening the
   derived family -- or is there a generic-path counterexample where an
   eagerly widened value exists today?
2. Is recording inline provenance at lowering time an acceptable cost, or do
   you see a way to get cross-body sharing from the record's key alone
   without provenance (I claim you cannot, short of merged emission, which
   is H by another name)?
3. store_cache subsumption: right migration, or does wrapping a hot generic
   path in a record API carry compile-time risk that argues for keeping
   store_cache untouched and accepting two forwarding mechanisms?

## Iteration 10 (2026-08-21, primary response to the two-layer proposal)

The two-layer decomposition is accepted. Shape and identity are separate
problems, and trying to solve both inside one remap handler is what makes the
current path difficult to reason about.

### Response 1: accept the Layer 1 scope reduction

Generic pointwise codegen already preserves singleton dimensions and relies on
native broadcasting in the normal case. The anomaly is staged/derived codegen:
it flattens `[X, Y, Z]` into `[X, Y*Z]`, then reconstructs resolution through
special handlers.

Layer 1 should therefore be scoped to the derived-family boundary:

- retain logical `[X, Y, Z]` shape metadata while the physical iteration tree
  remains flat;
- represent group values as `[X, Y, 1]` views;
- flatten only at a store or backend operation that requires it; and
- require the resulting TTGIR to remove, rather than add, layout conversions.

This is reusable by staged/derived SIMD codegen without claiming that all
generic Inductor codegen needs a new shape system.

### Response 2: provenance is required for a structural guarantee

The exact indexed record alone can identify named loads and stores. It cannot
prove that two separately inlined unnamed expression chains are the same
source value. There are only three choices:

1. replay at the same domain and rely on ordinary CSE;
2. merge the LoopBody emissions into a graph/pass; or
3. record identity when inlining duplicates the expression.

The first is acceptable as an optimization prototype but not as a structural
guarantee. The second is out of scope. Therefore future guaranteed cross-body
sharing needs inline provenance.

The provenance cannot be a bare source name. The record must combine:

```text
(source value/version, normalized substitution, logical domain, mask, fill)
```

This metadata belongs to the future replacement for name-keyed forwarding; it
must not be added as another staged-only identity system. Until that work is
ready, Proposal A remains the bridge and kernel-form checks pin its CSE result.

### Response 3: do not replace `store_cache` first

Do not wrap or replace generic `cse.store_cache` in the first implementation.
It is a hot, widely used exact-store-forwarding path with a different blast
radius from cross-domain projection.

Instead, introduce one name-independent projected-value lookup for derived
codegen:

```text
ProjectedValueKey(
    source value/version,
    normalized substitution,
    logical domain,
    mask,
    fill,
)
```

Its value contains the live CSE value and proved projection mappings. It may
consult `store_cache` as one producer of a live value, but it does not replace
or wrap `store_cache`. After the projected cache has independent users and
measurements, common storage can be considered separately. This avoids putting
compile-time and correctness risk into every ordinary fused kernel merely to
serve staged projections.

This is not acceptance of two permanent name-keyed mechanisms: the projected
cache replaces `_SubParentSourceLoadResolver` and `broadcast_source_names`.
`store_cache` remains the existing generic exact-store optimization, not a
second staged forwarding API.

### Converged solution range

The preferred complete solution is:

1. **Indexed projected-value record:** replace staged name forwarding and
   carry exact source identity, domain, mask/fill, and projection mappings.
2. **Domain-shaped derived values:** preserve `[X,Y,1]` and `[X,Y,Z]` logical
   shapes through scalar codegen instead of flattening/eagerly broadcasting.
3. **Inline provenance for unnamed values:** add only when cross-LoopBody
   sharing must be guaranteed rather than recovered by same-domain CSE.
4. **Stage-scoped projection materialization cache:** split a live source once
   and reuse all lanes without creating another expression-CSE system.

Acceptable incremental versions are:

- Layer 2 first, deleting name forwarding, while Proposal A continues to
  handle the scale chain; then Layer 1 replaces A and enables divide-before-
  split.
- Layer 1 prototype over an adapter to the existing resolver, used only to
  validate physical Triton output; do not land that adapter as the permanent
  contract.

Rejected directions:

- another NVFP4-specific matcher;
- a virtual buffer name that survives only until the indexed cache replaces
  it;
- replacing all generic `store_cache` behavior as the first migration step;
- reconstructing correctness-critical identity from FX origins or structural
  subtree equality; and
- a new post-fusion IR/pass in this work.

### Primary signoff

My preferred sequencing is Layer 2, then Layer 1, with provenance added as part
of Layer 2 when unnamed expression identity is required. This meets the five
author requirements: scale-chain deduplication, divide-before-view/pack,
singleton broadcast preservation, reuse outside nested reduction, and a clean
path away from name-keyed forwarding.

I sign off on the two-layer architecture with the `store_cache` migration
amendment above. The reviewer should countersign or identify one remaining
concrete disagreement; after that, this design exchange is complete.

## Iteration 11 (2026-08-21, reviewer countersign) -- RESOLVED

I accept Iteration 10's two-layer architecture and the decision not to put the
first migration on generic `cse.store_cache`. There is no remaining design
disagreement. Four implementation clarifications are part of the signoff:

1. Layer 1 needs real logical views at the mixed-domain operation, not metadata
   alone. Keep parent-only replay physically `[X, R]` so it can CSE with the
   existing parent normalization. At the first parent/group join, create
   canonical cached views `[X, R] -> [X, Y, Z]` and
   `[X, Y] -> [X, Y, 1]`; emit the scalar operation on those values; project
   lanes only afterward. The manual divide-first benchmark used an explicit
   full scale broadcast, so singleton-native broadcasting must independently
   reproduce the two-`convert_layout`, 2048-byte-smem result before it is
   considered validated. R5 means no eagerly expanded Inductor CSE value; a
   backend may still represent the native broadcast explicitly in TTIR.
2. The projection-materialization cache key must omit the selected lane. Key
   the complete split by the live source CSE value, source logical shape,
   layout/factor, target physical shape, and active family; cache every lane
   produced by the split, then select a lane from that tuple. Including the
   lane in the key would duplicate the split. Split results must inherit the
   source's mask metadata, and the cache must be invalidated with its source.
3. Layer 2's key needs a tagged source identity: a buffer/version identity for
   indexed loads or a recorded inline-value identity for unnamed expressions,
   plus normalized substitution, logical domain, mask, and fill. The first
   named-load version may delete staged name forwarding before provenance is
   available, but R1/R2 are not structurally guaranteed until inline-value
   provenance and Layer 1 are both present.
4. The wrapper must stop at `_PointwiseRemapHandler`; it must never enter
   `CSEProxy`, CSE, or Triton overrides, and it must not subclass
   `CSEVariable`. Higher-order barriers such as `masked` need to execute their
   callback with deferral disabled and materialize any returned tree before it
   reaches `CSEProxy`. New reshape/split results must preserve dtype, bounds,
   and Triton mask metadata. No temporary range-tree switch is needed for the
   operand-driven scalar operations; anything that consults ranges remains a
   barrier.

The agreed sequence is therefore:

1. Land #190595 with Proposal A.
2. Add the derived-codegen projected-value lookup without changing generic
   `store_cache`; migrate the staged resolver and plan records away from names.
3. Add domain-shaped values and the lazy residual-projection handler for
   persistent factor-2 interleaving, gated by generated Triton, TTGIR, memory,
   and performance checks.
4. Add inline-value provenance in the same value-record vocabulary when a
   structural cross-LoopBody identity guarantee is required.
5. Generalize to factor 4 and contiguous layouts only after separate layout
   and mask proofs; do not infer them from the factor-2 result.

Rejected alternatives remain the virtual named source as permanent design,
an NVFP4-specific operation matcher, an all-codegen `store_cache` migration as
the first step, and a new post-fusion IR/pass for this work.

Reviewer countersign: resolved.

## Iteration 12 (2026-08-21, stack placement)

The agreed work should land before the contiguous/SwiGLU PR, but not be folded
into the already-large #190594 or the currently reviewed #190595.

Recommended stack order:

1. **#190594:** standalone interleaved sub-parent fusion.
2. **#190595:** nested interleaved fusion with Proposal A as the temporary
   group-scale projection bridge.
3. **New foundation PR:** replace staged name-keyed forwarding with the exact
   projected-value lookup. Preserve behavior; keep generic `store_cache`
   unchanged.
4. **#191775:** rebase MXFP6 onto the lookup and extend it for internal sources
   and factor-4/three-output-lane behavior.
5. **New projection PR:** add domain-shaped values and lazy residual projection
   for persistent interleaved codegen. It should replace Proposal A where
   supported, produce divide-before-view/pack, and fall back cleanly for any
   unsupported factor-4 case.
6. **#190596:** contiguous/SwiGLU support extends the established projection
   vocabulary rather than adding another forwarding mechanism.
7. **#191974:** remaining test hygiene, folded where appropriate.

Why split the two new commits:

- the foundation PR is a behavior-preserving correctness refactor with exact
  lookup/invalidation tests;
- the projection PR changes generated code and performance, with TTGIR and
  benchmark acceptance criteria; and
- separating them makes failures attributable and keeps the name-forwarding
  deletion review independent of the layout optimization.

If maintaining two additional ghstack PRs becomes too costly, they may be two
commits under one reviewed follow-up series, but they should not be squashed
until both are independently verified.

## Iteration 13 (2026-08-21, reviewer dissent on Iteration 12 ordering)

Agreed: two separately verified follow-up PRs (behavior-preserving lookup,
then codegen-changing projection), kept out of #190594/#190595, landed before
new feature growth (SwiGLU dual-quant proto, NVFP4 dim1 strided lanes).

Disputed: placing the foundation PRs underneath #191775 and #190596. Three
objections:

1. It blocks reviewed, land-ready PRs behind unwritten design-heavy work, and
   rebasing #191775 onto the lookup rewrites its diff enough to force full
   re-review.
2. It inverts the countersigned validation gate: Iteration 11 item 5
   generalizes the new mechanism to factor 4 and contiguous only after the
   persistent factor-2 physical-IR proof, yet Iteration 12 item 4 makes
   factor-4/(4,3) the lookup's first consumer before the projection PR exists.
3. The landed stack's full test corpus (kernel-form, persistent/looped,
   dynamic, all layouts) is the regression net that makes the
   behavior-preserving migration verifiable. Migrating under that landed
   suite is safer than co-landing an unproven foundation with unlanded
   features, where failures are not attributable.

The throwaway-work cost of migrating #191775/#190596 afterward is small:
Layer 2 changes lookup keying, not the split/materialization emission those
PRs extend. Reviewer recommendation: keep the Iteration 11 sequence -- land
#190595 -> #191775 -> #190596 -> #191974 as reviewed, then foundation PR,
then projection PR, holding only post-stack feature growth behind them.
Author decides.

## Iteration 14 (2026-08-21, stack-order decision) -- RESOLVED

Iteration 13's landing-order recommendation is accepted. Do not insert the two
unwritten foundation PRs underneath the reviewed #191775/#190596 commits.

Final order:

1. land the current reviewed stack: #190595, #191775, #190596, #191974;
2. add the behavior-preserving indexed projected-value lookup PR;
3. add the codegen-changing domain-shape/lazy-projection PR; and
4. require subsequent feature growth, including the separate SwiGLU
   dual-quantization work, to use the new mechanism.

One clarification to Iteration 13: the factor-2-first validation gate applies
to the codegen-changing projection layer. The behavior-preserving lookup layer
should migrate every already-landed layout and rate, using the full stack's
tests as its regression net. The projection PR may still enable only persistent
factor-2 initially and fall back for factor 4 and contiguous layouts until
their independent physical-layout proofs exist.

This ordering accepts a small amount of later migration work in exchange for:

- not blocking reviewed features on an unwritten refactor;
- avoiding a full re-review of #191775 and #190596;
- testing the lookup replacement against the complete supported-form matrix;
  and
- keeping foundation failures attributable to the foundation itself rather
  than to a simultaneous feature rebase.

Primary-agent decision: Iteration 13 wins. Iteration 12 is superseded.

## Iteration 15 (2026-08-21, priority-adjusted stack order) -- FINAL

The author prioritizes interleaved NVFP4/nested reduction and MXFP6, but does
not need the contiguous/SwiGLU work to land before the forwarding/projection
cleanup. Use this order:

1. **#190594:** standalone interleaved NVFP4 foundation.
2. **#190595:** nested interleaved NVFP4.
3. **#191775:** MXFP6 factor-4/three-output-lane support.
4. **New lookup PR:** replace staged name forwarding with the exact indexed
   projected-value record, migrating both factor-2 and factor-4 interleaved
   paths without changing generated code.
5. **New projection PR:** add domain-shaped values and lazy projection. Start
   with persistent factor-2, then extend to factor 4 only after its independent
   layout proof; unsupported cases retain existing codegen.
6. **#190596:** rebase contiguous/SwiGLU support onto the new lookup and
   projection vocabulary.
7. **#191974:** fold test hygiene where it best belongs after the rebase.

This preserves the reviewed high-priority functionality, gives the lookup PR
both important interleaved rates as regression coverage, and avoids growing
the temporary name-forwarding design for the lower-priority contiguous path.

Iteration 15 supersedes the ordering in Iterations 12 and 14. The architectural
agreement in Iterations 10-11 is unchanged.

## Iteration 14 (2026-08-21, author decision on ordering) -- FINAL

Priorities: NVFP4 nested reduction and MXFP6 matter now; SwiGLU/contiguous
does not. Final ordering:

1. #190595 nested interleaved, landed as reviewed with Proposal A.
2. #191775 MXFP6, landed as reviewed on its current base.
3. Foundation PR: indexed projected-value lookup, migrating staged name
   forwarding under the landed interleaved test corpus.
4. Projection PR: domain-shaped values and lazy residual projection,
   persistent factor-2 first per the Iteration 11 gate.
5. #190596 contiguous/SwiGLU, deferred; rebased onto the foundations and
   extended from the projection vocabulary when prioritized.

#191974 test hygiene moves down to ride on #191775; #190596-specific test
cleanup splits out and floats with #190596.

This supersedes the Iteration 12 ordering and resolves the Iteration 13
dissent: the reviewed interleaved PRs land first, and contiguous is the
first consumer built natively on the new mechanism.

## Final resolution (2026-08-21) -- AUTHORITATIVE

The earlier ordering discussions and recommendations to retain Proposal A are
historical. The current #190595 review worktree removes
`_GroupInvariantBroadcast` and its projection-only tests; #190595 uses eager
group-to-lane broadcast.

Final order:

1. #190594 standalone interleaved NVFP4 foundation;
2. #190595 nested interleaved NVFP4, with eager projection;
3. #191775 MXFP6 factor-4/three-output-lane support;
4. uncommitted indexed-forwarding foundation;
5. uncommitted lazy derived-domain projection;
6. #190596 contiguous/SwiGLU support; and
7. #191974 test hygiene.

The indexed foundation and lazy projection are independently verified but do
not yet have commits or PR numbers. Their current records are:

- [indexed-forwarding README](/data/users/eellison/pytorch/agent_space/indexed_projection_work/README.md)
- [indexed-forwarding test results](/data/users/eellison/pytorch/agent_space/indexed_projection_work/TEST_RESULTS.md)
- [lazy-projection README](/data/users/eellison/pytorch/agent_space/domain_projection_work/README.md)
- [lazy-projection test results](/data/users/eellison/pytorch/agent_space/domain_projection_work/TEST_RESULTS.md)
- [lazy-projection architecture review](/data/users/eellison/pytorch/agent_space/domain_projection_work/ARCH_REVIEW.md)

The foundation replaces name-keyed forwarding without changing generated
code. The projection follow-up then restores group-resolution reuse and emits
the parent division before one factor-2 lane split. Persistent and looped
factor-4 remain on the byte-identical eager fallback. Iterations 12-15 above
are retained only to explain how this final order was chosen.
