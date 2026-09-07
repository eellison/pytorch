# F1 access-resolution design options

Status: design proposal only. No option in this document has been selected or
amended into the review commit.

## What problem are we solving?

The sub-parent planner permits an epilogue to run at a narrower iteration width
than its producer. Before fusion, it proves the exact relationship between each
source access and each epilogue read. Codegen must then reuse the already-emitted
register value instead of blindly reloading the buffer.

That reuse has four correctness requirements:

1. A parent-width value may need one lane selected for the narrower epilogue.
2. A group-width value may need broadcasting to the narrower epilogue width.
3. An epilogue-produced value may be forwarded directly to a later epilogue
   node.
4. A value produced in the same kernel cannot silently fall back to a memory
   reload. If its register value is unavailable, compilation must fail loudly.

The existing operation handler receives only:

```text
load(buffer_name, emitted_index)
```

The planner's proof is expressed as scheduler `MemoryDep`s in the node's loop
frame. The emitted index has already been remapped into the derived kernel
frame. The design question is therefore:

```text
How much of the planner's exact per-access proof must codegen retain, and how
should the specialized wrapper identify the register value for a load?
```

This is not a request for a general indexed-CSE redesign. The immediate goal is
the smallest sound bridge for sub-parent codegen, with no behavior change to
ordinary Triton codegen.

## Facts we can rely on

The planner already fails closed on the important ambiguity cases:

- For a parent input, every parent-side access to a tracked buffer must normalize
  to the same logical index.
- A reduced broadcast or epilogue-internal source has one unique writer.
- Every epilogue read of a tracked name is checked. One invalid read rejects the
  entire staged plan.
- Parent-lane reads are individually proved to be
  `parent_r = factor * child_r + lane`.
- Requiredness is known from whether the source is written inside the kernel.
- Codegen receives the final staged plan after loop normalization; it does not
  need to rediscover fusion legality from scratch.

These facts mean fusion requires exact per-access records, but codegen may be
able to consume a simpler, verified view of those records.

Put plainly, Option C is a principled return to name-keyed codegen contract
lookup. Parent-width values still use the actual replay index to derive and
validate a lane, so it is not purely name-only forwarding. It is also not a
return to name-based fusion permission. The earlier wrong-fusion class came from
letting a buffer name authorize a relationship that fusion had not proved.
Under Option C:

1. Fusion still requires exact plan membership for every residual read.
2. Codegen rebuilds the sub-parent relation plan from the final node
   representation and fails if the plan is lost. It does not rerun the complete
   fusion dependency proof.
3. Only then does the specialized wrapper collapse the proved relationships to
   the per-name operation it needs.

The current F1 runtime access lookup is therefore a third verification of a fact
already established at fusion and final-plan construction. Option C removes that
third check and its interpreter bridge; it does not weaken either load-bearing
planner check.

One property is intentionally lost: codegen will not independently detect a
same-name read whose runtime index is shifted but still derives a permitted
lane. Safety for that case rests on exhaustive `LoopBody` dependency capture,
fusion-time exact plan membership, and final relation-plan reconstruction. This
trade must remain explicit rather than being presented as identical runtime
checking.

## Required behavior for every option

- Fusion legality remains based on exact `MemoryDep` relationships.
- An unproved read must never receive a same-name register value.
- A required in-kernel source miss is a compiler error.
- An optional external source may issue an ordinary physical load.
- Existing `ops.masked` behavior owns predicates and fill values.
- Source values that are still live in CSE may be reused; stale values may not.
- Persistent and looped forms must generate the same kernels as the reviewed
  implementation unless a deliberate codegen improvement is separately
  approved.

## Option A: reconstruct the current `MemoryDep` on every operation

This is the current F1 design. The resolver combines the currently interpreted
FX operation with the current `SchedulerNode`'s `LoopBody`, reconstructs the
logical `MemoryDep`, and looks it up in `SubParentAccessRelation`.

Advantages:

- Exact per-read and per-write identity.
- Loud name-only misses.
- Already validated by the full suites and protected kernel corpus.
- No generic scheduler, `LoopBody`, or CSE API changes.

Costs:

- Codegen reaches through both `V.interpreter.current_node` and
  `kernel.current_node`.
- It duplicates part of dependency extraction inside `simd.py`.
- The logical operation is reconstructed on every load/store.
- It does not resemble the existing NestedReduction wrapper style.

Moving the helper onto `_SubParentValueResolver` contains the implementation but
does not remove the conceptual problem.

## Option B: bind finalized FX operations to relations before emission

The resolver scans the finalized `LoopBody` graphs once and creates maps such as:

```text
FX load node  -> planned consumer relation
FX load/store -> planned source access
```

Runtime resolution becomes a direct lookup of `V.interpreter.current_node`.

Advantages:

- Preserves exact per-operation identity.
- Removes repeated reconstruction and `kernel.current_node` coupling.
- Remains local to sub-parent codegen.

Costs:

- Introduces a new binding pass and requires passing all relevant scheduler
  nodes into the resolver.
- FX-node identity is not a durable planner/codegen contract; cloned
  `LoopBody`s may share graph nodes.
- It still depends on interpreter internals, just earlier.
- It adds machinery whose only purpose is adapting one exact representation to
  another.

The `_bind_access_relations` prototype is this option. It is more explicit than
Option A, but not simpler enough to justify the additional lifecycle and API.

## Option C: collapse the exact plan to the wrapper contract codegen needs

Keep `SubParentAccessRelation` as the fusion proof, but do not identify every FX
load again during emission. When constructing the specialized resolver, validate
and collapse the records into a per-buffer codegen contract:

```text
buffer name -> source names, requiredness, and allowed parent lanes
```

Then use the existing wrapper idiom:

- Source emission records live candidate values by buffer name.
- A derived-stage `load(name, index)` asks the resolver for that name.
- Materialization uses the live value's shape.
- If the value is parent-width, the wrapper derives the lane from the actual
  remapped load index using the existing `interleaved_sub_parent_lane` logic.
- The derived lane must be one of the lanes proved for that name by the planner.
- Direct and group-width values need no per-read identity at codegen.

This is sound only because the exact planner has already proved all accesses for
the name. The collapse must reject inconsistent records, including different
requiredness or a mixture of lane-selecting and non-lane relationships for one
buffer.

The current planner establishes those conditions:

- The parent-lane builder examines every parent read/write of a selected name,
  requires one parent-frame index, and emits a relation for every child read.
- The broadcast and internal builders require one writer and validate every
  same-name consumer.
- The three source-name classes are disjoint.
- `requires_live_source` is consequently constant for a name.

Multiple reads of one name are still supported. For example, the MXFP6 byte
nodes read adjacent lanes of the same source. Codegen derives each lane from
that load's real index and checks it against the planner-proved lane set. Reads
with different syntactic indices but the same non-lane relationship also reuse
the same materialized child-width value.

Advantages:

- Closest to the pre-F1 NestedReduction wrapper and existing Inductor style.
- Removes FX/interpreter access reconstruction and `_bind_access_relations`.
- Keeps exactness where it is needed: fusion planning.
- Keeps normal codegen unchanged.
- Likely deletes code rather than adding a new adapter layer.

Costs:

- Codegen trusts the planner's exhaustive coverage instead of independently
  matching each load again.
- Multiple live shapes for one equivalent source name must be retained as
  alternatives rather than overwritten. The existing flat and grouped source
  aliases demonstrate this case.
- Source capture must respect the planned role: optional external sources are
  captured from loads; required in-kernel sources are captured from stores. A
  later same-name store must not replace an external input.
- If a future name genuinely mixes lane-selecting and non-lane relationships,
  planning must reject it until the wrapper contract is extended.

This is the preferred option to prototype. It uses the exact relation as a proof
artifact without requiring codegen to replay the proof mechanism.

The collapse must also retain a pre-materialization seam. F2a needs to inspect a
live group-width source before broadcasting it so the narrowing cast remains at
group resolution. The exact API can change, but codegen must still be able to:

```text
find live candidates -> inspect candidate shape -> materialize only if needed
```

The F2a fast path remains restricted to a proved, live source with no parent
lane, an unmasked consumer, and a true group-width shape. The narrowed result is
then materialized immediately; masked, lane-selecting, stored, or already
materialized values remain on the eager path.

## Option D: give planner records stable operation identities

Extend relations with a durable source/consumer operation-site identity, then
pass those identities through fused scheduler nodes and `LoopBody` cloning.

Advantages:

- The cleanest exact handoff in principle.
- No codegen inference or ambient interpreter lookup.

Costs:

- Requires a new cross-layer identity and lifecycle contract.
- Fused scheduler nodes currently discard satisfied internal dependency edges.
- LoopBody cloning, subblocks, grouped reduction remapping, and mutation renames
  all need explicit treatment.
- Broad scheduler and codegen blast radius for a specialized feature.

This is not appropriate for F1. It belongs with a general retained-access or
indexed-CSE design if that work is pursued.

## Option E: derive the complete plan again in codegen

Codegen could inspect final node bodies and rebuild the forwarding relations.

Advantages:

- Operates on the final emitted representation.

Costs:

- Creates two planners or moves fusion legality into codegen.
- Risks fusion accepting a kernel that codegen later cannot emit.
- Duplicates the most complicated proof logic in the stack.

Reject this option.

## Option F: general indexed CSE / retained fused-node dependency matches

Teach the general compiler representation to retain exact internal access
matches, or make CSE key register values by logical access rather than buffer
name.

Advantages:

- A general solution that could remove the specialized bridge entirely.
- Potentially useful beyond sub-parent reductions.

Costs:

- Cross-cutting behavior change to ordinary fusion and codegen.
- Considerably larger review and regression surface.
- Not necessary to land the current feature.

Record this as a possible later direction, not part of F1.

Options D and F deliberately generalize the mechanism into core compiler state.
If pursued, that should be a separate project with an explicit decision to
change generic codegen, not incremental scope growth from F1.

## Comparison

| Option | Exact at runtime | Normal-codegen changes | New machinery | Fits existing wrapper style |
| --- | --- | --- | --- | --- |
| A. Reconstruct on use | Yes | None | Medium | No |
| B. Bind FX operations | Yes | None | Medium/high | No |
| C. Collapse proved plan | By planner contract | None | Low | Yes |
| D. Stable operation IDs | Yes | Broad | High | Partly |
| E. Replan in codegen | Yes | Medium | High | No |
| F. General indexed CSE | Yes | Broad | Very high | General solution |

## Recommendation

Prototype Option C before keeping either `_current_access` or
`_bind_access_relations`.

The intended boundary is:

```text
fusion planner
    proves every exact source/read relationship
    rejects ambiguity or incomplete coverage
        |
        v
sub-parent stage
    exposes a validated codegen view per buffer
        |
        v
specialized replay wrapper
    reuses live values, derives and validates a lane from the actual load index
    when needed, and otherwise reloads only optional external sources
```

If Option C cannot preserve the target kernels without adding comparable
machinery, retain Option A privately inside `_SubParentValueResolver`. Do not
take Option B merely to hide the same reconstruction behind a binding phase.

## Containment boundary

Option C is acceptable only if its implementation remains a sub-parent codegen
detail:

- No changes to `common.py`, generic `triton.py` paths, `LoopBody`, dependency
  extraction, or generic CSE behavior.
- No new state on `SIMDKernel`, `CSEProxy`, or generic iteration-range classes.
- Resolver construction and all new runtime branches remain guarded by the
  presence of a sub-parent stage.
- Ordinary pointwise, reduction, and nested-reduction kernels without a
  sub-parent stage remain behaviorally and textually unchanged.
- Keep the local load path that intentionally bypasses name-only store-cache
  forwarding. Its comment must enumerate the `CSEProxy.load` behaviors retained
  so future drift is reviewable; extracting a generic helper is not required for
  this PR.

## Acceptance gate for Option C

1. The containment boundary above is satisfied, including a neutral non-sub-
   parent kernel-form check.
2. The planner rejects a tracked name with incomplete or inconsistent consumer
   coverage before codegen.
3. Same-name source alternatives are all equivalent by planner proof and remain
   available until materialization.
4. Multiple reads of one name at different proved lanes remain supported.
5. The post-collapse desynchronization tripwires are explicit and tested:
   final relation-plan rebuild failure, inconsistent per-name collapse, derived
   lane outside the proved lane set, invalid materialized shape, and required-
   source materialization failure. Existing exact-runtime-miss tests are
   migrated to these walls rather than simply deleted.
6. Source capture is exhaustive and role-aware: every optional same-name load is
   a proved equivalent alternative, required values come from stores, and mixed
   requiredness for one name is rejected.
7. Optional fallback bypasses generic name-only store-cache forwarding.
8. Required-source misses still raise.
9. Masked sources and consumers retain existing behavior.
10. F2a retains a live-value-before-materialization seam and its admission
    corpus has no behavior disagreements after rebasing.
11. Option A and Option C agree on fusion/decline and kernel count across both
    the protected targets and the shape-generating adversarial corpus.
12. Full scheduler and nested-reduction suites pass.
13. All ten protected generated-kernel hashes remain identical.
14. The resulting production diff and cyclomatic complexity are lower than
   Option A, not merely rearranged.

Only after this gate passes should F1 be amended again.

## Decision process

Build Option C in a scratch worktree and compare it directly with the current
Option A snapshot. If every gate passes, amend F1 once, rebase F2a, re-attest the
corpus, refresh the review guide, and freeze F1 for review. If any gate requires
machinery comparable to Option A, discard the prototype, keep Option A with its
bridge documented locally, and freeze that version instead.
