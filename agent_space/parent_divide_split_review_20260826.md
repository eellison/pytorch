# Parent divide-before-split review

Date: 2026-08-27

Scope:

- final F1 exact-access resolver;
- final narrow F2a cast-before-widening change;
- the historical lazy-domain implementation and its benchmark artifacts; and
- the in-progress F2b prototype in
  `agent_space/followup_parent_divide_split_wt`.

This is a read-only design review. No production file was edited.

## Verdict

Reject F2b and retain F2a as the end state.

The final flat-parent prototype proves the optimization is real in a forced
persistent configuration, but it fails both landing gates:

- it is `+189/-12`, or `+177` net production lines over F2a; and
- a default-selected persistent workload is unchanged (`6.151 -> 6.144 us`).

The central helper has McCabe CC 15 (26 under the stricter branch counter) and
nesting depth 3. A clean relation trace shows
that the final FP8 scale is already an exact live group-width source, so the
prototype's generic group-only binary policy is unnecessary. The implementation
is therefore larger than the actual target requires even before considering
the failed default-performance gate.

There is a materially smaller representation:

```text
exact live parent source + proved lane
    -> _PendingParentLane(value, lane)
    -> replay only the measured parent-safe scalar ops on flat [X, R]
    -> at the first group join, use the existing group-to-parent broadcast
    -> execute the exact original op at flat parent width
    -> use the existing parent-to-child split once
    -> cache all lanes by the resulting CSE value
```

This needs one unregistered frozen dataclass and one stage-local split cache. It
does not need a domain enum, a structured-state bit, parent/group view helpers,
a view cache, a projection path, scheduler records, name buckets, or Triton
changes.

The design below remains the minimum acceptable shape for a future retry. It
can consume the existing exact final-scale relation; no planner expansion or
scale-expression interpreter is needed. A forced-persistent-only win does not
justify the current production surface.

## Prototype history

The first prototype used `_PendingSubParentLane` as a `NamedTuple` while its
dispatcher used pytree traversal. PyTorch pytree flattened the named tuple, so
the deferral and fallback checks did not see the wrapper. The final prototype
correctly uses an unregistered frozen dataclass.

The first structured-parent design was about `+214` net lines and was replaced
with the flat-parent design below. The flat rewrite reached the intended kernel
form, but its final stable snapshot is still `+177` net lines and does not help
the default-selected kernels.

## Minimal representation

The only new value state should be:

```python
@dataclasses.dataclass(frozen=True)
class _PendingParentLane:
    value: CSEVariable
    lane: int
```

The record means exactly: `value` is live at flat parent width and the current
consumer is entitled to select `lane` after any supported pointwise work. It
does not describe a general domain or layout.

The existing objects already own every other fact:

- `SubParentAccessRelation.parent_lane` is the planner proof for the selection;
- `_ResolvedSubParentSource` carries exact source access, guard, value, and lane;
- `CSEVariable.shape` distinguishes parent, group, singleton, and child width;
- `_GroupedReductionLayout` owns parent/group/child geometry;
- `_DerivedIterationFamily.set_value_masks()` owns target mask derivation; and
- `kernel.cse.contains_value()` owns lifetime.

The resolver may add one cache:

```text
parent_result_cse -> complete tuple of factor-2 lane values
```

The cache omits the lane. The resolver is already scoped to one stage, factor,
layout, and family, so repeating those facts in the key is unnecessary.

A clean F2a trace confirms the final FP8 scale is already recorded as an exact
live group-width `buf2` relation and resolved by both pack lanes. No new planner
record or scale-expression forwarding policy is needed. See
`agent_space/f2b_flat_factcheck_20260827/trace.log`.

## Minimal control flow

### Source load

Return `_PendingParentLane(resolved.value, resolved.parent_lane)` only when:

1. the relation is an exact F1 relation with `parent_lane is not None`;
2. the kernel is persistent and the stage factor is exactly 2;
3. the source CSE is still live;
4. the consumer has no extra load guard/fill to apply; and
5. the live source has exact flat parent width.

Otherwise retain F1's eager materialization or external reload. Do not infer
eligibility from a buffer name or scheduler node kind.

For persistent factor 2, `materialize_sources()` may leave lane relations lazy.
The later load remains the authoritative liveness check. Looped kernels retain
the current eager path because `codegen_body()` invalidates their parent CSE.

### Parent-only scalar work

Keep the pending value flat. This is required for ordinary CSE to match the
parent computation already emitted for the grouped reduction.

The first implementation should explicitly support only the operations shown
by the target kernels: `to_dtype`, `mul`, `truediv`, and the group-scale
`reciprocal` decomposition. An operation may remain at parent width when:

- every pending operand selects the same lane;
- every concrete tensor operand is scalar or parent-axis singleton; and
- the operation is pure, elementwise, and does not inspect indices or masks.

Unknown operations, mixed lanes, child-width operands, stores, reductions,
packed/impure inline assembly, indexing, and subgraphs materialize first.

Do not restore the historical global barrier set or a general scalar-op
allowlist. Add another operation only with a measured generated form that needs
it.

### Parent/group join

Do not create `[X, groups, G]` and `[X, groups, 1]` state initially. The fair
manual kernel that established the 1.18x opportunity used this simpler form:

```text
group scale [X, groups]
    -> existing group-to-parent broadcast [X, R]
flat parent value [X, R]
    -> exact original mul/div at [X, R]
    -> pending lane selection
```

Reuse `_GroupedReductionLayout._broadcast_value_to_axis_resolution()` through a
small, clearly named group-to-parent wrapper. It already handles the reshape,
broadcast, flatten, and FP8-safe bitcast path. This broadcast occurs once at
the consuming operation, not eagerly in each lane replay.

Only reconsider singleton structured views if the flat-parent prototype fails
the TTGIR or performance gate. They are not required by the historical physical
result and should not be paid for speculatively.

### Final lane materialization

The pending result remains flat parent width, so reuse
`_GroupedReductionLayout.materialize_value_at_sub_parent_resolution()`.
Cache the complete returned lane tuple by the result CSE and select the planned
lane. Check `kernel.cse.contains_value()` before every cache use. A stale
pending unnamed result is a compiler error, not a reload candidate.

Stores and unsupported operations call this materializer. A `masked` callback
must materialize any pending value before returning it to Triton's `masked`
implementation. A local callback wrapper is enough; no callback class or
disable counter is needed. New loads inside the callback are already excluded
by the nonempty guard condition.

The reciprocal override can emit its underlying constant/divide directly at
group width. It does not require the general binary helper to authorize
arbitrary group-only arithmetic chains.

## Proof obligations

### Index relation

The planner-provided `parent_lane` is the only selection witness. For factor 2:

```text
parent_r = 2 * child_r + lane
```

No codegen-side expression matching, source-name lookup, or new projection
record is permitted.

### Operation motion and rounding

The transformation commutes lane selection with the same elementwise operation:

```text
op(select_lane(parent), group_or_scalar)
    == select_lane(op(parent, broadcast_group_or_scalar))
```

This is valid only when every element sees the same operation, dtype conversion,
and scalar operands. The implementation must not algebraically rewrite divide
to reciprocal/multiply, reassociate arithmetic, remove casts, or move across a
rounding boundary. It emits the LoopBody's original operation once at parent
width. Bitwise equality against F2a under the same fixed configuration is the
acceptance criterion.

### Masks

Computing a pure elementwise operation on padded parent lanes is harmless only
because no side effect occurs and the final lane values receive masks derived
from the child shape. Split results must use
`family.set_value_masks(kernel, parts)`, replacing parent masks. Never copy or
union the parent `r0_mask` onto lane-shaped values.

Guarded source loads stay eager. Indirect indexing, assertions, stores,
reductions, random/stateful operations, and masked callback boundaries must see
materialized lane values.

### Liveness

Persistent selection is an opportunity, not a planner promise. Every raw
source and split-cache hit must satisfy `kernel.cse.contains_value()`. The cache
must remain owned by the resolver for one unflushed stage and must never survive
`codegen_body()`.

### Source identity

F1's exact `MemoryDep` plus `_AccessGuard` lookup remains authoritative. An
in-kernel required miss stays loud; an external miss keeps the existing reload
path. F2b must not add a parallel map keyed by name, FX node, expression text,
or scheduler ownership.

### CSE

Ordinary CSE may reunify the two separately replayed parent chains. A miss is
still numerically correct: each replay computes a full parent expression and
selects only its own lane. Kernel-form tests, rather than correctness logic,
make one parent chain and one split a performance contract. If that contract is
not stable, stop and revisit recorded inline provenance; do not add structural
expression matching.

## Machinery to reject

The following are unnecessary for the first F2b slice:

- a parent/group/structured domain enum;
- structured-parent state on the pending value;
- parent and group shape-construction helpers;
- a view cache;
- a custom structured split implementation;
- `emit_reshape_preserving_dtype` or any Triton change;
- a global projection barrier table;
- a generic domain factoring engine;
- scheduler-side expression reconstruction or provenance;
- layout or projection records beyond the existing `parent_lane` relation;
- name-keyed forwarding or cache keys; and
- launch-policy changes.

The current prototype's `is_structured_parent_shape`,
`can_structure_parent_value`, `structure_value`, and structured branch in
`materialize_parent_lane` should disappear under the flat-parent design.

## Benchmark audit

Historical numbers answer different questions:

1. Rewriting the Python expression produced byte-identical lane-first kernels.
   It demonstrated no compiler benefit.
2. The original manual `44.928 -> 30.656 us` (`1.47x`) result combined parent
   expression reuse with divide-before-split and is not the isolated ordering
   gain.
3. The fair manual comparison, with parent reuse on both sides, was
   `28.668 -> 24.279 us` (`1.181x`) for forced-persistent `4096x4096`. It also
   changed four to two TTGIR layout conversions and 4096 to 2048 bytes of shared
   memory. Reported spills increased from 168 to 194, so spill count must not be
   used as a monotonic correctness/performance gate.
4. The old compiler prototype's `42.880 -> 32.640 us` (`1.31x`) compared a much
   older foundation with the combined group and parent optimization and later
   mixed in launch-policy work. It is historical evidence, not an expected F2b
   delta.
5. Current matched F2a already measures `28.728 us`, 255 registers, 184 spills,
   and 4096 bytes shared at forced-persistent `4096x4096`. F2b's remaining gain
   was measured against the same wrapper/configuration: `28.667 -> 22.528 us`
   (`1.273x`), with 255 registers, 104 spills, and 2048 bytes shared. This
   clears the fixed-config gate.

The plan's 10% fixed-config gate is reasonable: F2b must beat 25.86 us on the
same current F2a wrapper/configuration. Replace the hard "no new spills" rule
with "no new spill cliff": record registers/spills, but decide using paired
latency plus the two-conversion/2048-byte physical target. The known good target
itself reported slightly more spills.

## Default-selection gate

All eight previously recorded F2a default rows (`NVFP4` and `MXFP4`,
D=4096/4608/8192) are looped and therefore cannot execute F2b. A compile-only
F2a probe now confirms the current default policy selects persistent code for
NVFP4 at D=512 and D=1024, and looped code at D=2048, for each of B=1, 128,
and 4096. The artifact is
`agent_space/f2b_default_probe_20260827/results.json`.

The confirmed default-persistent `4096x1024` row measured `6.151 -> 6.144 us`,
a `1.001x` tie. `4096x512` likewise measured `4.1014 -> 4.1015 us`. These
default configurations use small XBLOCK values and do not realize the layout
conversion benefit from the forced `XBLOCK=8` case. The useful compile matrix
for any future retry remains:

```text
B in {1, 128, 4096}
D in {512, 1024, 2048, 4096}
G = 16
```

F2b did not improve a representative default-selected row by 5%. Do not land it
without a separately reviewed launch-policy or planner change; configuration
changes are explicitly outside this slice.

Fresh generated wrappers confirm F2b did execute the desired order in both
default rows. The tie is physical rather than a missed optimization: default
uses XBLOCK 1 or 2, one warp, zero spills, and 512 bytes shared on both sides.
The forced XBLOCK=8 case is where split-first creates the extra shared-memory
and spill pressure that divide-before-split removes.

## Go/no-go rubric

Proceed to full validation only if the prototype satisfies all structural
conditions:

- simd-only production change;
- one unregistered frozen pending-lane record;
- one all-lanes split cache;
- no enum, structured state, view cache, projection/layout/name scaffolding, or
  scheduler/Triton/common change;
- target at most 100 net production lines, hard stop at 130;
- no new method above CC 10, no nesting above 2, and F2a+F2b below 200 net
  production lines; and
- unknown operations fail closed by materializing.

Keep F2b only if all behavior/performance conditions also pass:

- exact F2a/F2b outputs under each identical fixed configuration;
- persistent NVFP4 has one parent normalization, one scale conversion, one
  group-to-parent broadcast, one parent-width mul/div, and one split;
- TTGIR has at most two relevant `convert_layout` operations and about 2048
  bytes shared for fixed `4096x4096`;
- fixed persistent `4096x4096` is at least 10% faster than current F2a;
- at least one default-selected persistent workload improves at least 5%;
- no more than 2% repeatable regression in the paired default/fixed matrix;
- looped factor-2 and all factor-4/MXFP6 protected kernels are byte-identical to
  F2a after normalizing nondeterministic wrapper paths; and
- no new global traffic or kernel split appears.

The prototype fails the production-LOC, method-complexity, and default-benefit
gates. Keep F2a and record F2b as a measured future optimization. The current
F2a already delivers the large user-visible gain; F2b did not earn its
additional mechanism independently.

The formal method-by-method audit and the exact clean benchmark commands are in
`agent_space/parent_divide_split_complexity_audit_20260827.md`.
