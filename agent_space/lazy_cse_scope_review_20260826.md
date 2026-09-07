# F2a scope and proportionality review

Date: 2026-08-26

Worktree reviewed:
`/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt`

## Findings

### The measured optimization needs only `to_dtype`

All 12 matched NVFP4 wrappers have the same sole semantic source change:

```text
F1: broadcast FP8 through a uint8 bitcast, bitcast back to FP8, then cast to FP32
F2: cast the group-width FP8 value to FP32, then broadcast FP32
```

The next reciprocal/division and packing arithmetic is already child-width in
every target wrapper. All 12 MXFP4 wrappers are normalized-source identical
between F1 and F2. The factor-4 MXFP6, internal-source, swizzle, preshuffle, and
DCN captures are also unchanged.

The general policy for `abs`, arithmetic, comparisons, `where`, and pure pack-1
inline asm therefore produces no additional win in the real target corpus. Its
only visible beneficiary is the synthetic `factor2_chain` test.

### The target does not require masked group-width execution

Instrumented capture of all 12 NVFP4 cases, including the 4096x4608 looped tail,
showed that every preserved source has:

```text
parent_lane = None
consumer guard = _AccessGuard(mask=None, fill=None)
shape = (XBLOCK, nested_R0_REDUCED_BLOCK)
requires_live_source = True
```

The reduction and output tail masks are still required, but they are target
family masks installed by the existing materializer after widening. They are
not an `ops.masked` callback guard on the forwarded scale load.

Consequently, the measured path does not need to carry group-width values
through `masked()`. Masked and scalar-fill consumers can remain on F1's eager
materialization path without affecting any target source or performance result.

### The current generalized mechanism is disproportionate to its demonstrated use

The current production delta is `+119/-4`, net `+115`, and adds:

- a 16-operation positive allowlist;
- canonical shape-propagation preflight for every admitted operation;
- pure pack-1 inline-asm policy;
- a masked-callback wrapper and callback graph preservation;
- generic nested-pytree argument/result handling;
- tests for mixed widths, unknown shapes, packed asm, impure asm, arbitrary
  arithmetic chains, predicates, and callback returns.

These pieces are internally coherent, but most of them defend future behavior
rather than behavior needed by NVFP4, MXFP4, or MXFP6 today. The policy also
creates a second manually curated description of which operations are pure and
safe to execute at group width. That is a real maintenance surface even though
it fails closed for omitted operations.

## Recommended narrow contract

Keep the indexed F1 resolver and make F2a mean exactly:

> For an exact, live, lane-free, unguarded group-resolution source, perform one
> shape-preserving `to_dtype` before broadcasting it to child resolution.
> Materialize before every other operation or side effect.

Concretely:

1. `resolve_load()` returns a raw group value only when `parent_lane is None`,
   `consumer_guard.mask is None`, and the value has a true group-width shape.
2. `_PointwiseRemapHandler.to_dtype()` applies the cast to the group value,
   verifies that the returned `CSEVariable` is still group-width, and immediately
   calls the existing group-to-child materializer.
3. `_default()` materializes any group-width argument before delegating every
   other operation.
4. `store()` continues to materialize a directly stored group value.
   `store_reduction()` reaches the same fail-closed behavior through `_default()`.
5. Remove the custom `masked()` override. A load inside a masked callback is no
   longer eligible for raw deferral, while the target's ordinary tail masks are
   still assigned when the cast result is materialized.
6. Retain the `num_groups != child_block` distinction so the G==factor direct
   case is never mistaken for a deferred group value.

This is not an NVFP4 graph-name special case. It is a general algebraic rule:
an elementwise dtype conversion commutes with replication, so
`broadcast(cast(x))` is exactly equivalent to `cast(broadcast(x))`. It applies
to any planned sub-parent relation with that immediate use while declining all
other expression-domain motion.

## Expected simplification

The narrow implementation can remove the operation allowlist,
`ShapePropagationOpsHandler` dependency and exception path, inline-asm policy,
and masked callback wrapper. It should reduce the production delta from net
`+115` to roughly `+55` to `+70` lines, with lower cyclomatic complexity and no
new public API.

The scheduler unit-test addition can shrink from 163 lines to roughly 50-70:

- unguarded/lane-free/group-shaped resolver admission;
- masked, scalar-fill, lane-selected, child-shaped, and unknown-shaped eager
  fallbacks;
- cast-before-materialize ordering and result-shape assertion;
- generic non-cast and direct-store materialization;
- the G==factor direct case.

The existing NVFP4 kernel-form test remains the strongest positive oracle. The
generic arithmetic/compare/inline-asm policy test and its synthetic 500-recipe
fuzz matrix are no longer part of the production contract and can be deleted or
reduced to the narrow cast boundary.

## Tradeoff

The narrower implementation intentionally gives up delaying arbitrary
group-only arithmetic, compare/`where` chains, and pure inline asm. A future
graph containing such a chain will widen at its first non-cast operation and
remain correct, just less optimized. If a real workload later demonstrates a
gain, the policy can be expanded one operation family at a time with that
kernel as evidence.

This does not sacrifice a current MXFP4 optimization: MXFP4 already computes
its UE8M0 conversion and reciprocal in the grouped stage before the forwarding
boundary, so F1 and F2 emit identical code. It also does not advance the
separate divide-before-split idea; the current general policy does not move the
division in the measured NVFP4 graph either because the cast result reaches a
node/store boundary first.

## Recommendation

Narrow F2a before landing. The cast-only contract preserves the complete
measured NVFP4 gain, leaves all existing control kernels unchanged, removes the
unexercised generalized policy and masked-callback semantics, and is materially
easier to explain and review.

This recommendation is also consistent with avoiding one-off carve-outs. The
current 16-name allowlist is not a first-class lazy-CSE abstraction; it is 16
operation-specific permissions plus shape preflight. Replacing that list with
one exact algebraic rewrite is fewer carve-outs, not more. The code and PR should
name the optimization honestly as cast-before-sub-parent-broadcast rather than
claiming general delayed pointwise evaluation.

If general delayed broadcasting remains a goal, pursue it separately with an
explicit value-domain representation and a consumer that spans scheduler-node
boundaries. The current mechanism does not keep NVFP4's reciprocal or division
at group width because the converted scale reaches a node/store boundary first,
so retaining the larger allowlist would not actually complete that broader
goal.

Acceptance for the narrow prototype should be strict and cheap:

- all 12 NVFP4 normalized sources match the current F2 source exactly;
- all 12 MXFP4 and all protected factor-4 sources remain unchanged;
- exact F1/F2 numerics remain green;
- one representative default, looped, and persistent NVFP4 paired replay
  reproduces the current performance and resource result.

If the narrow prototype cannot meet those source-equality checks, retain the
current implementation. Based on the observed LoopBody and emitted kernels,
there is no expected reason for it to fail.
