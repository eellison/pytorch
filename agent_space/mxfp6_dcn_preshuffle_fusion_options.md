# MXFP6 DCN preshuffle fusion options

Date: 2026-08-26

This note separates the observed failure from the possible fixes. It does not
propose a DCN-specific compiler pattern.

## Desired result

For the production `(M, K) = (2048, 3072)`, group-size-32 workload, generate one
kernel that performs:

1. the optional DCN `addcmul`;
2. the group reduction and E8M0 scale calculation;
3. E2M3 conversion;
4. four-code-to-three-byte packing; and
5. row-major packed stores plus preshuffled scale stores.

The supplied hand-written kernel has exactly this structure. Its packed-output
tail mask needs a correction at this shape, but that does not change the target
kernel organization.

## What happens now

The reviewed `(4, 3)` worktree emits two kernels and never enters staged
sub-parent codegen (`codegen_nested_reduction == 0`):

| Kernel | Work | B200 time |
| --- | --- | ---: |
| 1 | DCN, group reduction, scale, E2M3 encode, then write all codes as `int32` | 17.11 us |
| 2 | Reload codes, pack 4 to 3, undo the packed-data permutation, write `uint8` | 13.09 us |

The end-to-end median is about 28-30 us. Kernel 1 writes a 25.2 MB temporary;
kernel 2 reads roughly 37.7 MB from that temporary because each packed result
uses four code values.

A one-kernel implementation is expected to take about 16-18 us. The corrected
hand-written DCN kernel takes about 24.1 us.

## Why the `(4, 3)` plan declines

The graph arranges the reduction in the scale-swizzled coordinate frame. The
encoded-code buffer is therefore produced with this logical shape:

```text
parent write sizes: [16, 4, 64, 16, 3, 32]
```

Its physical stride order is:

```text
[16, 16, 64, 3, 4, 32]
```

The packing node reads four consecutive code lanes in that physical order:

```text
child read sizes: [16, 16, 64, 3, 4, 8]
child addresses: 4 * child_ordinal + lane, lane in {0, 1, 2, 3}
```

These accesses cover exactly the same buffer. The relationship is a dense,
bijection-preserving permutation of the outer axes followed by the ordinary
factor-4 lane split.

The current proof in `_try_get_sub_parent_source_projections` is intentionally
narrower. It reduces both stages to `(x, r)` and requires:

```text
child(x, child_r) == parent(x, 4 * child_r + lane)
```

The outer `x` must be unchanged. In this workload the same physical element is
reached only after permuting the outer axes, so the proof rejects it. The
compiler then safely falls back to the two kernels above.

The realized code buffer makes this mismatch observable. The standalone form
can reload its external input in the derived frame and does not need to forward
this permuted internal buffer, so it reaches the existing one-kernel path.

## Proposal A: Keep the current two-kernel fallback

This is correct and requires no new compiler logic.

It is not a good performance result. It materializes a much larger `int32`
buffer than the final packed output and spends about 13 us packing it in a
second kernel.

Recommendation: reject as the intended production result, retain only as the
fail-closed behavior for unsupported layouts.

## Proposal B: Rewrite the source to swizzle only scales

Compute groups and packed bytes in row-major order, then reshape and permute the
small scale tensor. This is bit-exact to the existing pre-swizzled expression
and mirrors the hand-written kernel's logical dataflow.

Measured results:

- RMSNorm `(128, 384)`: one kernel, about 1.94 us.
- Standalone `(2048, 3072)`: two kernels, about 16.8 us.
- DCN `(2048, 3072)`: two kernels, about 18.0 us.

This is a useful source cleanup and performance baseline, but at production
size it still emits a separate scale-layout kernel. It therefore does not
preserve the one-kernel property of the successful `(4, 3)` cases.

Recommendation: useful independently, but insufficient as the compiler fix.

## Proposal C: Canonicalize the staged iteration frame

Use the exact producer write and consumer reads already present in the plan to
prove a dense relation:

1. one injective producer write;
2. all consumer reads address the same region;
3. the outer axes differ only by a bijective permutation;
4. the final source axis is split by the planned factor; and
5. every required lane is represented exactly.

When this proof succeeds, reorder the staged parent into the shared physical
frame, rebuild the complete plan, and keep the speculative loop changes only if
all ordinary fusion and staged dependency checks pass.

This is not a DCN rule. It is a general answer to:

```text
Can these exact producer writes and consumer reads share one iteration frame?
```

A scratch experiment established both parts of the mechanism. Applying the
producer write's actual stride permutation changes the failing source
projection from rejected to accepted. Canonicalizing the subsequent epilogue
chain into that frame then lets the unchanged exact planner fuse all nodes into
one nested kernel with four stores. Flat loop reindexing alone is not
sufficient; it preserves the old logical order and only rewrites it into
modular indexing.

The prototype has not yet been integrated or benchmarked. A production-quality
version is estimated at roughly 80-150 scheduler lines plus focused tests,
reusing the existing loop-mutation and older dependency-order machinery. It
does not require a new codegen projection layout.

Advantages:

- preserves the current `(4, 3)` representation;
- removes the large code temporary rather than hiding it;
- is driven by exact dependencies, not operation names;
- can be fail-closed by rerunning the existing full plan; and
- can later support other dense layout permutations.

Risks:

- every parent-stage node must be reindexed consistently;
- rollback must cover every unsuccessful fusion attempt;
- dynamic or unit-size axes need a conservative policy; and
- codegen must reconstruct the same plan from the retained loop state.

Recommendation: preferred direction, provided the implementation remains a
small exact-proof/canonicalization step rather than a new general layout system.

## Proposal D: Replay or recompute the producer in the child frame

Instead of forwarding the realized code buffer, replay its pure pointwise
producer chain after the reduction using the child iteration frame. External
inputs and completed reduction outputs would be loaded at the required indices.

This is general and avoids cross-thread register permutation, but it introduces
a larger compiler contract:

- which operations are safe to replay;
- how mutation, masks, guards, and side effects are excluded;
- how replay cost is bounded;
- how duplicated computation interacts with CSE; and
- how looped reductions keep the replay in the final pass.

Recommendation: plausible future indexed-forwarding/recomputation work, but too
large for the narrow production fix unless canonicalization proves impossible.

## Proposal E: Use three Concat slice stores

Cast the three packed lanes to `uint8`, lower the trailing stack as a
`ConcatKernel`, and fuse its three disjoint output slices.

This works for row-major packing. The prototype passes the focused MXFP6 suite
and removes about 145 scheduler/codegen lines. Transparent support, however,
adds a specialized post-grad cast rewrite and concat lowering rule, leaving only
about 45 net production lines removed.

For the pre-swizzled graph, the three permuted slice nodes must also be admitted
and scheduled atomically as one complete Concat partition. That adds alias
grouping on top of the same frame problem described above.

The frame canonicalization in Proposal C could eventually serve Concat too, but
Concat is not simpler for this failing case:

- `(4, 3)` has one epilogue node;
- Concat has three aliasing epilogue nodes;
- all three must be present before any one is safe to fuse; and
- the compiler must prove their slices form the complete disjoint output.

Recommendation: do not switch representations to fix this issue. Reconsider it
later only if Concat has an independent product requirement.

## Proposal F: Push the final permutation into output layout planning

Keep packing in the parent-aligned frame but make its stores target the final
permuted output layout directly, eliminating the copy.

This could be a useful general layout optimization, but it crosses FX/IR layout
propagation, output allocation, aliasing, and scheduler legality. It is broader
than the staged-reduction problem and would need its own design and regression
surface.

Recommendation: not for this PR.

## Proposed next step

Continue with Proposal C in an isolated prototype:

1. turn the successful scratch canonicalization into a small planner helper;
2. apply it speculatively through the scheduler's existing loop-mutation API;
3. rerun the unchanged staged plan and ordinary fusion legality;
4. require one kernel and no code-buffer allocation;
5. compare numerics against the unfused compiled reference; and
6. benchmark against the current 28-30 us path and corrected 24.1 us DCN kernel.

Stop if this requires general replay, arbitrary affine maps, or Concat-specific
grouping. In that case the reviewed `(4, 3)` design should remain unchanged and
the production graph should use the two-kernel fallback until the broader
indexed-replay work is ready.

## Acceptance criteria

- No `addcmul`, DCN, MXFP6, or fixed-shape checks in scheduler/codegen.
- The planner proves one exact dense permutation and complete lane coverage.
- Failure at any proof or legality step restores loop state and declines fusion.
- Production DCN+preshuffle emits one kernel with four final stores and no
  full-size `int32` code allocation.
- Standalone, RMSNorm, row-major, preshuffled, persistent, and looped tests stay
  correct.
- Expected production latency is approximately 16-18 us and must not regress
  the existing successful `(4, 3)` cases.
