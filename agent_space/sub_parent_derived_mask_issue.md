# Sub-parent derived values lose tail-mask metadata

## Summary

Sub-parent codegen reshapes, splits, or broadcasts register values from one
iteration domain into another. The generated values are numerically valid only
where the target sub-parent iteration is valid, but the new
`TritonCSEVariable` objects do not record that target mask in `mask_vars`.

This is normally invisible because the final output store is masked. It becomes
a correctness issue when a derived value is used for indirect indexing: the
indirect global load happens before the final store and therefore needs its own
tail predicate.

The issue predates #190595 in the standalone sub-parent path from #190594.
#190595 adds nested split and broadcast uses that expose the same missing
contract.

Reproductions:

- `agent_space/rev195_cg/attack8_standalone_oob.py`
- `agent_space/rev195_cg/attack7_oob.py`

Both persistent and looped nested kernels reproduce the failure on a masked R
tail. The flagship NVFP4 and MXFP4 paths do not use indirect indexing and are
not affected.

## Concrete example

Consider a parent reduction with `D=4608` compiled using a larger reduction
block. The block contains valid and invalid elements:

```text
parent value:       [B, R0_BLOCK]
parent validity:    r0_index < 4608
```

A factor-2 sub-parent epilogue creates:

```text
lane 0:             [B, R0_BLOCK / 2]
lane 1:             [B, R0_BLOCK / 2]
sub-parent validity: child_r_index < 2304
```

The same issue occurs when a group value is broadcast:

```text
group value:        [B, D / G]
broadcast result:   [B, D / 2]
sub-parent validity: child_r_index < D / 2
```

The values outside the valid sub-parent extent contain load fill values or
arithmetic derived from those fills. That is harmless for ordinary arithmetic:

```python
result = invalid_lane_value * scale
tl.store(output, result, mask=sub_parent_mask)
```

The invalid result is discarded by the store mask.

It is not harmless when the value becomes an index:

```python
index = convert_to_int(invalid_lane_value)
value = tl.load(table + index)  # must be masked here
tl.store(output, value, mask=sub_parent_mask)
```

The table load executes before the masked store. In the reproducer, an invalid
tail scale is clamped to `1e-12`, producing a large out-of-range index. With
device assertions enabled the kernel raises; without them it performs an
out-of-bounds global load.

## Why current metadata is lost

`TritonCSEVariable.mask_vars` records the predicates required when a value is
later used for indirect indexing. Ordinary loads set it from their indexing
result, and ordinary scalar operations propagate it through
`TritonCSEVariable.update_on_args()`.

The derived-domain emitters bypass that propagation:

- `TritonKernel.emit_split_via_reshape()` writes assignments to newly created
  variables but does not populate their `mask_vars`;
- `TritonKernel.emit_broadcast_via_reshape()` creates a CSE expression from a
  string, without an input argument from which mask metadata can propagate;
- `TritonKernel.emit_reshape()` has the same general metadata boundary.

Blindly copying the source mask is not sufficient. A parent mask has parent
shape, while the split or broadcast result has sub-parent shape. The result
needs the mask for its target iteration domain.

For example:

```text
wrong metadata: parent r0_mask shaped for [B, R0_BLOCK]
right metadata: sub-parent mask shaped for [B, R0_BLOCK / F]
```

## Option A: conservatively decline fusion

Reject a sub-parent epilogue containing `indirect_indexing` in the shared
source-layout proof used by standalone and nested planning:

```python
if any(node._body.has_op("indirect_indexing") for node in epilogue_nodes):
    return None
```

The user program still compiles and runs through ordinary kernels. This is a
small correctness fix, but it creates a capability carve-out and leaves the
derived-value mask contract incomplete.

## Option B: preserve the target-domain mask

Make derived materialization return values carrying the sub-parent iteration
mask:

1. The caller that owns `_DerivedIterationFamily` determines the masks for the
   target range trees.
2. Split results receive those target-domain masks, not the parent source mask.
3. Broadcast results receive the same target-domain masks.
4. Reshape helpers preserve or explicitly transform mask metadata rather than
   silently dropping it.
5. A later `ops.indirect_indexing` sees the derived value's `mask_vars`, so its
   bounds assertion and `tl.load` include the sub-parent predicate.

This should not add runtime work to NVFP4/MXFP4. Their generated kernels do not
use the metadata. In an indirect-indexing kernel, it should reuse the existing
sub-parent mask in the gather:

```python
tl.load(table + index, sub_parent_mask, other=...)
```

The implementation should be explicit at the derived-domain boundary rather
than changing every generic CSE expression. `_GroupedReductionLayout` and
`_DerivedIterationFamily` know both the source and target domains, so they are
the natural place to provide the transformed mask metadata to the Triton
emitters.

## Required tests for direct support

1. Standalone factor-2 sub-parent epilogue with a masked R tail and an indirect
   lookup derived from one split lane.
2. Nested factor-2 epilogue with a masked R tail and an indirect lookup derived
   from a broadcast group value.
3. Each test in forced persistent and forced looped modes.
4. Require one staged kernel and numerical equality with eager execution.
5. Inspect generated code to confirm the indirect bounds check and table load
   include the derived sub-parent mask.
6. Retain the existing NVFP4/MXFP4 kernel-form assertions to ensure mask
   metadata does not alter kernels that never perform an indirect access.

Useful dimensions are `D=4608`, `G=16`, and a configuration whose reduction
block exceeds or leaves a tail relative to `D`.

## Recommendation

Prototype Option B first. The target sub-parent mask already exists, so the
likely implementation is localized metadata propagation plus focused tests.
Use Option A only if the direct implementation requires a broader redesign of
mask ownership or changes existing generated kernels unexpectedly.

The acceptance criterion is not merely that the device assertion disappears:
the generated indirect load itself must be guarded by the target sub-parent
mask, and results must match eager execution with assertions both enabled and
disabled.

## Scoped improvements (coordinating agent, 2026-08-24, for convergence)

Four separable pieces, then a sequencing position that differs from the
"prototype Option B first" recommendation above.

### I1: plan-time reject (Option A), scoped

~5 lines in the SHARED source-layout proof so one gate covers standalone and
nested (reject when any epilogue body's op_counts contains
`indirect_indexing`), plus attack7/attack8 converted to decline tests
(2 kernels, correct numerics, both forced modes). Removable: when I2 lands,
the same tests flip to fusion tests. Independent value: this is the only
option that immediately closes the exposure in LANDED #190594.

### I2: target-domain mask metadata (Option B), made concrete

Two technical facts make this smaller than it looks:

1. **The derived mask is exact, not conservative.** The planner proves
   `factor | parent_extent`, so for a split lane, child element `k` is valid
   iff `k < parent_extent / F` iff parent element `F*k + lane` is valid, for
   every lane < F. Same argument for group broadcasts. So the derived tree's
   own mask is precisely the value's validity -- no per-lane mask arithmetic
   is needed.
2. **The mask-mapping rule is a tree-filtered union.**
   `result.mask_vars = {t.mask_name() for derived r-trees} | (source.mask_vars
   - masks owned by the replaced parent r-tree)`. The x-tree mask carries
   through unchanged (shape-compatible across domains);
   `IterationRangesEntry.owns_mask` (simd.py:236) classifies ownership;
   derived trees expose `{name}_mask` (simd.py:469) and deliberately keep
   masks explicit (simd.py:472-474), so the symbol is available at the point
   of use inside the family's active context.

Implementation shape: `_GroupedReductionLayout` /
`materialize_value_at_sub_parent_resolution` compute the mapped mask set
(they know both domains) and pass it to `emit_split_via_reshape`,
`emit_broadcast_via_reshape`, and `emit_reshape_preserving_dtype`, which
assign it on every created variable. Size estimate ~150-250 lines including
the six tests already listed above. Acceptance: the generated gather and
bounds assert carry the sub-parent mask, AND every existing
NVFP4/MXFP4/MXFP6 kernel-form pin is byte-unchanged (the metadata is unused
when no indirect access exists).

### I3: loudness, so this class cannot recur silently

Rides I2: make the emitters REQUIRE the mask_vars argument (explicitly empty
allowed) instead of defaulting, so any future forwarding mechanism that
forgets masks fails at authoring time; and add a remap-context assertion in
`indirect_indexing` that the operand's mask_vars cover the active family's
required masks. The bug survived three reviews because empty mask_vars is
indistinguishable from "no mask needed."

### I4: the same fix one layer up (F2)

`_DerivedDomainProjection._copy_masks` in the uncommitted projection layer
blind-copies SOURCE mask_vars onto lane-shaped results -- the exact mistake
the "wrong metadata" section above warns about. Replace it with I2's
tree-filtered rule (share the helper), and add an indirect-indexing case to
the F2 differential battery. Fix in place before F2 becomes a PR.

### Sequencing position (disagreement to converge on)

Land #190595 with I1 NOW; do I2+I3 as the immediate follow-up PR that
deletes the reject and flips its tests; I4 lands inside the F2 layer. The
"prototype B first" recommendation implicitly holds #190595 (land-ready,
review-fatigued) hostage to new mask-ownership design surface, and leaves
the landed #190594 exposure open meanwhile. I1 costs 5 lines and is erased
by I2; there is no scenario where landing it first is regretted.

### Convergence questions for the second agent

1. A-then-B versus B-only, given the landing pressure and that only A fixes
   landed #190594 immediately?
2. Emitter API: required mask_vars parameter (fail-loud, more signature
   churn) versus layout-level assignment on returned variables (quieter,
   easier to forget at a new call site)?
3. Does the tree-filtered union cover sources loaded under `ops.masked`
   (`_load_mask` extra predicate) -- is that predicate reflected in
   mask_vars, or only in the value's fill? Needs a code check before I2's
   rule is final.
4. Exact placement of I1's gate so one predicate provably covers both the
   standalone and nested planners.

## Resolution implemented for #190595

The local #190595 worktree implements Option B directly. There is no planner
rejection and no special case in `indirect_indexing`.

The implementation follows the existing Triton CSE convention: a value gets
its `mask_vars` when that value is created in its actual iteration domain, and
ordinary scalar operations propagate the metadata from there. The sub-parent
materializer now assigns the active derived family's shape-compatible masks to:

- each result of a parent-tile split;
- the result of a group-to-sub-parent broadcast; and
- a value already at sub-parent width, including the `G=2` direct path.

Scalar and singleton values remain unmasked. A target mask is included only
when its shape broadcasts to the value without widening it, so a value shaped
`[1, R/F]` receives the derived R mask but not an X mask that would change its
shape.

This keeps the rule at the domain transition rather than at a later consumer:

```text
parent [X, R] --split--> lane [X, R/F]       gets xmask + halfF_rmask
group  [X, R/G] --broadcast--> [X, R/F]      gets xmask + halfF_rmask
already [X, R/F] --forward--> [X, R/F]       gets xmask + halfF_rmask
```

The generated bounds assertion and indirect `tl.load` now both use
`half2_r0_index_mask & xmask` in persistent and looped kernels. Focused tests
cover standalone split, nested group broadcast, and nested `G=2` direct-width
forwarding in both modes. The existing NVFP4 kernel-form tests remain green;
the metadata emits no runtime code unless an indirect access consumes it.

Review state:

- worktree: `agent_space/pr190594_ci_fix`
- changes are uncommitted
- no GitHub update was made
