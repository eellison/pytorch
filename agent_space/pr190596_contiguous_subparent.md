# #190596 — Fuse contiguous sub-parent reduction epilogues

Local commit `1e2b8890c98` - #190596. The rebuilt commit was verified at the
stack tip on 2026-08-19 and resubmitted on 2026-08-21; earlier pre-amend patches remain under
`agent_space/stack_review_fixes_20260818/`.

| File | What it holds |
|---|---|
| `torch/_inductor/scheduler.py` | `CONTIGUOUS` layout, normalized projection proof and factor widening |
| `torch/_inductor/codegen/simd.py` | reshape-permute materialization and internal-source persistence gate |
| `torch/_inductor/codegen/triton.py` | optional permutation for the split emitter |
| `test/inductor/test_nested_reduction.py` | contiguous behavior, rejection and kernel-form coverage |

Read #190594 first.

## Stack evolution and reservations

Direct emitted-lane selection and the optional-permute extension to #191775's
recursive split emitter land here. MXFP6 already established multi-output
stages; this commit does not change its factor-4 interleaved path.

The lower-stack fixes are folded into #190595 and #191775: temporal planner
names survive later mutation, and nested append relaxation requires a per-read
index proof. CONTIGUOUS planning uses the same normalized source-access
pipeline as INTERLEAVED and inherits those corrected contracts.

**Remaining reservation at the final tip:** the nominal rank-1 permute path is
untested. Even a 1-D input is represented as the rank-2 tile `[1, R0_BLOCK]`;
the new `B=1` kernel-form test pins that behavior. INTERLEAVED and CONTIGUOUS
still have distinct parent-to-child relations and should not be collapsed into
one less-conservative formula.

## The problem

The first two PRs handle consumers that read *interleaved pairs*. A second
family of consumers reads **contiguous chunks** — SwiGLU and chunked gating:

```python
h        = x + r
v        = h.pow(2).mean(-1, keepdim=True)
h        = h * torch.rsqrt(v + 1e-6) * w
gate, up = h.chunk(2, dim=-1)          # first half / second half
out      = F.silu(gate) * up
```

Same resolution problem — the consumer is at `(numel*rnumel/2, 1)` — but the
addressing is different. Compare the deps against #190594's:

```
op0_op1   group=(32, 1024)   reduction=True
    read  arg0_1   1024*d0 + d1         size=(32, 1024)   <- reduction
    read  arg0_1   1024*d0 + d1         size=(32, 512)    <- gate: chunk 0
    read  arg0_1   1024*d0 + d1 + 512   size=(32, 512)    <- up:   chunk 1
    write buf1     512*d0 + d1          size=(32, 512)
```

`d1` and `d1 + 512` — a constant offset of half the row — where #190594 saw
`2*d2` and `2*d2 + 1`. The interleaved matcher cannot prove these, and it
shouldn't: they address a different permutation of the same tile.

## Design decisions

1. **A second layout with a distinct projection.** `SubParentSourceLayout`
   gains `CONTIGUOUS` alongside `INTERLEAVED`. Both use
   `_try_get_sub_parent_source_layouts`; only the parent-to-child coordinate
   relation differs.

2. **Contiguous is explicitly opt-in.** `allow_contiguous=False` keeps the
   nested append path interleaved-only. Standalone planning passes true,
   then additionally requires a static power-of-two parent extent because
   persistent code splits the padded Triton block.

3. **Validate the emitted lane.** The matcher computes the lane codegen will
   select from the constant offset and proves that exact lane's index. If more
   than one structural lane aliases the same address, either loads the same
   value, so uniqueness is not a correctness requirement.

4. **Internal contiguous sources require persistence.** External sources
   reload in a looped derived pass instead of projecting a partial R block. An
   internal contiguous source must be split
   from a resident parent tile, so fusion asks the ordinary persistent heuristic
   and codegen forces the matching persistent choice. This is tested with
   `triton.multi_kernel` both disabled and enabled.

Both kernel forms set `min_rblock` only to the lane factor. Persistence still
selects a full resident block through the ordinary reduction heuristic; the
sub-parent stage does not impose a second full-row minimum.

## What it generates

For a factor-2 source, the layout difference from #190594 is one `tl.permute`:

```python
# interleaved (#190594):
tmp1, tmp2 = tl.split(tl.reshape(tmp0, [XBLOCK, (R0_BLOCK//2), 2]))

# contiguous (this PR):
tmp1, tmp2 = tl.split(tl.permute(tl.reshape(tmp0, [XBLOCK, 2, (R0_BLOCK//2)]), (0, 2, 1)))
```

Interleaved reshapes with the lane axis **last**; contiguous reshapes with it
**first** and permutes it to the end. Both then call the same `tl.split`, and
both use the same `lane2_r0_index` derived range and the same masked stores.

#191775 already introduced `_emit_recursive_split` for factor-4 MXFP6. This
commit reuses it for factors 8/16 and adds the optional permutation needed by
CONTIGUOUS layouts. The two layouts share the split strategy and differ only in
which axis is made trailing first.

## The flow: where the layout is chosen, and where the lane is derived

Two layouts now exist, and the whole PR is about keeping the *planner's* lane
and the *emitter's* lane identical.

```
  FUSION
  NestedReduction.sub_parent_epilogue_plan
    -> _try_get_sub_parent_source_layouts(..., allow_contiguous=True)
       -> normalize shared MemoryDeps into parent (X, R)
          and child (X, R/factor) domains
       -> INTERLEAVED first (factors through 4)
          prove child == parent[R := factor*child_r + lane]
       -> CONTIGUOUS for an allowed static power-of-two extent
          prove child == parent[R := child_r + lane*child_extent]

  CODEGEN
  _DerivedIterationFamily.resolve_load(name, index)
    -> INTERLEAVED: interleaved_sub_parent_lane(...)
    -> CONTIGUOUS: sub_parent_contiguous_lane(...)  # same helper
    -> _select_lane(parts, lane)
```

`sub_parent_contiguous_lane` appearing on both sides is an important invariant
in this rebased commit. Earlier revisions spelled out
`FloorDiv(Mod(offset, factor*child_extent), child_extent)` independently, and
they agreed only because `local_reduction_size` happened to equal
`parent_rnumel`. Nothing enforced that; drift would have produced a **silently
wrong lane** (gate/up swapped in a SwiGLU), not an assertion.

Interleaved is tried first. CONTIGUOUS remains reachable at factor 2 and 4 when
the interleaved index proof fails, as it does for chunk offsets. When
`child_extent == 1`, the formulas can coincide and the first successful proof
wins without changing the loaded value.

## How to read it

1. **`MemoryDep.normalize_with_ranges`** - the guarded domain reindex used for
   both parent and child accesses.
2. **`_try_get_sub_parent_source_layouts`** - shared-source selection, one
   normalized parent access, and the two projection proofs.
3. **`sub_parent_contiguous_lane`** - the emitted-lane formula shared by
   planning and codegen.
4. **`simd.py` -> `materialize_value_at_sub_parent_resolution`**, CONTIGUOUS arm.
5. **`triton.py` -> `emit_split_via_reshape`** - `permute_dims` is the only
   layout-specific branch.

## What to pay attention to

- **The lane formula is now written once.** It used to exist twice: the planner
  computed `FloorDiv(Mod(offset, parent_rnumel), parent_rnumel // factor)` while
  codegen computed the same thing from `local_reduction_size // factor`, and
  they agreed *only* because the standalone path passes `parent_rnumel` as the
  local reduction size. The `FusedNestedReductions` path uses the group size `G`
  instead, so enabling contiguous there would have silently diverged them. Both
  sides now call `NestedReduction.sub_parent_contiguous_lane(index, factor,
  parent_extent)`, and `_ContiguousSubParentRemappedValue` carries the parent
  extent rather than a pre-divided child extent so the division happens in one
  place. This was the highest-risk item in the PR.

- **The second, unvalidated lane candidate is gone.** The lane resolver
  (`_DerivedIterationFamily.resolve_load`, a free function named
  `_resolve_remapped_value` before #190594's cleanup) used to try `(offset, sizevars.simplify)` and then
  `(index, kernel.simplify_indexing)`. Only the first is what the planner
  validates, so the second could fire only on index shapes nobody checked and
  then silently pick an unproven lane. Live instrumentation recorded **zero
  hits in historical instrumentation over the then-current 304 tests**, so it
  was removed; a lane that does not resolve is
  now a loud assert, which is the correct signal for a planner-invariant
  violation.

- **Lane uniqueness is intentionally not required.** The planner checks the
  exact lane codegen selects. Multiple structural matches can exist only when
  their addresses coincide, in which case they load the same value.

- **Non-canonical sources are rejected by construction.** The emitted lane is a
  *byte-offset bucket*, not a structural lane, so it only coincides with the
  structural lane when the reduction's read is unit-stride from offset 0. For
  `base[:, 520:]` the bucket is 1 while the structural lane is 0, and the fusion
  is declined. `test_rejects_noncanonical_contiguous_sub_parent_source` enshrines
  this as expected — worth confirming that's intended and not just observed.

- **The capability switch is explicit.** Nested append passes
  `allow_contiguous=False`; standalone planning passes true. Static extent
  eligibility is checked independently rather than encoded as a `None`
  sentinel.

## Test coverage

The commit was tested independently during the rebase. `swiglu` (factor 2) and
`gating` (factor 4) discriminate well -
I verified 23 of 24 lane permutations are detected for gating.

Factors 8 and 16 now have lane-asymmetric numeric tests using
`sum(i * parts[i])` under both forced kernel forms. A factor-8 INTERLEAVED
negative test separately pins the smaller interleaved capability limit.

The cross-products are exercised rather than rejected speculatively:

- a factor-4 CONTIGUOUS source feeding one- and three-output-lane stages;
- distinct source buffers using INTERLEAVED and CONTIGUOUS layouts in one plan;
- the same source buffer requesting conflicting layouts, which still declines.

Dynamic batch sizes fuse. A real symbolic reduction extent runs three runtime
values through one compiled fallback graph and asserts two generic kernels and
no staged codegen. Static non-power-of-two reduction extents also decline.

Internal CONTIGUOUS sources fuse only when the ordinary heuristic selects a
persistent reduction. The tests cover that decision under both ordinary and
multi-kernel choice generation, plus the forced-looped fallback.

A mixed looped plan with an internal source and an external CONTIGUOUS source
is covered at `D=2^20`, forcing a genuine multi-trip reduction. Strict
reductions reject CONTIGUOUS projection because their persistent R block may be
larger than the logical extent; INTERLEAVED remains valid.

The nominal rank-1 contiguous branch (`permute_dims = (1, 0)`) remains
unreached: `B=1` and even a genuinely 1-D input retain the rank-2 kernel tile
`[1, R0_BLOCK]` and use `(0, 2, 1)`.

## Changed during the stack rewrite

- Source matching was replayed onto #190594's normalized
  `MemoryDep.normalize_with_ranges` pipeline. The old normal/flat 2x2 matcher
  matrix is not present in this commit.
- The contiguous lane formula is
  `NestedReduction.sub_parent_contiguous_lane`, called by both the planner and
  `resolve_load`. `_ContiguousSubParentRemappedValue` carries the parent extent
  instead of a pre-divided child extent.
- The unvalidated second lane candidate was deleted (0 hits in the historical
  304-test instrumentation run).
- The split emitters were merged. `emit_split_via_reshape` accepts an optional
  permutation and documents the `tl.split` trailing-axis constraint.
- The bounded one-output factor classifier widens from 4 to 16 here; the cheap
  prefilter uses that classifier, so factor-8/16 cases reach planning.

## Later-stack resolution and remaining follow-ups

The explicit `allow_contiguous` capability, direct emitted-lane check and
single split emitter are all present in this commit.

Still open: determine whether the rank-1 permute path is reachable. Dynamic or
non-power-of-two contiguous R remains a conservative fallback because resident
persistent splitting is defined over the padded block. Nested append supports
interleaved factors through 4, including `(4, 3)`, but not CONTIGUOUS.
