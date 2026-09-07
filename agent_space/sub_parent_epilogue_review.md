# Sub-Parent Epilogue — Review Summary

State-of-the-work writeup consolidating the review. Companion docs:
`sub_parent_epilogue_design.md` (design contract) and
`sub_parent_epilogue_split_plan.md` (how to land it).

Base: `96ac987dbfb [inductor] Fuse NVFP4 nested-reduction packing`.
Working tree: ~1255 insertions across `simd.py`, `scheduler.py`, `triton.py`,
`test_nested_reduction.py`.

---

## 1. What it is

Extends nested-reduction epilogue fusion to consumers whose iteration domain is a
proper sub-domain of the reduction's parent tile. Instead of writing the normed
tensor to global memory and reloading it, the consumer values are derived from
the parent tile that is already live in the fused reduction kernel.

Two proven source layouts, attached to the source buffer name through legality
and codegen via `SubParentSourceLayout`:

- INTERLEAVED: `parent_r = factor*child_r + lane` (NVFP4 even/odd packing).
  This is 96ac's factor-2 case, now a strict special case.
- CONTIGUOUS: `parent_r = lane*child_extent + child_r` (RMSNorm chunk/SwiGLU).

Generalizes 96ac along two axes: factor 2 -> {2,4,8,16}, and a second layout.

## 2. How it works (architecture)

The fused kernel runs the reduction at *parent* resolution and the sub-domain
consumers at *sub-parent* resolution in one kernel, mapping between the two via
derived iteration families. The whole thing rests on a single contract: prove
how each consumer read maps back to a parent-tile load already live in the
kernel, and carry that proof (the layout) from legality into codegen.

Key components:

- `_GroupedReductionLayout` -- geometry of the consumer grouped reduction: how
  the outer tile reshapes into `[num_groups, local_reduction_size]`, which axis
  is grouped (`local_reduction_in_r`), and how values move between resolutions.
  Owns `materialize_value_at_sub_parent_resolution`.
- `_DerivedIterationFamily` -- an iteration space temporarily swapped onto the
  kernel while a consumer stage runs. The sub-parent family
  (`make_parent_half_family(factor)`) has a derived range tree with numel/block
  divided by `factor`; activating it makes ordinary load/store/index code
  interpret indices in that space.
- `SubParentSourceLayout` (scheduler enum) -- INTERLEAVED / CONTIGUOUS; the proof
  of how a consumer read maps to the parent tile, keyed by source buffer name.
- `HalfResolutionEpiloguePlan` -- the scheduler's plan: the sub-parent consumer
  nodes (`half_nodes`), the `sub_parent_factor`, and
  `source_layouts: dict[name, SubParentSourceLayout]`.
- Ops handlers: `_ParentHalfSourceLoadMaterializer` wraps the reduction body's
  loads and splits the parent-tile load into per-lane parts;
  `_ParentHalfPointwiseRemapHandler` runs each consumer in the derived family and
  resolves its loads to the right lane via `_resolve_remapped_value`.

Flow:

1. Detect + prove (scheduler). `half_resolution_epilogue_plan` finds pointwise
   consumers whose numel is `parent_numel / factor`, requiring a static pow2
   reduction size, pow2 factor, and R-axis grouping. For each consumer read of a
   reduction source it proves the read is a clean sub-region of the reduction's
   read -- `_interleaved_...` (`r = factor*child + lane`) or `_contiguous_...`
   (`r = child + lane*child_extent`), both using `_unique_trailing_sub_parent_dim`
   to find the split axis. Conflicting layouts on one buffer, ambiguous reads,
   half-output readers, and source-reading siblings are rejected. Guards:
   `outputs_unread` (plan) + `leaf_violation` (schedule-time).
2. Materialize the source at sub-parent resolution (codegen, persistent). The
   parent tile is live in registers; `materialize_value_at_sub_parent_resolution`
   splits it into `factor` lane-parts -- INTERLEAVED via `reshape + tl.split`,
   CONTIGUOUS via `reshape + tl.permute + recursive tl.split`
   (`emit_split_via_reshape_permute`). Parts are stored in `remapped_values` by
   buffer name.
3. Run the consumers in the derived family. Each consumer load of the source
   resolves to the right lane-part: INTERLEAVED by `index % factor`, CONTIGUOUS
   by `(offset % (factor*child_extent)) // child_extent` (the chunk's constant
   offset). Stores emit at sub-parent resolution.

Persistent vs looped codegen:

- Persistent: full parent tile in registers -> the in-register split above.
- Looped: the tile is never fully live, so `source_layouts` is empty and
  consumers *recompute* from the original inputs at their sub-parent indices
  (verified: looped kernels load only `in_ptr`, never `out_ptr`/`in_out_ptr`).
  One kernel, at the cost of re-reading inputs -- the amortization concern in S6.

Constraints (from the design contract): static pow2 parent reduction size, pow2
sub-parent factor in {2,4,8,16}, R-axis grouped reductions, non-padded tile.
Unsupported cases fall back rather than reloading or storing intermediates.

## 3. Status

- `test_nested_reduction.py`: 244 pass (was 234 at review start).
- `test_inductor_scheduler.py`: 30 pass.
- Functionally complete and correct. Everything remaining is packaging/polish.
- Cleanups that landed during review (each verified behavior-preserving):
  - materializer unified into `materialize_value_at_sub_parent_resolution`
  - dead contiguous RBLOCK guard removed
  - `outputs_unread` guard de-duplicated (was 3 copies)
  - `_unique_trailing_sub_parent_dim` extracted (shared half-dim finder)
- Still open: finish `half` -> `sub_parent` rename (residual ~76 simd / ~29
  scheduler refs), full `test_torchinductor.py` + lint (clean env only), execute
  the commit split, add the contiguous profitability guard.

## 4. Correctness (the strong part)

Across the review: 244 unit tests + ~50 adversarial compile/numeric cases.
**Zero miscompiles in the current code.** Every unsupported pattern degrades to a
safe non-fusion; nothing silently produces wrong results.

Paths exercised:
- standalone persistent, standalone looped (forced via choices handler)
- producer-consumer (two-reduction nested) path
- dynamic shapes (dynamic batch fuses; dynamic reduction dim correctly rejected)
- dtypes: fp32, bf16, fp16, fp8; NVFP4 integer packing verified **bit-exact**
- cooperative reductions (safe fallback to non-nested)
- B=1, 3D, tiny D=16
- index-proof stress: transposed / sliced / permuted / prologue-computed /
  stacked / combo_kernels -- non-contiguous sources correctly rejected,
  contiguous slices and computed sources fuse correctly
- restriction checks: overlap/shifted reads, mixed layouts on one buffer,
  factor>16, non-pow2 group sizes -- all rejected by falling back, never wrong

Regression check on shared code: 17 ordinary reduction/fusion patterns (softmax,
layernorm, argmax, cumsum, matmul->softmax, fp8 quant, group_norm, ...) all
match eager. This matters because `_half_resolution_epilogue_leaf_violation`
runs inside the generic `can_fuse` for *every* reduction+pointwise fusion when
`nested_reduction` is on (the default) -- it does not regress non-nested codegen.

## 5. The one confirmed defect -- in shipped code, not this PR

On base HEAD (96ac), the interleaved factor-2 NVFP4 pattern with a
full-resolution reader of the even/odd halves **miscompiles**: reconstructing
`full = stack(even, odd)` reads the half-output buffers before their late stores
(read-before-write). `triton.nested_reduction` defaults on, so this is a live
miscompile in shipped PyTorch. This PR fixes it.

Scope, verified precisely:
- The **full-resolution pointwise reader** is the live bug.
- The **reduction reader** of a half output does *not* miscompile on 96ac (it
  simply does not fuse) -- so the reduction-reader guard/tests are hardening, not
  a reproduction of the shipped bug.
- The fix needs **both** guards, at different stages: the schedule-time
  `leaf_violation` prevents the bad fusion from forming, and the plan-time
  `outputs_unread` makes codegen fail closed. With only the plan guard, the
  silent miscompile becomes a hard AssertionError instead of a clean fallback
  (verified by disabling `leaf_violation` at runtime). Neither alone suffices.

## 6. Design assessment

Sound. The `SubParentSourceLayout` spine (proof attached to source name, carried
through legality and codegen) is the right abstraction -- the evidence is that
96ac's factor-2 interleaved path falls out as a strict specialization, which is
what a good generalization looks like.

Most of the line count is **essential** complexity, not bloat: you are running
two iteration resolutions in one kernel and must *prove* each consumer read maps
back to the parent tile. The proofs and guards earn their lines -- the simpler
96ac version miscompiled precisely because it lacked one. A fundamentally smaller
design would require a more general inductor capability (generic multi-resolution
tiling / provable sub-tile aliasing) that does not exist; the special case is the
tractable thing.

The real lever is **scope, not design**. Two pieces carry disproportionate
complexity for incremental coverage:
- factor > 2 (the factor scan, recursive permute-split, multi-lane proof)
- the looped/non-persistent path
The core -- persistent, factor-2, interleaved + contiguous -- already covers
NVFP4 quant and SwiGLU, the two workloads that justify the project. The rest can
ride behind the same abstraction.

The risk worth more attention than line count is **fire-rate**: pattern-match
fusions only fire on the exact op shape. Before investing further, confirm the
real target graphs (actual model RMSNorm+quant and SwiGLU, not synthetic repros)
hit the fusion -- a slightly different decomposition silently falls back.

## 7. Performance

Persistent path (small/medium D): clean, consistent win (~1.5x at D=1024).
NVFP4/RMS+FP8 interleaved packing is the headline (2-3x vs hand-written in
earlier benchmarking).

Looped path (large D, after persistent-forcing was removed): a real win at
medium D (~1.3-1.4x at D=4096) but erodes to break-even or slightly negative
(0.83-0.9x) at large D with small batch. Measurements are noisy at this scale
(sub-20us kernels), but the trend is real and expected.

Root cause = amortization, the same XBLOCK-vs-group tradeoff the XBLOCK-grouped
reduction work manages with `min_xblock`/`min_rblock`. These fusions trade a
global round-trip for extra per-program work (staged split in persistent mode,
input recompute in looped mode), which only pays off with enough rows (XBLOCK)
per program. Interleaved enforces `min_xblock = 128`; contiguous does not. For
large reduction sizes the tile caps XBLOCK regardless, giving a built-in
profitability ceiling.

Implication: contiguous needs a profitability guard (see split plan commit 4) so
it only fires where it is net-positive; past the ceiling, prefer not fusing over
forcing persistent.

## 8. Landing plan (see split_plan.md for detail)

1. Fix the 96ac full-res-reader miscompile (both guards + minimal factor-2
   helper). Independently valuable / backportable; land first.
2. Generalize factor/layout data model, interleaved-only, persistent.
3. Looped/non-persistent support, interleaved (interleaved keeps its
   `min_xblock` floor, so it stays profitable).
4. Contiguous chunk/SwiGLU + mandatory profitability guard (inherits looped from
   #3; real SwiGLU is large-D, so it depends on looped to be useful).

Notes: looped precedes contiguous on purpose (contiguous's real workload is
large-D / non-persistent). Commit 1 must not depend on the final data model --
use a minimal factor-2 helper, generalize in commit 2. Keep the factor scan (do
not rewrite as a quotient -- it preserves symbolic-shape fusions).

## 9. Open items

- Add the contiguous profitability guard (blocks commit 4).
- Confirm real-model SwiGLU / RMSNorm+quant graphs actually hit the fusion.
- Finish the `half` -> `sub_parent` rename (or scope it per commit).
- Full `test_torchinductor.py` + `lintrunner -a` in a proper env.
- Execute the 4-commit split.
