# Blockwise-2D tile-local group reductions: design proposal

Date: 2026-08-22 (rev 2, centered on the band-persistent form). Status: DRAFT
for author review. No implementation exists.

Goal: a generic Inductor codegen capability for reductions whose groups are
local to an aligned sub-extent of a non-R iteration axis, such that the
combined dim0+dim1 quantization kernel emerges from ordinary fusion. Coverage
bar: NVFP4 16-groups, MX 32-groups, MXFP6 (4,3) rates, both dims, row-major
and transposed outputs, dynamic shapes, producer fusion.

## Motivating baseline

RMSNorm feeding both quant dims (`rms_dual_quant` probe, 16384x7168, nested
ON): **3 kernels, 267.2 us**. K0 is the flagship staged kernel (norm + dim0
amax + pack, one raw-x read, 64 KB stats + dim0 outputs). K1/K2 are the dim1
amax and dim1 pack, each re-reading RAW x and recomputing the normalization
from the stats: Inductor already uses recompute-from-stats and never
materializes the normed intermediate, so the entire waste is the 3x raw-x
read (~700 MB vs 234 MB ideal; single-load floor 45.9 us). The flagship
norm+dim0-only graph is 1 kernel at 110.6 us; the handwritten dual-quant
kernel (no producer) is 83.7 us kernel-only. **Acceptance band for the fully
fused producer case: 120-150 us.**

Secondary anchors: standalone NVFP4 dual quant is 3 kernels / 169-179 us
(cd) vs the handwritten 83.7 us kernel-only / 95.1 us wall; single-dim NVFP4
dim1 is 2 kernels / 99.4 us (cd) vs ~65 us estimated fused
(dim1 investigation); mix-order is structurally inapplicable (group-local
shapes never reverse-match its contract).

Architectural frame: this proposal extends the staged-reduction stack
(#190594/#190595 phase rules: typed fused identity approved at fusion, plan
rebuilt post-merge_loops, legality on normalized MemoryDeps, ordinary fusion
gates re-entered on append, producer-first ordering) and is specified against
the foundation vocabulary (derived_domain_projection_proposals.md Iterations
8-14: indexed projected-value record keyed by source value/version +
normalized substitution + domain + mask + fill; domain-shaped [X,Y,1] values
with native broadcasting; requirements R1-R5). No name-keyed forwarding.

## 0. The primary form, and what already exists

**The primary form is the band-persistent (d2) topology: the existing
staged/nested schedule with its row band widened from 1 to lcm-of-group
rows, and the grouped axis admitted on the row axis.** Concretely: X = rows
(band, XBLOCK a multiple of the row-group size G1), R = full K (looped in
practice: a G1-row band times large K exceeds the register budget, so the
persistent form is only viable at small K), stats pass then quantize pass -
i.e. the landed
`FusedNestedReductions` pipeline plus a second grouped family whose groups
live on X. The one-shot 2D tile (d1) is the standalone materialized-input
form of the same tile-local grouping proof, and the single-dim dim1 case
(the old strided-lane item (c)) is the degenerate no-producer, no-dual form.
One proof and one lane machine serve all three.

Two facts about the landed stack scope this work down:

- **X-grouped reduction STAGES already exist.** Grouped-axis discovery
  classifies the `[X/G, R] reduce G` geometry as `GroupedAxis.X`
  (scheduler.py ~1846), and the nested emitter fuses such a grouped amax
  into the staged kernel today: the pinning test
  `test_producer_consumer_rejects_sub_parent_grouped_axis_x`
  (test/inductor/test_nested_reduction.py:1504) asserts
  `codegen_nested_reduction == 1` - the norm + X-grouped amax ARE one
  kernel. Only its packing epilogue falls out (`generated_kernel_count == 2`).
- **The rejection is one guard.** `_nested_sub_parent_rate`
  (scheduler.py:941) opens with
  `if domain_context.grouped_axis is not cls.GroupedAxis.R: return None`
  (line 946). Everything after that guard - the numel-ratio rate
  (`_sub_parent_epilogue_rate`), the factor bound, `group_size % factor`,
  and the expected-groups domain match - is axis-agnostic arithmetic.

**What replaces the rejection** (the planner change under review): delete the
axis guard and add the X-lane legality it was standing in for:

1. the lane proof runs in the family's transposed normalized view
   (k, row-group, lane), where the X-lane is an ordinary trailing-axis lane
   and the landed INTERLEAVED/CONTIGUOUS matchers apply verbatim;
2. block alignment becomes legality: the nested path's existing load-bearing
   `min_xblock` (simd.py ~3144) additionally requires
   `XBLOCK % group_size == 0` so a row group never straddles programs
   (the X analog of `min_rblock = factor`);
3. codegen gains X-lane emission (capability N2 below).

**The review anchor is the test flip**:
`test_producer_consumer_rejects_sub_parent_grouped_axis_x` changes from
asserting `generated_kernel_count == 2` to `== 1` (with a bitwise
nested-vs-unnested check, which `check_nested_matches_unnested` already
performs). That single assertion change is the observable contract of the
planner relaxation.

### The two new capabilities

- **N1: non-R grouped reduction family.** In-register reduction over an
  aligned sub-extent of a grid axis tile (reshape that axis to
  `(BLOCK//G, G)`, reduce the sub-axis). No loop or CTA interaction. Status:
  EXISTS on X in the nested emitter (above); NEW on Y for the standalone
  tiled form.
- **N2: non-R lane-split stores.** Packed epilogue stores whose lane axis
  derives from a grid axis. Emission: transpose the register view so the
  lane axis is trailing, then reuse `emit_split_via_reshape` and the
  existing rate machinery verbatim (INTERLEAVED factors 2..4, MXFP6 (4,3);
  CONTIGUOUS deferred). Status: NEW; **subsumes the strided-lane sub-parent
  follow-up** - in the flattened 1D domain the dim1 lane sits at stride K
  and the landed proof correctly fails; in the transposed per-family view
  the same lane is trailing and no new lane algebra exists anywhere in this
  proposal. N2 is designed standalone-usable: a lone dim1 quant (no
  producer, no dual) is admitted through the same relaxed rate path.

The handwritten kernel is the physical blueprint: its dim1 side is
`tl.trans(x)` followed by the dim0 code (measured ~10% faster than
permute/split, one layout conversion per family, bitwise-equal).

## 1. Recognition

### Banded/nested cases: no new recognition

For any graph the nested planner already admits - producer + one grouped
reduction, R- or X-grouped - `PointwiseDomainContext` already carries the
grouped axis and group size. B1 (below) only relaxes the rate guard and adds
the per-family lane proof in the transposed view. The standalone single-dim
dim1 case reuses the standalone sub-parent entry
(`NestedReduction.sub_parent_epilogue_plan`) with the same transposed-view
normalization replacing the trailing-axis assumption.

### Dual-family cases: the joint 2D factoring proof

The flattened (x, r) groups of the two amax families are IDENTICAL, not
transposed - both present `(M*K/16, 16)` at 16384x7168 (probe evidence; this
is why `has_mix_reduction_orders` and the 1D lane proof both decline). The
2D structure is visible only in the UNFLATTENED node bodies: each
`SchedulerNode._body` retains original `var_ranges`, and both families' read
indices are linear in those vars. Given sibling nodes A and B (each a
reduction, or a Fused/staged node containing one) with a common full-access
read of one buffer version:

1. Normalize each node's read over its own ranges (existing
   `MemoryDep.normalize_with_ranges` + `map_kernel_groups_to_node_sizes`).
   Require a linear, zero-offset index; indirect indexing or remaining
   ModularIndexing rejects.
2. From A recover candidate axes: the trailing stride-1 run is the k axis
   (extent K); the remaining triples must form one axis at stride K
   (extent M). Gaps, extra axes, or non-divisible stride towers reject.
3. Require B's triples to factor onto the SAME (M at stride K, K at
   stride 1) axes. Windows, shifts, transposed or broadcast views produce a
   stride/extent mismatch and reject (mismatched-views rule).
4. Each node's reduction var must sit aligned inside one axis (sub-extent
   G, its stride times G divides the next stride up), and the two reduction
   axes must differ (orthogonality). Dual quant: A reduces G0=16 at stride 1
   in k; B reduces G1=16 at stride K in m.
5. Divisibility in the stack's style (`try_get_sub_parent_extent_subs` /
   sizevars): `K % G0 == 0`, `M % G1 == 0`, statically or by guard.
6. Per-family epilogue legality reuses the landed vocabulary unchanged,
   inside that family's own normalized view: dim0 epilogues in
   (m, k-group, lane) with the trailing-axis matcher; dim1 epilogues in the
   transposed (k, m-group, lane) with the SAME matcher. Candidate
   classification, extent substitutions, outputs-unread, and the
   temporal-name / mutation rules (#190595 correction) carry over per
   family.

### Rejection fallbacks

Every rejection is a silent `return None` in the established style; the pair
falls through to ordinary fusion (which declines as today) and the schedule
degrades to the current one. Blockwise admission is attempted in addition
to, never instead of, standalone staged planning, so a rejected dim1 side
leaves the dim0 staged fusion intact - non-orphaning is structural. Mapped
rejections: non-divisible groups (step 5), mismatched views (step 3),
mutated sources (temporal names + has_aliasing_or_mutation), indirect or
offset reads (step 1), extra or non-orthogonal axes (steps 2/4).

No pattern matching on quant ops anywhere: legality is index arithmetic on
MemoryDeps, per the #190594 design rule.

## 2. Fusion identity

Single-family cases keep their existing identities: the relaxed X-grouped
nested form stays `FusedNestedReductions` (its topology record already
carries grouped axis and group size), and the standalone single-dim dim1
form stays `FusedStagedReduction`.

Dual-family cases get **one new typed identity,
`FusedBlockwiseReductions(FusedStagedReduction)`**, carrying stable
topology: the 2D axis map (normalized extents/strides), per-family records
(grouped axis, group size, rate `(factor, output_lanes)`), and optional
producer topology. Justification: the standalone staged path may rebuild its
plan from scratch because 1D rediscovery is lossless; the joint 2D factoring
is not recoverable after `merge_loops` - the same argument that made
`FusedNestedReductions` carry topology and rebuild via `plan_from_topology`.
The mutable plan (`StagedReductionPlan` widened from "zero or one
SubParentEpilogueStage" to one grouped stage + one epilogue stage PER
family) is rebuilt inside each phase, never carried across the boundary.
Subclassing `FusedStagedReduction` inherits every existing exclusion (combo
grouping, benchmark fusion, the generic-codegen assert) for free.

Sibling admission enters from the mixed candidate path beside
`_sub_parent_epilogue_decision` when both candidates contain reductions, are
siblings (no ancestor overlap), and share a full-access common read. Either
side may already be staged or nested; admission REPLANS THE COMBINED LEAVES
(the stack's existing rule for extending an exact staged node) and upgrades
the identity, folding nested topology in as producer topology. Ordinary
fusion gates re-run by re-entering `_can_fuse` with producer-first
normalization, exactly as `_can_fuse_nested_reduction_append` does.
Append-order independence (dim0-first, dim1-first, producer-first) is a
required test, mirroring the existing nested append-order tests.

## 3. Producer fusion: the primary form

A row-norm producer reduces over full K, so its stats are not computable in
a free 2D tile; producer + dual quant therefore takes the banded form, which
IS the existing nested topology:

```text
X = rows, banded: XBLOCK % lcm(G1) == 0        (was: 1 row per x element)
R = K, looped or persistent                     (unchanged)
stage 1: outer reduction  -> stats              (unchanged)
stage 2: R-grouped family -> dim0 scales        (unchanged)
stage 3: X-grouped family -> dim1 scales        (exists: N1-on-X)
stage 4: R-lane epilogue  -> dim0 packed        (unchanged)
stage 5: X-lane epilogue  -> dim1 packed        (new: N2, via the relaxed
                                                 _nested_sub_parent_rate)
```

The dim1 stages complete per R-chunk - their reduction and lane axes are X,
so looped R needs no cross-iteration accumulation; each iteration finishes
its own k-columns. The dim0 chain is byte-for-byte untouched; the flagship
norm+dim0 test suite must pass unchanged, and a graph whose dim1 side is
rejected reproduces today's schedule exactly (the staged dim0 fusion is
never orphaned).

Degenerate forms, in decreasing generality:

- **No producer, both dims (standalone dual, the (d1) form):** no full-K
  stats pass is needed, so the iteration space is the free y/x/r tiled
  persistent reduction Inductor already emits (mxfp8_dim1_t is the existing
  proof of the form): y = rows (YBLOCK % G1 == 0), x = k-groups, r0 = G0
  persistent, dim0 on R, dim1 via N1-on-Y + N2-on-Y. This is the handwritten
  kernel's exact shape. It shares recognition (section 1), identity
  (section 2), and the lane machine with the banded form; only the axis
  carrying rows differs.
- **No producer, one dim (the old (c) item):** a lone dim1 quant admits
  through the same relaxed rate path and N2, producing one kernel from
  today's two. Nothing dual-specific is required - this is the degenerate
  case, not a separate project.

## 4. Tiling and launch

- **Group alignment is legality, not heuristics** (the #190594 lesson: the
  deleted standalone `min_xblock` was a throughput heuristic riding a
  legality mechanism; here the multiples ARE the mechanism). Banded form:
  `min_xblock = lcm(existing_nested_min_xblock, G1)` and
  `XBLOCK % G1 == 0`; `min_rblock = lcm(G0, factor)` continues the landed
  rule that a lane group cannot straddle an iteration. Tiled form:
  `YBLOCK % G1 == 0`, r0 = G0 persistent, X free. General statement: each
  grouped axis's block is a multiple of the lcm of the group extents on
  that axis.
- **Decided once.** Like `_sub_parent_tiling_is_2d`, fusion-time planning
  records the form (banded vs tiled) and the multiples; codegen forces that
  tiling. An incompatible parent tiling demand declines the fusion.
- **Candidate generation:** `tiling_scores` / `_match_target_block_product`
  keep distributing the block product; the plan contributes hard multiples
  filtering candidates plus seeds from the measured band: row block = G1
  exactly (16/32), elements-per-thread 64-128, warps derived from
  elements-per-thread rather than scaled with block product (winners
  16x256/w2 and 16x512/w4; 64-row tiles lost ~45%; warp scaling lost ~20%).
- **Dynamic shapes:** group extents are per-dtype constants; axis extents
  may be symbolic. `M % G1` / `K % G0` proven or guarded in the
  `try_get_sub_parent_extent_subs` style. Tiles that do not divide a
  group-divisible extent are legal with masks because partial tiles contain
  only whole groups (proven bitwise by the handwritten kernel at 528x1040
  and 2064x4112); group-indivisible extents reject at recognition.

## 5. Codegen

- **Per-tile group reductions in registers** (N1): the non-R family is
  emitted on a transposed register view of the already-loaded tile,
  structurally identical to the trailing-axis family (trans-then-rowwise
  blueprint). One load feeds both families: within one kernel body both
  families' loads normalize to the same physical index and ordinary CSE
  dedupes them - external sources need no forwarding machinery.
- **Scale stores at group resolution** as domain-shaped values: group values
  are `[X, Y, 1]`-style singletons in the foundation vocabulary, widened
  only by native broadcasting at the divide, flattened only at the store
  (Layer 1 rule). The dim1 scale is `[Y/G1, X_k, 1]` in the transposed view.
- **Lane-split packed stores** (N2): transposed view -> trailing-axis
  `emit_split_via_reshape` -> existing rate/output_groups machinery. Factor
  2 first; factor 4 and MXFP6 (4,3) inherit `_nested_sub_parent_rate`'s
  arithmetic unchanged. CONTIGUOUS non-R layouts deferred (same positioning
  as CONTIGUOUS nested append in #190595).
- **Both output layouts:** row-major dim1 outputs transpose the packed tile
  back before storing (all stores coalesced along k); transposed (t) outputs
  store the transposed view directly. Both are store-index arithmetic; the
  ~1.4x t-layout cost is a measured property, not a legality difference.
- **Numerics: exact by default.** Per-element division by the fp8-rounded
  scale, amax trees, fp8 conversion, and the e2m1x2 inline asm are the ops
  the references execute; acceptance is bitwise equality on u8 views, the
  standard the handwritten kernel meets everywhere. The per-group
  reciprocal-multiply variant (+20%, zero differing bytes across ~470 MB) is
  opt-in only, never default.
- **Foundation prerequisites, per part:**
  - B1 (guard relaxation + N2 single-family) and B2's MXFP8 slice need
    NEITHER foundation PR: sources are external loads deduped by CSE, and
    scale broadcasting may use the #190595 eager-projection precedent.
  - N2's projection-clean form (divide-before-split, no eager broadcast,
    R2/R5) wants the **lazy derived-domain projection PR (F2)**: the
    transposed-view split is a new residual-projection suffix, and its
    materialization cache (keyed without the lane, Iteration 11 item 2) is
    where the split tuple lives. Shipping first with eager projection is
    acceptable; the performance acceptance assumes F2.
  - B3 (producer + dual) requires the **indexed projected-value lookup PR
    (F1)**: stats and inlined scale-chain values cross stage boundaries into
    two derived families, and only the record's key (source value/version +
    normalized substitution + domain + mask + fill) can serve one emission
    to both the R-view and the X-group view. Building on the name-keyed
    resolver is rejected: it is scheduled for deletion and cannot express
    one buffer under two normalized views.

## 6. PR decomposition

Sizes in stack units: 1.0 x #190594 = the standalone foundation
(+2148/-214); 1.0 x #190595 = the nested integration commit. Foundation
order stands (#190595, #191775 land as reviewed; then F1 lookup; then F2
projection; #190596 deferred).

- **PR-S (side, independent, not on this critical path): mix-order
  reduction-type widening.** max/min into the `{sum, prod}` allowlist;
  codegen already generic. ~0.05 x #190594. Acceptance: canonical row+col
  amax pair 3 kernels -> 1, bitwise, ~152.6 -> ~70 us at 16384x7168
  (measured 2026-08-22).
- **PR-B1: grouped-axis guard relaxation + N2 (single-family).** Deletes the
  `GroupedAxis.R` guard in `_nested_sub_parent_rate`, adds the
  transposed-view lane proof and X-block alignment legality, and adds N2
  emission. Covers the nested X-grouped epilogue AND the standalone
  single-dim dim1 quant (the degenerate (c) case). Prerequisites: none
  (eager projection); F2 re-gates performance. Size ~1.0 x #190595.
  Acceptance: (i) THE TEST FLIP -
  `test_producer_consumer_rejects_sub_parent_grouped_axis_x` asserts
  `generated_kernel_count == 1` (bitwise via the existing nested-vs-unnested
  check); (ii) standalone NVFP4 dim1_rm 2 kernels -> 1, <= ~70 us at
  16384x7168 (2-kernel cd baseline 99.4 us, single-read estimate ~65 us);
  (iii) kernel-form checks: one x load site, one e2m1x2 pack site, lane
  stores masked by the derived range.
- **PR-B2: dual-family recognition + FusedBlockwiseReductions + tiled
  standalone form.** The joint 2D factoring proof, the new identity, plan
  widening to per-family stages, and N1-on-Y for the tiled form. Ships in
  two acceptance slices: MXFP8 dual cast first (full-res payloads, no N2,
  no foundation deps), then NVFP4/MXFP6 dual using B1's N2. Prerequisites:
  B1; F2 for the packed slice's performance gate. Size ~1.0 x #190594.
  Acceptance: MXFP8 dim0+dim1_rm ONE kernel, bitwise, <= the cd-tuned
  separate-kernel sum (~164 us = 58.4 + 105.5); NVFP4 dual ONE kernel, all
  four outputs bitwise vs the compiled references, kernel-only <= 1.25x the
  handwritten 83.7 us (i.e. <= ~105 us; status quo 169-179 us); TTGIR: one
  x load, no workspace, layout-conversion count matching the handwritten
  trans form, exactly two e2m1x2 pack sites.
- **PR-B3: producer + dual (the banded primary form).** Multiple grouped
  families in a nested plan, identity upgrade/replan rules, band alignment
  (`min_xblock` lcm), append-order independence. Prerequisites: B1, B2
  vocabulary, F1 (hard). Size ~0.5-1.0 x #190595 - smaller than a fresh
  emitter because the X-grouped stage emission already exists; the work is
  plan plumbing and admission. Acceptance: `rms_dual_quant` 3 kernels -> 1,
  **120-150 us** at 16384x7168 (baseline 267.2 us, floor 45.9 us,
  norm+dim0-only reference 110.6 us); flagship norm+dim0 suite unchanged;
  dim1-rejected graphs reproduce today's schedule.
- **PR-B4: transposed output layouts + dynamic-shape widening + config
  seeding.** t-layouts for payload and scale, symbolic M/K with guards,
  masked partial tiles, measured-band seeds. Size ~0.5 x #190595.
  Acceptance: dual quant with dim1_t outputs ONE kernel, bitwise, <= the
  handwritten combined_t band (129-148 us); one compiled dynamic graph
  serving multiple runtime shapes (existing dynamic-test pattern); edge
  shapes 528x1040 / 2064x4112 bitwise.

Ordering: B1 may proceed against the landed stack immediately and is
independently valuable (test flip + single-dim dim1). B2 follows B1; its
packed slice re-gates on F2. B3 waits for F1. #190596 (contiguous/SwiGLU)
remains deferred and independent; CONTIGUOUS non-R layouts become a B-series
follow-on after it, not part of this proposal.

## 7. Non-goals

- No TMA, warp specialization, or persistent-pipeline codegen: the
  handwritten kernel is L1/compute-bound at 52% DRAM; memory-side machinery
  attacks the wrong limit (the naive TMA variant measured 2x slower in the
  combined-MXFP8 work).
- No cross-CTA machinery: no workspace, no grid barriers, no split
  accumulation. Tile-locality is the defining constraint; mix-order remains
  the home for full-axis sibling reductions and is not generalized beyond
  PR-S.
- No pattern matching on NVFP4/MX ops: legality is memory-dep index
  arithmetic only.
- No changes to generic `cse.store_cache` (Iteration 10 decision stands).
- No new post-fusion IR or pass.
- No swizzled-scale blocked layouts in this series (a store-index transform
  that can ride N2 later; not in the coverage bar).
- No default-numerics change: reciprocal-multiply stays opt-in regardless of
  measured bitwise equality.
- No launch-config heuristic overhaul: this series contributes hard
  multiples and seeds only; the default-config work for tiled persistent
  reductions (the mxfp8_dim1_t cd-dependence) remains the separate
  benchmark-corpus item in the followups.
- No CPU or non-Triton backends: capability-gated like the staged path
  (cuda_combined_scheduling forwarding pattern).
