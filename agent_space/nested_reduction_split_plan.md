# Nested Reduction Split Plan

Source branch state:

- Base: `d69ec4e17a07ec00e3f6944b1013cc2acd8f393f`
- Current split stack:
  - `0a7a34058e6` (`[inductor] Make Triton range metadata root-owned`)
  - `fb9ab05e6d6` (`[inductor] Add derived SIMD range roots`)
  - `048cda46947` (`[inductor] Thread block-size floors through Triton autotuning`)
  - `5109e041b75` (`[inductor] Add nested reduction scheduler legality`)
  - `dd5ec144010` (`[inductor] Lower nested reductions in SIMD codegen`)
- Follow-up commits parked for later:
  - `d7fd134fa01` (`[inductor] Fuse NVFP4 nested-reduction packing`)
  - `0eff320375b` (`[inductor] Checkpoint follow-up fusion work`)
- Useful old refs:
  - `nested_reduction_v3_test_before_resplit_20260505_175823` is the backup
    branch for the previous six-commit split.
  - `6280052fd47` is the previous pre-resplit final tree; current `HEAD` should
    remain tree-identical to it.
  - `38d787d3823` (`[inductor] Fuse dependent cross-axis reductions`) is the
    last one-commit version before the split.
  - `037949e5ead` / `1907b69aeef` / `09290721cff` are the earlier split
    commits that still carried the XBLOCK grouped-reduction path and its
    scheduler support.

Goal: split the current nested-reduction commit into reviewable pieces while
keeping the first landing stack focused on the FP8/MXFP8-style nested reduction.

## Current Scope Decision

The first landing stack intentionally supports nested grouped reductions where
the local group splits the parent reduction axis (`RBLOCK`). This covers the
block-quant pattern we need first:

```text
[B, D] -> [B, D / G, G].amax(-1)
```

The earlier prototype also supported grouped reductions where the local group
splits the parent pointwise axis (`XBLOCK`). That path is not fundamentally
blocked, but it brought extra scheduler-side classification, scoring,
coalescing/profitability, and codegen branching into the first PR. For the first
landing stack, XBLOCK grouped reductions should fall back to the existing
multi-kernel path.

Follow-up requirement: restore the full equivalent XBLOCK grouped-reduction
case after the RBLOCK path lands, preferably with generic vertical dependency
matching/scoring so the support does not need the earlier nested-specific
broadcast-dep machinery. The old refs above are the implementation breadcrumbs.

## Recommended Landing Stack

### 1. Triton Range Metadata Refactor

Scope:

- Root-owned `block_size()`, `block_offset()`, `mask_name()`, `owns_mask()`,
  `full_range()`

Framing:

This should read as a pure codegen cleanup. Existing range roots already own
this information logically; the commit just makes that ownership explicit so
later commits do not need to special-case root metadata by symbol name.

### 2. Derived SIMD Range Roots

Scope:

- `DerivedIterationRangesRoot`
- Active range-tree handling (`active_range_trees`, `use_range_trees`)
- Mask/index classification by owning range tree instead of symbol prefix only
- Generic reshape/reduce/broadcast emit helpers

Framing:

This is still codegen plumbing, not nested scheduler policy. It gives SIMD
codegen a way to temporarily view the same physical loop through a derived
iteration space.

### 3. Autotune Block Floors

Scope:

- `min_xblock` / `min_rblock` metadata on kernels/configs
- Reduction config generation honoring minimum block sizes
- Persistent reduction config threading
- Coordinate descent `value_too_small` handling
- Coordinate descent unit test

Why separate:

Grouped reshape legality requires the chosen block to be at least the local
group size. The first landing stack only consumes `min_rblock`, but
`min_xblock` belongs here too so the autotune API is symmetric and the future
XBLOCK grouped-reduction follow-up does not need to reopen this plumbing.

### 4. Nested Scheduler Legality

Scope:

- `config.triton.nested_reduction`
- `metrics.codegen_nested_reduction`
- `NestedReduction`
- `FusedNestedReductions`
- Nested-scoped score bridge with TODO for future generic normalized scoring
- Scheduler legality/rejection tests

Why separate:

This commit answers "when is this fusion legal?" It should be reviewable without
also reviewing the large SIMD lowering. The scheduler commit can introduce the
legality model before the lowering commit exercises it end-to-end.

Keep nested-scoped:

The score bridge should stay nested-specific for now. A broader generic scoring
change can accidentally affect fusion ordering and skip loop-reindex repair.

### 5. Nested Codegen and End-to-End Tests

Scope:

- `_GroupReductionLayout`
- Reduced-output `_DerivedIterationFamily`
- `_GroupedReductionOpsHandler`
- Basic grouped reshape + reduce lowering
- Reduced-output prologue/epilogue support
- Pointwise schedule reuse/remapping
- Lazy full-resolution load resolver
- Register/CSE broadcast-back from reduced grouped values to parent resolution
- End-to-end tests, generated-kernel checks, FP8/MXFP8-style quant epilogues,
  B=1, masks, dynamic shapes, and non-persistent coverage

Why separate:

This commit answers "how do we emit it?" Keep the behavior tests next to the
lowering so reviewers can read generated-code expectations beside the code that
produces them.

## Deferred Generic Prep

### Generic Vertical Fusion Prep

Scope:

- `MemoryDep.normalize_without_broadcast()`
- Generic vertical read/write matching that understands broadcast-stripped deps
- Intermediate-dependency safety after relaxed dep matching
- Loop reindex retry rollback/repair needed by pointwise/reduction vertical fusion
- Focused scheduler/loop-ordering tests

Status note:

See `agent_space/vertical_broadcast_fusion_mechanism.md`. The broad
`normalize_without_broadcast()` / `fusable_read_and_write()` fallback should not
lead the stack unless we find a real non-nested compiled-kernel case that needs
it. Ordinary broadcast epilogues already fuse through existing normalization,
shared-input scoring, and loop reindexing paths.

Why separate:

These are not intrinsically nested-reduction features. They are generic fusion
correctness improvements exposed by nested reduction. Splitting them out reduces
the amount of scheduler irregularity in the nested PR.

## Test Placement

Prefer folding tests into the commit that introduces the behavior. Avoid a
single final test-only commit unless review tooling makes that necessary.

Suggested placement:

- Range metadata: helper/codegen stability tests if isolated; otherwise rely on
  later nested codegen tests
- Derived range roots: helper/codegen stability tests if isolated; otherwise rely
  on later nested codegen tests
- Autotune floors: coordinate descent test
- Nested scheduler detection: legality/rejection tests and metric checks
- Nested codegen: reduced-output numerical/kernel-form tests plus FP8/MXFP8,
  prologue/epilogue, B=1, mask and non-persistent coverage

## Follow-Up Work Not Blocking This Stack

- Revisit generic normalized-dep scoring so the nested score bridge can be
  removed.
- Restore XBLOCK grouped nested reductions as a follow-up. The future feature
  should cover the same shape family as the current RBLOCK support, but with
  the local group split out of `XBLOCK`; use the older split refs above as
  breadcrumbs rather than re-discovering the codegen shape mapping.
- Generalize full-resolution epilogues for group-size-in-X with
  dependency-aware coordinate mapping as part of that XBLOCK follow-up.
- Replay `d7fd134fa01` for NVFP4 half-resolution consumers after the base stack
  lands.
- Replay or rework `0eff320375b` after the base and NVFP4 paths are stable.
