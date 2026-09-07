# Indexed forwarding rebase

Current local review commit: `fc09fb38287`, on local review-base snapshot
`d4167dba5b8`. Nothing has been submitted.

## Target contract

- Planning records exact source accesses and exact consumer reads.
- Each consumer record carries only the proved lane when it selects one lane
  from a parent-width value. Direct and reduced-width values are distinguished
  from lane projections by their emitted CSE shape, not by a layout enum.
- Codegen reconstructs the logical `MemoryDep` for the current load and resolves
  only that planned read.
- A source written earlier in the kernel must resolve from a live exact access;
  a miss is a compiler error. External sources may reload normally, bypassing
  name-based store forwarding.
- Guarded sources remain on the ordinary load path. An unguarded source may be
  forwarded only when existing `TritonKernelOverrides.masked()` owns any outer
  predicate and fill.
- Only lane-selecting internal sources suppress the parent/epilogue pass-boundary
  flush.

## Simplification goal

Delete `SubParentSourceLayout`, `source_layouts`, `broadcast_source_names`,
`internal_dependency_names`, `_SubParentSourceLoadResolver`, and
`_DerivedIterationFamily.remapped_values`. The value materializer should infer
direct, reduced-width broadcast, or parent-width lane split from the live value
shape plus the consumer's optional proved lane.

## Final design

- `SubParentAccessRelation` is the only planner/codegen forwarding record. Each
  record contains source alternatives, one exact consumer access, an optional
  parent lane, and planner-derived `requires_live_source` state.
- `_SubParentValueResolver` records exact source loads/stores, indexes relations
  by consumer name and exact normalized `MemoryDep`, and tries source
  alternatives until one live value can be materialized.
- Requiredness comes from raw temporal writer membership in the planned kernel.
  It is not inferred from buffer names or runtime stores.
- A planned external relation may reload if no source CSE is live. A required
  in-kernel relation fails loudly and reports both source and consumer accesses.
- The resolver records only unguarded sources. It forwards while `_load_other`
  is `None`, which means either the consumer is unmasked or existing
  `TritonKernelOverrides.masked()` will emit the outer `where`. When a concrete
  fill belongs on a physical load, optional external sources take the normal
  reload path and required in-kernel sources fail loudly.
- Only required relations with a parent-lane witness suppress the standalone
  parent/body flush. Nested codegen eagerly materializes all lane relations
  before the parent body can be invalidated. These are intentionally separate
  from general requiredness.
- Source recording wraps parent/local/grouped emission, but consumer resolution
  is enabled only while replaying sub-parent output groups. This prevents an
  earlier same-name parent read from being mistaken for a planned epilogue read.
- Direct child-width values are checked before group-width values. When
  `num_groups == child_block`, the value remains a direct use because one child
  element corresponds to one group.

## Deleted compatibility surface

The unstaged F1 diff removes:

- `NestedReduction.SubParentSourceLayout` and all
  `INTERLEAVED`/`BROADCAST`/`IDENTITY` branches;
- `ProjectedSourceAccess`, grouped consumer records, and the runtime flattened
  projected-load record;
- stage `source_projections`, `source_layouts`, `broadcast_source_names`, and
  `internal_dependency_names` views;
- `_SubParentSourceLoadResolver`, `_DerivedIterationFamily.remapped_values`, and
  the name-keyed `forwarded_store_names`/`masked_forward_names` adapters;
- duplicated standalone/nested replay loops, replaced by
  `_codegen_sub_parent_output_groups`.

The replacement surface is one planner record (`SubParentAccessRelation`), one
runtime resolver (`_SubParentValueResolver`), and one small live-source record
(`_ResolvedSubParentSource`). The resolver has 11 methods including `__init__`;
no new function exceeds CC 12.

## Retained scoped path

A trial that moved the inherited parent-to-grouped equivalence proof into plan
records was reverted. Disabling the legacy branch lost more than twenty
established nested fusions, and broad normalized matching could equate unsafe
traversals or fail after fused-loop reconstruction. F1 therefore makes exact
records the sole authority for sub-parent forwarding while retaining the old
proof only for nested-stage legality; codegen never consumes it.

The follow-up is a source-anchored parent-to-grouped proof that records exact
relations before fusion and reconstructs them after fusion without stride-order
normalization. The inline TODO names that endpoint.

## Size and complexity

All figures below compare the F1 review commit with its local review-base
parent.

| File | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| `torch/_inductor/codegen/simd.py` | 316 | 208 | +108 |
| `torch/_inductor/scheduler.py` | 131 | 181 | -50 |
| **Production** | **447** | **389** | **+58** |
| `test/inductor/test_inductor_scheduler.py` | 235 | 40 | +195 |
| `test/inductor/test_nested_reduction.py` | 18 | 7 | +11 |

The current runtime hotspots are `_logical_memory_access` (CC 12, nesting 2)
and `_SubParentValueResolver.materialize_sources` (CC 6, nesting 2). Across
changed scheduler functions, total function-body LOC falls from 775 to 750,
aggregate CC falls from 160 to 150, and maximum nesting falls from 5 to 4.
`_prove_staged_fusion_dependencies` falls from CC 33 to 32.

Permanent coverage adds four scheduler test methods plus one parameterized
extension of the existing nested internal-source test. It covers exact access
identity, temporal mutation names, live/stale CSE behavior, source alternatives,
masked-load ownership, atomic/TMA exclusion, external fallback, the
`num_groups == child_block` ambiguity, and resolver scope.

## Verification

Full files:

```bash
PYTORCH_WORKTREE="$PWD" TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib \
conda run --no-capture-output -n pytorch-3.12 \
python ../run_wt.py \
test/inductor/test_inductor_scheduler.py -q
# 130 passed, 6 skipped

PYTORCH_WORKTREE="$PWD" TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
LD_LIBRARY_PATH=/home/eellison/.conda/envs/pytorch-3.12/lib \
conda run --no-capture-output -n pytorch-3.12 \
python ../run_wt.py \
test/inductor/test_nested_reduction.py -q
# 399 passed, 8 skipped
```

Focused post-format resolver run: 20 passed, 110 deselected. Earlier post-
flatten focused runs passed 30 scheduler tests and 34 nested tests (4 skipped).

Static checks:

```bash
conda run --no-capture-output -n pytorch-3.12 spin quicklint
conda run --no-capture-output -n pytorch-3.12 python -m py_compile \
  torch/_inductor/scheduler.py torch/_inductor/codegen/common.py \
  torch/_inductor/codegen/simd.py torch/_inductor/codegen/triton.py \
  test/inductor/test_inductor_scheduler.py \
  test/inductor/test_nested_reduction.py
git diff --check
```

All pass. The compatibility-symbol scan is empty. The exact final snapshot was
also re-attested against the protected generated-kernel corpus; see
`agent_space/f1_f2a_protected_kernel_reattest_20260827.md`.

Normalized generated sources match the frozen worktree for all ten cases:

| Case | SHA256 |
| --- | --- |
| factor 2 persistent | `98b111d6f7798393c16439e9ed2c770ad98ddb8dee6cb731cb2d1640a75bc59c` |
| MXFP6 4:3 persistent | `b0fe88ce8d8fb83adc55d9d9d9e7c4c16bc360160bd14db3afe6343b42b57a79` |
| internal source persistent | `2efc561292dd1ba54ae8576515638189fcf7671122fcafbb64c0f987f3e9c887` |
| factor 2 looped | `6edb0230b628013056895dcbddd3643228608bb4be95d40250de15d7ec906961` |
| MXFP6 4:3 looped | `fcd5bda8ac08084965dfc92ad7142b0dadde8f3615469f3c4a8b28a44f6a976e` |
| internal source looped | `fb769f5dca3089dce0684cef2b3f4fcc6a5d9c232d736570d89aedd434be52ba` |
| standalone reduced broadcast | `a907b3fbb5d83fdf8a61c7f292a4729d26aa01d4c8f9c9ed61f8562bb83b8258` |
| scale swizzle | `7f72ab3924703863f8e92edcf6849f6c70f9270e31f696cde87fd115571b48e9` |
| preshuffle | `710d6b80635a0add97076be321f2c19dd82301319f8cc85224f6421c99871497` |
| DCN preshuffle | `f3a02e5c076058f991b61a2ebb5bb8a54fb5d7c0e8f08f110018c2f99a869d19` |

The DCN source contains two nested random temporary-directory components; the
scratch comparison normalizes the complete prefix. The remaining source is
identical.

## Handoff

`_ResolvedSubParentSource.value`, `resolve_sources`, and `materialize_source`
remain separate intentionally: immediate F2 can inspect a live lane-free source
before eager width materialization. F2 must keep the existing `ops.masked`
callback boundary and defer only for unmasked consumers.

The F1 review commit is local only. No ghstack or GitHub action was performed.
