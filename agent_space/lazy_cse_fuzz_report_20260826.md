# F2a cast-only group-width CSE: fuzz report

Date: 2026-08-26

Prototype tested:
`/data/users/eellison/pytorch/agent_space/f2a_narrow_prototype_wt`

Canonical worktree:
`/data/users/eellison/pytorch/agent_space/followup_lazy_projection_wt`

The two worktrees are byte-identical across the complete working-tree diff:

```text
sha256(git diff --binary) = 7ab016dc72657354c894e0252d74bc3d0bd8a9c5f4ab35423bf36785236e82b0
```

No production or checked-in test file was changed by this fuzz pass.

## Verdict

The narrow contract passes the CPU policy fuzz and a two-seed F1-versus-F2 GPU
differential. No semantic mismatch, fusion loss, extra kernel, or protected
factor-4 source change was found.

The tested contract is:

1. only an unguarded, lane-free, true group-width load may remain narrow;
2. exactly one `to_dtype` executes at group width;
3. that cast result materializes immediately to child width; and
4. masked, scalar-fill, non-cast, store, lane, unknown-shape, and `G == factor`
   paths retain eager behavior.

## CPU policy fuzz

Runner:
`/data/users/eellison/pytorch/agent_space/lazy_cse_policy_fuzz_20260826.py`

Result:
`/data/users/eellison/pytorch/agent_space/lazy_cse_policy_fuzz_narrow_20260826.json`

- 5 tests passed in 1.55 seconds.
- 500 deterministic non-cast recipes covered 20 operation shapes, 25 times
  each, including arithmetic, comparisons, `where`, reciprocal, square,
  inline asm, unknown operations, mixed widths, and nested pytrees.
- 6 resolver cases covered the complete lane/guard/shape admission boundary.
- 3 cast cases covered cast-then-materialize, an ordinary child-width cast, and
  loud failure when the cast unexpectedly changes shape.
- 2 guarded cases covered masked-callback and scalar-fill loads; both eagerly
  materialized before entering the callback.
- 2 store cases covered ordinary and reduction stores.
- The direct `num_groups == child_block` geometry remained child-width.

All 7 scratch mutants failed their intended oracle:

| Mutant | Detected property |
| --- | --- |
| `disable_raw_load` | the sole unguarded group source must stay narrow |
| `allow_masked_raw` | masked and scalar-fill sources must materialize eagerly |
| `skip_cast_materialization` | the cast result must widen immediately |
| `materialize_before_cast` | conversion must precede widening |
| `allow_noncast_raw` | every non-cast op must widen first |
| `group_equals_child` | direct-width coincidence must not be called group-width |
| `skip_store_materialization` | side effects may not receive a group-width value |

## GPU differential

Runner:
`/data/users/eellison/pytorch/agent_space/lazy_cse_fuzz_20260826.py`

F1 baseline:
`/data/users/eellison/pytorch/agent_space/followup_indexed_rebase_wt`

Results are under:
`/data/users/eellison/pytorch/agent_space/lazy_cse_fuzz_20260826/`

The aggregate comes from `narrow_f2_seed0.json`, `narrow_f2_seed2.json`,
`narrow_f2_bool_seed0.json`, and `narrow_f2_bool_seed2.json`.

- 2 deterministic input sets: seed offsets 0 and 100000.
- 34 compiled cases and 38 runtime invocations in aggregate.
- All 80 compared tensors matched F1 bit-for-bit.
- Every case emitted one staged kernel and one total kernel.
- The dynamic case used one graph for `D=4096,4608,5120`.
- Physical GPU 3, NVIDIA B200.

Coverage per seed:

- persistent and looped NVFP4;
- persistent and looped cast-then-non-cast chains;
- the exact `test_standalone_sub_parent_mismatched_masked_source` graph;
- persistent and looped boolean non-cast barriers;
- persistent and looped `G == factor` direct-width graphs;
- persistent and looped factor-4 chains and MXFP6 packing;
- looped reciprocal, indirect-tail, and dynamic-R cases.

## Source oracles

NVFP4 has exactly one FP8 conversion, no uint8/FP8 bitcast round-trip, and this
order in both persistent and looped kernels:

```text
FP8 conversion -> FP32 conversion -> group-to-child broadcast
```

The synthetic chain additionally verifies that its later comparison,
arithmetic, and `where` occur after that broadcast. This distinguishes the
narrow contract from the earlier broad allowlist design.

The boolean source has no intervening cast. Its generated source contains the
second group-to-child broadcast before the consuming `tl.where`; allowing the
non-cast operation to see the raw group value instead triggers a compile-time
shape assertion.

These cases are normalized-source identical to F1 for both seeds:

- masked/scalar-fill, persistent and looped;
- `G == factor`, persistent and looped;
- factor-4 chain, persistent and looped;
- MXFP6 4:3 packing, persistent and looped; and
- indirect tail masking.

This pins that the narrow optimization does not perturb the reviewed factor-4,
direct-width, or guarded paths.

## GPU mutation tripwires

Three mutations have end-to-end GPU tripwires:

- disabling narrow source resolution restores the old FP8 bitcast broadcast;
- materializing before the cast restores the same old source form; and
- allowing a boolean non-cast consumer to remain narrow fails compilation on
  incompatible group/child shapes.

The remaining mutations are intentionally direct-policy tripwires. Skipping
immediate cast materialization can be repaired by the next real boundary in the
current NVFP4 graph, and the available masked graph has no live masked
group-width candidate. Their CPU tests inspect the policy boundary directly
rather than claiming an end-to-end distinction that the graphs do not expose.

## Separate prerequisite issue

The removed sliced-and-padded group-source repro remains an inherited #191775
fail-to-fallback. Eager and `nested_reduction=False` pass; enabling nested
reduction raises `sub-parent reduction plan was lost before codegen` on the
frozen #191775 layer, F1, and F2. It is not an F2 regression.

Details:
`/data/users/eellison/pytorch/agent_space/masked_group_source_repro_20260826.md`
