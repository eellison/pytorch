# F2a review guide: keep scale work at group width

Date: 2026-08-27

This is an AI-assisted local review guide. It is not intended to be pasted into
GitHub without human review and the disclosure required by `AI_POLICY.md`.

## Snapshot

Worktree:

`/data/users/eellison/pytorch/agent_space/f2a_final_wt`

Base F1 commit:

`859385382d515b81c551a60faac10a749491e377`

F2a is an uncommitted five-file diff. It does not change scheduler legality,
generic CSE, or generic Triton codegen. The only shared addition is metadata:
`register_pointwise()` records the lowered deterministic scalar operation name
that the sub-parent wrapper may consult.

## Purpose

The natural NVFP4 graph computes the same group scale in two fused epilogue
bodies: packing and scale swizzle. If the first body broadcasts the amax result
to child width immediately, its scale expression has a different shape from the
second body's group-width expression, so normal CSE cannot recognize them as the
same computation. That produced two FP8 conversions.

F2a now keeps the exact scalar scale chain at group width:

```text
group amax
  -> casts / multiply / clamp / FP8 cast / float cast / reciprocal
  -> one broadcast when the scale first mixes with child-width values
```

Both epilogue bodies therefore emit the same group-width expression and ordinary
CSE reuses it. There is no NVFP4-specific or dtype-specific codegen rule.

## Review order

1. Raw group-width admission:
   [`_SubParentValueResolver.resolve_load`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/codegen/simd.py:2598)

   A source stays narrow only for a direct relation (`parent_lanes is None`), an
   unmasked load, and an unambiguous group-width live shape. Lane projections and
   the `group_width == child_width` case retain the existing materialization path.

2. Narrow positive rule:
   [`registered_pointwise_ops`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/ops_handler.py:56),
   [`register_pointwise`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/lowering.py:1141), and
   [`_operation_preserves_group_width`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/codegen/simd.py:2350)

   Deterministic scalar operations emitted by `register_pointwise()` may remain
   narrow. The manual `to_dtype`, `mul`, and `truediv` lowerings register the
   same property explicitly. RNG, memory, reduction, masked, and inline-assembly
   operations are not registered. `ShapePropagationOpsHandler` is the second
   gate: it must also prove that this invocation's result remains group-width.
   A child-width operand or unknown shape materializes before emission.

3. Materialization boundaries:
   [`_PointwiseRemapHandler._default`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/codegen/simd.py:2363),
   [`store`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/codegen/simd.py:2323), and
   [`masked`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/codegen/simd.py:2380)

   Unsupported operations, stores, and values leaving a masked callback widen
   through the existing sub-parent materializer. When no resolver is present,
   the wrapper delegates directly, so ordinary remapped codegen is unchanged.

4. Shape and liveness handling:
   [`is_group_width_shape`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/codegen/simd.py:2516) and
   [`materialize_group_width`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/torch/_inductor/codegen/simd.py:2524)

   Widening reuses `_materialize`, including its CSE-liveness check, cache, target
   family masks, and canonical broadcast implementation.

5. Focused policy tests:
   [`test_sub_parent_group_width_raw_resolution`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/test/inductor/test_inductor_scheduler.py:674) and
   [`test_sub_parent_group_width_materialization_boundaries`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/test/inductor/test_inductor_scheduler.py:710)

6. End-to-end regression:
   [`test_nvfp4_scale_swizzle_reuses_group_scale`](/data/users/eellison/pytorch/agent_space/f2a_final_wt/test/inductor/test_nested_reduction.py:3452)

   This uses the real `/6`, min/max clamp, emulated-cast, reciprocal, pack, and
   swizzle chain at hidden size 4608. It requires one fused kernel, exactly one
   numeric FP8 conversion across the whole source, and reciprocal-before-broadcast
   ordering. The numeric test at line 1113 compares the same tail/swizzle graph
   against non-nested compilation.

## Validation

- Full scheduler suite: 136 tests passed, 6 skipped.
- Full nested-reduction suite: 403 tests passed, 8 skipped. The registry rerun
  reproduced the known OpenSSL checksum environment error in one AOT test; that
  test passed when rerun with the Conda executable directory first in `PATH`.
- Focused NVFP4 tests: 18 passed.
- Group-width policy tests: 12 passed, including registered `abs` and excluded
  RNG boundaries.
- Masked-source tests: 4 passed.
- Indirect-index mask tests: 6 passed.
- `spin quicklint`: clean.
- Protected source corpus: all 10 hashes unchanged, including MXFP6 persistent
  and looped, scale swizzle, preshuffle, and DCN preshuffle.
- All 24 NVFP4/MXFP4 generated sources are unchanged from the final pre-registry
  F2a snapshot after metadata normalization.
- The earlier per-operation mutations establish that every operation in the
  target scale chain must remain narrow. The policy test now additionally proves
  that a registered operation outside that chain (`abs`) is admitted and RNG is
  still a materialization boundary.

## Performance

B200, FlashInfer 0.6.14, natural `quant(); swizzle(scale)`, coordinate descent
enabled, 100 warmup / 500 repetitions:

| Format | Shape | Inductor | FlashInfer | Relative |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 | 4096x4096 | 20.48 us | 20.45 us | parity |
| NVFP4 | 4096x4608 | 24.77 us | 24.54 us | 0.9% slower |
| NVFP4 | 4096x8192 | 32.74 us | 36.83 us | 11.1% faster |
| MXFP4 | 4096x4096 | 20.45 us | 22.50 us | 9.1% faster |
| MXFP4 | 4096x4608 | 24.64 us | 26.59 us | 7.3% faster |
| MXFP4 | 4096x8192 | 32.74 us | 36.83 us | 11.1% faster |

NVFP4 geometric mean is 1.036x faster; MXFP4 is 1.101x faster. Every case is
one nested kernel with one scale conversion and exact row-major-equivalent
output. Artifact:
`agent_space/f2a_final_structural_swizzle_cd_post_todtype_20260827.json`.
The immutable final-code snapshot in
`agent_space/f2a_final_bab8_snapshot_swizzle_cd_gpu3_20260827.json` emits the
same normalized sources and reproduces five timings within 0.52%; its small
NVFP4 case landed on a known noisy 22.5 us plateau.

Without coordinate descent, the same kernels remain one-conversion and correct,
but configuration selection leaves NVFP4 about 13% slower and MXFP4 about 7%
slower geometrically. That is separate from this duplicate-computation fix.
