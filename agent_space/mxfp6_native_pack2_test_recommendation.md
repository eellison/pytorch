# Minimal native MXFP6 regression test for #191775

## Recommendation

Add one CUDA/SM100 integration test to `test/inductor/test_nested_reduction.py`.
Do not change production code and do not make this benchmark-only.

The current tree already supports the needed single-output
`inline_asm_elementwise(..., pack=2)` path in:

- `torch/_inductor/lowering.py:9676-9710`
- `torch/_inductor/codegen/triton.py:1805-1869`

Generic `pack=2` behavior and padding are already covered in
`test/higher_order_ops/test_inline_asm_elementwise.py:242-279` and
`:601-770`. The missing coverage is its composition with the new `(4,3)`
staged epilogue and a static scale-only swizzle.

## Test-only delta

1. Near `test/inductor/test_nested_reduction.py:30`, add the approximately
   five-line `E2M3X2_UNPACK_ASM` constant used by
   `agent_space/bench_aiter_vs_inductor_grouped_native_subagent.py`.
2. Immediately after `_swizzle_scale` at line 226, add an approximately
   22-line helper named `_mxfp6_native_pack_scale_swizzle`:
   - reduce row-major `[B, D / 32, 32]` blocks;
   - scale them;
   - form AITER field order with
     `stack((scaled[..., :16], scaled[..., 16:]), -1)`;
   - retain shape `[B, D / 32, 32]`;
   - convert with the native asm using `dtype=torch.int32, pack=2`;
   - call `_mxfp6_pack_four_to_three(values, realize=False)`;
   - apply `_swizzle_scale` only to the uint8 scale.
3. Near the existing MXFP6 tests around line 1980, add one approximately
   18-line CUDA/SM100-only test in `_NestedReductionBase`. Use static
   `(B,D,G)=(128,384,32)`, compare exactly against a nested-disabled compile,
   call `self.check_fusion()`, and inspect `run_and_get_code` for:
   - exactly one `tl.inline_asm_elementwise`;
   - exactly one `cvt.rn.satfinite.e2m3x2.f32`;
   - exactly four `tl.store` calls;
   - exactly three `tl.split` calls.

Because `_NestedReductionBase` is inherited by `NestedReductionTest` and
`NestedReductionNonPersistentTest`, one test method covers forced persistent
and forced looped codegen. Total estimated delta is 45-50 test LOC.

Do not parameterize or replace
`test_producer_consumer_mxfp6_preshuffled_four_to_three_pack`. That case covers
the older full-data prepermutation and its shifted negative control; the native
test should preserve the distinct scale-only layout and grouped full-resolution
representation.

## Validation

The proposed graph was run against the unmodified reviewed worktree:

| Mode/shape | Exact | Kernels | Nested | Loads | Stores | Asm | Splits |
|---|---|---:|---:|---:|---:|---:|---:|
| Persistent, 128x384 | yes | 1 | 1 | 3 | 4 | 1 | 3 |
| Looped, 128x384 | yes | 1 | 1 | 3 | 4 | 1 | 3 |
| Persistent, 8192x128 | yes | 1 | 1 | 3 | 4 | 1 | 3 |

Probe: `agent_space/probe_mxfp6_native_pack2_scale_swizzle.py`.

The test belongs in #191775 because it locks the interaction among the grouped
full-resolution producer, `(4,3)` staged output, and swizzled reduced output.
An exhaustive test of E2M3 instruction semantics belongs with the HOP/native
conversion work instead.
