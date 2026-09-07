# MXFP6 packing, preshuffle, and slice-store prototype report

Date: 2026-08-26

Scope: read-only review of D112902012, D112910493, and related D112902015; local B200 experiments against the pre-prototype Inductor checkout and `pr191775_slice_store_proto`; and a public-source survey. No production files or GitHub state were changed.

## Conclusions

1. The compact MXFP6 byte stream used by the internal DCN kernel, Sonar, Quack, MSLK, CUTLASS sub-byte references, and AITER's GEMM path is the same canonical little-endian stream for every four E2M3 codes:

   ```python
   b0 = q0 | ((q1 & 0x03) << 6)
   b1 = ((q1 >> 2) & 0x0F) | ((q2 & 0x0F) << 4)
   b2 = ((q2 >> 4) & 0x03) | (q3 << 2)
   ```

   Hand-written kernels issue three direct byte stores. The prototype makes the equivalent three output slices expressible from PyTorch `stack`; it does not change the file format.

2. D112902012's externally visible packed tensor is row-major even in the preshuffled path. Only the scale is shuffled. The cleanest source formulation is therefore to quantize and pack row-major, then reshape/permute only the scale. It is exactly equivalent when it keeps D112902012's preshuffled scale rule.

3. D112910493 already lowers its matched graph to D112902015's hand Triton kernel, which has three direct packed-byte stores and a scale-only `PRESHUFFLE` address transform. It needs no 4-to-3/slice-store change.

4. The prototype is neutral to faster on the representative row-major, standalone-preshuffle, DCN-preshuffle, persistent, and looped cases. The original D112902012-style RMSNorm preshuffle graph regresses from 2 kernels / 3.166 us to 5 kernels / 6.760 us. Rewriting it as scale-only preshuffle restores 1 kernel / 1.94 us.

5. The supplied D112902015 kernel has a missing packed-output tail mask for launch configurations where `GROUPS_PER_THREAD` is not a multiple of `GROUP_LOAD`. D112910493's actual caller does not enforce that precondition. At `(2048,3072)`, `GROUPS_PER_THREAD=79`, so each program's second iteration has 15 valid groups but stores 64, racing into the next program's output. Adding the per-program output limit makes the reference exact.

6. The reviewed `(4,3)` lowering should be retained. With the scale-only source rewrite and a one-condition staged-append fix, it emits one kernel with direct packed and scale stores for production DCN, standalone, and RMSNorm. The three-slice lowering produces the same result and essentially the same latency, so it adds no benefit for this workload.

## Internal layouts

### D112902012

The source implementation uses `torch.stack((b0, b1, b2), dim=-1).to(torch.uint8)`. Its preshuffled path first changes the reduction order from logical

```text
[row/128, row_tile4, row32, K/192, sub3, K-pair2, 32]
```

to

```text
[row/128, K/192, K-pair2, row32, sub3, row_tile4, 32].
```

It reduces in that order so scale bytes are already in the consumer layout, then inverse-permutes packed data back to row-major. For logical scale `(row, kg)`, the flat preshuffle index is equivalent to:

```text
[row//128, kg//6, (kg%2)*32 + row%32, (kg//2)%3, (row//32)%4]
```

The row-major and preshuffled functions in D112902012 do not use identical scale selection: row-major uses `floor(log2(max_abs)) - 2`, while preshuffle uses `ceil(log2(max_abs / 7.5))`. The latter must be retained when checking exact equivalence to the preshuffled graph.

Minimal three-lane rewrite of the existing pre-permuted graph:

```python
# Before: the byte axis participates in the later view/permute.
packed = torch.stack((b0, b1, b2), dim=-1).to(torch.uint8)
packed = packed.reshape(g, kt, 2, 32, 3, 4, 8, 3)
packed = packed.permute(0, 5, 3, 1, 4, 2, 6, 7)

# After: transform each byte lane, then concatenate the final output axis.
lanes = tuple(
    byte.to(torch.uint8)
    .reshape(g, kt, 2, 32, 3, 4, 8)
    .permute(0, 5, 3, 1, 4, 2, 6)
    for byte in (b0, b1, b2)
)
packed = torch.stack(lanes, dim=-1)
```

This is exact, but it exposes three disjoint `ConcatKernel` aliases and triggers the RMSNorm fragmentation described below.

Recommended scale-only rewrite:

```python
blocks = value.reshape(rows, width // 32, 32)
exponent, codes = quantize_with_preshuffle_scale_rule(blocks)
packed = pack_fp6(codes.reshape(rows, width))
scale = (exponent.to(torch.int32) + 127).to(torch.uint8)
scale = scale.reshape(rows // 128, 4, 32, width // 192, 3, 2)
scale = scale.permute(0, 3, 5, 2, 4, 1).reshape(-1)
```

This formulation was bitwise equal to the original D112902012-style graph at `(256,576)`. At `(2048,3072)`, standalone output was also bitwise equal to the corrected D112902015 reference.

### D112910493 and D112902015

D112910493 is the graph-pattern replacement. D112902015 is the related revision containing the actual `_dcn_mxfp6_quantize_kernel` and launcher. The kernel directly stores `b0`, `b1`, and `b2` at `o0`, `o0+1`, and `o0+2`. `PRESHUFFLE` changes only the scale store address. Packed bytes are identical for `PRESHUFFLE=False/True`; scale bytes are a permutation of the same values.

The caller uses:

```python
num_threads = ceil(sqrt(numel))
GROUP_LOAD = 64
GROUPS_PER_THREAD = max(ceil(num_groups / num_threads), 64)
```

For `(2048,3072)`, the launch is grid 2509, `GROUPS_PER_THREAD=79`, two loop iterations, and four warps. There is no multiple-of-64 check or rounding in the caller.

The existing packed stores are masked only against the global output size. The missing condition is a per-program limit:

```python
output_limit = initial_output_start + GROUPS_PER_THREAD * 24
store_mask = global_output_mask & (byte_offset < output_limit)
```

The scale store already has an equivalent per-program bound. With the added packed bound, standalone row and preshuffle are exact. Without it at production shape, standalone differs in 2,914,085 of 4,718,592 packed bytes while all 196,608 scales match. The fused-DCN comparison differs in 2,880,310 packed bytes. At `(32,1024)`, `GROUPS_PER_THREAD=64`, so supplied and corrected kernels are identical.

## B200 benchmark matrix

Times are CUDA-graph replay medians in microseconds. Each replay batches 20 calls (100 for the small FMHA case); `triton.testing.do_bench` used `warmup=20, rep=100`. "Current" is the `current` expression on the installed pre-prototype checkout. "Slice" is the source three-lane expression on `pr191775_slice_store_proto`. Kernel counts are shown as `time / kernels`.

| Case | Shape | Current | Slice | Delta | Exactness |
|---|---:|---:|---:|---:|---|
| Standalone row, G32 | 2048x3072 | 15.253 / 1 | 14.845 / 1 | -2.7% | Both exact to eager and corrected D |
| Standalone original preshuffle | 2048x3072 | 20.067 / 1 | 17.613 / 1 | -12.2% | Both exact to eager and corrected D |
| Fused DCN row, G32 | 2048x3072 | 15.874 / 1 | 15.870 / 1 | neutral | Current and slice exact to each other; both differ from eager/corrected D by 67 scales and 16,359 packed bytes |
| Fused DCN original preshuffle | 2048x3072 | 28.363 / 2 | 25.704 / 2 | -9.4% | Same fused-DCN mismatch as row |
| Realized-fp16 DCN row | 2048x3072 | 19.347 / 2 | 19.096 / 2 | -1.3% | Exact to eager and corrected D |
| Realized-fp16 DCN original preshuffle | 2048x3072 | 24.154 / 2 | 21.710 / 2 | -10.1% | Exact to eager and corrected D |
| Fused DCN persistent, G32 | 32x1024 | 1.634 / 1 | 1.638 / 1 | +0.3% | Current=slice; 66 packed-byte differences from eager/D due fused arithmetic rounding |
| Fused DCN looped compiler stress, G16384 | 8x16384 | 7.264 / 1 | 7.170 / 1 | -1.3% | Current=slice; two generated reduction loops; not a legal MXFP6 G32 case |
| RMSNorm row, G32 | 128x384 | 1.939 / 1 | 1.944 / 1 | neutral | Current=slice; compiled fusion differs from eager by 1 scale and 146 packed bytes |
| RMSNorm original preshuffle | 128x384 | 3.166 / 2 | 6.760 / 5 | +113.5% | Current=slice; same RMS reduction-order differences from eager |
| Realized RMSNorm row | 128x384 | 3.371 / 2 | 3.072 / 2 | -8.9% | Exact to eager |
| Realized RMSNorm original preshuffle | 128x384 | 3.576 / 2 | 3.074 / 2 | -14.0% | Exact to eager |

The non-realized DCN and RMSNorm differences are not packing-layout errors. Fusion removes the explicit fp16 materialization and changes arithmetic/reduction rounding. Explicitly realizing the fp16 producer makes current, slice, eager, and corrected hand-kernel outputs agree exactly.

Corrected D112902015 reference latencies at `(2048,3072)` were 22.73 us row and 24.06 us preshuffled for DCN, and 23.14 us row and 23.86 us preshuffled standalone. It is one kernel with four stores in all cases.

### Scale-only preshuffle

The source rewrite is bitwise equal to the original D112902012-style expression for both standalone and DCN at `(2048,3072)`. It uses ordinary `reshape`/`permute` operations on the scale; no `ir.Scatter` or extra copy is required.

| Case | Shape | Reviewed `(4,3)`, no fix | `(4,3)` + planner fix | Slice + planner fix | Corrected D |
|---|---:|---:|---:|---:|---:|
| Standalone | 2048x3072 | 17.611 / 2 | 15.456 / 1 | 15.050 / 1 | 23.859 / 1 |
| Fused DCN | 2048x3072 | 17.611 / 2 | 16.280 / 1 | 16.275 / 1 | 24.054 / 1 |
| RMSNorm | 128x384 | 1.944 / 1 | 1.939 / 1 | 1.941 / 1 | n/a |

Every one-kernel compiler result above is a nested reduction with four stores. The `(4,3)` dump is `agent_space/dumps/dcn_prod_scale_only_43_append_guard/source_0.py`. Its scale store directly uses the R128c4-equivalent output index, followed by three direct packed-byte stores to one output pointer. There is no scale temporary or swizzle kernel.

Standalone is bitwise exact to eager and corrected D. Fused DCN differs from eager/corrected D by 67 of 196,608 scales and 16,359 of 4,718,592 packed bytes because compiled fusion removes the explicit fp16 `addcmul` materialization. The `(4,3)` and slice outputs are bitwise equal. Explicitly realizing the fp16 producer restores exactness but adds a kernel.

This is the preferred D112902012 rewrite. It is faster than the original pre-permuted source at production shape, avoids the RMSNorm 2-to-5-kernel regression, and does not require changing the packed representation.

## Production DCN planner diagnosis

The original pre-permuted DCN graph at `(2048,3072)` emits 2 kernels, no nested kernel, and takes 28.461 us. The candidate `(4,3)` grouping is recognized, but `_try_get_sub_parent_source_projections` rejects `buf7` because the producer and pack consumer use different outer iteration frames.

The normalized producer access is:

```text
parent_r
+ 393216*(parent_x//12288)
+ 128*ModularIndexing(parent_x,1,3)
+ 24576*ModularIndexing(parent_x,3,16)
+ 384*ModularIndexing(parent_x,48,64)
+ 32*ModularIndexing(parent_x,3072,4)
```

The first pack lane reads:

```text
4*child_r + 32*parent_x
```

Replacing `parent_r` with `4*child_r + lane` cannot remove the outer-X permutation, so accepting the current projection would be incorrect. This is specific to the pre-permuted multi-input DCN producer layout, not tensor size: the production and small DCN graphs both decline, while standalone/SILU preshuffle graphs fuse at both production and small shapes. A test-only realization of row-major codes proves the layout diagnosis by producing one nested kernel at 20.890 us, but it is not an appropriate source API.

A shape-specific scratch reindex of the pack and final output epilogue into the producer frame also proves that the original full-preshuffle graph can be one kernel: it is byte-identical to the compiled baseline and measures 18.22 us persistent or 21.09 us looped, versus 28.67/28.46 us without reindexing. Generalizing that experiment requires a dense-permutation proof, coordinated closure mutation, and rollback, so it is broader and slower than the scale-only source rewrite plus the reduced-output allowance.

After the scale-only rewrite, packed-code projection succeeds. The remaining production failure happens later: a `FusedStagedReduction` already contains the pack, and appending the reduced scale output is rejected as `consumer is not wholly in the sub-parent epilogue`. This is an initial-formation rule being reapplied to an existing staged group. The Triton backend already distinguishes these cases, so the narrow scheduler change is:

```python
elif type(node1) is not FusedStagedReduction and not all(
    node in plan.sub_parent_stages[0].epilogue_nodes
    for node in node2.get_nodes()
):
    return False
```

Complete plan reconstruction remains the general invariant. It classifies every node into a supported domain, validates all projections, orders post-reduction work, and prevents reads of sub-parent outputs. The staged dependency proof, ordinary fusion legality, backend plan/tiling check, and codegen-time replanning still run. A shifted-scale negative mutant remains at two kernels and is rejected by `staged fusion dependency proof failed`.

Scheduler order cannot avoid this change: before the pack creates the staged group, the swizzled reduced scale and reduction are not an ordinary fusion candidate because their index frames have zero shared-data score. The scale therefore arrives as a later append. The dedicated `FusedNestedReductions` append path already handles the analogous case; standalone `FusedStagedReduction` did not.

Scratch worktrees:

- Reviewed `(4,3)` plus the clean one-condition fix: `agent_space/mxfp6_43_append_guard_fix`
- Slice prototype plus the same condition: `agent_space/mxfp6_slice_scale_fusion_fix`

Recommendation: keep `(4,3)`, change D112902012 to scale-only preshuffle, and use the already-staged guard with positive exact-append and negative shifted-append tests. Do not relax the original cross-frame source projection.

Verification: the exact MXFP6 append changes from 2 kernels to 1; a shifted-scale mutant remains at 2 kernels and fails the staged dependency proof; the focused MXFP6 suite passed 26 tests with 2 skipped; scheduler subsets passed 24 `nested` and 12 `sub_parent` tests; and the full nested-reduction file passed 391 tests with 8 skipped for the equivalent already-staged guard.

## RMSNorm regression

Generated-code samples:

- `agent_space/dumps/rmsnorm_preshuffle_baseline_current/source_0.py`
- `agent_space/dumps/rmsnorm_preshuffle_proto_slice/source_0.py`
- `agent_space/dumps/standalone_prod_preshuffle_proto_slice/source_0.py`
- `agent_space/dumps/dcn_prod_preshuffle_proto_slice/source_0.py`

The pre-prototype graph emits one RMS variance kernel and one fused normalize/quantize/scale/packed-store kernel. The prototype emits the RMS kernel, a quantize/scale/temporary kernel, and one pointwise kernel for each of the three packed output aliases.

The candidate nested-reduction plan is found, but `_sub_parent_aliasing_is_supported` rejects each incremental fusion because only one alias writer is present while it requires every input of the three-input `ConcatKernel`. The siblings share no data edge, so the scheduler never groups them before this check. Merely relaxing the completeness check reduced 5 kernels to 4 but then failed staged dependency proof; adding artificial lane dependencies reduced it only to 3. Supporting partial aliases safely requires atomic sibling grouping or a more general staged-alias design, not a one-line legality relaxation.

Smallest concrete fix: keep packed data row-major throughout, preshuffle only the scale, and allow the already planned reduced scale output to join an existing staged reduction. If the source cannot be changed, retain/fall back to the existing 4-to-3 lowering when the complete sibling group cannot be planned.

## Public implementations

| Project/reference | Packing and output layout | Stores / representative shape |
|---|---|---|
| Sonar `mxfp6_online_utils.py`, commit `5d9c85a1` | Canonical compact row-major data; optional scale-only R128c4 swizzle | Three direct byte stores; `BLOCK_K=256`. B200 port at 2048x3072: 23.755 us, one kernel, exact to corrected internal standalone reference |
| AITER GEMM `gemm_op_a6w6.py`, commit `a8d090cb` | Canonical bytes, then both data and scales are retiled into a padded 256x128 C0/C1 GEMM blob. Uses floor-minus-2 scales and half-up E2M3 rounding | Three direct byte stores, `BLOCK_M=128`. B200 structural port at 2048x3072: 19.760 us, one kernel, exact to its layout oracle |
| AITER FMHA `mxfp6_fmha_pack.py`, commit `a21d8ad1` | Q/K field order is `q0,q16,q1,q17,...,q15,q31`, then canonical 6-bit streaming. K has a second gather into 16 KiB LDS-order tiles; V uses 12,288 data + 512 scale bytes per 128-token tile | Three direct byte stores, `BLOCK_N=128`. B200 Q/K port at `[1,256,5,128]`: 55.485 us, grid 40, one kernel, exact to its oracle |
| AITER CK `pk_fp6.hpp` | `pk_fp6_t<16/32>` places element `i` at bit `i*6`, the canonical little-endian stream | Packed numeric container, not a quantization launcher |
| Quack `blockscaled/quantize.py`, commit `c8ec3170` | Pure-Torch canonical `stack(b0,b1,b2)`; data stays compact row-major, scale uses R128c4 | Direct target for this Inductor lowering |
| MSLK `mx_mixed_dtype_utils.py`, commit `7ba5a48b` | Pure-Torch canonical `stack(b0,b1,b2)`; R128c4 scale | Example GEMM shapes include `(1,256,2048)`, `(64,512,2048)`, `(256,1024,4096)` |
| CUTLASS `float_e2m3_t` / `SubbyteReference`, commit `e05f953a` | Six-bit element `i` starts at bit `i*6`; `Sm1xxBlockScaledTensorConfig` scale atom is R128c4 | Consumer/layout definition rather than an online quantizer |
| torchao historical commit `29488018` | Triton kernel used three direct stores but plane-split bytes within each MX row (`all b0`, then `all b1`, then `all b2`), unlike canonical interleaving. The path was later removed | Historical only; current torchao has no compact MXFP6 pack path found |
| FlashInfer commit `ede7a275` | No online MXFP6 packing kernel found. Vendored TRT-LLM documents Linear/R8c4/R8c16/R128c4 scale layouts | Consumer support only |
| FourOverSix commit `dadfad...` | FP6 values occupy byte containers (`packing_factor=1`); scales are Blackwell-blocked | Not a compact 4-to-3 target |

Public references used read-only:

- https://github.com/dphnAI/sonar/blob/5d9c85a1beedb6a704c03ddc1aba5fc0835c8c37/aphrodite/model_executor/layers/quantization/utils/mxfp6_online_utils.py
- https://github.com/Dao-AILab/quack/blob/c8ec3170057987da0ec99883736f381ea1937cf3/quack/blockscaled/quantize.py

For a direct AITER Q/K-style comparison, keeping the 32-value fields grouped through packing and removing artificial `realize` barriers makes reviewed `(4,3)` Inductor exact and one-kernel. Inductor/AITER B200 times were 3.748/4.710 us at `8192x128`, 10.793/12.891 us at `32768x128`, 3.767/4.363 us at `1024x1024`, and 5.161/5.899 us at `128x12288`. The earlier flattened/four-realize harness understated Inductor performance.

## Commands

```bash
conda run -n pytorch-3.12 python agent_space/run_internal_mxfp6_baseline.py \
  --case standalone_prod_row --case standalone_prod_preshuffle \
  --case dcn_prod_row --case dcn_prod_preshuffle \
  --case dcn_realized_prod_row --case dcn_realized_prod_preshuffle \
  --case dcn_persistent --case dcn_looped \
  --case rmsnorm_row --case rmsnorm_preshuffle \
  --case rmsnorm_realized_row --case rmsnorm_realized_preshuffle

conda run -n pytorch-3.12 python agent_space/run_internal_mxfp6_compare.py \
  --case standalone_prod_row --case standalone_prod_preshuffle \
  --case dcn_prod_row --case dcn_prod_preshuffle \
  --case dcn_realized_prod_row --case dcn_realized_prod_preshuffle \
  --case dcn_persistent --case dcn_looped \
  --case rmsnorm_row --case rmsnorm_preshuffle \
  --case rmsnorm_realized_row --case rmsnorm_realized_preshuffle

MXFP6_SCALE_ONLY_PRESHUFFLE=1 conda run -n pytorch-3.12 \
  python agent_space/run_internal_mxfp6_compare.py \
  --case standalone_prod_preshuffle --case dcn_prod_preshuffle \
  --case rmsnorm_preshuffle

MXFP6_SCALE_ONLY_PRESHUFFLE=1 conda run -n pytorch-3.12 \
  python agent_space/run_internal_mxfp6_43_append_guard_fix.py \
  --case standalone_prod_preshuffle --case dcn_prod_preshuffle \
  --case rmsnorm_preshuffle

conda run -n pytorch-3.12 python agent_space/run_mxfp6_43_append_guard_script.py \
  agent_space/dump_internal_mxfp6_case.py --case dcn_prod_preshuffle --scale-only \
  --output agent_space/dumps/dcn_prod_scale_only_43_append_guard

conda run -n pytorch-3.12 python agent_space/run_mxfp6_43_append_guard_tests.py -k mxfp6

conda run -n pytorch-3.12 python \
  agent_space/run_mxfp6_43_append_guard_script.py \
  agent_space/probe_staged_parent_append_gate.py mxfp6_shifted_reduced

SEED=123 REINDEX_EPILOGUE=1 VERIFY_OUTPUT=1 BENCH=1 SOURCE_KIND=dcn \
  TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 conda run -n pytorch-3.12 \
  python agent_space/diagnose_dcn_preshuffle_43_subagent.py

conda run -n pytorch-3.12 python agent_space/diagnose_mxfp6_rmsnorm_plan.py
conda run -n pytorch-3.12 python agent_space/bench_sonar_mxfp6.py
conda run -n pytorch-3.12 python agent_space/bench_aiter_mxfp6_gemm_pack.py
conda run -n pytorch-3.12 python agent_space/bench_aiter_mxfp6_fmha_pack.py
conda run -n pytorch-3.12 python agent_space/bench_aiter_vs_inductor_grouped_subagent.py
```

## Remaining matrix gaps

- AITER GEMM/FMHA timings here are faithful CUDA/B200 structural ports, not native gfx950 measurements. Their exact layout checks are useful; their latency is not an AMD performance claim.
- No public AITER MXFP6 RMSNorm-fused or DCN-fused online quantizer was found, so those rows use the internal/synthetic graphs.
- D112902015 preshuffle requires rows divisible by 128 and width divisible by 192. It cannot reference the `(32,1024)` persistent or `(8,16384)` looped shapes.
- `(8,16384), G16384` is deliberately a nested-reduction loop stress case, not a valid MXFP6 block-size-32 workload.
- The original and scale-only source formulations were tested on B200 only; MI350 should be rerun before using the B200 ranking to choose the AMD production implementation.
