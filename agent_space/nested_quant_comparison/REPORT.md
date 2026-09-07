# Main-only nested reduction and quantization benchmark report

Date: 2026-08-31

## Summary

- The old static RMSNorm-to-FP8 gap is not present on PyTorch main. The earlier report used `2.13.0a0+gitcc07682` and did not use the CUDAGraph protocol below, so its 2.1-6.8x statement is excluded here.
- For the clean non-padded fused-add RMSNorm + block-FP8 case (`128x4096`, BF16), Inductor main is 3.437 us by default and 2.277 us with coordinate descent, versus FlashInfer main at 2.270 us with PDL disabled. Tuned Inductor is effectively tied (0.3% slower).
- Padding remains the clear block-FP8 gap. Coordinate-descent Inductor uses three kernels and is 1.11-1.71x slower than FlashInfer on the BF16 padded shapes. `D117896257` directly targets this path, but it is unpublished/unlanded and is not included in any result here.
- Coordinate descent is critical for FP4 and MXFP8. For the tested BF16 FP4 matrix it changes several default 1.4-1.7x deficits into ties or wins. For fused MXFP8 it gives Inductor 1.55-2.16x wins over the fastest external comparison in the table, and up to a 3.80x win over Transformer Engine.
- Padding the 128x4 blocked scale layout is a separate structural cliff on main. The ordinary `F.pad` formulation uses three kernels for row-only padding. Coordinate descent helps the producer but does not remove the extra launches; the existing `force_pointwise_cat` diagnostic reduces this to two kernels and brings small/medium FP4 within 3-9% of PDL-disabled FlashInfer. MXFP8 remains faster than the composed FlashInfer path even when padded.
- vLLM's current Helion dynamic-per-token kernel is competitive. Tuned Inductor wins the large non-add case, but the large fused-add case (`4096x4096`) remains a real gap: 27.507 us versus 21.525 us (1.28x slower).
- The cross-implementation correctness checks pass at the expected quantization/reduction-order tolerances. MXFP8 scale bytes match Transformer Engine and torchao exactly for every tested shape and layout.
- A current-main AITER MXFP6 runner is ready, but this host is an NVIDIA B200 and cannot produce valid ROCm/gfx950 measurements.

## Revisions and environment

| Project | Revision |
|---|---|
| PyTorch main | `cdd22ade2948699188c3e2d0d80b5a396a8489ae` |
| FlashInfer main | `faf7c6aecebcfa9b6dd7bff13c358b434a7d7ce9` |
| Transformer Engine main | `4fba94ae7d22697feae69949f3d28106e60940b8` |
| torchao main | `4e42c02f0f9a48f514167c27656cd7905251902b` |
| vLLM main | `39e276eaeb9daed06a180f6a8d187bbb8790e97b` |
| SGLang main | `9cf157c2521bf9a1f866fd2a8c4a9ff83696085a` |
| AITER main | `5853cf35377eb4e1433a075ce7a9d6fa0beed15d` |

Hardware/software: NVIDIA B200 (SM100), CUDA 13.0, PyTorch `2.15.0a0+gitcdd22ad`, Transformer Engine `2.20.0.dev0+4fba94a`, and Helion 1.4.0 as required by vLLM main.

Every timing below is the median of 50 CUDA-event samples after 20 warmups. Each sample replays a CUDA graph containing 100 operation calls, and the reported value is divided by 100. Inductor's internal CUDAGraph integration is disabled so every implementation is measured through the same external graph wrapper. `triton.nested_reduction=True` is enabled. Coordinate-descent tuning is timed separately and compilation/tuning time is excluded.

PDL is disabled for every FlashInfer call (`enable_pdl=False`). No PDL-enabled number appears in this report. The other measured APIs do not expose a PDL switch on these paths.

All times are microseconds. `CD` means `coordinate_descent_tuning=True`.

## Static per-tensor FP8, BF16

| Operation | Shape | Inductor | Inductor CD | FlashInfer, PDL off |
|---|---:|---:|---:|---:|
| RMSNorm | 1x4096 | 1.468 | 1.475 | 1.689 |
| RMSNorm | 19x4096 | 1.571 | 1.683 | 1.810 |
| RMSNorm | 99x4096 | 1.614 | 1.717 | 1.901 |
| RMSNorm | 128x4096 | 1.639 | 1.641 | 1.922 |
| RMSNorm | 989x4096 | 4.363 | 3.700 | 3.298 |
| RMSNorm | 989x8192 | 7.321 | 6.161 | 4.980 |
| RMSNorm | 989x16384 | 13.092 | 10.878 | 12.472 |
| Fused add + RMSNorm | 1x4096 | 1.772 | 1.905 | 1.811 |
| Fused add + RMSNorm | 19x4096 | 1.855 | 1.849 | 2.137 |
| Fused add + RMSNorm | 99x4096 | 1.927 | 1.929 | 2.242 |
| Fused add + RMSNorm | 128x4096 | 1.946 | 2.110 | 2.267 |
| Fused add + RMSNorm | 989x4096 | 5.448 | 5.453 | 4.795 |
| Fused add + RMSNorm | 989x8192 | 8.729 | 8.732 | 7.367 |
| Fused add + RMSNorm | 989x16384 | 28.236 | 24.496 | 19.250 |

Current Inductor emits one kernel throughout. Coordinate descent helps the large plain RMSNorm cases, but has little effect on fused-add except at hidden size 16384. FP16 results are included in the raw JSON and show the same pattern.

## Fused add + RMSNorm + block-FP8 g128

The scale output is column-major and padded to a multiple of four rows, matching FlashInfer's contract.

| BF16 shape | Padding | Inductor | Inductor CD | FlashInfer, PDL off | CD/FI |
|---|---:|---:|---:|---:|---:|
| 1x4096 | yes | 4.353 | 3.239 | 1.943 | 1.67x |
| 19x4096 | yes | 4.844 | 3.514 | 2.058 | 1.71x |
| 99x4096 | yes | 5.278 | 3.902 | 2.282 | 1.71x |
| 128x4096 | no | 3.437 | 2.277 | 2.270 | 1.00x |
| 989x4096 | yes | 9.568 | 8.077 | 5.540 | 1.46x |
| 989x8192 | yes | 15.203 | 14.646 | 11.233 | 1.30x |
| 989x16384 | yes | 42.718 | 36.800 | 33.051 | 1.11x |

The non-padded case is one Inductor kernel. Nsight Compute shows coordinate descent selecting 512 threads/block instead of the default 1024; FlashInfer also uses 512. The remaining structural difference is that the generated Inductor kernel reloads input/residual for its second pass, while FlashInfer keeps its per-thread vectors resident across the RMS and block-max reductions. In normal CUDAGraph timing that difference is only 0.3% after tuning at this shape.

The padded cases are three kernels on current main. `D117896257` changes the padded scale write to an unsafe in-bounds write plus masked scatter so the producer can write the final allocation and padding can be handled without rereading the full scale output. Since the diff is not landed, it was not applied.

## Fused RMSNorm + FP4, BF16

FlashInfer uses its fused CuTe DSL kernel. Both row-major and 128x4 swizzled scale layouts are covered.

| Format | Shape | Layout | Inductor | Inductor CD | FlashInfer |
|---|---:|---|---:|---:|---:|
| NVFP4 | 128x4096 | row | 2.804 | 1.993 | 2.845 |
| NVFP4 | 128x4096 | swizzled | 2.871 | 2.029 | 2.984 |
| NVFP4 | 1024x4096 | row | 5.125 | 4.645 | 4.038 |
| NVFP4 | 1024x4096 | swizzled | 5.383 | 4.930 | 4.430 |
| NVFP4 | 256x8192 | row | 4.949 | 3.304 | 3.303 |
| NVFP4 | 256x8192 | swizzled | 5.144 | 3.552 | 3.606 |
| NVFP4 | 128x16384 | row | 7.810 | 3.913 | 4.667 |
| NVFP4 | 128x16384 | swizzled | 7.882 | 4.093 | 4.899 |
| MXFP4 | 128x4096 | row | 2.841 | 1.991 | 2.759 |
| MXFP4 | 128x4096 | swizzled | 2.867 | 2.001 | 2.935 |
| MXFP4 | 1024x4096 | row | 5.176 | 4.754 | 5.025 |
| MXFP4 | 1024x4096 | swizzled | 5.326 | 4.877 | 5.082 |
| MXFP4 | 256x8192 | row | 5.061 | 3.342 | 3.568 |
| MXFP4 | 256x8192 | swizzled | 5.259 | 3.589 | 3.715 |
| MXFP4 | 128x16384 | row | 8.023 | 3.739 | 4.739 |
| MXFP4 | 128x16384 | swizzled | 7.947 | 4.201 | 4.943 |

All Inductor cases are one kernel. With coordinate descent, Inductor ties or beats FlashInfer except NVFP4 at `1024x4096`, where it is 11-15% slower. FP16 results are in the raw JSON.

## Fused RMSNorm + MXFP8, BF16

FlashInfer and torchao are composed baselines because current main does not expose their RMSNorm + MXFP8 operation as one fused call. Transformer Engine's `rmsnorm_fwd` is fused. A dash means that source has no directly comparable swizzled composed path in this benchmark.

| Shape | Layout | Inductor | Inductor CD | FlashInfer composed | TE fused | torchao composed |
|---|---|---:|---:|---:|---:|---:|
| 128x4096 | row | 2.951 | 1.933 | 4.408 | 4.170 | 6.241 |
| 128x4096 | swizzled | 3.035 | 2.459 | 4.516 | 4.180 | - |
| 1024x4096 | row | 6.338 | 4.674 | 7.235 | 7.396 | 7.957 |
| 1024x4096 | swizzled | 6.617 | 4.784 | 7.458 | 7.409 | - |
| 256x8192 | row | 5.046 | 3.401 | 6.167 | 5.741 | 8.669 |
| 256x8192 | swizzled | 5.205 | 3.602 | 6.299 | 5.744 | - |
| 128x16384 | row | 7.998 | 3.714 | 7.886 | 14.184 | 12.484 |
| 128x16384 | swizzled | 8.109 | 3.788 | 8.001 | 14.238 | - |

Inductor emits one fused kernel. FlashInfer and torchao each execute separate norm and quant kernels. The coordinate-descent result at `256x8192` was repeated in a fresh dedicated process (3.401 us); importing TE before compiling that graph sometimes caused the tuner to retain the default-like 5.07 us configuration, so the dedicated result is used above. This tuning sensitivity is itself worth following up.

## Padded 128x4 scale layout, BF16

This section uses the explicit `F.pad` plus reshape/permute representation of
the final 128x4 scale layout. Each Inductor mode was compiled and timed in a
fresh process; retaining default and coordinate-descent wrappers for the same
Python frame in one process caused both wrappers to resolve to the later
compiled graph, so those earlier rows were discarded.

| Format | Shape | Inductor | Inductor CD | FlashInfer, PDL off | CD/FI | CD kernels |
|---|---:|---:|---:|---:|---:|---:|
| NVFP4 | 1x4096 | 5.594 | 5.641 | 2.747 | 2.05x | 5 |
| NVFP4 | 19x4096 | 4.808 | 3.896 | 2.857 | 1.36x | 3 |
| NVFP4 | 99x4096 | 4.797 | 4.038 | 2.912 | 1.39x | 3 |
| NVFP4 | 129x4096 | 4.941 | 4.071 | 2.957 | 1.38x | 3 |
| NVFP4 | 989x4096 | 7.501 | 7.101 | 4.395 | 1.62x | 3 |
| NVFP4 | 129x4128 | 4.629 | 4.020 | 3.379 | 1.19x | 2 |
| MXFP4 | 1x4096 | 4.771 | 3.859 | 2.639 | 1.46x | 3 |
| MXFP4 | 19x4096 | 4.785 | 3.879 | 2.830 | 1.37x | 3 |
| MXFP4 | 99x4096 | 4.750 | 3.914 | 2.891 | 1.35x | 3 |
| MXFP4 | 129x4096 | 4.881 | 3.923 | 2.931 | 1.34x | 3 |
| MXFP4 | 989x4096 | 7.419 | 6.701 | 5.057 | 1.33x | 3 |
| MXFP4 | 129x4128 | 4.598 | 3.936 | 3.342 | 1.18x | 2 |
| MXFP8 | 1x4096 | 4.811 | 3.954 | 4.096 | 0.97x | 3 |
| MXFP8 | 19x4096 | 4.873 | 3.924 | 4.371 | 0.90x | 3 |
| MXFP8 | 99x4096 | 4.915 | 3.978 | 4.716 | 0.84x | 3 |
| MXFP8 | 129x4096 | 5.025 | 3.993 | 4.545 | 0.88x | 3 |
| MXFP8 | 989x4096 | 8.541 | 6.835 | 7.448 | 0.92x | 3 |
| MXFP8 | 129x4128 | 4.570 | 3.622 | 5.417 | 0.67x | 2 |

The 3-to-2-kernel difference is caused by main's padding lowering. Row-only
padding is rewritten as a `ConcatKernel`: the nested reduction writes valid
scales into a slice, a second kernel zero-fills the remaining rows, and a third
kernel rereads and swizzles the full allocation. When both rows and columns need
padding (`129x4128`), `_pad_as_cat` declines the multi-dimensional case and the
generic masked pad fuses with the swizzle.

As a main-only diagnostic, `force_pointwise_cat=True` also changes row-only
padding to two kernels. With coordinate descent it measures 2.932/3.111/3.220
us for NVFP4 at 19/99/129 rows, versus FlashInfer at 2.857/2.912/2.957 us.
MXFP4 measures 2.970/3.053/3.091 us versus 2.830/2.891/2.931 us. This is a
substantial improvement, but it still materializes logical scales and performs
the full padded transform; it is not equivalent to a direct final-layout store.

Correctness comparisons unswizzle and crop to the logical scale matrix. This is
necessary because FlashInfer writes every logical scale value but leaves its
padding slots unspecified. MXFP8 logical scales are bit exact on all six shapes.
MXFP4 differences are at most one E8M0 exponent code, and NVFP4 differences are
adjacent E4M3 buckets; their rates are stable across shapes and tuning modes and
show no tile-correlated layout error.

## vLLM Helion dynamic per-token FP8, BF16

| Operation | Shape | Inductor | Inductor CD | vLLM Helion |
|---|---:|---:|---:|---:|
| RMSNorm | 1x4096 | 1.685 | 1.807 | 1.739-1.744 |
| RMSNorm | 128x4096 | 1.928 | 1.967 | 1.945-1.949 |
| RMSNorm | 1024x4096 | 5.547 | 5.556 | 4.706-4.777 |
| RMSNorm | 4096x4096 | 15.340 | 12.162 | 14.392-14.415 |
| RMSNorm | 1024x5120 | 6.881 | 6.303 | 6.197-6.210 |
| Fused add + RMSNorm | 1x4096 | 2.456 | 2.000 | 1.984 |
| Fused add + RMSNorm | 128x4096 | 2.680 | 2.232 | 2.171-2.176 |
| Fused add + RMSNorm | 1024x4096 | 7.404 | 6.332 | 6.251-6.353 |
| Fused add + RMSNorm | 4096x4096 | 28.196 | 27.507 | 21.525-23.303 |
| Fused add + RMSNorm | 1024x5120 | 9.321 | 9.112 | 8.904-8.974 |

Inductor emits one kernel for all these graphs. Coordinate descent closes the `1024x4096` fused-add gap and wins the large non-add case, but the large `4096x4096` fused-add case remains 1.28x slower.

The first pass accidentally omitted `emulate_precision_casts=True`; those numbers were discarded. With it enabled, Inductor and vLLM are bit exact on most shapes and cross-dequantized mean error is at most `6.1e-7`.

This is not a CSE bug in the compiled kernel. The generated Python/Triton text spells the RMS-derived scalar expression twice, once in the amax loop and once in the quantization loop. Triton's optimizer already hoists and merges them: both PTX variants contain exactly one `rsqrt.approx.ftz.f32`, and both cubins contain exactly one `MUFU.RSQ`. Manually hoisting the expression was bit exact and did not improve CUDAGraph time at an identical `XBLOCK=1`, `R0_BLOCK=512`, eight-warp configuration (29.64 us unhoisted versus 30.06 us hoisted in the isolation run).

The larger issue is that Inductor uses one reduction block size for all three passes. Its coordinate-descent choice is `512/512/512`, while Helion independently tunes the passes to `4096/4096/2048`. A hand-generated Inductor variant using the latter sizes reduced isolated time from 29.96 to 25.21 us; an interleaved paired run measured 29.13 us for the original, 24.98 us for the mixed-block prototype, and 23.73 us for Helion. A single shared `R0_BLOCK=2048` reached 27.70 us, while changing the mixed variant from one to eight stages had no material effect.

Nsight Compute supports this diagnosis. The original, mixed-block prototype, and Helion execute 19.66M, 13.99M, and 15.93M instructions respectively. Their global-load request counts are 2.10M, 0.39M, and 0.52M, reflecting the vectorization unlocked by the larger pass-specific tiles. The remaining normal-timing difference is not explained by arithmetic CSE: the mixed prototype uses 42 registers/thread versus Helion's 34 and exhibits different cache/code scheduling. General pass-specific reduction tiling is therefore the actionable Inductor opportunity; a source-level scalar-hoisting patch is not justified.

## Correctness

- Static FP8: all small BF16 cases are bit exact. Non-exact large cases have mean absolute differences below `5.8e-8`. Fused-add residual outputs are bit exact for all tested shapes.
- Block FP8: residual outputs are bit exact; the maximum logical-scale mean absolute difference is `4.9e-10`; the maximum cross-implementation dequantized mean absolute difference is `5.8e-5`.
- FP4: payload mismatch fraction is at most 1.70%. Against the independent RMSNorm reference, Inductor and FlashInfer have nearly identical expected FP4 quantization error; maximum mean absolute errors are 0.1057 and 0.1059, respectively.
- MXFP8 versus TE/torchao: scale bytes are exact for every case. Payload mismatch fraction is at most `7.2e-6`, and cross-dequantized mean absolute difference is at most `5.6e-7`.
- vLLM dynamic FP8: cross-dequantized mean absolute difference is at most `6.1e-7`. Both implementations have approximately 0.014 mean absolute error versus the unquantized RMSNorm reference. Fused-add residuals are bit exact.

The raw files include exact/max/mean differences and p20/median/p80/min/max timing distributions.

## Coverage and exclusions

- SGLang main's NVIDIA static fused path directly calls FlashInfer's `rmsnorm_quant` / `fused_add_rmsnorm_quant`, so timing it separately would duplicate the FlashInfer rows. Its ROCm paths delegate to AITER.
- vLLM main is represented by its current Helion 1.4 dynamic-per-token fused kernel. No stale installed vLLM extension was used.
- torchao main is represented by its Triton MXFP8 quantizer composed with ATen RMSNorm. Its direct Triton API is row-major only.
- Transformer Engine was built from current main against the pinned PyTorch main build and cuDNN Frontend 1.27.0; no older installed binary was used.
- AITER MXFP6 is gfx950-specific. `bench_rocm_aiter_mxfp6.py` compares Inductor main compiling AITER's torch reference against AITER main's HIP implementation and uses the same 100-call graph protocol, but it intentionally refuses to run on this CUDA host. ROCm numbers therefore remain unmeasured.

## Validation and artifacts

PyTorch main tests:

```text
python test/inductor/test_nested_reduction.py -k mxfp8_scale_swizzle
# 2 passed

python test/inductor/test_nested_reduction.py -k mxfp
# 38 passed, 2 skipped

python test/inductor/test_nested_reduction.py -k nvfp4
# 18 passed
```

Primary artifacts:

- `bench_main_cuda.py`: FlashInfer static FP8, block FP8, NVFP4, MXFP4, and MXFP8; PDL is off by default.
- `bench_te_main.py`: Transformer Engine main MXFP8.
- `bench_torchao_main.py`: torchao main composed MXFP8.
- `bench_vllm_helion_main.py`: vLLM main Helion dynamic per-token FP8.
- `bench_rocm_aiter_mxfp6.py`: ROCm/AITER main MXFP6 runner.
- `results/main_default_nopdl.json`: fresh no-PDL static and block-FP8 data.
- `results/main_coordesc_nopdl.json`: full coordinate-descent sweep.
- `bench_padding_fresh_mode.py`, `results/padding_fresh_idle_consolidated.json`: isolated-process padded 128x4 main sweep and launch/config metadata.
- `PADDING_KERNEL_ANATOMY_AGENT.md`, `traces_padded_anatomy_agent/`: padded-layout IR and generated-kernel anatomy.
- `bench_padded_quant_main.py`, `results/padded_quant_correctness_logical_cdd22ad.json`: logical-scale correctness checks that exclude unspecified padding slots.
- `results/mxfp8_coordesc_repeat.json`: dedicated repeated coordinate-descent MXFP8 sweep.
- `results/te_default.json`, `results/te_coordesc.json`: Transformer Engine comparisons.
- `results/torchao_default.json`: torchao comparison.
- `results/vllm_helion_default.json`, `results/vllm_helion_coordesc.json`: vLLM comparisons.
- `bench_vllm_cse_isolation.py`, `bench_vllm_cse_paired.py`: forced-config and interleaved CUDAGraph follow-up for the large vLLM gap.
- `results/vllm_cse_fp_fusion.json`, `results/vllm_cse_paired.json`: raw CSE, block-size, stage-count, and paired timing results.
- `results/ncu_vllm_original.csv`, `results/ncu_vllm_mixed.csv`, `results/ncu_vllm_helion.csv`: Nsight Compute evidence for the follow-up.
- `generated_nonpadded_block_fp8_default.py`, `generated_nonpadded_block_fp8_coordesc.py`: generated kernels for the clean non-padded case.
- `generated_vllm_gap_default.py`, `generated_vllm_gap_coordesc.py`: generated Inductor kernels for the large dynamic fused-add gap.
- `results/ncu_inductor_nonpadded_default.csv`: Nsight Compute profile for the default non-padded kernel.

No source file outside `agent_space/` was modified, and no result from the older PyTorch checkout or an unlanded diff is included.
