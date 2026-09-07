# RMSNorm -> MXFP8 vs Transformer Engine

> Superseded by `agent_space/rmsnorm_mxfp8_native_nan_writeup_20260716.md`,
> which includes the native NaN min/max fix and final NCU results.

## Correction

Transformer Engine has two relevant MXFP8 output paths in 2.17:

1. Fused cuDNN RMSNorm -> MXFP8 with compact E8M0 scales: one kernel.
2. GEMM-optimized swizzled E8M0 scales: an unfused RMSNorm kernel followed by
   an MXFP8 quantization/swizzle kernel.

The fused path requires `NVTE_NORM_FWD_USE_CUDNN=1`. TE explicitly rejects a
preallocated GEMM-swizzled output on that path with `MXFP8 output must have
scales in compact format, not swizzled for GEMM`.

The installed TE wheel was compiled with cuDNN 9.24 while the PyTorch
environment contained cuDNN 9.10.2. The fused graph uses a reshape attribute
that the older runtime rejects. The fused measurements below use cuDNN
9.24.0.43 loaded from `agent_space/cudnn_924` without changing the environment.

## Methodology

- GPU: NVIDIA B200, SM100
- CUDA runtime / driver: 12.8 / 580.82.07
- PyTorch source HEAD: `e552f86629fdb358e2e6ed784a30fc4b3abae3da`
- PyTorch tracked-worktree diff SHA256: `5cb5260833d08224d1c90588acb883d4148ea9f3abf2a165e6d43d68c860c426`
- PyTorch binary build commit: `e1067a5ad33609f9486adefbbd627e18281844c2`
- Transformer Engine: 2.17.0, tag commit `2e559f062497bef768dfbe9d7e45548fadeca80a`
- Timing: CUDA graph replay via `triton.testing.do_bench`
- Warmup / rep: 25 ms / 100 ms, three measurements per compiled graph
- Clock preconditioning: untimed `torch.cuda._sleep(200_000_000)` before timing
- Timed outputs: raw E4M3 payload and E8M0 scales; no dequantization or `q.float()`
- Correctness: dequantized outside timing against FP32-accumulating RMSNorm,
  `atol=0.05`, `rtol=0.05`; every case passed with zero mismatches
- Input: BF16, epsilon `1e-5`, MXFP8 block size 32
- Compile mode: `max-autotune-no-cudagraphs`

## Fused Compact-Scale Comparison

This is the correct one-kernel TE API comparison. Both implementations return
compact, unswizzled E8M0 scales.

| Shape | TE fused compact | Inductor nested + asm compact | TE / Inductor | Kernels |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 8.06 us | 8.06 us | 1.00x | 1 / 1 |
| `1024x4096` | 12.19 us | 12.16 us | 1.00x | 1 / 1 |
| `256x8192` | 10.14 us | 10.11 us | 1.00x | 1 / 1 |
| `4096x8192` | 32.77 us | 40.83 us | 0.80x | 1 / 1 |

The three requested shapes are effectively tied. At the production-size
`4096x8192` row, TE is 1.25x faster. Across all four rows, TE is 1.06x faster
by geomean.

Compact-layout Inductor ablations:

| Shape | Nested + asm | Nested, no asm | No nested + asm | No nested, no asm |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 8.06 us (1) | 12.16 us (1) | 10.11 us (2) | 10.14 us (2) |
| `1024x4096` | 12.16 us (1) | 12.16 us (1) | 14.21 us (2) | 14.24 us (2) |
| `256x8192` | 10.11 us (1) | 10.11 us (1) | 12.16 us (2) | 12.16 us (2) |
| `4096x8192` | 40.83 us (1) | 40.83 us (1) | 49.09 us (2) | 49.06 us (2) |

The `128x4096` non-inline result is compile-choice sensitive: a separate smoke
compile measured 8.16 us. The other compact rows show no measurable inline-asm
benefit once the whole expression remains in one kernel. Nested reduction still
removes one launch and improves the larger rows by roughly 17-20%.

## GEMM-Swizzled Comparison

For GEMM-ready swizzled scales, TE 2.17 deliberately uses two kernels while
Inductor fuses the scale swizzle into its one nested kernel.

| Shape | TE swizzled | Inductor nested + asm swizzled | Speedup | Kernels |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 10.27 us | 8.16 us | 1.26x | 2 / 1 |
| `1024x4096` | 16.35 us | 12.26 us | 1.33x | 2 / 1 |
| `256x8192` | 14.30 us | 10.21 us | 1.40x | 2 / 1 |
| `4096x8192` | 53.22 us | 42.78 us | 1.24x | 2 / 1 |

Geomean Inductor speedup for GEMM-swizzled output: 1.31x. This is a layout and
fusion advantage, not evidence that TE's compact-output API needs two kernels.

Swizzled-layout Inductor ablations:

| Shape | Nested + asm | Nested, no asm | No nested + asm | No nested, no asm |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 8.16 us (1) | 8.16 us (1) | 10.21 us (2) | 10.21 us (2) |
| `1024x4096` | 12.26 us (1) | 14.30 us (2) | 14.34 us (2) | 16.35 us (3) |
| `256x8192` | 10.21 us (1) | 12.26 us (2) | 12.26 us (2) | 14.27 us (3) |
| `4096x8192` | 42.78 us (1) | 45.02 us (2) | 51.30 us (2) | 53.22 us (3) |

Here inline asm matters at the larger shapes because it avoids realizing the
scale conversion as a separate pointwise kernel. XBLOCK nested reductions are
not involved; all measured groups split the hidden/R axis.

## Tuning

On the swizzled `128x4096` case, coordinate descent alone was effectively flat
(8.10 us, one kernel). Coordinate descent with `triton.multi_kernel=3`
regressed it to 10.21 us and two kernels, so neither is used in the results.

## Artifacts

- Fused compact raw JSON: `agent_space/rmsnorm_mxfp8_te_cudnn_compact_20260716.json`
- GEMM-swizzled raw JSON: `agent_space/rmsnorm_mxfp8_rerun_20260716_clocked.json`
- Benchmark harness: `agent_space/rmsnorm_mxfp8_writeup_bench.py`
- TE 2.17 source checkout: `agent_space/transformer_engine_v2.17`
