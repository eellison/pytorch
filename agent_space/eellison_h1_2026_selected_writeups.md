# eellison H1 2026 - Selected Write-ups

## PyTorch Compiler / Inductor Local GPU Performance

Landed a series of Inductor fusion, reduction, quantization, and wrapper-overhead
optimizations focused on local single-GPU inference/training kernels. The main
theme was removing intermediate GPU kernels and wrapper work from quantized model
paths, especially RMSNorm -> grouped quant and MXFP8-style scale/payload
generation.

## RMSNorm -> MXFP8 / Grouped Quant Fusion

Landed nested-reduction support for RMSNorm -> grouped/block quant patterns,
allowing RMSNorm results to feed grouped reductions and payload conversion
without materializing intermediate kernels. (PRs #182891, #182892, #182893,
#182896, #183432, #182897, #182898)

- RMSNorm -> swizzled MXFP8: one-kernel nested+inline-asm path is 1.33x geomean
  faster than Transformer Engine and 1.37x faster than the prior PyTorch
  nested-off decomposition across four B200 shapes.
- Representative MXFP8 cases: `1024x4096` improves 16.26us -> 12.16us;
  `4096x8192` improves 53.15us -> 40.83us.
- RMSNorm -> grouped FP8 quant, residual RMSNorm -> grouped FP8 quant, and
  hidden-state-view grouped FP8 quant show similar speedups and one-kernel
  fusion versus the prior PyTorch decomposition.
- Benchmark: `agent_space/rmsnorm_mxfp8_writeup_swizzled_after_index_simplify.json`

## Grouped Quant Fusion / Loop Reindexing

- Fused grouped-quant patterns through loop reindexing and better fusion
  candidate selection.
- Representative grouped-quant case: 6.20us -> 4.15us (1.49x), with amax/scale
  and quant payload collapsed from 2 kernels -> 1. (PRs #176927, #179090)

## E8M0 Scale Encoding

- Added support for SM100 PTX conversion through the inline-asm HOP for
  `cvt_e8m0_rceil`, replacing software ceil/log2 encoding with
  `cvt.rp.satfinite.ue8m0x2.f32`.
- Encode-only benchmarks improve 4.0us -> 2.2us (`1024x128`) and
  8.2us -> 4.1us (`8192x512`); full bf16 MXFP8 scale generation improves
  8.2us -> 6.1us (`2048x4096`).
- This is required for the RMSNorm -> MXFP8 quant-fusion speedups above.
  (PR #172497)

## Pointwise Cat / RoPE-Style Fusion

- Fused pointwise-cat RoPE/QKNorm patterns so split branches that recombine the
  same data stay in one fused kernel.
- Representative QKNorm + split RoPE cat case improves 0.0388ms -> 0.0232ms
  (1.67x) in direct compiled-call timing, with 2 kernels -> 1. (PR #179091)

## Deferred Wrapper Work

Deferred generated-wrapper size/stride assertions and `copy_if_misaligned` work
until the first real use of an input, reducing Python-wrapper overhead before the
first GPU kernel. In local wrapper microbenchmarks, pre-first-mm size/stride
assertions drop from 6 -> 2 and time-to-first-mm improves 129.01us -> 69.16us;
misaligned-copy work similarly drops from 6 -> 2 pre-first-mm copies and
improves 218.18us -> 36.41us to first mm. The reference host-overhead result
reported 6-10% improvement across HuggingFace models, with smaller timm and
TorchBench gains. (PRs #177783, #179039, #180599; reference #178489)

## Supporting Artifacts

- Full local impact doc: `agent_space/eellison_h1_2026_local_impact_final.md`
- Representative kernel table: `agent_space/eellison_h1_2026_representative_kernel_table.md`
- RMSNorm MXFP8 focused summary: `agent_space/eellison_h1_2026_rmsnorm_mxfp8_te_swizzled.md`
