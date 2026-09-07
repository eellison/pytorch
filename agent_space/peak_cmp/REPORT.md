# Peak vs peak: torch.compile (nested reduction) vs QuACK / flashinfer kernels

2026-09-05, B200, bf16 inputs, median us of 50 samples, each a CUDA graph of
100 calls (Inductor cudagraphs off, flashinfer PDL off). One cell per process.
Harness: `peak_cell.py`, `peak_run.py`, `peak_analyze.py`, `results.jsonl`
(164 cells, 0 errors).

"Ours" is the optimal formulation (inline-asm `cvt.rn.satfinite.e2m1x2`,
`cvt_e8m0_rceil`, swizzled 128x4 scales written as a pure permute) compiled
with `triton.nested_reduction=True`; modes default / coordinate descent (cd) /
forced persistent / persistent+cd / multi-kernel (mk); "best" is the fastest.
"Theirs" is the fastest available kernel: flashinfer's fused CuTe-DSL
`rmsnorm_fp4quant` (NVFP4/MXFP4), flashinfer's TRT-LLM CUDA `fp4_quantize` /
`mxfp4_quantize` / `mxfp8_quantize`, QuACK's CuTe `rmsnorm_fwd`, and QuACK's
`gemm(out_dtype=...)` with the fused SFD quantized-output epilogue. QuACK has no
standalone or RMSNorm-fused CuTe quantizer (its `blockscaled.quantize` is
torch.compile of a torchao port; the fused-RMSNorm-quant path named in its
docstring does not exist in 0.6.4 or upstream main as of 2026-08-30).

| workload | shape | ours best (mode) | theirs (which) | ours/theirs |
|---|---|---|---|---|
| rmsnorm | 1024x4096 | 3.01 (cd) | 3.49 quack / 3.86 fi | 0.86 |
| rmsnorm | 8192x4096 | 22.56 (default) | 25.37 quack / 24.13 fi | 0.93 |
| rmsnorm | 65536x2048 | 100.0 (cd) | 95.7 quack / 99.8 fi | 1.05 |
| rmsnorm+nvfp4 | 1024x4096 | 4.59 (cd) | 4.44 fi fused | 1.03 |
| rmsnorm+nvfp4 | 2048x3072 | 5.43 (cd) | 5.73 fi fused | 0.95 |
| rmsnorm+nvfp4 | 8192x4096 | 24.64 (cd) | 21.22 fi fused | 1.16 |
| rmsnorm+nvfp4 | 65536x2048 | 111.4 (persistent_cd) | 108.0 fi fused | 1.03 |
| rmsnorm+mxfp4 | 1024x4096 | 4.28 (persistent_cd) | 5.07 fi fused | 0.84 |
| rmsnorm+mxfp4 | 8192x4096 | 23.64 (cd) | 27.12 fi fused | 0.87 |
| rmsnorm+mxfp4 | 65536x2048 | 101.7 (cd) | 126.7 fi fused | 0.80 |
| rmsnorm+mxfp8 | 1024x4096 | 4.18 (cd) | 6.98 quack+fi composed | 0.60 |
| rmsnorm+mxfp8 | 8192x4096 | 24.57 (persistent_cd) | 48.65 quack+fi composed | 0.51 |
| rmsnorm+mxfp8 | 65536x2048 | 100.8 (persistent_cd) | 193.1 quack+fi composed | 0.53 |
| nvfp4 quant | 1024x4096 | 3.41 (default) | 5.88 fi | 0.58 |
| nvfp4 quant | 8192x4096 | 18.57 (cd) | 13.79 fi | 1.35 |
| nvfp4 quant | 65536x2048 | 82.3 (cd) | 67.9 fi | 1.21 |
| mxfp4 quant | 8192x4096 | 18.15 (cd) | 20.43 fi | 0.89 |
| mxfp8 quant | 8192x4096 | 16.29 (default) | 17.88 fi | 0.91 |
| gemm+nvfp4 out | 2048x3072x4096 | 51.86 (default; cuBLAS mm + 1 kernel) | 47.47 quack fused (bf16 mm alone 45.2) | 1.09-1.15 |
| gemm+nvfp4 out | 8192x4096x4096 | 276.0 (cd) | 249.0 quack fused (bf16 mm alone 263.2) | 1.11 |
| gemm+mxfp8 out | 8192x4096x4096 | 286.8 (default) | 280.1 quack fused | 1.02-1.13 |

Full table: `python peak_analyze.py`.

Correctness: standalone quantizers are byte-identical to flashinfer's kernels
(0.0000% payload and scale mismatch, all formats and shapes). Fused RMSNorm
variants differ from flashinfer by 1.1-3.4% of payload bytes with identical
dequantized error to four digits (RMSNorm rounding, not quantization). QuACK's
CuTe RMSNorm is bit-exact with `F.rms_norm`. GEMM quant-out dequant errors
match QuACK's where both were measured (4.573 vs 4.571 for nvfp4).

## Where we stand

- MXFP8 and MXFP4: we lead or tie everywhere, including against flashinfer's
  fused CuTe RMSNorm->MXFP4 kernel (13-20% faster).
- NVFP4: tie at small/medium shapes, behind at 8192x4096 (16% fused, 35%
  standalone) and 65536x2048 (3% fused, 21% standalone). Since the 2026-08-31
  report the fused 1024x4096 gap shrank from 11% to 3%.
- RMSNorm alone: within +-7% of QuACK's and flashinfer's CuTe/CUDA kernels.
- GEMM with quantized output: 9-15% behind QuACK's fused SFD epilogue.

## Fusion misses and inefficiencies recorded

1. **GEMM + block-scaled quantized output.** Inductor runs cuBLAS mm then a
   separate quant kernel; the block-amax epilogue is a reduction and cannot
   fuse into a mm template (extern or Triton). QuACK's fused nvfp4 output GEMM
   is as fast as or faster than a plain bf16 mm because it never writes bf16.
2. **FP4 quant kernel efficiency.** Our NVFP4/MXFP4 kernels reach ~4.5 TB/s at
   8192x4096 while our MXFP8 kernel reaches 6.0 TB/s and flashinfer's NVFP4
   kernel 5.9 TB/s. The generated kernel loads x once (no reload), so it is
   not a memory-traffic issue; the FP4 path differs by `tl.reshape` +
   `tl.split` over the pair axis and masked r0 loads (rnumel 16). Layout
   conversion from the split is the leading suspect; needs ncu.
3. **Reduction-form and tuning choice.** Persistent vs looped is
   shape/format dependent (persistent wins MXFP8 at K=4096, looped+cd wins
   NVFP4/MXFP4 at K=4096). Multi-kernel's pick is not always <= the best
   single form (nvfp4 8192x4096: mk 30.8 vs looped 27.2 / persistent 28.8),
   and coordinate descent on whichever form is the larger lever: default vs
   best is 1.4x at 65536x2048 (157.7 vs 111.4 nvfp4; 177.8 vs 100.8 mxfp8).
4. Not a fusion issue but a gap: standalone NVFP4 at large shapes is where
   flashinfer's TRT-LLM kernel is clearly ahead; every other standalone
   format we win.

## Update 2026-09-05: lane-fold codegen fix

Root cause of the FP4 inefficiency (finding 2 above), from ncu and TTGIR: the
sub-parent lane replay recomputed `x*rstd*w` per lane from split raw loads,
and the per-group scale broadcast landed in a different Triton layout than
the split data, forcing a full-tile shared-memory `convert_layout`. Fixed in
`_SubParentValueResolver` (fold lane ops onto the parent value; route
group-width broadcasts through the parent tile before splitting). Results
in `results_fold_nvfp4.jsonl` / `results_fold_mxfp4.jsonl`:

| RMSNorm -> | shape | ours before | ours after | fi fused | after/fi |
|---|---|---|---|---|---|
| NVFP4 | 1024x4096 | 4.59 | 3.77 | 4.43 | 0.85 |
| NVFP4 | 2048x3072 | 5.43 | 4.87 | 5.75 | 0.85 |
| NVFP4 | 8192x4096 | 24.6 | 18.8 | 21.2 | 0.89 |
| NVFP4 | 65536x2048 | 111 | 81.4 | 109 | 0.75 |
| MXFP4 | 1024x4096 | 4.28 | 3.32 | 5.07 | 0.65 |
| MXFP4 | 2048x3072 | 5.79 | 4.79 | 6.96 | 0.69 |
| MXFP4 | 8192x4096 | 23.6 | 17.7 | 27.1 | 0.65 |
| MXFP4 | 65536x2048 | 102 | 79.4 | 127 | 0.62 |

Outputs are byte-identical to the pre-change kernels. DCN MXFP6 (4,3) is
unchanged (9.60 -> 9.63 us, identical outputs). Standalone quant unchanged.

## Update 2026-09-05: MXFP8

The lane fold does not apply to MXFP8 (no pair split). ncu at 8192x4096 showed
the fused kernel issue-bound at 25.5 instructions per element (standalone
quant: 10.8, flashinfer NVFP4: 12.3). Two levers:

1. **Formulation.** Our MXFP8 formulation divided every element by the
   power-of-two scale (a full fp32 division); multiplying by
   `recip_ue8m0(scale)` is bit-identical and drops the fused kernel to 16.6
   instructions per element. `peak_cell.py` now uses it. Results in
   `results_fold_mxfp8.jsonl`:

   | | 1024x4096 | 2048x3072 | 8192x4096 | 65536x2048 |
   |---|---|---|---|---|
   | RMSNorm->MXFP8 before -> after (best) | 4.18 -> 3.40 | 5.67 -> 4.80 | 24.6 -> 21.2 | 101 -> 94.0 |
   | vs QuACK RMSNorm + fi mxfp8_quantize | 0.48x | 0.49x | 0.44x | 0.49x |
   | standalone MXFP8 before -> after | 3.19 -> 2.98 | 4.21 -> 3.55 | 16.3 -> 15.5 | 77.8 -> 77.1 |
   | standalone vs fi mxfp8_quantize | 0.85x | 0.74x | 0.84x | 0.84x |

   The same division is in QuACK/torchao's `to_mx`.
2. **fp32 register residency (not fixed).** Loads are upcast to fp32 at the
   load, so a persistent row costs twice the registers it needs (96 at 2
   warps, 60 at 4 for K=4096), which forces 4-8 warps per row and cross-warp
   reductions; flashinfer holds 32 registers with the bf16 row in smem.
   `triton.codegen_upcast_to_fp32=False` is not the fix: it is slower and
   changes numerics because intermediate arithmetic becomes bf16. The fix is
   codegen keeping the bf16 load live and upcasting per use. A config sweep
   of the generated kernel confirmed coordinate descent's pick is within 1%
   of the best config, so config is not the lever.

### MXFP8 instruction budget (hand kernel, 2 warps, 8192x4096; ncu opcode histogram)

11.6 warp-instructions per element: FMNMX 2.44, IMAD 2.20, FMUL+FMUL2 2.46,
PRMT 1.03, F2FP 0.56, LOP3 0.33, loads/stores 0.33, rest < 0.3 each. The
standalone quant is 8.0. Two observations:

- About two of the FMNMX per element are the `clamp(-448, 448)` before the
  e4m3 cast. The cast is `cvt.rn.satfinite`, so the clamp is redundant, and
  Inductor's clamp lowering propagates NaN, so removing it is bit-identical
  for finite and special inputs (verified). Compiled effect is modest and
  config-dependent: fused persistent+cd 21.7 -> 20.8us, persistent default
  23.6 -> 21.5us, but the standalone default-config kernel got slower
  (15.4 -> 19.7us, autotune picked a different config), so the bench
  formulation keeps the clamp for now.
- IMAD + PRMT (~3.2 per element, vs ~1.8 standalone) are register-level data
  movement from reshapes, broadcasts and bf16/fp8 packing, i.e. Triton
  layout plumbing rather than arithmetic. This, the cross-warp reductions
  and the per-row weight reload are what separates 21us from the ~15us
  roofline; none of it is reachable from the Inductor side without a
  different tiling model.
- bf16 residency was tested by hand (non-pure asm upcasts to defeat CSE):
  registers are not the limiter (the fastest config has the most), and
  recomputing y per pass is slower. Not pursued.

## Update 2026-09-06: Gluon layout-control prototype (row-wise MXFP8)

Question: is Triton's inferred layout what keeps the fused RMSNorm->MXFP8
kernel at ~21us vs a ~15us roofline? Prototype in
`agent_space/gluon_proto/rms_mxfp8_gluon.py` (Triton 3.8 Gluon), 8192x4096,
all variants correct (scales byte-identical, <10 payload bytes from rsqrt
rounding):

| layout | graph us | ncu us | instr/elem | L1 ld sectors | smem wavefronts |
|---|---|---|---|---|---|
| Triton generated, cd (8 warps) | 21.2 | 31.5 | 16.6 | 5.2M | 3.8M |
| Triton hand, 2 warps | 19.5-21.0 | - | 11.6 | - | - |
| Gluon thread-owns-group (32/thread, 4 warps) | 24.3 | 25.9 | 11.6 | 8.4M | 0.34M |
| Gluon coalesced load + convert_layout | 25.5 | 28.6 | 13.8 | 4.2M | 2.7M |
| Gluon 16/thread, 8 warps | 22.5 | - | - | - | - |

Thread-owned groups remove the shuffles and shared-memory reductions but
double L1 load sectors (each warp load touches 32 scattered 16B pieces);
staging through `convert_layout` restores coalescing but adds the smem round
trip and 2.4M instructions. No explicit layout beats the best Triton config.
Conclusion: for the row-wise (R-grouped) kernels layout control is NOT the
lever; they are at the practical floor for this algorithm on Triton/Gluon
without deeper restructuring (packed math, TMA + smem pipelining). Layout
control matters for the column-group (band/dual) geometry, where the 32-row
max crosses warps in Triton's row-major layout; that is the handwritten dual
kernel's advantage and remains a multi-week Gluon-emitter project.
