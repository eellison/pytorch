# Padded 128x4 scale swizzle vs FlashInfer -- 2026-09-02

All numbers: `graph_bench` (100 calls captured in one CUDAGraph, replayed,
event-timed), GPU 6, worktree `agent_space/padded_perf_wt` (nested-reduction
stack + both index_put-stack commits), coordinate descent off.

## 0. Methodology correction (invalidates earlier v2-v5 matrices)

Benching all variants in one process inflates every variant except the last one
in the dict by 20-70%. Forward+reverse sweep with min does not fix it. All
numbers below are **one variant per process** (`results/iso_*.json`).

Evidence: v4 (zeros_scatter last) and v5 (pad_scatter last) each showed the last
variant ~1.5-2x ahead; in isolation all variants land within 2%. For the last
variant, interleaved == isolated to ~1%.

## 1. Fused rmsnorm + fp4 quant + padded swizzle (median us, kernel count)

| suite | shape | fpad | index_put | predicated | zeros_tail | zeros_scatter | pad_scatter | FI | best/FI |
|---|---|---|---|---|---|---|---|---|---|
| mxfp4 | [19,4096]  | 4.77(3) | 3.94(2) | 3.63(2) | 3.56(2) | 3.55(2) | 3.67(2) | 2.77 | 1.28x |
| nvfp4 | [19,4096]  | 4.94(3) | 4.14(2) | 3.84(2) | 3.77(2) | 3.77(2) | 3.88(2) | 2.84 | 1.33x |
| mxfp4 | [129,4096] | 4.94(3) | 4.10(2) | 3.77(2) | 3.72(2) | 3.72(2) | 3.79(2) | 2.87 | 1.29x |
| nvfp4 | [129,4096] | 5.10(3) | 4.28(2) | 3.95(2) | crash   | 3.91(2) | 4.04(2) | 2.95 | 1.33x |
| mxfp4 | [989,4096] | 7.45(3) | 6.56(2) | 6.25(2) | 6.20(2) | 6.20(2) | 6.29(2) | 5.10 | 1.22x |
| nvfp4 | [989,4096] | 8.21(3) | 7.44(2) | 7.19(2) | crash   | 6.93(2) | 7.05(2) | 4.41 | 1.57x |

All variants byte-identical to `fpad` including pad lanes.

Conclusions:
- Kernel count is the only thing that matters: 3 -> 2 is worth 1.18-1.34x.
- `predicated` / `zeros_scatter` / `pad_scatter` are within 2% of each other.
- `pad_scatter` (zero only the pad rectangles, tiny grid) is NOT faster than
  the full-domain predicated store. The grid was never the cost.
- `zeros_scatter` (plain `torch.zeros` + `_unsafe_index_put`) reaches the same
  2-kernel result with no compiler change.

## 2. Where the residual FlashInfer gap is (profiler, device us)

| case | rmsnorm+quant+scatter | pad kernel | total | FI (1 kernel) |
|---|---|---|---|---|
| nvfp4 [129,4096] | 3.45 | 1.37 | 4.82 | 3.33 |
| mxfp4 [129,4096] | 3.33 | 1.36 | 4.69 | 3.31 |
| mxfp4 [989,4096] | 5.92 | 1.39 | 7.32 | 5.51 |
| nvfp4 [989,4096] | 6.75 | 1.46 | 8.21 | 4.85 |

- The main kernel is already at parity with FlashInfer on 3 of 4 cases
  (+0.6%, +3.6%, +7.4%). nvfp4 [989,4096] is the exception: +39%, a separate
  kernel-quality problem.
- The pad kernel is a flat ~1.4us regardless of shape. It moves ~64KB; at
  3 TB/s that is 0.02us. It is pure launch cost.
- **Any 2-kernel formulation is stuck ~1.4us behind FlashInfer.** Reaching
  parity requires collapsing to one kernel, not making kernel 2 cheaper.

## 3. Swizzle-only scaling (no rmsnorm), median us

| shape | MB | pad% | fpad | index_put | predicated | zeros_scatter | pad_scatter |
|---|---|---|---|---|---|---|---|
| 989x256    | 0.26  | 3.4 |  1.44 |   3.26 |   2.99 |  2.94 |  2.91 |
| 8000x250   | 2.03  | 1.6 |  2.89 |   8.19 |   7.69 |  7.12 |  8.06 |
| 32700x512  | 16.78 | 0.2 | 17.06 |  42.30 |  42.18 | 38.50 | 34.33 |
| 100000x512 | 51.25 | 0.1 | 50.23 | 115.18 | 114.19 | 114.76 | 88.97 |
| 32768x512  | 16.78 | 0.0 | 17.22 |  47.89 |  38.87 | 29.99 | 29.99 |

Standalone, `fpad` wins everywhere: it is one kernel with a masked gather load
and a **coalesced** store (~2 TB/s at 51MB), whereas every scatter formulation
has a **scattered** store (~0.9 TB/s). The scatter variants only win in the
fused setting, where they buy a kernel by riding the rmsnorm epilogue.

## 4. Bug: `tl.full` with an fp8 dtype fails Triton compilation

A masked fill of a `float8_e4m3fn` tensor emits

    tmp4 = tl.full([1], 0.0, tl.float8e4nv)
    tmp7 = tl.full([1], float("nan"), tl.float8e4nv)

Triton lowers this to `fptrunc float -> fp8e4nv`; fp8e4nv is `i8` in LLVM, so
the verifier rejects it:

    'llvm.fptrunc' op result #0 must be floating point LLVM type ..., but got 'i8'
    RuntimeError: PassManager::run failed   (triton_poi_fused_fill_0)

Fix direction: Inductor knows the constant at compile time, so it should
compute the fp8 bit pattern in Python and emit an integer constant plus a
bitcast, never a float->fp8 truncation.

Repro: `zeros_tail` variant, nvfp4 [129,4096] and [989,4096]. Plain
`torch.zeros` on fp8 does NOT hit it -- it needs the masked fill. The
predicated store avoids it structurally (no `where(mask, val, nan)` else-arm).

## Files

- `blocked_variants.py` -- all seven padding formulations, layout-identical
- `bench_padded_variants.py` -- fused rmsnorm matrix (use `--variants` one at a time)
- `bench_swizzle_scaling.py` -- swizzle-only size sweep
- `profile_vs_fi.py` -- per-kernel breakdown vs FlashInfer
- `results/iso_*.json`, `results/scal_*.json`

## 5. Reconciliation with `agent_space/padded_quant_review_20260902.md`

That report's per-variant deltas came from the interleaved harness described in
section 0. `pad_scatter` is the last Inductor variant in every one of its raw
result files, i.e. the one position that reads accurately; `predicated` sits
mid-list and was inflated. That is the whole reason the two looked different.

Isolated re-measurement on that report's own tree
(`agent_space/index_put_nested_integration`, PR #191974 head) and its own
config:

| suite | shape | fpad | predicated | pad_scatter | FI |
|---|---|---|---|---|---|
| mxfp4 | [19,4096]  | 3.96 (1.44x) | 3.07 (1.11x) | 3.16 (1.14x) | 2.76 |
| nvfp4 | [19,4096]  | 3.93 (1.38x) | 2.90 (1.02x) | 2.94 (1.03x) | 2.84 |
| mxfp4 | [99,4096]  | 4.70 (1.63x) | 2.89 (1.00x) | 2.78 (0.97x) | 2.88 |
| nvfp4 | [99,4096]  | 3.89 (1.33x) | 2.89 (0.99x) | 2.98 (1.02x) | 2.93 |
| mxfp4 | [129,4096] | 4.41 (1.53x) | 2.87 (1.00x) | 2.88 (1.00x) | 2.87 |
| nvfp4 | [129,4096] | 4.05 (1.37x) | 3.00 (1.02x) | 3.09 (1.05x) | 2.95 |

Corrections to that write-up (its correctness, testing, opcheck and hardening
sections are unaffected):

- "FP4 is still 8-55% behind FlashInfer" -- wrong. With coordinate descent the
  generic predicated store is at parity, 0.99-1.11x. (`1x4096` not rechecked.)
- "Tuned row-only pad_scatter brings common FP4 shapes within +/-3% of
  FlashInfer" -- true of pad_scatter, but not special to it. Isolated,
  `predicated` and `pad_scatter` are indistinguishable (max delta 3%, sign
  varies) under both tuning modes. The recommended caller-side rectangular
  fast path buys nothing and should be dropped.
- "Generic predicated padding is 9-27% faster than F.pad" -- holds, and is if
  anything understated: isolated it is 26-63% faster at these shapes.

Coordinate descent, not the tree, is what closes the gap. Same tree, same
variant, isolated both ways:

| config | predicated vs FI | pad_scatter vs FI |
|---|---|---|
| coordinate descent | 0.99-1.11x | 0.97-1.14x |
| default tuning     | 1.27-1.32x | 1.26-1.34x |

## 6. Big shapes: fused rmsnorm at 16k x 8k (isolated, default tuning)

| suite | shape | fpad | index_put | predicated | zeros_scatter | pad_scatter | FI |
|---|---|---|---|---|---|---|---|
| mxfp4 | [16000,8192] | 133.40(1) | 136.84(2) | 132.44(2) | 138.48(2) | 134.85(1) | 113.05 |
| nvfp4 | [16000,8192] | 164.61(1) | 173.03(2) | 170.61(2) | 169.11(2) | 163.81(1) | 107.33 |
| mxfp4 | [16384,8192] | 125.98(1) | 128.16(2) | 126.19(2) | 129.09(2) | 123.87(1) | 120.29 |
| nvfp4 | [16384,8192] | 166.15(1) | 170.81(2) | 171.24(2) | 171.27(2) | 163.40(1) | 108.97 |

The ranking inverts: `fpad` collapses to one kernel (the coalesced gather fuses
into the rmsnorm epilogue) and is within 1% of the best, while `zeros_scatter`
becomes the worst as the double-write penalty finally costs bandwidth. Total
spread across all five strategies is ~4%. Padding choice is a small-shape
phenomenon.

The remaining big-shape gap is in the main quant kernel and is nvfp4-specific:
mxfp4 runs at 2.7 TB/s vs FlashInfer's 2.8 (1.03-1.17x), but nvfp4 is at
2.1 TB/s vs 3.15 (1.50-1.53x).

## 7. NVFP4 large-shape kernel anatomy

The packed output is not split into two stores. The generated kernel combines
each pair with one `cvt.rn.satfinite.e2m1x2.f32`, converts the result to one
`uint8`, and emits one contiguous `tl.store`. Its other store writes the
separate scale output.

The generated NVFP4 kernel instead duplicates the scale path across iteration
domains. It converts and stores the FP8 scale once in the reduced domain, then
replays the conversion and reciprocal in the full pair domain before packing.
MXFP4 computes the inverse once in the reduced domain and broadcasts it.

A direct generated-kernel A/B at `[16000,8192]` replaced only the duplicated
NVFP4 replay with reduced-domain reciprocal plus broadcast. Outputs were
byte-identical:

| kernel | median us |
|---|---:|
| generated baseline | 154.68 |
| reuse reduced scale | 129.02 |

This is a 16.6% kernel-time reduction and explains a substantial part, but not
all, of the roughly 1.5x FlashInfer gap. The remaining comparison is about
1.17-1.23x depending on the FlashInfer reference run. The scratch A/B is
`bench_nvfp4_scale_reuse.py`.

## 8. Open items

- fp8 `tl.full` bitcast bug (section 4) -- standalone Inductor fix.
- nvfp4 large-shape kernel quality: 1.5x off FlashInfer at 16k x 8k.
- The single-kernel probe assertion was a stale-stack artifact, not a
  current-main crash. It reproduces on the pre-#191974 nested feature stack,
  including its first commit, while replaying a `where` whose operands arrive
  in parent-full (`R0_BLOCK`) and grouped-reduced
  (`nested_R0_REDUCED_BLOCK`) domains. Current `origin/main` at
  `fd23f0c97b4`, including the landed #191974 correctness guard, declines that
  unsafe fusion and compiles the probe with one nested reduction and two
  kernels. A true one-kernel implementation remains future mixed-domain work,
  but there is no open correctness crash on main.
