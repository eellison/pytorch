# Final stack performance audit: NVFP4 and MXFP4 RMSNorm

Audit target: `02da48d4f73205966cfb8b44dcf10c3fa9c4198d` on 2026-08-05.

## Bottom line

- The user-important paths do fuse at the final SHA: nested-on is one generated
  kernel with `codegen_nested_reduction == 1`; nested-off is three kernels with
  the metric at zero.
- With coordinate-descent tuning enabled, final-shape NVFP4 is effectively tied
  with FlashInfer row-major at `4096x8192` (34.752 vs 34.528 us, +0.65%). MXFP4
  is 16.6% faster (30.624 vs 36.736 us).
- The previously observed roughly 6% NVFP4 loss at `4096x8192` is not stable.
  The old short run had FlashInfer at 32.704 us, while a long rerun had it at
  34.688 us; Inductor stayed near 34.7 us. The fresh final-SHA run is again a
  tie. Treat the 6% result as clock/measurement-regime variance, not a robust
  regression.
- Coordinate descent is load-bearing for the large cases. Without it, NVFP4 is
  45.024 us and MXFP4 36.768 us at `4096x8192`; with it they are 34.752 and
  30.624 us respectively.
- There is one measured kernel-choice opportunity: forced persistent NVFP4 at
  `4096x4096` is 20.416 us versus 22.400 us looped (8.9% faster). At
  `4096x8192` they tie, and persistent is worse for some MXFP4 shapes, so this
  argues for preserving/benchmarking both choices, not forcing persistence.
- Inductor emits row-major scales only. These timings omit the cost of producing
  a downstream GEMM-ready swizzled scale layout. FlashInfer's swizzled result is
  produced inside its one kernel; an Inductor comparison that needs that layout
  must include a swizzle or teach the fused kernel to write it directly.

## Environment and method

- GPU: NVIDIA B200, SM100, CUDA 12.8 (`CUDA_VISIBLE_DEVICES=1`)
- FlashInfer: 0.6.14
- `torch.__version__`: `2.14.0a0+git6e40f92`; Python source was loaded from this
  checkout at exact HEAD `02da48d4f73`. The version suffix describes the
  existing C-extension build, not the Python source SHA.
- No rebuild was required or performed.
- Primary timing is one CUDA-graph replay per `triton.testing.do_bench` sample,
  100 warmups and 500 repetitions. Caches were force-disabled for compilation.
- The existing FlashInfer harnesses use `F.rms_norm(..., eps=None)` for Inductor
  but `eps=1e-6` for FlashInfer. That changes values, not kernel topology or the
  timing comparison. The dedicated correctness and repeated-measurement scripts
  in this audit explicitly use `eps=1e-6` on both sides.
- FlashInfer was run with PDL disabled, matching the existing harness.

## Primary one-replay results

All numbers are median microseconds. `off` is the three-kernel Inductor
reference; `on` is the fused final-stack result.

### NVFP4

| Shape | CD | Flash row | Flash swizzled | Inductor off | Inductor on | On vs row |
|---|---:|---:|---:|---:|---:|---:|
| 128x4096 | off | 10.016 | 10.176 | 10.176 | 8.160 | -18.5% |
| 1024x4096 | off | 10.336 | 12.256 | 16.352 | 10.336 | tied |
| 256x8192 | off | 8.224 | 10.176 | 12.224 | 10.208 | +24.1% |
| 4096x8192 | off | 33.888 | 36.832 | 63.616 | 45.024 | +32.9% |
| 128x4096 | on | 8.320 | 10.112 | 10.144 | 6.144 | -26.2% |
| 1024x4096 | on | 10.272 | 12.192 | 16.288 | 10.144 | -1.2% |
| 256x8192 | on | 8.160 | 10.080 | 12.192 | 8.096 | -0.8% |
| 4096x8192 | on | 34.528 | 36.768 | 63.520 | 34.752 | +0.65% |

### MXFP4

| Shape | CD | Flash row | Flash swizzled | Inductor off | Inductor on | On vs row |
|---|---:|---:|---:|---:|---:|---:|
| 128x4096 | off | 8.096 | 8.096 | 10.144 | 8.096 | tied |
| 1024x4096 | off | 12.192 | 12.192 | 16.288 | 10.144 | -16.8% |
| 256x8192 | off | 8.064 | 8.064 | 12.192 | 10.144 | +25.8% |
| 4096x8192 | off | 36.736 | 36.768 | 63.424 | 36.768 | +0.09% |
| 128x4096 | on | 8.064 | 8.064 | 10.144 | 6.080 | -24.6% |
| 1024x4096 | on | 12.192 | 12.192 | 16.288 | 10.112 | -17.1% |
| 256x8192 | on | 8.064 | 8.064 | 12.160 | 8.064 | tied |
| 4096x8192 | on | 36.736 | 36.768 | 63.360 | 30.624 | -16.6% |

For MXFP4, source inspection also confirmed the intended code collapse. The
fused kernel contains one E8M0 conversion, one reciprocal sequence, and one FP4
pack instruction; nested-off emitted one conversion, three reciprocal sequence
occurrences, and two pack occurrences across three kernels.

## Correctness

NVFP4 nested-on is byte-for-byte equal to the nested-off Inductor reference and
has exactly equal scales for every shape and both CD settings in the primary
runs.

MXFP4 was audited separately at `4096x8192`, bf16, fixed seed, and `eps=1e-6`:

- CD off: zero differences across 16,777,216 packed bytes / 33,554,432 FP4
  values; scales exactly equal.
- CD on: exactly one byte and one FP4 nibble differ (2.98e-8 of values); scales
  are exactly equal. The FP4 value step is 0.5, max dequantized error is 0.125,
  dequantized RMSE is 2.16e-5, and relative L2 error is 2.20e-5.
- Recompiling the CD-on fused kernel reproduces exactly, including that one
  differing nibble.

The isolated, deterministic one-nibble change is consistent with a value at a
quantization boundary crossing because CD chooses a different reduction/block
order. It is not consistent with a lane/layout error, which would affect a
structured fraction of lanes. Bitwise equivalence is therefore not guaranteed,
although the numeric effect is negligible.

FlashInfer is not used as the correctness oracle. Even after matching epsilon,
its packed bytes differ from Inductor and its scales can differ (NVFP4 max 0.25
in float scale value; MXFP4 up to one exponent code). The nested-off Inductor
path is the relevant same-graph reference.

## Repeated/interleaved 4096x8192 sensitivity

I also rotated method order over 21 rounds and timed batches of 100 identical
CUDA-graph replays with CUDA events. This drives the GPU continuously and lets
the roughly 86 MB input/output working set become L2-hot. It is a sensitivity
experiment, not the primary latency result: production usually supplies new
activations, although a preceding producer may make some data hot.

| Format | CD | Flash row | Flash swizzled | Inductor off | Inductor on |
|---|---:|---:|---:|---:|---:|
| NVFP4 | off | 20.525 | 23.880 | 38.967 | 38.967 |
| NVFP4 | on | 20.525 | 23.955 | 28.716 | 28.717 |
| MXFP4 | off | 26.781 | 28.717 | 28.726 | 28.717 |
| MXFP4 | on | 26.793 | 28.717 | 24.619 | 24.601 |

Per-round p10-p90 bands are narrow (for example NV CD-on Inductor on:
28.716-28.756 us), so order rotation removes the earlier several-percent
run-to-run spread. The absolute regime changes substantially, however: repeated
same-address replay boosts clocks and cache residency. It should not replace the
one-replay comparison.

## Persistent versus looped

Forced variants used the same fused graph, CD enabled, `triton.multi_kernel=0`,
and a choices handler that disabled cooperative reduction and selected either
the persistent or looped form.

| Shape | NV looped | NV persistent | MX looped | MX persistent |
|---|---:|---:|---:|---:|
| 4096x1024 | 10.112 | 10.144 | 10.144 | 10.112 |
| 4096x2048 | 14.208 | 14.208 | 12.160 | 14.208 |
| 4096x4096 | 22.400 | 20.416 | 20.352 | 20.352 |
| 4096x8192 | 36.768 | 36.768 | 30.624 | 32.640 |

The generic INNER-reduction heuristic uses a persistent threshold of 1024 when
multi-kernel is disabled, so the default selects looped above D=1024. This is
the right direction for MXFP4 and neutral at NVFP4 D=8192, but it misses the
measured NVFP4 D=4096 win.

The sub-parent path calls `create_kernel_choices(...)[0]`. When multi-kernel is
enabled, `add_multi_kernel_choices` constructs both forms and sorts
non-persistent kernels first; `[0]` therefore keeps the looped form and discards
the persistent alternative before the ordinary runtime selector can benchmark
it. At D=2048, static source capture confirmed both `multi_kernel=0` and `=1`
emit only `triton_red_fused...`; the latter increments generated-kernel metrics
to two during choice construction but still emits the first looped source.

This leaves a plausible 8.9% shape-specific NVFP4 opportunity. It is not a
blanket regression: persistence is 16.8% slower for MXFP4 D=2048 and 6.6%
slower at D=8192. Any change should let the existing multi-kernel mechanism
retain and benchmark both choices for this path, or add a measured format-aware
heuristic, rather than force persistent codegen.

## Layout accounting

Inductor's output scale is row-major. FlashInfer's
`is_sf_swizzled_layout=True` writes the scale-factor swizzle inside its RMSNorm
quantization kernel. At `4096x8192`, the primary NVFP4 FlashInfer row/swizzled
times are 34.528/36.768 us; the repeated hot-input times are 20.525/23.955 us.
For MXFP4 the primary medians quantize to 36.736/36.768 us, while the repeated
experiment resolves 26.793/28.717 us.

Therefore:

- NVFP4's row-major tie does not imply a tie for a GEMM-ready swizzled pipeline.
  Inductor has no timing headroom for a separate swizzle.
- MXFP4 has about 6.1 us of primary timing headroom over FlashInfer swizzled,
  but the cost of a separate Inductor swizzle was not measured and could consume
  part or all of it through memory traffic plus another launch.
- If the downstream consumer accepts row-major scale, the row-major comparison
  is the correct one. Otherwise direct swizzled stores belong in the fused
  kernel for an end-to-end comparison.

## Why coordinate descent fixes large NVFP4

An isolated autotune-cache rerun at `4096x8192` records the exact selected
configs:

| Setting | Kernel | XBLOCK | R0_BLOCK | warps | stages | grid |
|---|---|---:|---:|---:|---:|---:|
| CD off | looped `triton_red` | 1 | 1024 | 8 | 1 | 4096 CTAs |
| CD on | looped `triton_red` | 1 | 2048 | 8 | 1 | 4096 CTAs |

The CD winner is recorded verbatim in
`final_nv_config_cd1_cache/77/9282d4fde5fb1f7e8035683a9d4bc91f97c48bc6946f20a4863a0efc9504634f.best_config`:

```json
{"XBLOCK": 1, "R0_BLOCK": 2048, "num_warps": 8, "num_stages": 1,
 "found_by_coordesc": true}
```

Nothing about row tiling, persistence, warp count, stages, or grid changes. The
entire runtime win is the doubled reduction block. This looped kernel traverses
the 8192-wide row twice: once for RMS accumulation and once for scaling/packing.
R0_BLOCK=1024 therefore executes 8 loop chunks per traversal (16 total), while
2048 executes 4 per traversal (8 total). It halves loop/control and partial-tile
overhead and gives each CTA more reduction work without reducing grid
parallelism.

Resource use supports that explanation rather than an occupancy tradeoff.
`cuobjdump --dump-resource-usage` reports 31 registers/thread for R0_BLOCK=1024
and 32 for the selected R0_BLOCK=2048 kernel, with no stack/local spill and
1024 bytes static shared memory for both. With 8 warps/CTA, the one-register
increase does not change the thread/register occupancy limit. The larger block
mainly improves useful work and memory/reduction instruction amortization per
CTA, producing the measured 45.024 -> 34.752 us improvement (1.30x).

## Commands

Primary runs:

```bash
CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/bench_flashinfer_rmsnorm_nvfp4.py \
  --shapes 128x4096 1024x4096 256x8192 4096x8192 \
  --warmup 100 --rep 500 \
  --out agent_space/final_stack_nvfp4_cd0_20260805.json

CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/bench_flashinfer_rmsnorm_nvfp4.py --coordesc \
  --shapes 128x4096 1024x4096 256x8192 4096x8192 \
  --warmup 100 --rep 500 \
  --out agent_space/final_stack_nvfp4_cd1_20260805.json

CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/bench_flashinfer_rmsnorm_mxfp4.py \
  --shapes 128x4096 1024x4096 256x8192 4096x8192 \
  --warmup 100 --rep 500 \
  --out agent_space/final_stack_mxfp4_cd0_20260805.json

CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/bench_flashinfer_rmsnorm_mxfp4.py --coordesc \
  --shapes 128x4096 1024x4096 256x8192 4096x8192 \
  --warmup 100 --rep 500 \
  --out agent_space/final_stack_mxfp4_cd1_20260805.json
```

Repeated/interleaved example:

```bash
CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/final_stack_interleaved_bench.py \
  --format nvfp4 --shape 4096x8192 --coordesc \
  --rounds 21 --inner-reps 100 --warmup 100 \
  --out agent_space/final_stack_nvfp4_4096x8192_interleaved_cd1_20260805.json
```

Correctness and forced persistence:

```bash
CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/final_stack_mxfp4_correctness_audit.py

CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/final_stack_persistent_looped_bench.py \
  --format nvfp4 \
  --out agent_space/final_stack_nvfp4_persistent_looped_cd1_20260805.json

CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python agent_space/final_stack_persistent_looped_bench.py \
  --format mxfp4 \
  --out agent_space/final_stack_mxfp4_persistent_looped_cd1_20260805.json
```

## Artifacts

- `final_stack_{nvfp4,mxfp4}_{cd0,cd1}_20260805.json`
- `final_stack_{nvfp4,mxfp4}_4096x8192_interleaved_cd{0,1}_20260805.json`
- `final_stack_mxfp4_correctness_20260805.json`
- `final_stack_{nvfp4,mxfp4}_persistent_looped_cd1_20260805.json`
- Scratch runners: `final_stack_interleaved_bench.py`,
  `final_stack_mxfp4_correctness_audit.py`, and
  `final_stack_persistent_looped_bench.py`
