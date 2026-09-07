# Padded quantization benchmark findings

Date: 2026-09-03

## Method

The primary run is PyTorch `origin/main` at
`f8546ee64c3b9ed5cfdef3686d4f465d9af6c869`, which contains the landed nested
reduction enablement. It completed 208/208 cells on NVIDIA B200 with no compile
failures or incorrect padding values.

Each cell runs in a fresh process. Timings are the median of 50 CUDA-event
samples after 20 warmups. Each sample replays a CUDA graph containing 100 calls.
Inductor's internal CUDA-graph wrapper and graph caches are disabled.
FlashInfer PDL is disabled. Default and coordinate-descent tuning are compiled
and measured separately. All implementations of a workload/shape are assigned
to the same physical GPU.

The matrix separates:

- padding and swizzle alone;
- quantization plus padded/swizzled scale output;
- RMSNorm plus quantization plus padded/swizzled scale output;
- DCN `addcmul` plus MXFP6 plus padded/swizzled scale output.

It covers MXFP4, NVFP4, MXFP8, and MXFP6. The 128x4 layout uses zero padding.
The DCN XDL layout maps 96 logical rows to 128 physical rows and uses the
required `127` padding sentinel.

## Current-main result

The padding microbenchmark does not predict end-to-end performance. In
isolation, `F.pad` produces one kernel and beats fill+scatter by 1.57-2.88x.
Inside a padded RMSNorm quantization graph, fill+scatter often removes one of
the kernels produced by `F.pad` and becomes faster.

Selected coordinate-descent medians follow. `Best` is the fastest of `F.pad`,
fill+scatter, and explicit padding-region scatter. Ratios below 1.00 beat the
PDL-disabled FlashInfer comparison.

| workload | shape | best | us | kernels | FlashInfer us | best/FI |
|---|---:|---|---:|---:|---:|---:|
| MXFP4 quant | 19x4096 | pad scatter | 3.346 | 3 | 1.439 | 2.32x |
| MXFP4 quant | 128x4096 | `F.pad` | 1.506 | 1 | 1.568 | 0.96x |
| MXFP4 quant | 129x4128 | `F.pad` | 3.093 | 2 | 2.076 | 1.49x |
| MXFP4 quant | 989x4096 | fill+scatter | 6.567 | 3 | 3.808 | 1.72x |
| RMSNorm -> MXFP4 | 19x4096 | fill+scatter | 2.610 | 2 | 2.690 | 0.97x |
| RMSNorm -> MXFP4 | 128x4096 | pad scatter | 2.061 | 1 | 2.897 | 0.71x |
| RMSNorm -> MXFP4 | 129x4128 | fill+scatter | 3.625 | 2 | 3.225 | 1.12x |
| RMSNorm -> MXFP4 | 989x4096 | pad scatter | 5.406 | 2 | 5.063 | 1.07x |
| RMSNorm -> NVFP4 | 19x4096 | pad scatter | 3.206 | 2 | 2.766 | 1.16x |
| RMSNorm -> NVFP4 | 128x4096 | `F.pad` | 2.128 | 1 | 2.940 | 0.72x |
| RMSNorm -> NVFP4 | 129x4128 | fill+scatter | 3.590 | 2 | 3.378 | 1.06x |
| RMSNorm -> NVFP4 | 989x4096 | fill+scatter | 5.732 | 2 | 4.376 | 1.31x |
| MXFP8 quant | 19x4096 | pad scatter | 2.253 | 2 | 2.308 | 0.98x |
| MXFP8 quant | 128x4096 | `F.pad` | 1.564 | 1 | 2.296 | 0.68x |
| MXFP8 quant | 129x4128 | `F.pad` | 2.784 | 2 | 3.176 | 0.88x |
| MXFP8 quant | 989x4096 | fill+scatter | 4.428 | 2 | 3.488 | 1.27x |
| RMSNorm -> MXFP8 | 19x4096 | fill+scatter | 2.692 | 2 | 4.211 | 0.64x |
| RMSNorm -> MXFP8 | 128x4096 | `F.pad` | 2.034 | 1 | 4.523 | 0.45x |
| RMSNorm -> MXFP8 | 129x4128 | fill+scatter | 3.350 | 2 | 5.420 | 0.62x |
| RMSNorm -> MXFP8 | 989x4096 | pad scatter | 5.571 | 2 | 7.499 | 0.74x |

For the XDL MXFP6 layout there is no external CUDA baseline in this run:

| workload | shape | best current-main form | us | kernels |
|---|---:|---|---:|---:|
| MXFP6 quant | 95x3072 | `F.pad` | 3.227 | 2 |
| MXFP6 quant | 96x3072 | `F.pad` | 2.747 | 2 |
| MXFP6 quant | 97x3104 | `F.pad` | 2.946 | 2 |
| MXFP6 quant | 2048x3072 | `F.pad` | 8.954 | 2 |
| RMSNorm -> MXFP6 | 95x3072 | fill+scatter | 3.916 | 2 |
| RMSNorm -> MXFP6 | 96x3072 | fill+scatter | 3.904 | 2 |
| RMSNorm -> MXFP6 | 97x3104 | `F.pad` | 4.057 | 2 |
| RMSNorm -> MXFP6 | 2048x3072 | `F.pad` | 13.658 | 2 |
| DCN -> MXFP6 | 95x3072 | `F.pad` | 2.948 | 2 |
| DCN -> MXFP6 | 96x3072 | `F.pad` | 2.953 | 2 |
| DCN -> MXFP6 | 97x3104 | `F.pad` | 3.210 | 2 |
| DCN -> MXFP6 | 2048x3072 | `F.pad` | 9.516 | 2 |

The simple XDL fill+scatter formulation is not suitable for plain or DCN
MXFP6 on current main: it creates four kernels and is 1.39-2.29x slower than
`F.pad`. RMSNorm -> MXFP6 is different: both forms use two kernels and the
winner varies by shape and tuning.

## Experimental mechanisms

The predicated masked-fill stack was rebased onto the same main revision in a
clean scratch worktree (`f2ac1f47f60e`). All 112 measured cells succeeded and
matched the other padding forms byte-for-byte. On the principal RMSNorm ->
MXFP4 coordinate-descent cases it was never the fastest: it was 4-21% slower
than the best existing formulation. It has isolated wins for small MXFP8 and
NVFP4 cases, but no broad advantage that justifies selecting it as the single
padding strategy.

The XDL auxiliary-write prototype was measured in its existing, dirty
`5bd51f563588` integration tree, so its absolute times must not be mixed with
the current-main table. Its within-tree comparisons are useful:

| workload | shape | auxiliary us | best ordinary us | auxiliary/ordinary |
|---|---:|---:|---:|---:|
| MXFP6 quant | 95x3072 | 2.778 | 2.998 | 0.93x |
| MXFP6 quant | 96x3072 | 2.893 | 2.729 | 1.06x |
| MXFP6 quant | 97x3104 | 2.788 | 2.920 | 0.95x |
| MXFP6 quant | 2048x3072 | 8.844 | 9.184 | 0.96x |
| RMSNorm -> MXFP6 | 95x3072 | 3.583 | 3.928 | 0.91x |
| RMSNorm -> MXFP6 | 96x3072 | 3.653 | 3.656 | 1.00x |
| RMSNorm -> MXFP6 | 97x3104 | 3.766 | 3.902 | 0.97x |
| RMSNorm -> MXFP6 | 2048x3072 | 14.268 | 12.881 | 1.11x |
| DCN -> MXFP6 | 95x3072 | 2.978 | 2.956 | 1.01x |
| DCN -> MXFP6 | 96x3072 | 3.002 | 3.020 | 0.99x |
| DCN -> MXFP6 | 97x3104 | 3.449 | 3.193 | 1.08x |
| DCN -> MXFP6 | 2048x3072 | 9.195 | 9.511 | 0.97x |

The auxiliary path removes the two-kernel penalty of ordinary fill+scatter for
plain/DCN MXFP6, but it does not reduce the complete graph below two generated
kernels and its coordinate-descent gain over `F.pad` is generally small. It is
not uniformly better for RMSNorm -> MXFP6.

## Recommendation

1. Do not select padding from the standalone swizzle result. Fusion topology is
   decisive and reverses the ranking in several RMSNorm pipelines.
2. Bypass padding machinery when the 128x4 output is already aligned; the
   one-kernel `F.pad`/identity path wins clearly.
3. For padded 128x4 RMSNorm outputs, use the simple initialized destination plus
   valid-value scatter as the baseline. It consistently removes the extra
   `F.pad` kernel and the predicated primitive has no consistent end-to-end win.
4. For plain/DCN MXFP6 XDL output, retain `F.pad` unless the auxiliary-write
   representation is available. Do not use ordinary fill+scatter: its four
   kernels are a large regression. The auxiliary representation is promising
   because it restores the two-kernel topology, but its small and mixed gains
   argue for a focused implementation rather than a general scheduler feature.
5. Treat one-kernel padded quantization as a separate follow-up. None of these
   mechanisms makes the full MXFP6 pipelines one kernel on the measured stack;
   changing the number or coalescing of stores alone does not solve the pack and
   reduction staging constraints.

One coordinate-descent MXFP4 sparse-scatter result at 989x4096 selected a bad
configuration (21.269 us). An isolated 100-sample rerun measured 6.924 us, and
the latter is used for conclusions; fill+scatter remains faster at 6.567 us.

ROCm/AITER MXFP6 is not measured here: this host is an NVIDIA B200. The existing
ROCm runner remains the appropriate cross-vendor follow-up on gfx950.

## Artifacts

- `bench_worker.py`: one benchmark cell, correctness metadata, codegen counts,
  and CUDA-graph timing.
- `run_matrix.py`: resumable process-isolated matrix runner.
- `analyze.py`: Markdown table generator.
- `results_main_f854_core_v2.jsonl`: 208 current-main cells.
- `results_predicated_f2ac_core.jsonl` and
  `results_predicated_f2ac_extra_formats.jsonl`: 112 predicated-stack cells.
- `results_auxiliary_5bd51_smoke.jsonl`: 84 within-tree XDL comparisons.
- `REPORT_main_f854_core_v2.md`, `REPORT_current_main_and_predicated.md`, and
  `REPORT_auxiliary_5bd51_core.md`: generated detailed tables.
