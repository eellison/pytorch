# eellison H1 2026 local remaining benchmarks

Generated: 2026-07-13

## Artifacts

- Harness: `agent_space/bench_h1_local_remaining.py`
- Raw results: `agent_space/h1_local_remaining_results.json`
- Smoke results: `agent_space/h1_local_remaining_smoke.json`
- Compile caches and generated code: `agent_space/h1_local_remaining_cache/`

No files outside `agent_space/` were modified. Distributed and multi-GPU overlap work was intentionally excluded.

## Environment

- Source HEAD: `d59fc87a47b0848573c67c25481a8eff3f31df98` (`[inductor] Fuse contiguous sub-parent epilogues`)
- Imported torch: `2.13.0a0+gite1067a5` from `/data/users/eellison/pytorch/torch/__init__.py`
- GPU: NVIDIA B200, capability `[10, 0]`, CUDA runtime 12.8
- Visible GPUs: 8, but all benchmark tensors ran on local `cuda:0`
- Timing: CUDA-event direct timing, synchronized wall timing, and `torch.cuda.CUDAGraph` replay timing where external replay was valid.

Interpretation notes:
- CUDA-event direct timing is the primary GPU-body timing for ordinary compiled callables.
- PDL rows use synchronized wall time as primary because direct CUDA events can undercount device-side dependent launch tail work.
- `mode="reduce-overhead"` cudagraph-policy rows are measured as Inductor internal CUDA graph behavior. External graph recapture was not used there.

## Loop reindexing and pointwise cat refinement

These refine the previous weak/shape-sensitive loop and cat rows with larger local B200 shapes.

| Case | Direct event | Direct speedup | Graph replay event | Graph speedup | Kernels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Loop reindex transposed RMSNorm, bf16 `x=(64,16384)` stride `(1,64)` | 8.35 us -> 31.48 us | 0.27x | 6.26 us -> 4.23 us | 1.48x | 2 -> 1 |
| Pointwise cat split RoPE, `B=8,H=16,S=256,D=128` | 28.46 us -> 12.46 us | 2.29x | 10.33 us -> 12.48 us | 0.83x | 2 -> 1 |
| Pointwise cat interleaved RoPE, same shape | 30.55 us -> 23.01 us | 1.33x | 14.52 us -> 18.52 us | 0.78x | 2 -> 1 |

Takeaway: the larger split-RoPE case strengthens #179091 direct GPU evidence. Replay still shows the fused cat kernels are heavier than the unfused bodies at this shape, so the strongest benefit remains launch/materialization reduction rather than pure kernel-body speed. The transposed RMSNorm row confirms the 2 -> 1 kernel collapse; its replay improves, while direct event timing remains noisy/shape-sensitive.

## PDL combo kernels

Current-head toggle: `combo_kernels=True`, `triton.enable_pdl=False/True`. Each case produced one combo kernel; PDL-enabled code emitted `launch_pdl=True` and 8 `gdc_launch` sites.

| Elements per input | Direct synchronized wall | Wall speedup | Direct event | Graph replay event | PDL code |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 30.24 us -> 35.74 us | 0.85x | 31.79 us -> 37.04 us | 2.80 us -> 3.87 us | `False -> True`, `gdc 0 -> 8` |
| 65536 | 30.40 us -> 79.87 us | 0.38x | 32.39 us -> 2.08 us | 4.06 us -> 4.18 us | `False -> True`, `gdc 0 -> 8` |
| 1048576 | 36.57 us -> 86.96 us | 0.42x | 32.36 us -> 9.93 us | 8.58 us -> 12.37 us | `False -> True`, `gdc 0 -> 8` |

Takeaway: local B200 codegen for #174232 is present, but this synthetic 8-way independent pointwise combo regresses under PDL by synchronized wall time. CUDA-event undercounting on PDL rows is visible for larger shapes, so wall time is the safer number here. This still does not prove the intended dependent-launch win; it is negative coverage for a simple pointwise combo.

## Low-precision primitives

### `cvt_e8m0_rceil`

Current-head comparison: `inductor_prims.cvt_e8m0_rceil` PTX lowering versus a compiled bitwise software fallback over `1<<20` elements. PTX rows contained `cvt.rp.satfinite.ue8m0x2.f32`.

| dtype | Direct event fallback -> PTX | Speedup | Graph replay fallback -> PTX | Speedup | PTX present |
| --- | ---: | ---: | ---: | ---: | --- |
| `float32` | 23.09 us -> 4.24 us | 5.45x | 4.15 us -> 2.90 us | 1.43x | yes |
| `float16` | 4.19 us -> 4.20 us | 1.00x | 4.22 us -> 4.19 us | 1.01x | yes |
| `bfloat16` | 22.29 us -> 22.05 us | 1.01x | 3.97 us -> 3.81 us | 1.04x | yes |

Takeaway: #172497 has a clear local primitive win for fp32 input. The fp16/bf16 cases are flat in this microbenchmark, likely because conversion/upcast overhead dominates and both variants stay single-kernel.

### `inline_asm_elementwise`

Current-head compiled HOP rows over `1<<20` elements:

| Case | Direct event | Graph replay event | Generated code contains inline asm |
| --- | ---: | ---: | --- |
| f32 add | 4.25 us | 4.28 us | yes |
| fp16 upcast double | 4.20 us | 4.22 us | yes |
| bf16 upcast double | 4.20 us | 4.21 us | yes |

Takeaway: #177922's HOP path is locally measurable and graph-capturable. Exact shipped attribution still requires pre/post because current HEAD has no off toggle for removing the HOP.

## Autotune and `aten.mm`

### `aten.mm` max-autotune

Current-head toggle: default compile versus `mode="max-autotune"` with `max_autotune=True`, `max_autotune_gemm_backends="TRITON"`.

| Shape | Direct event default -> max-autotune | Speedup | Kernels | External graph replay |
| --- | ---: | ---: | ---: | --- |
| bf16 `(64,256) x (256,128)` | 4.21 us -> 42.45 us | 0.10x | 0 -> 20 | max-autotune side failed external capture with `cudaErrorStreamCaptureInvalidated` |
| bf16 `(1024,1024) x (1024,1024)` | 33.65 us -> 48.26 us | 0.70x | 0 -> 22 | max-autotune side failed external capture with `cudaErrorStreamCaptureInvalidated` |

Takeaway: the current local max-autotune path is measurable, but it regressed on these two shapes. This is not a clean #175278 attribution because the current toggle changes tuning mode and generated choices, not the historical application of custom-op autotuning to `aten.mm`.

### Custom-op autotune API

Current-head custom fp16 matmul op, `x=(64,256)`, `weight=(256,128)`, `benchmark_with_cudagraphs=True`.

| Threshold | Direct event | Graph replay event | `benchmark_gpu_with_cuda_graph` calls | Generated kernels |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 30.10 us | 4.17 us | 2 | 19 |
| 1.5 | 56.23 us | 4.18 us | 2 | 0 |

Takeaway: the #175275/#175276 CUDA graph benchmark API path is active in both threshold settings. The threshold sweep is API/selection validation, not a landed speedup proof.

## CUDA graph policy current-head probes

These validate current behavior only. They are not pre/post attribution because the old behavior is not exposed by stable config toggles.

| Case | Direct event | Direct wall | Rerecord count | Key counters |
| --- | ---: | ---: | ---: | --- |
| Self-overlap expand input, `x=base.expand(256,4096)`, pointwise + `mm` | 58.43 us | 56.83 us | 0 | `cudagraph_recorded_non_static_inputs=2`, no `cudagraph_skips` |
| Fresh `nn.Parameter` pointer each call, limit `1` | 1540.81 us | 4501.74 us | 0 | `cudagraph_recorded_non_static_inputs=70`, no `cudagraph_skips` |
| Retained live outputs | 48.55 us | 46.60 us | 0 | no `cudagraph_skips`; retained-output smoke did not force an alias hazard |

Takeaway: current HEAD handles the #182524 self-overlap and #182531 parameter-pointer-churn smoke cases without rerecord-limit fallback. The live-output row is only a smoke test for #188117/#188078; it did not force the mutation/alias hazard needed to prove clone behavior.

## Host assert/copy deferral probe

Current-head generated-wrapper probe with 16 CUDA fp32 inputs:

- Generated kernels: 2
- Generated Triton defs: 2
- `assert_size_stride` occurrences: 18
- `copy_misaligned` occurrences: 0

Takeaway: current generated code is inspectable, but #177783/#179039/#180599 cannot be attributed with a current-head toggle. The expected win is host placement before first GPU launch, which requires parent/landed checkout comparison plus launch-time instrumentation.

## Pre/post-only plans

No pre/post checkout or rebuild was performed in this pass.

- #182524 `10efc0f4c5701389a7842aa12def560b679ee797`: compare `10efc0f4c57^` vs `10efc0f4c57` with `cudagraph_self_overlap_expand_current_head`. Current HEAD validates behavior, but cannot re-enable the old complex-memory-overlap cudagraph gate.
- #182531 `7089ea8477d00624f66460c197ca2d288036efd6`: compare `7089ea8477d^` vs `7089ea8477d` with `cudagraph_param_pointer_churn_current_head`. Current HEAD can set the rerecord limit but cannot make parameter-pointer churn count as unexpected again.
- #188117 `d46712a5fb2` and #188078 `d877745fe08`: compare each `commit^` vs `commit` with a stronger retained-live-output hazard that feeds a later compiled mutation/aliasing step. Current smoke does not prove clone insertion, and #188078 has only medium local inventory confidence.
- #177783 landed sequence `f0c3582ef2d...`, `779c78b3d7d...`: compare each `commit^` vs `commit` using the many-input assert harness plus CPU wrapper-entry-to-first-kernel instrumentation. Current `TORCHINDUCTOR_SIZE_ASSERTS=1` cannot emulate old top-of-wrapper placement.
- #179039 landed sequence `ddaac926c33...`, `55fc17f8653...` and #180599 `13a6d0ddbc7...`: compare each parent/landed pair with a late-use non-contiguous or misaligned input. Measure wrapper-entry-to-first-kernel time, total first iteration, and generated copy placement. Current HEAD has no old/new deferral toggle.
- #175275 `dd3c11bbb3b`, #175276 `1bb77ba1b09`, #175277 `9e85a6ead69`, #175278 landed sequence `b9b20336385...`, `e8120f9062c...`, `d7f39875e57...`, and #175422 `67a21876c88`: run the custom-op and `aten.mm` rows commit-by-commit. Current max-autotune toggles exercise the path, but do not isolate the historical API, cleanup, scoped-patch, `aten.mm`, or fallback-normalization changes.
- #172497 `b79269a4308961b681c6afb734054128d953e7c2`: compare `b79269a4308^` vs `b79269a4308` with the `cvt_e8m0_rceil` primitive rows. Current synthetic fallback is useful for throughput but is not the historical pre-primitive code path.
- #177922 landed sequence `a9be19830db...`, `1ae64875e7e...`: compare each parent/landed pair with the inline-asm HOP rows and the NVFP4 pack path. Current HEAD cannot remove the HOP with a config toggle.
- #184904 `eeccb459bcc`: compare `eeccb459bcc^` vs `eeccb459bcc` through `CapabilityBasedPartitioner(skip_horizontal_fusion=False/True)` on the unsupported-producer/four-supported-consumers FX graph from the queue. Current HEAD can call the API but old code did not expose it.
- #184905 `d79ee01a97f`: compare `d79ee01a97f^` vs `d79ee01a97f` with 4-8 local worker processes contending on the same GPU and a registered file/process lock hook. Current default behavior is a no-op unless an external harness registers the hook.

