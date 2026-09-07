# H1 2026 Loop/Fusion/Autotuning Benchmark Slice

Input inventory: `agent_space/eellison_h1_2026_pr_inventory.md`

Artifacts:
- Harness: `agent_space/bench_h1_loop_fusion_autotune.py`
- Raw results: `agent_space/h1_loop_fusion_autotune_results.json`

## Scope

I independently filtered the H1 inventory for non-nested Inductor/PT2 runtime-performance PRs touching loop ordering, reindexing, pointwise cat, combo kernels, tiling/autotuning, and cudagraph runtime/benchmarking. I excluded nested-reduction-only work from this slice unless the same current-head toggle directly applies to non-nested cases.

Benchmarked with current-head toggles/APIs:
- #176927: `loop_reindexing_after_fusion`
- #179090: `loop_ordering_after_fusion` and loop-order candidate behavior
- #179091: pointwise cat via `max_pointwise_cat_inputs`
- #174232: combo kernels and `triton.enable_pdl`
- #175271: `triton.cudagraph_min_partition_size`
- #175275, #175276, #175277, #175278, #175422: custom-op autotuning, scoped config patches, min speedup threshold, and CUDA graph benchmarking
- #182895, #183585: tiling/coalescing surface through `triton.coalesce_tiling_analysis`; exact #182895 block-size-floor attribution still needs pre/post

Surfaced but not directly benchmarkable with an equivalent current-head toggle:
- #184904: FX `CapabilityBasedPartitioner(skip_horizontal_fusion=...)`, not a `torch.compile` runtime config.
- #184905: GPU benchmark lock hooks are no-op unless an external harness registers a lock.
- #182524, #182531, #188117, #188078: cudagraph correctness/runtime policy changes need workloads with self-overlap, ptr churn, or live-output cloning and should be measured pre/post.
- #177783, #179039, #180599: input assert and copy-misaligned deferral/renaming do not have a stable current-head before/after toggle.
- #176345, #181275, #181276: compile-time symbolic/indexing fast paths, not CUDA runtime replay benchmarks.

## Methodology

Each case compiled with fresh `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` under `agent_space/`, `fx_graph_cache=False`, `benchmark_kernel=True`, and `triton.unique_kernel_names=True`. The harness checks eager-vs-compiled correctness, records `torch._inductor.metrics.generated_kernel_count`, counts generated Triton definitions and launch sites in emitted code, then times the compiled callable.

Timing method:
- Direct timing: CUDA events, 10 warmup calls, 5 reps of 50 iterations, report median milliseconds.
- CUDA graph replay: warm up compiled callable, capture one static replay with `torch.cuda.CUDAGraph`, then time graph replay with CUDA events using the same reps/iters.
- For `mode="reduce-overhead"` cudagraph cases, external recapture may be invalid because Inductor is already using CUDA graphs internally; those rows keep direct timing and mark graph replay unavailable.

Environment:
- Source HEAD: `d59fc87a47b0848573c67c25481a8eff3f31df98` (`[inductor] Fuse contiguous sub-parent epilogues`)
- Imported torch: `2.13.0a0+gite1067a5`
- Torch path: `/data/users/eellison/pytorch/torch/__init__.py`
- GPU: NVIDIA B200, capability `(10, 0)`, CUDA runtime 12.8
- Visible GPUs: 8

## Results

Speedup is left/right, so values above 1.0 mean the enabled/new setting was faster.

| PRs | Case and shape | Toggle | Kernels | Direct median | CUDA graph replay median | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| #179090 | outer sum + pointwise, `x=[32, 2^20]` | `loop_ordering_after_fusion False -> True` | 2 -> 1 | 0.1080 -> 0.0804 ms, 1.34x | 0.1048 -> 0.0800 ms, 1.31x | `num_loop_reordering` 0 -> 1 |
| #179090 | outer softmax, `x=[32, 2^20]` | `loop_ordering_after_fusion False -> True` | 2 -> 1 | 0.1421 -> 0.1340 ms, 1.06x | 0.1496 -> 0.1332 ms, 1.12x | Fuses to one Triton kernel |
| #176927 | RMSNorm reshape slice, bf16 `x=[16,8192]` from wider contiguous qkv | `loop_reindexing_after_fusion False -> True` | 2 -> 1 | 0.0248 -> 0.0203 ms, 1.22x | 0.00416 -> 0.00230 ms, 1.81x | Launch-count win is clearest under graph replay |
| #176927, #179090 | RMSNorm reshape transposed, bf16 `x=[16,8192]`, stride `(1,16)` | `loop_reindexing_after_fusion False -> True` | 2 -> 1 | 0.0253 -> 0.0211 ms, 1.20x | 0.00424 -> 0.00413 ms, 1.03x | Kernel count improves; replay body time nearly flat |
| #179091 | QKNorm + split RoPE cat, `B=4,H=8,S=128,D=64` | `max_pointwise_cat_inputs 0 -> 8` | 2 -> 1 | 0.0388 -> 0.0232 ms, 1.67x | 0.00420 -> 0.00413 ms, 1.02x | Current cat gate reproduces the intended fusion |
| #179091 | QKNorm + interleaved RoPE stack/flatten, same shape | `max_pointwise_cat_inputs 0 -> 8` | 2 -> 1 | 0.0321 -> 0.0231 ms, 1.39x | 0.00416 -> 0.00620 ms, 0.67x | Direct improves; graph replay shows fused kernel body is heavier for this small case |
| #174232 | Six independent pointwise ops, `6 * [2^22]` | `combo_kernels False -> True` | 6 -> 1 | 0.0504 -> 0.0338 ms, 1.49x | 0.0412 -> 0.0329 ms, 1.25x | Combo kernel cuts launches and generated Triton defs |
| #174232 | Two pointwise + two reductions, `[1024]`, `[32,1024]` | `combo_kernels_pointwise_only False -> True` | 2 -> 3 | 0.0271 -> 0.0315 ms, 0.86x | 0.00416 -> 0.00621 ms, 0.67x | Restricting combo to pointwise hurts this mixed case |
| #174232 | Combo pointwise PDL, `6 * [2^20]` | `triton.enable_pdl False -> True` | 1 -> 1 | 0.0275 -> 0.0321 ms, 0.86x | 0.00824 -> 0.01028 ms, 0.80x | Emitted code has `launch_pdl=True`; this synthetic B200 case regresses |
| #175271 | Cudagraph partition threshold, small matmul chain `x=[128,128]` | `cudagraph_min_partition_size 0 -> 10` | 1 -> 1 | 0.0386 -> 0.0318 ms, 1.21x | unavailable -> 0.00671 ms | Threshold 0 was already internally cudagraphed; external recapture failed |
| #182895, #183585 | Contiguous + transposed add, `4096x4096` | `coalesce_tiling_analysis False -> True` | 1 -> 1 | 0.0346 -> 0.0347 ms, 1.00x | 0.0340 -> 0.0335 ms, 1.02x | No meaningful runtime change |
| #182895, #183585 | Permute clone view amax, bf16 `[1024,2048] -> [2048,1024]` | `coalesce_tiling_analysis False -> True` | 1 -> 1 | 0.0204 -> 0.0205 ms, 1.00x | 0.0123 -> 0.00620 ms, 1.99x | Replay isolates improved kernel execution |
| #175275/#175276/#175277/#175278/#175422 | Custom op autotune, fp16 `x=[8,128,256]`, weight `[256]` | `benchmark_with_cudagraphs=True` API | 4 | 0.0225 ms | 0.00287 ms | Autotune log picked best decomp around 0.0058 ms vs fallback around 0.0079 ms |
| #175275/#175276/#175277/#175278/#175422 | Same custom op, independent registration | `benchmark_with_cudagraphs=True` API | 4 | 0.0215 ms | 0.00277 ms | Repeat verifies stable API path |

Generated-code metrics:
- Kernel count and Triton definition count matched in all simple fusion cases.
- `num_loop_reordering` was only nonzero for the loop-ordering-enabled outer reduction cases.
- `metrics.num_bytes_accessed` stayed `0` in this harness, so I did not use it as evidence.

## Caveats

The imported `torch.__version__` reports `gite1067a5` while the source checkout HEAD is `d59fc87`; the import path is the checkout itself, but this should be recorded when comparing against CI or other local builds.

CUDA graph replay removes Python and launch overhead, which is useful for kernel-body comparisons, but it can obscure launch-latency features such as PDL. For PDL, the direct and replay timings both regressed on the synthetic pointwise combo case; a more representative dependent-launch benchmark should use a short sequence of many small dependent kernels outside an enclosing CUDA graph.

For `mode="reduce-overhead"`, external `torch.cuda.CUDAGraph` capture can fail because Inductor has already captured internally. The `cudagraph_min_partition_size=0` row hit `cudaErrorStreamCaptureInvalidated`; direct timing is still valid, and the threshold-10 row was externally replayable because the small partition was skipped.

## Required Pre/Post Benchmarks

#184904: Compare parent of `eeccb459bcc` vs `eeccb459bcc` with an FX graph where one unsupported producer feeds multiple independent supported consumers. Run `CapabilityBasedPartitioner(..., skip_horizontal_fusion=False/True)`, report partition count, max partition size, and if this is wired into a runtime path, end-to-end compiled runtime.

#184905: Register a real external benchmark lock hook and run concurrent Inductor autotuning workers, for example the custom-op autotune case above or a max-autotune GEMM case. Measure autotune variance and wall time with and without the hook; default current-head behavior is no-op.

#182895/#183585: For exact attribution, compare parent vs landed commits with an autotuned Triton kernel whose candidate configs hit block-size floor constraints, and separately measure scheduler compile time for repeated coalescing-analysis queries. The current-head toggle above exercises the tiling surface but does not isolate block-size-floor propagation or cache lookup cost.

#182524/#182531/#188117/#188078: Use `torch.compile(..., mode="reduce-overhead")` workloads that exercise self-overlapping inputs, parameter pointer churn across iterations, and live user outputs. Report graph re-record counts, manager graph ids, allocations/clones, and steady-state replay time on commits before and after each PR.

#177783/#179039/#180599: Use graphs with many inputs and either deferred `assert_size_stride` opportunities or misaligned inputs requiring copies. The relevant measurements are CPU time before first kernel launch, first-iteration time, and steady-state direct/CUDA-event time on pre/post commits; current HEAD does not expose equivalent toggles.

#176345/#181275/#181276: These are compile-time symbolic/indexing performance PRs. Use a wide-symbol after-AOT repro with 50+ symbolic terms, measure compile wall time and time in SymPy/static reasoning on pre/post commits. CUDA graph replay is not the right benchmark surface.
