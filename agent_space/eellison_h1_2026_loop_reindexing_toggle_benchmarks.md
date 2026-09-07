# H1 2026 Loop Reindexing/Ordering/Cat Toggle Benchmarks

Generated: 2026-07-13

Artifacts:
- Harness: `agent_space/bench_h1_loop_reindexing_toggle.py`
- Raw results: `agent_space/h1_loop_reindexing_toggle_results.json`
- Generated code snippets: `agent_space/h1_loop_reindexing_toggle_code/`
- Compile caches: `agent_space/h1_loop_reindexing_toggle_cache/`

No files outside `agent_space/` were modified. All tensors ran on `cuda:0`.

## Environment

- Source HEAD: `d59fc87a47b0848573c67c25481a8eff3f31df98` (`[inductor] Fuse contiguous sub-parent epilogues`)
- Imported torch: `2.13.0a0+gite1067a5` from `/data/users/eellison/pytorch/torch/__init__.py`
- GPU: `NVIDIA B200`, capability `(10, 0)`, CUDA runtime `12.8`
- Visible devices: `8`; selected CUDA device: `0`

## Methodology

The benchmark cases are derived from `test/inductor/test_loop_ordering.py` and the existing local H1 benchmark notes in `agent_space/`. Each row uses a fresh Inductor/Triton cache, `fx_graph_cache=False`, `benchmark_kernel=True`, and `triton.unique_kernel_names=True`. The harness checks eager-vs-compiled correctness, records `torch._inductor.metrics.generated_kernel_count`, generated Triton definitions, wrapper launch sites, and `num_loop_reordering`, then reports external `torch.cuda.CUDAGraph` replay where capture is valid. The raw JSON also contains direct CUDA-event timings, but this report uses graph replay for kernel-performance claims.

For #179090 candidate selection, current HEAD has no config toggle to re-enable the old candidate picker. The grouped-quant row uses a scratch monkey-patch inside the harness to emulate the previous largest-buffer candidate choice; this is kept local to the benchmark process and is not a repo edit.

## Results

Speedup is before/after, so values above 1.0 mean the enabled/current setting was faster.

| PR / case | Toggle | Kernels | CUDA graph replay median | Notes |
| --- | --- | ---: | ---: | --- |
| Background loop-ordering sanity, outer sum + pointwise, x=(32, 2^20) | `loop_ordering_after_fusion False -> True` | 2 -> 1 | 104.84 us -> 79.93 us (1.31x) | Broad loop-ordering toggle from test_loop_ordering; not credited as #179090 impact. |
| Background loop-ordering sanity, outer softmax, x=(32, 2^20) | `loop_ordering_after_fusion False -> True` | 2 -> 1 | 150.07 us -> 133.20 us (1.13x) | Broad loop-ordering toggle from test_loop_ordering; not credited as #179090 impact. |
| #179090 candidate selection grouped quant, x=(8,7168) bf16 | `scratch legacy candidate selection -> current HEAD` | 2 -> 1 | 6.20 us -> 4.15 us (1.49x) | Monkey-patch emulates old largest-buffer candidate selection; closest isolation for #179090. |
| #176927 RMSNorm over sliced QKV/hidden-state view, x=(16,8192) bf16 | `loop_reindexing_after_fusion False -> True, loop_ordering_after_fusion=False` | 2 -> 1 | 4.20 us -> 3.04 us (1.38x) | Clean reindexing toggle isolation. |
| #176927 RMSNorm over transposed/non-contiguous hidden-state view, x=(16,8192) bf16 stride=(1,16) | `loop_reindexing_after_fusion False -> True, loop_ordering_after_fusion=True` | 2 -> 1 | 6.18 us -> 4.09 us (1.51x) | Combined with loop-ordering path; distinct from the grouped-quant candidate-selection row. |
| #179091 QKNorm + split RoPE cat, B=4,H=8,S=128,D=64 | `max_*_pointwise_cat_inputs 0 -> 8` | 2 -> 1 | 4.21 us -> 4.14 us (1.02x) | Normal current HEAD pointwise-cat gate. |
| #179091 QKNorm + interleaved RoPE stack/flatten | `max_*_pointwise_cat_inputs 0 -> 8` | 2 -> 1 | 4.19 us -> 6.19 us (0.68x) | Normal current HEAD pointwise-cat gate. |
| #179091 split RoPE force pointwise cat | `force_pointwise_cat False -> True with max_* limits at 0` | 2 -> 1 | 4.21 us -> 4.14 us (1.02x) | Sanity check of the explicit force toggle; not the PR heuristic. |

## Pointwise-Cat Toggles

Current HEAD exposes `max_complex_pointwise_cat_inputs`, `max_pointwise_cat_inputs`, and `force_pointwise_cat`. The report uses `max_* = 0 -> 8` as the closest toggle-based measurement for #179091 because that exercises the normal heuristic path. `force_pointwise_cat=True` is included only as a sanity check that bypasses the heuristic.

## Generated-Code Evidence

- `loop_ordering_outer_sum_False`: kernels=2, Triton defs=2, run calls=4, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_ordering_outer_sum_False`
- `loop_ordering_outer_softmax_False`: kernels=2, Triton defs=2, run calls=4, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_ordering_outer_softmax_False`
- `loop_ordering_outer_sum_True`: kernels=1, Triton defs=1, run calls=2, loop_reorders=1, code=`agent_space/h1_loop_reindexing_toggle_code/loop_ordering_outer_sum_True`
- `loop_ordering_outer_softmax_True`: kernels=1, Triton defs=1, run calls=2, loop_reorders=1, code=`agent_space/h1_loop_reindexing_toggle_code/loop_ordering_outer_softmax_True`
- `loop_ordering_candidate_grouped_quant_legacy_True`: kernels=2, Triton defs=2, run calls=4, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_ordering_candidate_grouped_quant_legacy_True`
- `loop_ordering_candidate_grouped_quant_legacy_False`: kernels=1, Triton defs=1, run calls=2, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_ordering_candidate_grouped_quant_legacy_False`
- `loop_reindex_rmsnorm_slice_False`: kernels=2, Triton defs=2, run calls=4, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_reindex_rmsnorm_slice_False`
- `loop_reindex_rmsnorm_transposed_False`: kernels=2, Triton defs=2, run calls=4, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_reindex_rmsnorm_transposed_False`
- `loop_reindex_rmsnorm_slice_True`: kernels=1, Triton defs=1, run calls=2, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_reindex_rmsnorm_slice_True`
- `loop_reindex_rmsnorm_transposed_True`: kernels=1, Triton defs=1, run calls=2, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/loop_reindex_rmsnorm_transposed_True`
- `pointwise_cat_split_rope_disabled`: kernels=2, Triton defs=2, run calls=4, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/pointwise_cat_split_rope_disabled`
- `pointwise_cat_split_rope_enabled`: kernels=1, Triton defs=1, run calls=2, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/pointwise_cat_split_rope_enabled`
- `pointwise_cat_split_rope_forced`: kernels=1, Triton defs=1, run calls=2, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/pointwise_cat_split_rope_forced`
- `pointwise_cat_interleaved_rope_disabled`: kernels=2, Triton defs=2, run calls=4, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/pointwise_cat_interleaved_rope_disabled`
- `pointwise_cat_interleaved_rope_enabled`: kernels=1, Triton defs=1, run calls=2, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/pointwise_cat_interleaved_rope_enabled`
- `pointwise_cat_interleaved_rope_forced`: kernels=1, Triton defs=1, run calls=2, loop_reorders=0, code=`agent_space/h1_loop_reindexing_toggle_code/pointwise_cat_interleaved_rope_forced`

## Caveats

- #176927 is directly benchmarkable with `loop_reindexing_after_fusion=False/True`.
- The broad `loop_ordering_after_fusion` rows are background sanity checks, not
  #179090 impact. #179090's candidate-selection change has no public
  current-head config toggle; the grouped-quant row is the closest isolated
  scratch-patch measurement.
- #179091 has pointwise-cat gates but no PR-specific toggle. The `max_*_pointwise_cat_inputs` comparison is the closest current-head toggle-based measurement.

## Errors

None.
