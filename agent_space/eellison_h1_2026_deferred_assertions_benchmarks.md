# H1 2026 deferred assertions/copy-until-use benchmarks

Local scratch benchmark evidence for deferred input size/stride assertions (#177783) and deferred `copy_if_misaligned` until first use (#179039/#180599).

## Environment

- repo: `/data/users/eellison/pytorch`
- git HEAD: `d59fc87a47b0848573c67c25481a8eff3f31df98`
- torch: `2.13.0a0+gite1067a5`
- torch git version: `e1067a5ad33609f9486adefbbd627e18281844c2`
- CUDA available: `True`
- CUDA runtime: `12.8`
- device: `NVIDIA B200`
- capability: `[10, 0]`

No repo files were modified. The eager comparisons are scratch-only monkey-patches to generated wrapper codegen because HEAD has no config toggle for eager-vs-deferred placement.

## Metric labels

- `first_mm_us`: Python `perf_counter` time from entering the compiled function until the generated wrapper invokes the first external matmul (`extern_kernels.mm` or `extern_kernels.addmm`); this is a host-wrapper first-launch latency probe.
- `sync_wall_us`: Python `perf_counter` with CUDA synchronization after a call batch; this includes host enqueue plus device completion and is noisier for this question.
- Host timings do not use CUDA graph replay. `graph_partition=True`/`triton.cudagraphs=True` is used only for the partition-wrapper codegen probe.

## Size/stride assertion first-launch latency

Pattern: `y = x @ w`, then a later pointwise kernel consumes `tail_count` additional tensor inputs. Those tail tensors are graph inputs but unused on the first-kernel hot path. Matrix size `64x64`.

| tail inputs first used after mm | deferred median first_mm_us | eager-wrapper median first_mm_us | eager - deferred | ratio |
|---:|---:|---:|---:|---:|
| 0 | 22.17 | 22.28 | 0.11 | 1.00x |
| 64 | 33.61 | 65.32 | 31.72 | 1.94x |
| 256 | 69.16 | 129.01 | 59.85 | 1.87x |

Synchronized wall medians for the same cases:

| tail inputs | deferred sync_wall_us | eager-wrapper sync_wall_us | eager - deferred | ratio |
|---:|---:|---:|---:|---:|
| 0 | 41.38 | 42.41 | 1.03 | 1.02x |
| 64 | 85.74 | 116.35 | 30.60 | 1.36x |
| 256 | 249.44 | 258.06 | 8.62 | 1.03x |

Generated wrapper placement probe:

- deferred: `2` `assert_size_stride` calls before the first kernel, snippet `agent_space/eellison_h1_2026_deferred_assertions_snippets/size_asserts_deferred.py.txt`
- eager scratch: `6` `assert_size_stride` calls before the first kernel, snippet `agent_space/eellison_h1_2026_deferred_assertions_snippets/size_asserts_eager_size_asserts.py.txt`

## Misaligned-input copy first-launch latency

Pattern: aligned `x,w` feed the first `mm`; `tail_count` later inputs are deliberately misaligned views and first used by the later pointwise kernel, so they are also unused on the first-kernel hot path. Matrix size `256x256`.

| misaligned tail inputs | deferred median first_mm_us | eager-wrapper median first_mm_us | eager - deferred | ratio |
|---:|---:|---:|---:|---:|
| 1 | 35.54 | 34.56 | -0.98 | 0.97x |
| 8 | 29.87 | 78.89 | 49.02 | 2.64x |
| 32 | 36.41 | 218.18 | 181.77 | 5.99x |

Synchronized wall medians for the same cases:

| misaligned tail inputs | deferred sync_wall_us | eager-wrapper sync_wall_us | eager - deferred | ratio |
|---:|---:|---:|---:|---:|
| 1 | 60.29 | 60.34 | 0.05 | 1.00x |
| 8 | 107.21 | 108.26 | 1.05 | 1.01x |
| 32 | 265.71 | 265.13 | -0.58 | 1.00x |

Generated wrapper placement probe:

- deferred: `2` `copy_if_misaligned` calls before the first kernel, snippet `agent_space/eellison_h1_2026_deferred_assertions_snippets/alignment_deferred.py.txt`
- eager scratch: `6` `copy_if_misaligned` calls before the first kernel, snippet `agent_space/eellison_h1_2026_deferred_assertions_snippets/alignment_eager_alignment_copies.py.txt`

## Graph partition/subgraph probe

With `graph_partition=True` and `triton.cudagraphs=True`, a CUDA/CPU/CUDA case generated partition wrappers. The outer `Runner.call` unpacks args, while `copy_if_misaligned` appears inside `partition_0`/`partition_1` before the kernels that read those inputs.

- partitions found: `2`
- total `copy_if_misaligned` in generated partition code: `6`
- snippet: `agent_space/eellison_h1_2026_deferred_assertions_snippets/graph_partition_deferred.py.txt`

## Raw artifacts

- raw JSON: `agent_space/eellison_h1_2026_deferred_assertions_raw.json`
- harness: `agent_space/eellison_h1_2026_deferred_assertions_harness.py`
- snippets: `agent_space/eellison_h1_2026_deferred_assertions_snippets/`
