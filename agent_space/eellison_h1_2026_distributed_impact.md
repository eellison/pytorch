# H1 2026 eellison Distributed/Comms Impact Slice

## Scope and Method

- Scope: eellison-authored commits reachable from current `HEAD`, with commit dates from 2026-01-01 through 2026-06-30.
- Source: local git history and local commit bodies/files. `gh pr view` failed with a proxy/TLS error, and the document loader was blocked by input filtering, so PR closed/merged timestamps from GitHub were not available. Dates below are local landed commit dates (`%cd`).
- Candidate set: 64 H1 2026 reachable commits. Relevant PRs are those directly in distributed/collective overlap, bucketing, NCCL estimation, or CI/perf trace infrastructure.
- Exclusions: CUDA graph/autotuning PRs such as #175275/#175276 and cudagraph fixes such as #182524/#182531 are H1 landed perf-related work, but they are outside the corrected distributed/comms/CI trace slice unless noted under benchmark timing applicability.

## Relevant PRs

| PR | Landed date | Representative commit | Area | Files |
| --- | --- | --- | --- | --- |
| #172510 | 2026-01-16 | `ddfbbc49854` - `[inductor] Log NCCL estimations to tlparse in CSV format` | NCCL/collective estimation logging | `torch/_inductor/fx_passes/node_runtime_estimation.py`, `torch/_inductor/fx_passes/overlap_scheduling.py` |
| #172460 | 2026-01-21 | `3599c9635ac` - `Add perfetto trace event to graph pass application + ennvvar for disable` | Perfetto graph-pass trace infra | `torch/fx/passes/graph_transform_observer.py`, `torch/_inductor/fx_passes/pre_grad.py`, `torch/_inductor/config.py`, `test/dynamo/test_utils.py` |
| #173653 | 2026-01-28 | `95892933ff8` - `Add foreach_groups optimization to _pre_bucket_all_gather` | Collective bucketing/pre-bucket all-gather | `torch/_inductor/fx_passes/bucketing.py`, `test/distributed/test_overlap_bucketing_unit.py` |
| #183340 | 2026-05-13 | `794fb28641f` - `[ci] Fix tlparse artifact collection and enable torch trace on A100/B200 perf jobs` | CI perf trace artifacts | `.ci/pytorch/test.sh`, `.github/workflows/inductor-perf-test-b200.yml`, `.github/workflows/inductor-perf-test-nightly.yml` |

## PR Impact Analysis

### #172510 - NCCL estimations to tlparse CSV

Impact:
- Adds parseable CSV artifacts for collective estimates even when compile-time collective benchmarking is disabled.
- Logs both NCCL estimator and Inductor analytical estimator values, keyed by collective type, group, group size, and input bytes.
- Expected model-level speedup: none directly. The value is attribution: it makes overlap/bucketing decisions and exposed collective estimates easier to validate in production traces.

Benchmarkability:
- Needs multi-GPU to validate estimator quality against real collectives.
- Single-GPU is only useful for CSV formatting and trace artifact tests, not for NCCL model accuracy.

Multi-GPU scenarios:
- Sweep `all_gather`, `reduce_scatter`, and `all_reduce` sizes from small KB tensors to large bucket-sized tensors across 2, 4, 8, and multi-node world sizes.
- Compile an FSDP/DTensor transformer block with overlap scheduling enabled and collect tlparse artifacts; compare `fx_collectives_analytical_estimation` against CUDA-event measured collectives.
- Run with and without overlap-preserving bucketing to compare estimate deltas before and after bucket formation.

`torch.cuda.Graph` replay:
- Not directly applicable to this PR. The PR logs estimates and optional CUDA-event benchmark comparisons; graph replay is not required for validating the estimator.
- If an end-to-end model benchmark already uses CUDA graphs, traces may include that runtime mode, but the estimator artifact itself should be validated against explicit collective timings.

Confidence:
- High for functional impact from local diff and commit body.
- Medium for model-level implications because no benchmarks were run and GitHub PR metadata was unavailable.

### #172460 - Perfetto graph-pass trace events and pass disable envvar

Impact:
- Wraps graph pass application in `dynamo_timed`, making pass-level compile time visible in Perfetto/tlparse-style traces.
- Adds `TORCHINDUCTOR_DISABLED_PASSES`, allowing named passes going through `GraphTransformObserver` to be skipped for diagnosis.
- Expected model-level speedup: none directly. It improves compile-time attribution and pass bisection, including for distributed graphs that run bucketing/overlap-related passes.

Benchmarkability:
- Single-process compile-time benchmarks are sufficient to validate trace event overhead and pass attribution.
- Multi-GPU only matters when diagnosing rank-local compile variance or distributed graph pass behavior.

Multi-GPU scenarios:
- Compile a representative FSDP/DTensor training step on each rank with TORCH_TRACE enabled, then inspect pass durations and rank variance.
- Use the disable envvar to bisect a suspected pass-level regression in an overlap/bucketing model compile, comparing compile time and resulting graph behavior.
- Pair with #172510 artifacts to correlate graph-pass time with collective-estimation output.

`torch.cuda.Graph` replay:
- Not applicable. This is compile/trace infrastructure, not GPU runtime timing.

Confidence:
- High for trace/diagnostic impact.
- Low for direct runtime performance impact, by design.

### #173653 - foreach groups in `_pre_bucket_all_gather`

Impact:
- Pre-computes groups by `(src_dtype, dst_dtype, shape)` at trace time and calls `torch._foreach_copy_` per compatible group.
- Avoids slow-path behavior when `_pre_bucket_all_gather` packs heterogeneous tensors for a bucket.
- Expected model-level speedup: workload-dependent. Most likely visible for FSDP/DTensor all-gather buckets composed of many small tensors, mixed dtypes, or shape groups where local packing/casting is exposed. For large buckets dominated by NCCL bandwidth, impact may be small.

Benchmarkability:
- Single-GPU microbenchmarks can isolate local pack/cast overhead in `_pre_bucket_all_gather`.
- Multi-GPU is required for model-level impact because the win depends on how local packing interacts with collective latency, wait placement, and overlap.

Multi-GPU scenarios:
- FSDP model with many small parameters and mixed dtype all-gathers; compare step time, exposed comm time, and bucket count with this optimization enabled versus disabled or emulated by forcing no groups.
- Sweep tensor count per bucket: 8, 32, 128, 512. Sweep same-shape homogeneous dtype, same-shape mixed dtype, and heterogeneous-shape mixed dtype.
- Test single-node NVLink and multi-node IB/EFA separately; local packing may matter more when communication is already well hidden or when buckets are small.
- Include overlap-scheduled training step traces and NCCL traces to distinguish local pre-bucket copy cost from NCCL transfer time.

`torch.cuda.Graph` replay:
- Applies to the isolated local packing microbenchmark if the callable can be captured; replay removes Python/kernel launch overhead and measures steady-state device work.
- Applies to end-to-end compiled steady state only if the model path is actually captured. It should not be used as the only evidence for NCCL overlap behavior.

Confidence:
- High for local packing/copy mechanism.
- Medium for model-level impact; it depends on bucket composition and whether the local pack is exposed on the critical path.

### #183340 - CI tlparse artifact collection and TORCH_TRACE on A100/B200 perf jobs

Impact:
- Fixes tlparse artifact generation by preserving raw gzipped trace logs and invoking `tlparse` per log file instead of passing a directory.
- Enables profiler trace export and TORCH_TRACE for A100 and B200 Inductor perf workflows.
- Expected model-level speedup: none directly. The impact is improved regression forensics and availability of raw trace artifacts from CI perf jobs.

Benchmarkability:
- Best validated as CI/integration infrastructure: confirm perf jobs produce raw trace logs and tlparse HTML artifacts.
- Single-GPU or multi-GPU depends on the perf job being traced; the PR itself is not a model optimization.

Multi-GPU scenarios:
- Run distributed Inductor perf jobs on A100/B200 with TORCH_TRACE enabled and inspect whether each rank produces raw trace logs and tlparse artifacts.
- Use the artifacts to analyze a distributed regression involving compile time, graph passes, collective estimation logs, and runtime kernels.
- Compare trace completeness across H100, A100, and B200 perf jobs after this PR.

`torch.cuda.Graph` replay:
- Not directly applicable. The CI trace infrastructure can capture jobs that use CUDA graphs, but replay timing is not part of this PR.

Confidence:
- High for artifact collection impact from local diff.
- Low for direct model performance impact, because this is observability infrastructure.

## Benchmark Plan Summary

Multi-GPU required:
- NCCL estimator accuracy and overlap validation (#172510).
- `_pre_bucket_all_gather` end-to-end impact under real FSDP/DTensor collectives (#173653).
- Trace/artifact usefulness for distributed perf regressions (#172460, #183340).

Single-GPU sufficient:
- `_pre_bucket_all_gather` local packing/casting microbenchmarks (#173653).
- Perfetto graph-pass event overhead and pass-disable behavior (#172460).
- CI trace artifact smoke tests if the job is not specifically distributed (#183340).

CUDA Graph replay useful:
- Isolated `_pre_bucket_all_gather` packing microbenchmarks, to remove launch overhead and approximate captured steady state.
- End-to-end model timing only when the production/deployment path also uses CUDA graphs.

CUDA Graph replay not central:
- NCCL estimator CSV logging (#172510), graph-pass tracing (#172460), and CI tlparse artifact collection (#183340).

## Caveats

- GitHub PR API/page access was blocked, so this report uses local landed commit dates and commit bodies. Actual GitHub closed or merged timestamps may differ slightly.
- The analysis is read-only and no benchmarks were run.
- PRs outside the corrected distributed/comms/CI trace scope were intentionally excluded even if they are H1 2026 eellison-authored performance work.
