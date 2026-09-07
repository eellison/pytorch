# Eellison H1 2026 Compile-Time, Debug, and Maintenance PR Impact

Scope: `pytorch/pytorch` PRs by `eellison` that landed or closed in H1 2026, filtered to test infrastructure, repro scripts, stack traces, bisector, Perfetto/tlparse, compile-time fast paths, SymPy/sizevar fast paths, lint, CI, and benchmarking infrastructure.

Sources used:

- Local HEAD-reachable git history: `git log --since=2026-01-01 --until=2026-06-30 --author=eellison`
- GitHub PR search: `repo:pytorch/pytorch is:pr author:eellison closed:2026-01-01..2026-06-30`
- Per-PR `gh pr view` metadata for candidate PRs

Raw slice size:

- 64 local H1 commits authored by `eellison`
- 134 GitHub PRs authored by `eellison` closed in H1 2026
- 40 topical candidate PRs after filtering

Notes:

- Date is the landed commit date when available; otherwise the PR closed date.
- "Closed only" means I did not find a mainline/local landed commit or a `Merged` label for that PR. These should not be credited as shipped impact.
- A few ghstack PRs have confusing GitHub state, for example `OPEN` with `Merged`/`Reverted` labels. I used the local commit history plus labels and explicit revert commits where visible.

## Highest-impact takeaways

- Best user-facing compile-time win: #181275 and #181276. #181275 reports a real wide-symbol compile moving from over 50 minutes to about 6 minutes by avoiding catastrophic SymPy polynomial GCD paths; #181276 adds a cheap hint-disproves fast path for repeated `statically_known_true/false` calls on the same class of expressions.
- Best debug/repro win: #181277. It makes after-AOT repro scripts preserve symbolic relationships, which is essential for reproducing the same wide-shape compile-time failures that motivated #181275/#181276.
- Best observability/CI wins: #172460, #172510, #183340, and #183718. These improve graph-pass Perfetto visibility, tlparse collective-estimation artifacts, and perf-job torch trace collection.
- Best test-infra measurement, but de-emphasized for current upstream impact: #181617. The PR body measured 4.0x faster collection and 1.7x faster execution for sampled Inductor tests, but the PR also carries a `Reverted` label and I found a later revert commit in refs.
- Most items are maintenance/debug infrastructure rather than broad end-user runtime performance. Only #181275/#181276, #177783, #183585, and #185951 should be framed as compile-time or overhead performance work, and even those are targeted/niche except #181275.

## PR table

| PR | Date | Status | Category | Impact | How to measure impact | De-emphasize? |
|---:|---|---|---|---|---|---|
| #169863 | 2026-03-08 closed | Closed only, stale | SymPy/test | Attempted test/behavior change around avoiding assertions from SymPy `Mod` simplification. No landed impact found. | If revived, run the linked SymPy `Mod` simplification repro and test that `.is_constant` pessimizes instead of asserting. | Yes. Closed-only maintenance. |
| #170551 | 2026-03-29 closed | Closed only, stale | SymPy robustness | Attempted less aggressive assertions in SymPy `Mod.eval`; intended to make simplification APIs pessimize rather than throw. No landed impact found. | Reproduce issue #163876 and measure assertion/crash rate before/after. | Yes. Closed-only robustness work. |
| #170677 | 2026-03-20 closed | Closed only, stale | Logging | Attempted JSON logging change. No landed impact found and PR body has little detail. | Check emitted logs for machine-parseable JSON and measure downstream parser simplicity/error rate. | Yes. Closed-only maintenance. |
| #170729 | 2026-01-28 landed; PR closed 2026-07-07 | Landed, then explicitly reverted 2026-01-29 | Constant folding compile time | Moved uniform 1-element joint-graph constant folding to CPU to avoid CUDA syncs. Intended to reduce compile latency caused by syncs during graph cleanup, but it was reverted the next day. | Measure joint-graph compile time and CUDA sync count on graphs with many folded uniform CUDA tensors. Current upstream impact should be treated as zero due revert. | Yes for current impact; mention only as attempted compile-time optimization. |
| #172460 | 2026-01-21 landed | Merged | Perfetto/debug | Adds Perfetto trace events around graph pass application and an env var to disable passes. Improves attribution of compile time to FX/Inductor graph passes. | Capture a compile trace and verify per-pass Perfetto events and disabled-pass behavior; compare time spent by pass before/after for slow compiles. | Mostly. Developer-facing diagnostics, not model performance. |
| #172461 | 2026-01-14 landed | Merged | Bisector/debug | Adds CUDA graph application to the compiler bisector. Helps isolate correctness/performance issues caused by cudagraph application. | Run the compiler bisector on a cudagraph-specific failure and compare time/manual steps to isolate the bad stage. | Mostly. Debug tooling. |
| #172508 | 2026-01-14 closed | Closed only, superseded by #172510 | tlparse/debug | Earlier attempt to log NCCL/Inductor collective estimations to tlparse as CSV. Superseded by #172510. | Same as #172510, but do not credit separately. | Yes. Duplicate/closed-only. |
| #172510 | 2026-01-15 landed | Merged | tlparse/debug | Always logs analytical NCCL/Inductor collective estimations to tlparse, even without compile-time benchmarking, and switches the artifact to parseable CSV. | Confirm `fx_collectives_analytical_estimation` appears in tlparse output with valid CSV; validate parser against the regression test. | Mostly. Debug/analysis infrastructure. |
| #172841 | 2026-01-20 landed | Merged | Bisector/debug | Fixes bisector run mode by using a file-based mechanism instead of a hard-cap bisect range. Makes the bisector more usable/reliable on long or dynamic runs. | Run bisector on a known failing compile with run-mode enabled; compare range selection and successful isolation rate. | Yes. Maintenance/debug. |
| #173550 | 2026-01-27 closed | Closed only | Profiler/trace | Proposed `export_chrome_trace` callback API so other subsystems could augment Chrome traces. No landed impact found. | Verify callbacks can mutate exported trace JSON, preserve ordering, and fail non-fatally. | Yes. Closed-only proposed API. |
| #173561 | 2026-01-28 closed | Closed only | Profiler/trace | Later duplicate attempt for the same `export_chrome_trace` callback API. No landed impact found. | Same as #173550. | Yes. Closed-only duplicate. |
| #173585 | 2026-01-28 closed | Closed only | tlparse/benchmarking | Proposed logging all benchmarked collectives/compute nodes to tlparse and loading them as a reproducible overlap-scheduling estimator. No landed impact found. | Run overlap scheduling with benchmark logging enabled, extract tlparse JSON, then replay scheduling with the `BenchmarkEstimator` and compare decisions. | Yes. Closed-only infrastructure. |
| #173586 | 2026-01-28 closed | Closed only | Profiler/benchmarking | Proposed `ProfilerTraceEstimator` to load runtime estimates from Chrome traces for overlap scheduling. No landed impact found. | Feed a PyTorch profiler Chrome trace into the estimator and compare predicted vs measured collective/compute times. | Yes. Closed-only infrastructure. |
| #173664 | 2026-06-20 closed | Closed only, stale | Profiler/trace | Another stale attempt at the `export_chrome_trace` callback API. No landed impact found. | Same as #173550. | Yes. Closed-only duplicate. |
| #173665 | 2026-06-20 closed | Closed only, stale | Profiler/perf analysis | Proposed profiler utilization annotations with FLOPS/bandwidth/roofline metrics and kernel byte-estimation fixes. No landed impact found. | Export Chrome traces and verify achieved FLOPS/bandwidth annotations against known kernels and device bandwidth/TFlops metadata. | Yes. Closed-only analysis tooling. |
| #174706 | 2026-02-11 landed | Merged | Build/CI maintenance | Fixes hipify import behavior for non-HIP CUDA builds in `torch.utils.cpp_extension`. Prevents build/import breakage in CUDA-only environments. | Import `torch.utils.cpp_extension` and build a minimal CUDA extension on a non-HIP CUDA build. | Yes. Build maintenance, not user performance. |
| #175275 | 2026-02-20 landed | Merged | Benchmarking/autotune | Adds CUDA graph benchmarking for `ExternKernelCaller` and `min_speedup_threshold` so custom op autotuning compares choices under cudagraph replay when relevant. Improves autotune decision quality. | Compare selected algorithms and benchmark variance with/without `benchmark_with_cudagraphs`; run `test_min_speedup_threshold` and end-to-end custom-op autotune tests. | Mixed. Infrastructure that can affect user performance indirectly. |
| #175276 | 2026-02-24 landed | Merged | Benchmarking/autotune | Cleans up cuBLAS workspaces allocated during CUDA graph benchmarking to avoid memory retention/leaks during autotuning. | Run repeated cudagraph autotune benchmarks and track CUDA memory/workspace lifetime before/after. | Mostly. Maintenance for benchmark reliability. |
| #176345 | 2026-04-14 landed | Merged | Sizevar/index simplification | Simplifies `FloorDiv(ModularIndexing(...))`, generalizes zero-term removal, and adds base-less-than-divisor simplification. Produces cleaner index expressions for reshape/reduction patterns. | Run `test/inductor/test_indexing.py`; measure generated index expression size and compile/codegen time on reshape-plus-reduction repros. | No for compile-time/codegen quality, but it is targeted rather than broad. |
| #176953 | 2026-03-16 landed | Merged | Stack traces/debug | Uses output stack traces on output nodes instead of intermediates, improving cudagraph/tree error attribution. | Run `test/inductor/test_cudagraph_trees.py`; inspect stack trace source locations on graph outputs and cudagraph runtime errors. | Mostly. Debuggability, not speed. |
| #177783 | 2026-03-30 landed | Merged; earlier version reverted and relanded | Runtime overhead/asserts | Defers generated input size/stride assertions until first use so CPU overhead before the first kernel is reduced or hidden by GPU work. | Measure wrapper CPU time before first kernel and end-to-end latency for graphs with many inputs/asserts; verify generated wrapper assert placement. | No if discussing overhead performance; not compile-time. |
| #177784 | 2026-06-17 closed | Closed only, stale | Runtime overhead/asserts | Proposed skipping redundant subgraph input asserts already checked by the parent graph. No landed impact found. | Measure generated subgraph wrapper overhead and assert count on nested/subgraph-heavy Inductor outputs. | Yes. Closed-only follow-up. |
| #177789 | 2026-03-19 closed | Closed only, superseded by #177783 | Runtime overhead/asserts | Earlier closed attempt at deferring input size/stride assertions. Superseded by #177783. | Same as #177783. | Yes. Duplicate/closed-only. |
| #177790 | 2026-03-19 closed | Closed only, related to #177784 | Runtime overhead/asserts | Earlier closed attempt at skipping redundant subgraph asserts. No landed impact found. | Same as #177784. | Yes. Closed-only follow-up. |
| #177890 | 2026-06-30 closed | Closed only, stale | Stack traces/debug | Proposed broader cudagraph stack trace coverage, a `cudagraph_assert_stack_traces` config, and tests for graph outputs from user code. No landed impact found. | Enable the assertion config and run cudagraph tree tests with custom graph passes; compare missing-stack-trace rate. | Yes. Closed-only debug tooling. |
| #178802 | 2026-03-30 landed | Merged | Test infrastructure | Turns off shape padding for opinfo tests to avoid flaky stride issues while a more general fix was pending. | Track opinfo flake rate before/after and run `test/inductor/test_torchinductor_opinfo.py`. | Yes. Test stabilization. |
| #180340 | 2026-04-14 closed | Closed only | Codegen/symbolic fast path | Proposed short-circuiting `ops.masked` for statically true/false masks. PR body says it was a no-op until sub-range iteration infrastructure existed. No landed impact found. | Once sub-range iteration exists, measure generated code size and eliminated masked ops on relevant fusion patterns. | Yes. Closed-only/enabling hook. |
| #181275 | 2026-05-04 landed | Merged | SymPy/sizevar compile time | Skips polynomial `sympy.gcd` on very wide shape expressions and adds width-based bailouts in hot symbolic reasoning paths. PR body reports a backward compile improving from over 50 minutes to about 6 minutes. | Re-run the wide-concat/64-plus-symbol backward compile repro; profile time in `sympy.gcd`, `statically_known_*`, and `remove_zero_terms`. | No. This is the clearest user-facing compile-time performance win. |
| #181276 | 2026-05-04 landed | Merged | Sizevar compile time | Adds hint-disproves fast path to `statically_known_true/false`, avoiding expensive full SymPy reasoning when current hints already refute the claim. | Count calls/time in `_maybe_evaluate_static` on wide-symbol repros; compare scheduler/codegen compile time with #181275 alone vs #181275 plus #181276. | No. User-facing compile-time fast path. |
| #181277 | 2026-05-18 landed | Merged; first landing reverted, later relanded | Repro scripts/debug | Preserves symbolic relationships in after-AOT repro scripts by serializing SymInt expressions and rebuilding symbolic wrappers. Makes compile-time pathologies from derived wide expressions reproducible. | Generate an after-AOT repro for a derived-symbol graph, rerun with `tracing_mode="symbolic"`, and verify the FX graph keeps derived expressions instead of independent symbols. | Mostly. Debug/repro infrastructure, but high leverage for compile-time bugs. |
| #181617 | 2026-05-18 landed locally | Merged label but also `Reverted`; explicit revert found in refs | Test infrastructure | Speeds Inductor tests via ISA load caching, lazy imports, shared Triton cache, and reduced test autotune configs. PR body measured import 19.0s to 4.4s, collection 20.9s to 5.3s, 22 GPU tests 51.7s to 30.2s, and estimated about 15 minutes CI wall-clock saved. Current upstream impact should be de-emphasized due revert. | Repeat the PR body measurements: import time, pytest collection for 2106 tests, selected 22 GPU tests, flex attention test, and CI aggregate wall time. | Yes for current impact; strong maintenance measurement but reverted. |
| #183340 | 2026-05-13 landed | Merged | CI/tlparse | Fixes tlparse artifact collection by saving gzipped raw trace logs and running tlparse on individual log files; enables torch trace on A100/B200 perf nightly workflows. | Inspect perf job artifacts for raw logs and non-empty tlparse HTML zips; verify A100/B200 jobs set trace envs. | Yes. CI/debug infrastructure. |
| #183585 | 2026-05-13 landed | Merged | Compile-time cache | Caches scheduler coalescing analysis on scheduler nodes so repeated tiling decisions reuse normalized read/write analysis while invalidation still works after loop rewrites. | Profile nested-reduction scheduler/codegen paths and count coalescing-analysis recomputations before/after. | Mostly no for compile-time impact, but niche to scheduler-heavy cases. |
| #183632 | 2026-05-13 landed | Merged | Lint | Fixes set-linter nested f-string parsing by separating f-string nesting state from token-index stack state; adds a regression case. | Run the set-linter tests and a lint pass on files with nested f-strings. | Yes. Maintenance. |
| #183718 | 2026-05-14 landed | Merged | CI/tlparse | Fixes `ENABLE_TORCH_TRACE` propagation on OSDC H100/B200 runners and updates `perf_cli.py` for `test-osdc` job names/S3 artifact paths. | Trigger or inspect H100/B200 OSDC perf jobs and verify trace artifacts are collected/discoverable by `perf_cli.py`. | Yes. CI/debug infrastructure. |
| #184815 | 2026-05-22 closed | Closed only | Benchmarking | Attempted to serialize Inductor GPU benchmark timing. No landed impact found and PR body lacks detail. | If revived, measure benchmark variance/contention with concurrent benchmark workers before/after serialization. | Yes. Closed-only benchmark maintenance. |
| #184821 | 2026-05-21 landed | Merged | Test infrastructure | Enables loop ordering in nested-reduction tests to match expected dependency form without changing fbcode production default. | Run `test/inductor/test_nested_reduction.py` and compare generated-code expectation stability. | Yes. Test maintenance. |
| #184905 | 2026-06-02 landed in GH refs | Merged | Benchmarking infrastructure | Adds process-local GPU benchmark lock hooks so external harnesses can coordinate Inductor GPU benchmark sections, including CUDA graph benchmarking and profiling-based benchmarking. | Run multi-process benchmark harnesses with a registered lock context; measure contention failures, variance, and throughput. | Mostly. Benchmark infrastructure, indirect performance relevance. |
| #185845 | 2026-06-01 closed | Closed only | Benchmarking/dynamic shapes | Proposed converting SymInt scalar args to concrete ints when benchmarking collectives, fixing `SymInt::expect_int` in dynamic-shape `all_to_all_single` benchmark paths. No landed impact found. | Run the 4-GPU dynamic-shape `all_to_all_single` repro with `collective_estimator="benchmark"` and overlap scheduling. | Yes unless revived; closed-only correctness for benchmark path. |
| #185951 | 2026-06-02 landed in GH refs | Merged | Compile-time/logging overhead | Lazily formats missing-op fallback INFO logs so expensive diagnostic strings are built only when INFO logging is emitted. Avoids repeated large IR formatting when logging is disabled. | Run the custom-op fallback repro and `CpuTests.test_missing_op_info_log_is_lazy_cpu`; measure time spent in `operator_str` with INFO disabled. | No for compile-time overhead on affected custom-op graphs; otherwise targeted. |

## Suggested emphasis

Emphasize as user-facing compile-time/performance:

- #181275
- #181276
- #183585, with the caveat that it is scheduler/nested-reduction specific
- #185951, with the caveat that it affects custom-op fallback logging cases
- #177783 if runtime wrapper overhead is in scope

Emphasize as high-leverage debug/observability:

- #181277
- #172460
- #172510
- #176953
- #183340
- #183718

De-emphasize as maintenance, closed-only, duplicate, or reverted:

- Reverted/current-impact caveat: #170729, #181617
- Closed-only/stale/prototype: #169863, #170551, #170677, #172508, #173550, #173561, #173585, #173586, #173664, #173665, #177784, #177789, #177790, #177890, #180340, #184815, #185845
- Maintenance/test/lint/CI: #174706, #175276, #178802, #183632, #184821

