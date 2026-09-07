# eellison H1 2026 PR impact summary

## Source baseline

This synthesis uses the provided subagent outputs under `agent_space/`, with
`agent_space/eellison_h1_2026_pr_inventory.md` as the inventory baseline.

Confirmed inventory counts:

- 137 GitHub/local PR-numbered rows.
- 68 rows with a landed or merged signal: 63 high-confidence rows plus 5
  medium-confidence rows.
- 69 closed-only rows without landed proof.
- 0 H1 landed commits without a PR number in the inventory baseline.

Source caveats:

- PyTorch ghstack PRs often have sparse GitHub `merged_at`; the inventory uses
  GitHub merged/label data plus local `origin/main` first-parent proof.
- Local fallback is `origin/main`, not the detached/current `HEAD`, because the
  checkout has non-main work.
- The local `origin/main` snapshot ends at 2026-06-23. Late-June PRs without
  local proof rely on GitHub metadata and are lower confidence when the
  inventory marks them that way.
- Several subagent slices used current reachable `HEAD` rather than the
  inventory's `origin/main` baseline, and some GitHub metadata fetches failed
  through the local proxy. Treat those as useful impact/benchmark context, not
  as changes to the inventory counts.
- The nested-reduction impact slice mentions PR #183638 and two no-PR
  current-HEAD commits for sub-parent epilogues. They are useful benchmark
  context, but they are not credited in the inventory count unless separately
  confirmed against the PR inventory/mainline source of truth.
- A `Reverted` label is not automatically zero impact because some ghstack
  changes relanded. Explicitly de-emphasized current-upstream impact from the
  subagent reports: #170729 and #181617.

## Ranked landed impact clusters

1. Loop reindexing, loop ordering, and pointwise cat fusion:
   #176927, #179090, #179091, with #176345 as enabling index simplification and
   #182890 as related fusion-state reliability. This is the clearest naturally
   surfaced non-nested runtime-impact cluster. It turns RMSNorm/qknorm/reshape,
   grouped quantization, and RoPE cat patterns from multi-kernel graphs into
   one fused Triton kernel in representative cases.

2. Nested reductions for RMSNorm/LayerNorm plus grouped quantization:
   #182891, #182892, #182893, #182895, #182896, #183432, #182897, #182898, plus
   low-precision enablers #177922 and #172497. This stack adds scheduler
   legality, derived range roots, autotune block-size floors, codegen lowering,
   XBLOCK coverage, inline PTX, and MX E8M0 conversion support. The strongest
   shipped effect is launch/materialization reduction for RMSNorm -> grouped
   amax/scale/quant patterns; runtime wins are shape-dependent in the current
   microbenchmarks.

3. Compile-time symbolic reasoning and repro quality:
   #181275 and #181276 are the highest-impact compile-time wins. #181275
   reported a wide-symbol backward compile improving from over 50 minutes to
   about 6 minutes by avoiding catastrophic SymPy polynomial GCD. #181276 adds
   a hint-disproves fast path for `statically_known_true/false`. #181277 makes
   after-AOT repros preserve symbolic relationships, which is high leverage for
   reproducing these failures. #183585 and #185951 add targeted compile-time
   overhead reductions in scheduler coalescing analysis and missing-op logging.

4. CUDA graph runtime, policy, and benchmarking correctness:
   #175275, #175276, #175271, #174103, #176620, #182524, #182531, #188117, and
   #188078 improve cudagraph benchmarking decisions, workspace cleanup,
   graph-partition thresholds, opt-out controls, saved-activation handling,
   self-overlap support, pointer-churn rerecord behavior, and live-output
   cloning. These are important reliability/performance enablers, but many need
   pre/post commit checkout benchmarks to isolate their effect.

5. Autotuning and generated-kernel performance controls:
   #175275, #175277, #175278, #175422, #174232, #182895, #184905. This includes
   CUDA graph benchmarking for extern/custom choices, scoped config patches,
   custom-op autotuning applied to `aten.mm`, fallback normalization, combo
   kernel PDL, block-size floors, and GPU benchmark lock hooks. Impact is mostly
   indirect through better choice selection and benchmark stability.

6. Distributed/comms and trace observability:
   #173653 is the main runtime candidate, grouping `_pre_bucket_all_gather`
   copies via foreach to reduce local packing overhead. #172510 logs collective
   estimates to tlparse CSV, #175204 adds compute-estimator selection for
   overlap scheduling, and #172460/#183340/#183718 improve graph-pass and perf
   trace artifacts. These are high-leverage for distributed diagnosis; only
   #173653 is a direct runtime optimization, and it needs multi-GPU validation.

7. Targeted correctness, numerics, and maintenance:
   Examples include #185847, #185840, #177546, #181793/#181795, #182139/#182178,
   #172951, #175203, #175281, #174706, #183632, #178802, and #184821. These
   preserve dtype semantics, decomposition numerics, dynamic tiling, stream
   naming, attention selection, fusion crash behavior, build/lint/test health,
   and test expectation stability. They matter, but they are not broad
   performance-impact leaders.

## Benchmark evidence: nested reductions, NVFP4, MXFP8, RMSNorm

Methodology from `eellison_h1_2026_nested_benchmarks.md`:

- Each case compiled twice with `torch.compile(..., fullgraph=True,
  dynamic=False)`.
- Compared `torch._inductor.config.patch({"triton.nested_reduction": False})`
  against the same patch set to `True`.
- Used fresh compile caches under `agent_space/.bench_cache_h1_2026_nested/`.
- Timing used 20 warmup calls, captured one compiled callable invocation inside
  `torch.cuda.CUDAGraph`, then measured 1000 graph replays with CUDA events.
- Environment: B200, capability 10.0, CUDA 12.8, Triton 3.6.0, imported torch
  `2.13.0a0+gite1067a5`.

Findings:

- RMSNorm block amax/scale, `B=128,D=4096,G=16`: kernels 2 -> 1,
  `codegen_nested_reduction` 0 -> 1, 8.58 us -> 8.54 us. This confirms the core
  #182897 launch/materialization reduction, with neutral runtime at this shape.
- RMSNorm NVFP4 pack, `B=128,D=4096,G=16`: kernels 3 -> 1, nested metric
  0 -> 1, 8.54 us -> 8.51 us. This validates the intended single-kernel
  RMSNorm/amax/scale/inline-asm pack shape on SM100-class hardware, but PR
  attribution to #183638 is outside the inventory baseline.
- RMSNorm MXFP8-style path, `B=128,D=4096,G=32`: kernels 3 -> 2, nested metric
  0 -> 1, 14.08 us -> 14.18 us. This is an Inductor-expressible MXFP8-style
  microbenchmark, not proof of a final production swizzled MXFP8 pack.
- Half-resolution sub-parent epilogue, `B=128,D=4096,G=16`: kernels 2 -> 1,
  8.54 us -> 8.35 us. Useful context, but the enabling commits were no-PR
  current-HEAD commits in the nested slice and should not be credited as
  inventory landed PR impact without confirmation.
- Weighted RMSNorm reduce-K, chunk SwiGLU, and chunk4 gated consumers all
  collapsed from 2 kernels to 1 with `codegen_nested_reduction` incrementing.
  Runtime was essentially flat around the measured shapes.

Bottom line: the nested stack clearly proves codegen coverage and launch-count
reduction. The current graph-replay microbenchmarks do not yet prove a large
steady-state kernel-body speedup on B200 for the selected static shapes.

## Benchmark evidence: non-nested loop/fusion/autotune

Methodology from `eellison_h1_2026_loop_fusion_autotune_benchmarks.md`:

- Fresh `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` under `agent_space/`.
- `fx_graph_cache=False`, `benchmark_kernel=True`, and
  `triton.unique_kernel_names=True`.
- Correctness checked against eager, Inductor generated-kernel counts recorded,
  generated Triton defs/launches counted, and timing collected with CUDA events.
- Direct timing used 10 warmups and 5 reps of 50 iterations. CUDA graph replay
  captured one static compiled invocation and replayed with CUDA events.
- Environment: B200, source HEAD `d59fc87a47b`, imported torch
  `2.13.0a0+gite1067a5`.

Findings:

- #179090 loop ordering: outer sum plus pointwise over `x=[32,2^20]` fused
  2 -> 1 kernels and improved direct time 0.1080 -> 0.0804 ms (1.34x) and graph
  replay 0.1048 -> 0.0800 ms (1.31x). Outer softmax also fused 2 -> 1 and
  improved 1.06x direct, 1.12x replay.
- #176927 loop reindexing: RMSNorm reshape slice over bf16 `[16,8192]` from a
  wider qkv buffer fused 2 -> 1 kernels, direct 0.0248 -> 0.0203 ms (1.22x),
  graph replay 0.00416 -> 0.00230 ms (1.81x). The transposed variant also
  fused 2 -> 1 and improved direct time 1.20x, with nearly flat replay time.
- #179091 pointwise cat: QKNorm + split RoPE cat fused 2 -> 1 kernels and
  improved direct time 0.0388 -> 0.0232 ms (1.67x); replay was flat at this
  small shape. The interleaved stack/flatten variant improved direct time 1.39x
  but replay regressed for the small case, so shape selection matters.
- #174232 combo kernels: six independent pointwise ops collapsed 6 -> 1 kernels
  and improved direct time 1.49x, graph replay 1.25x. A mixed pointwise/reduction
  restriction and a synthetic PDL pointwise case regressed, so PDL needs a more
  representative dependent-launch benchmark.
- #175271 cudagraph partition threshold: small matmul chain direct time improved
  0.0386 -> 0.0318 ms (1.21x). External graph recapture is not always valid when
  Inductor already captures internally.
- #182895/#183585 coalescing surface: one case was flat; one permute/clone/amax
  replay improved 0.0123 -> 0.00620 ms, but exact attribution needs pre/post
  commits.
- #175275/#175276/#175277/#175278/#175422 custom-op autotune path: API and CUDA
  graph benchmarking path were stable, with graph replay around 0.0028 ms and
  logs selecting a faster decomposition than fallback. This verifies behavior
  more than it proves a landed pre/post speedup.

Bottom line: the non-nested loop reindexing/order/cat cluster has the strongest
measured runtime evidence in the provided artifacts.

## Closed-only and no-landed-proof PRs

Closed-only/no-landed-proof rows are excluded from the landed impact ranking.
There are 69 such rows in the inventory. They can show design exploration or
superseded attempts, but they should not be credited as shipped PR impact unless
new landed proof is found.

Representative closed-only groups:

- Superseded attempts: #172508 -> #172510, #177789 -> #177783, #173554/#173545
  -> #173653, #172494 -> #172497, #184814 -> #184904.
- Inline asm WIP series before #177922: #177886, #177913, #177915, #177916,
  #177917, #177919, #177920, #177921, #175814.
- Profiler/export/utilization prototypes: #173550, #173561, #173664, #173665,
  #173585, #173586.
- Cudagraph/live-output/sym-shape prototypes without inventory landed proof:
  #188118, #178895, #178915, #175288, #173966.
- SymPy/indexing/debug prototypes: #169863, #170551, #180340, #180339,
  #184810, #184811.
- Distributed/benchmark/fuzz prototypes: #185845, #175803, #175812, #175799,
  #184815.

Reverted/current-impact caveat:

- #170729 and #181617 had meaningful reported intent or measurements, but the
  compile/debug slice found explicit revert evidence and recommends treating
  current upstream impact as zero or de-emphasized.

## Next benchmark gaps needing pre/post checkout

- #184904: compare parent of `eeccb459bcc` against the landed commit on an FX
  graph where unsupported producers feed multiple supported consumers; report
  partition count, max partition size, and runtime if wired into compile.
- #184905: register a real external GPU benchmark lock hook and run concurrent
  Inductor autotuning workers; measure variance and wall time with and without
  the hook.
- #182895/#183585: isolate block-size-floor propagation and coalescing-analysis
  caching with parent-vs-landed commits. Current-head toggles exercise the
  surface but do not separate the two PRs.
- #182524/#182531/#188117/#188078: use `torch.compile(..., mode="reduce-overhead")`
  workloads for self-overlapping inputs, parameter pointer churn, and live user
  outputs; report rerecord counts, graph ids, clones/allocations, and steady
  replay time before and after each PR.
- #177783/#179039/#180599: measure CPU time before first kernel launch,
  first-iteration latency, and steady-state CUDA time for many-input graphs and
  misaligned-copy graphs. There is no stable current-head off toggle.
- #176345/#181275/#181276: use wide-symbol after-AOT repros with 50+ symbolic
  terms; measure compile wall time and time in SymPy/static reasoning on
  parent-vs-landed commits. CUDA graph replay is not the right surface.
- #173653: run both single-GPU local packing microbenchmarks and real multi-GPU
  FSDP/DTensor all-gather bucket scenarios; sweep tensor counts, shapes, dtypes,
  and single-node vs multi-node interconnect.
- #172510/#175204: validate estimator quality and overlap scheduling on real
  multi-GPU collective traces; single-GPU only proves artifact formatting.
- #172497/#177922/#183638: run SM100+ primitive and production-flow benchmarks
  for E8M0 conversion and NVFP4 packing. For #183638 specifically, first confirm
  inventory/mainline landed proof because it was not present in the PR inventory.
- No-PR sub-parent nested epilogue commits from the nested slice: keep benchmark
  results as context only until a PR number or mainline landed proof is found.
