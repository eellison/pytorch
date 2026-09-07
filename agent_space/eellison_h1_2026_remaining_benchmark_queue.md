# eellison H1 2026 remaining benchmark queue

Generated: 2026-07-13

This queue is based only on the existing subagent outputs under `agent_space/`.
No benchmarks were run while preparing this file.

## Source status

Inventory baseline:
- `eellison_h1_2026_pr_inventory.md` reports 137 PR-numbered rows, 68 with a
  landed or merged signal, and 69 closed-only rows without landed proof.
- Closed-only rows are not queued for shipped-impact benchmarking unless they
  are needed as superseded context for a landed PR.
- #183638 and the no-PR sub-parent nested epilogue commits are useful benchmark
  context, but they are not inventory-credited landed PRs in the provided
  baseline.
- #170729 and #181617 are de-emphasized for current upstream impact because the
  subagent reports found revert evidence.

Status terms used below:
- Benchmark done: a subagent actually ran a benchmark and reported timing or
  kernel/codegen metrics.
- Analyzed only: a subagent identified the impact and proposed a benchmark, but
  did not run it.
- Current-HEAD toggle enough: the next useful benchmark can be run on current
  HEAD with a config/API knob that approximates old versus new behavior.
- Pre/post required: current HEAD can exercise the surface, but parent versus
  landed checkouts are required to attribute impact to the PR.

## Already benchmarked

### Nested reductions and low-precision-adjacent nested paths

Benchmark done with current-HEAD `torch._inductor.config.patch({"triton.nested_reduction": False/True})`
and CUDA graph replay timing:

| PRs or context | Case | Shape | Result status |
| --- | --- | --- | --- |
| #182891, #182892, #182893, #182895, #182896, #183432, #182897, #182898 | RMSNorm block amax/scale | `x=(128,4096)` bf16, `weight=(4096)` bf16, `group=16` | Kernels `2->1`, nested metric `0->1`, runtime flat at about `8.58->8.54 us`. |
| #177922 plus nested stack; #183638 context only | RMSNorm NVFP4-style pack | `x=(128,4096)` bf16, `group=16` | Kernels `3->1`, nested metric `0->1`, runtime flat at about `8.54->8.51 us`; #183638 landed proof is not in inventory. |
| #172497 plus nested stack | RMSNorm MXFP8-style path | `x=(128,4096)` bf16, `group=32` | Kernels `3->2`, nested metric `0->1`, runtime slightly slower at about `14.08->14.18 us`; not a production swizzled MXFP8 pack. |
| No-PR current-HEAD nested context | Half-resolution sub-parent epilogue | `x=(128,4096)` bf16, `group=16` | Kernels `2->1`, runtime `8.54->8.35 us`; do not credit without PR/mainline proof. |
| Nested stack | Weighted RMSNorm reduce-K | `x=(64,16,4096)` bf16, `w=(64,16)` bf16 | Kernels `2->1`, runtime flat around `54.8 us`. |
| Nested stack | Chunk SwiGLU consumer | `x/residual=(128,1024)` bf16, `weight=(1024)` bf16 | Kernels `2->1`, runtime flat around `8.35 us`. |
| Nested stack | Chunk4 gated consumer | `x/residual=(128,1024)` bf16, `weight=(1024)` bf16 | Kernels `2->1`, runtime flat around `8.3 us`. |

Remaining nested gap: production MXFP8/NVFP4 packing and pre/post attribution for
low-precision primitives are still open; see P1-4.

### Non-nested loop, fusion, PDL, and autotune surfaces

Benchmark done with current-HEAD toggles/APIs and CUDA event timing:

| PRs | Case | Shape | Result status |
| --- | --- | --- | --- |
| #179090 | Outer sum plus pointwise | `x=(32, 2^20)` | Kernels `2->1`; direct `0.1080->0.0804 ms`; graph replay `0.1048->0.0800 ms`. |
| #179090 | Outer softmax | `x=(32, 2^20)` | Kernels `2->1`; direct `1.06x`; graph replay `1.12x`. |
| #176927 | RMSNorm reshape slice | bf16 `x=(16,8192)` from wider contiguous qkv | Kernels `2->1`; direct `1.22x`; graph replay `1.81x`. |
| #176927, #179090 | RMSNorm reshape transposed | bf16 `x=(16,8192)`, stride `(1,16)` | Kernels `2->1`; direct `1.20x`; replay nearly flat. |
| #179091 | QKNorm plus split RoPE cat | `B=4,H=8,S=128,D=64` | Kernels `2->1`; direct `1.67x`; replay flat. |
| #179091 | QKNorm plus interleaved RoPE stack/flatten | `B=4,H=8,S=128,D=64` | Kernels `2->1`; direct `1.39x`; replay regressed on this small shape. |
| #174232 | Combo kernels, six independent pointwise ops | `6 * (2^22,)` | Kernels `6->1`; direct `1.49x`; graph replay `1.25x`. |
| #174232 | Combo pointwise-only restriction | `[1024]` and `[32,1024]` mixed pointwise/reduction | Regressed; useful negative coverage. |
| #174232 | Synthetic combo PDL | `6 * (2^20,)` | Emitted `launch_pdl=True`, but regressed on B200 synthetic case. |
| #175271 | Cudagraph partition threshold | small matmul chain, `x=(128,128)` | Direct `0.0386->0.0318 ms`; external graph replay was not valid for one side because Inductor already captured internally. |
| #182895, #183585 | Coalescing runtime surface | `4096x4096` contiguous/transposed add | Runtime flat. |
| #182895, #183585 | Coalescing runtime surface | bf16 permute/clone/amax `[1024,2048]->[2048,1024]` | Direct flat; graph replay `0.0123->0.00620 ms`; exact PR attribution still open. |
| #175275, #175276, #175277, #175278, #175422 | Custom-op autotune API path | fp16 `x=(8,128,256)`, `weight=(256)` | API stable; replay about `0.0028 ms`; this verifies behavior more than landed pre/post speedup. |

These are the strongest completed runtime results. The remaining queue should not
re-benchmark them first unless a larger/model-level shape is specifically needed.

## Analyzed only, not benchmarked

The subagent reports analyzed but did not benchmark these shipped-impact areas:
- CUDA graph policy and runtime behavior: #182524, #182531, #188117, #188078,
  #176620, #174103.
- Compile-time symbolic fast paths and repro preservation: #176345, #181275,
  #181276, #181277.
- Distributed local packing and real collective overlap: #173653, #172510,
  #175204.
- Pre/post attribution for tiling/coalescing: #182895, #183585.
- Partitioning, benchmark-lock, and wrapper-overhead PRs: #184904, #184905,
  #177783, #179039, #180599.
- Low-precision primitive and production pack throughput: #172497, #177922,
  #183638 context only.
- Targeted compile/logging overhead and observability: #185951, #172460,
  #183340, #183718.

The rest of the landed inventory is mostly correctness, diagnostics, CI, or test
maintenance. It is not first-order benchmark queue material unless a separate
correctness-validation matrix is desired.

## Remaining benchmark queue

### P0-1: Wide-symbol compile-time fast paths

PRs/clusters:
- #181275: skip catastrophic polynomial `sympy.gcd` on very wide shape
  expressions.
- #181276: hint-disproves fast path for `statically_known_true/false`.
- #176345: prerequisite-style index simplification for
  `FloorDiv(ModularIndexing(...))` and zero-term removal.
- #181277: after-AOT repro preservation needed to make the wide-symbol cases
  reproducible.

Why it matters:
- This is the clearest reported user-facing compile-time win. The compile/debug
  slice reports a real backward compile improving from over 50 minutes to about
  6 minutes for #181275.
- These PRs affect dynamic-shape and wide-symbol workloads where normal
  microbenchmarks miss the cost.

Toggle or checkout:
- Pre/post required for attribution. Current HEAD has no stable off toggle for
  #181275 or #181276.
- Current HEAD is useful for generating and validating an after-AOT repro with
  #181277, then replaying the same repro across parent and landed commits.

Benchmark shape/model:
- Generate an after-AOT repro with 50 to 70 symbolic terms and derived symbolic
  relationships, including nested `FloorDiv`, `ModularIndexing`, and wide
  additive expressions.
- Use a backward graph with wide concat/slice or reshape/reduction indexing so
  `remove_zero_terms`, `safe_gcd`, and `statically_known_true/false` are hot.
- Record end-to-end compile wall time, time in `sympy.gcd`, time in
  `statically_known_true/false`, call counts, generated code size, and whether
  derived symbolic relationships survived in the repro.

`torch.cuda.Graph` replay:
- Not applicable. This is compile-time CPU/symbolic work, not steady-state GPU
  replay.

Blockers:
- Need a stable after-AOT repro or the original wide-symbol model.
- Parent/landed checkouts are required. If changing commits invalidates the
  local build, rebuild only with `pip install -e . -v --no-build-isolation` per
  repo rules.
- Runs may be long on the pre-PR side; isolate caches and cap the first attempt.

### P0-2: CUDA graph policy, rerecord, and live-output behavior

PRs/clusters:
- #182524: self-overlapping inputs and storage-copy fallback.
- #182531: do not count normal parameter pointer churn against the unexpected
  rerecord limit.
- #188117 and #188078: live user output cloning and opt-in tree cloning.
- #176620: saved activations should not be marked static in partitioned
  backward.
- #174103: graph-level cudagraph opt-out annotations.

Why it matters:
- These determine whether `torch.compile(..., mode="reduce-overhead")` keeps
  using CUDA graphs in realistic training/inference loops.
- The risk is not just runtime: skipped capture, excessive rerecords, or unsafe
  live outputs can silently erase expected reduce-overhead wins.

Toggle or checkout:
- Pre/post required for shipped-impact attribution. Current HEAD configs can
  validate behaviors, but they are not equivalent old/new toggles.
- Use current HEAD first only to stabilize the harness and expected counters.

Benchmark shape/model:
- Self-overlap: `base=(1,1024)` CUDA tensor, `x=base.expand(64,1024)`, then
  pointwise and `mm`/reduction consumers under `mode="reduce-overhead"`. Scale
  to `base=(1,4096)` and `x=(256,4096)` for timing.
- Parameter pointer churn: compile `fn(x, p): return (x * p).sin().sum()` with
  `x=(1024,1024)` and pass a fresh `nn.Parameter` or fresh parameter storage
  each step. Set `triton.cudagraph_unexpected_rerecord_limit=1` and report
  rerecords/skips.
- Live output cloning: compile a function returning a tensor that remains live
  in user code, then run later compiled steps that would otherwise alias or
  mutate graph-managed memory. Use `x=(1024,1024)` and a smaller `(4,4)` smoke
  case for allocation/counter clarity.
- Saved activations: training graph with a partitioned forward and backward,
  `x=(1024,1024)` requiring grad, plus a graph break or unsupported op to force
  forward partitioning. Track backward graph rerecords.
- Opt-out annotations: `x=(1024,1024)` segmented graph with one unsafe segment
  annotated by `torch._dynamo.override_cudagraphs(fwd=..., bwd=...)`; verify
  only the intended segments skip capture.

`torch.cuda.Graph` replay:
- Internal Inductor CUDA graph behavior is the benchmark surface.
- External `torch.cuda.CUDAGraph` replay is often invalid for
  `mode="reduce-overhead"` and should not be the only timing. Report direct
  steady-state CUDA event timing plus Inductor graph ids, rerecord counts,
  skip reasons, clone/allocation counts, and replay time when available.

Blockers:
- CUDA GPU required.
- True attribution needs parent/landed checkouts; #188078 has only medium
  inventory confidence in this checkout and may need the GitHub commit rather
  than `origin/main`.
- Some cases may be Python-only changes, but after checkout use the repo's
  standard editable build command if the import/build is stale.

### P0-3: Distributed pre-bucket all-gather local packing and model impact

PRs/clusters:
- #173653: foreach groups optimization in `_pre_bucket_all_gather`.
- Superseded closed-only context: #173545, #173554.

Why it matters:
- This is the main direct distributed runtime optimization in the provided
  reports.
- It can reduce local pack/cast overhead for buckets with many same-shape or
  same-dtype tensors before NCCL all-gather.

Toggle or checkout:
- Current HEAD is enough for a local mechanism benchmark if the harness compares
  grouped foreach copy against `groups=None` or an equivalent ungrouped path.
- Pre/post required for landed attribution and for end-to-end FSDP/DTensor
  impact.

Benchmark shape/model:
- Local microbenchmark: call `_pre_bucket_all_gather` on tensor lists with
  counts `8, 32, 128, 512`.
- Sweep bucket composition:
  - homogeneous: all tensors `(1024,)` fp16.
  - same-shape mixed dtype: alternating `(1024,)` fp16/bf16/fp32.
  - heterogeneous: `(10,)` fp32, `(20,)` fp16, `(10,)` fp32 from the unit-test
    shape, scaled to dozens or hundreds of tensors.
  - larger bucket: tensors totaling 16 MB, 64 MB, and 256 MB.
- End-to-end: FSDP or DTensor transformer block with many small parameters and
  mixed dtype all-gather buckets. Sweep world sizes 2, 4, and 8 on one node;
  add multi-node if available.
- Report local packing time, foreach copy kernel count, NCCL time, exposed comm
  time, overlap gaps, step time, bucket count, and trace artifacts.

`torch.cuda.Graph` replay:
- Useful for the isolated local packing microbenchmark if the callable can be
  captured.
- Not sufficient for end-to-end collective overlap; use normal compiled steady
  state and traces as primary evidence.

Blockers:
- Multi-GPU required for model-level evidence. Single GPU only proves the local
  packing mechanism.
- Multi-node validation is useful because local packing importance changes with
  interconnect.
- Pre/post checkout needed for attribution; use the standard editable build if
  the checkout switch requires it.

### P1-1: Tiling/coalescing attribution

PRs/clusters:
- #182895: thread block-size floors through Triton autotuning.
- #183585: cache scheduler coalescing analysis.

Why it matters:
- Current benchmarks exercised the surface and found one replay win, but they
  did not separate block-size-floor config validity from compile-time caching.
- This cluster affects nested reductions, reductions with constrained block
  sizes, and scheduler-heavy codegen.

Toggle or checkout:
- Current-HEAD `triton.coalesce_tiling_analysis` is enough for more surface
  exploration.
- Pre/post required to attribute #182895 versus #183585 separately.

Benchmark shape/model:
- #182895 runtime/config validity: reductions whose candidate configs hit
  `min_rblock=16`, `128`, and `1024`, plus `min_xblock=128`. Include a
  nested/block-floor case with `x=(128,4096)` bf16 and grouped reduction
  `group=16/32`.
- #183585 compile-time caching: coalescing-heavy shapes from the runtime impact
  report: `(128,384,196)`, `(768,64,196)`, and fused
  `(128,6,64,196)`. Enable `loop_ordering_after_fusion=True` and
  `triton.coalesce_tiling_analysis=True`.
- Continue the existing runtime surface: bf16 permute/clone/amax
  `[1024,2048]->[2048,1024]` and `4096x4096` contiguous/transposed add.
- Report selected Triton configs, generated kernels, compile wall time,
  coalescing-analysis call count, and steady GPU time.

`torch.cuda.Graph` replay:
- Applies to the runtime portions.
- Not applicable to the compile-time cache measurement except as a separate
  steady-state check.

Blockers:
- CUDA GPU required for runtime.
- Pre/post checkout and isolated compile caches are needed for attribution.
- May require instrumentation around scheduler coalescing-analysis calls.

### P1-2: Input assert and misaligned-copy deferral

PRs/clusters:
- #177783: defer input size/stride assertions to first use.
- #179039: defer `copy_misaligned_inputs` to first use.
- #180599: rename/update `copy_misaligned` to `copy_if_misaligned`.

Why it matters:
- These target wrapper CPU overhead before the first GPU kernel and can improve
  overlap by moving checks/copies closer to the first actual use.
- CUDA graph replay alone misses the host-side effect.

Toggle or checkout:
- Pre/post required. Current HEAD does not expose an equivalent old/new toggle.
- `TORCHINDUCTOR_SIZE_ASSERTS=1` can keep asserts present, but it cannot emulate
  pre-deferral placement.

Benchmark shape/model:
- Many-input assert case: compile a graph with 10, 32, and 128 CUDA bf16 inputs,
  each `(4096,4096)`. The first kernel should use only inputs 0 to 2; later
  kernels should use the remaining inputs after independent GPU work.
- Misaligned-copy case: two-stage graph where stage 1 is a contiguous
  `(4096,4096) @ (4096,4096)` bf16 matmul, and stage 2 uses a late
  transposed/narrowed non-contiguous input in pointwise and reduction consumers.
- Smaller smoke shapes: `(512,512)` for quick iteration.
- Report CPU time from wrapper entry to first kernel launch, first-iteration
  latency, total steady direct time, generated wrapper assert/copy placement,
  and steady GPU time.

`torch.cuda.Graph` replay:
- Not primary because the expected win is host work before or overlapping with
  GPU work.
- Useful only as a secondary steady-state GPU-body check.

Blockers:
- Needs launch-time instrumentation or profiler markers to measure CPU time to
  first kernel.
- Pre/post checkout required; likely Python-heavy, but rebuild if the checkout
  import is stale.

### P1-3: Horizontal partition fusion skip

PRs/clusters:
- #184904: `CapabilityBasedPartitioner(skip_horizontal_fusion=...)`.
- Superseded closed-only context: #184814, #170191.

Why it matters:
- The intended win is avoiding overly large fused horizontal partitions when an
  unsupported producer feeds multiple independent supported consumers.
- It can affect partition count, partition size, compile time, and runtime if
  the partitioner is in the compile path.

Toggle or checkout:
- Current HEAD can compare `skip_horizontal_fusion=False/True` directly through
  the FX partitioner API.
- Pre/post required to prove landed impact because the old commit did not have
  the same API.

Benchmark shape/model:
- FX graph with one unsupported producer creating `x=(1024,1024)` feeding four
  independent supported branches: `sin`, `relu`, `add/mul`, and a small
  `mm`/reduction branch, then a final supported combine.
- Run `CapabilityBasedPartitioner(..., skip_horizontal_fusion=False/True)`.
- Report partition count, max partition node count, number of duplicated
  unsupported boundaries, compile time, and generated graph readability.
- If wired into a runtime path, time a compiled CUDA version with
  `x=(1024,1024)` and `x=(4096,4096)`.

`torch.cuda.Graph` replay:
- Not applicable for the pure partitioner measurement.
- Applies only if the partitioned graph is compiled into a static CUDA runtime
  benchmark.

Blockers:
- Requires a small FX harness with a supported-op callback and an intentionally
  unsupported producer.
- Pre/post checkout for attribution.

### P1-4: Low-precision primitives and production pack paths

PRs/clusters:
- #172497: `cvt_e8m0_rceil` primitive with SM100+ PTX lowering.
- #177922: `inline_asm_elementwise` higher-order operator.
- #183638: NVFP4 pack context from the nested slice, but not inventory-credited
  until landed proof is confirmed.

Why it matters:
- The nested benchmarks proved launch-count reductions for Inductor-expressible
  microbenchmarks, not production swizzled MXFP8/NVFP4 packing throughput.
- #172497 and #177922 are low-precision building blocks that need primitive and
  end-to-end measurements on Blackwell-class hardware.

Toggle or checkout:
- Current HEAD is enough to benchmark primitive throughput and compare against
  hand-written software fallbacks.
- Pre/post required for PR attribution because there is no stable off toggle for
  the landed primitive/operator.
- #183638 must first be confirmed against the inventory/mainline source of
  truth before being credited.

Benchmark shape/model:
- Primitive E8M0: `inp=(1<<20,)` with dtype float32, float16, and bfloat16 on
  CUDA. Compare `inductor_prims.cvt_e8m0_rceil` against a software
  `log2/ceil/clamp` implementation; inspect generated PTX for the SM100
  conversion instruction.
- Inline asm HOP: `inline_asm_elementwise` over `(1<<20,)` tensors for
  float32, float16, and bfloat16, comparing eager/Jiterator-style fallback
  against compiled Triton.
- Production pack: RMSNorm to amax/scale to final MXFP8 or NVFP4 pack/swizzle
  with `x=(128,4096)` bf16, `group=16` for NVFP4-style and `group=32` for
  MXFP8-style. Use the actual production pack primitive, not only an
  Inductor-expressible approximation.
- Report throughput, generated PTX, kernel count, memory layout correctness,
  and replay timing.

`torch.cuda.Graph` replay:
- Applies and is useful for primitive and pack throughput on static shapes.

Blockers:
- SM100/B200 or equivalent required for the E8M0 PTX path.
- Production pack/swizzle primitive must be available.
- #183638 landed proof is unresolved in the provided inventory.
- Pre/post checkout for attribution; rebuild if the historical checkout needs it.

### P1-5: Representative PDL combo-kernel benchmark

PRs/clusters:
- #174232: PDL support to combo kernels.

Why it matters:
- The completed benchmark proved the code path and found a regression on a
  synthetic B200 pointwise case.
- PDL is supposed to help dependent-launch workloads; an enclosing CUDA graph
  can hide the launch-latency surface.

Toggle or checkout:
- Current-HEAD toggle is enough: compare `triton.enable_pdl=False/True` with
  `combo_kernels=True`.
- Pre/post is optional unless strict PR attribution is needed.

Benchmark shape/model:
- Build a short sequence of 16 to 64 small dependent pointwise/combo kernels
  outside an enclosing CUDA graph, where each output feeds the next.
- Shapes: `(4096,)`, `(65536,)`, and `(2^20,)` CUDA tensors; include a mixed
  pointwise/reduction variant with `x=(32,1024)`.
- Report direct CUDA event time, CPU launch overhead if measurable,
  `launch_pdl=True` presence, generated kernel count, and correctness.

`torch.cuda.Graph` replay:
- Not primary. CUDA graph replay hides the launch-latency effect PDL is meant to
  improve.
- A replay row can be included as a secondary kernel-body sanity check.

Blockers:
- Hopper or newer GPU with PDL support; B200 is acceptable.
- Need to avoid outer CUDA graph capture for the primary measurement.

### P2-1: Custom-op and `aten.mm` autotune attribution

PRs/clusters:
- #175275: CUDA graph benchmarking for extern/custom choices and
  `min_speedup_threshold`.
- #175276: CUDA graph benchmark memory cleanup.
- #175277: scoped autotuning config patches.
- #175278: custom-op autotuning applied to `aten.mm`.
- #175422: normalize custom-op autotuning fallback choice.

Why it matters:
- The current benchmark verified API stability and a custom-op selection path,
  but it did not prove landed pre/post speedups or cover real `aten.mm`
  selection and memory-retention behavior.

Toggle or checkout:
- Current HEAD is enough to exercise APIs and thresholds.
- Pre/post required for attribution and for #175278's change to `aten.mm`.

Benchmark shape/model:
- Custom op: reuse the completed fp16 case `x=(8,128,256)`, `weight=(256)` and
  add `min_speedup_threshold` sweep `1.0`, `1.05`, `1.5`.
- Matmul choices: `(64,256) x (256,128)`, `(128,4096) x (4096,4096)`, and
  `(4096,4096) x (4096,11008)` bf16 under `max_autotune`.
- Scoped config: custom op with `CustomOpConfig(config_patches={"coordinate_descent_tuning": True})`.
- Fallback normalization: two repeated custom-op calls with different kwargs on
  `(64,256) x (256,128)`.
- Memory cleanup: repeat CUDA-graph benchmarked `torch.mm` with
  `(1024,1024)` bf16 and float32 for 100 to 1000 autotune iterations; record
  allocated/reserved memory and cuBLAS workspace behavior.

`torch.cuda.Graph` replay:
- Central for #175275 and #175276.
- Use replay timing for choice comparison, plus direct timing for launch-sensitive
  decisions.

Blockers:
- Need custom autotune registrations and isolated caches.
- Pre/post checkout for attribution; no multi-GPU required.

### P2-2: Distributed estimator quality and overlap scheduling

PRs/clusters:
- #172510: NCCL/Inductor collective estimates to tlparse CSV.
- #175204: compute estimator option for overlap scheduling.

Why it matters:
- These PRs do not directly optimize kernels, but they affect overlap-scheduling
  decisions and make distributed regressions diagnosable in production traces.

Toggle or checkout:
- Current HEAD is enough to validate artifacts and compare
  `compute_estimator={analytical,benchmark}` and collective estimator options.
- Pre/post required only if quantifying the landed change versus no artifact or
  no option.

Benchmark shape/model:
- Collective microbenchmarks: `all_gather`, `reduce_scatter`, and `all_reduce`
  from small KB tensors through 16 MB, 64 MB, and 256 MB buckets.
- World sizes: 2, 4, and 8 on one node; add multi-node if available.
- Overlap model: FSDP/DTensor transformer block with overlap scheduling enabled,
  plus a synthetic all-reduce plus `mm` chain using `(4096,4096)` CUDA matrices.
- Report estimator-predicted time, CUDA-event measured collective time, schedule
  choice, exposed comm time, GPU sync count, compile time, and tlparse CSV
  contents.

`torch.cuda.Graph` replay:
- Not central. Explicit collective timing and traces are the validation surface.
- If the end-to-end model uses CUDA graphs in production, include that mode as a
  separate runtime row.

Blockers:
- Multi-GPU required for estimator quality.
- Multi-node is useful for real network behavior.
- Requires `TORCH_TRACE`/tlparse artifact collection.

### P2-3: Benchmark lock hooks under contention

PRs/clusters:
- #184905: Inductor benchmark GPU lock hooks.
- Closed-only context: #184815.

Why it matters:
- Default current-head behavior is a no-op unless an external harness registers
  a lock. The value is lower benchmark variance and less GPU contention under
  concurrent autotuning/profiling workers.

Toggle or checkout:
- Current HEAD is enough to write and register a real lock hook.
- Pre/post required to prove landed impact because the old commit lacks the
  hook.

Benchmark shape/model:
- Launch 4 to 8 worker processes on the same GPU. Each worker repeatedly compiles
  and autotunes either:
  - the custom-op case `x=(8,128,256)`, `weight=(256)`, or
  - max-autotune GEMMs `(1024,1024) x (1024,1024)` and
    `(4096,4096) x (4096,4096)` bf16.
- Compare no hook versus a process-local/file-lock or external scheduler lock.
- Report autotune wall time, selected configs, benchmark variance, failures,
  GPU utilization, and total throughput.

`torch.cuda.Graph` replay:
- Applies inside Inductor benchmark sections if those choices use CUDA graph
  benchmarking.
- External replay is not the main measurement.

Blockers:
- Need a multi-process harness and registered lock hook.
- Single GPU is enough; multi-GPU contention tests are optional.
- Pre/post checkout for attribution.

### P2-4: Missing-op fallback log laziness

PRs/clusters:
- #185951: lazily format missing-op fallback logs.

Why it matters:
- Avoids expensive diagnostic string construction when INFO logging is disabled,
  especially on custom-op fallback graphs with large IR strings.

Toggle or checkout:
- Pre/post required for speed attribution. Current HEAD can only compare logging
  levels, not old eager formatting.

Benchmark shape/model:
- Use the custom-op fallback repro from the compile/debug slice and
  `CpuTests.test_missing_op_info_log_is_lazy_cpu` style coverage.
- Build a graph with many fallback/missing-op sites and large readable IR, then
  run compile with INFO logging disabled and enabled.
- Report compile wall time, `operator_str` or equivalent formatting call count,
  time spent formatting, and emitted log correctness when INFO is enabled.

`torch.cuda.Graph` replay:
- Not applicable. This is compile/logging CPU overhead.

Blockers:
- CPU-only benchmark is sufficient.
- Need instrumentation or monkeypatching around the expensive formatting helper.
- Pre/post checkout for attribution.

### P3-1: Observability and CI trace artifact validation

PRs/clusters:
- #172460: Perfetto graph-pass trace events and disabled-pass envvar.
- #183340: tlparse artifact collection and `TORCH_TRACE` on A100/B200 perf jobs.
- #183718: OSDC H100/B200 trace propagation and `perf_cli.py` support.
- #172461 and #172841: compiler bisector reliability around cudagraph
  application/run mode.

Why it matters:
- These do not directly speed up models, but they are high-leverage for
  diagnosing compile/runtime regressions found by the higher-priority
  benchmarks.

Toggle or checkout:
- Current HEAD is enough for smoke validation.
- Pre/post only needed for before/after artifact availability or bisector
  behavior.

Benchmark shape/model:
- Compile a representative Inductor model with `TORCH_TRACE` enabled and inspect
  `pass.*` Perfetto/tlparse events from #172460.
- Use `TORCHINDUCTOR_DISABLED_PASSES=PASSNAME` on a known graph pass and verify
  pass skipping plus resulting graph behavior.
- Run or inspect A100/B200/H100 perf jobs and verify raw gzipped logs,
  per-log tlparse HTML artifacts, and `perf_cli.py repro <run-id> --model ...`
  lookup.
- Run compiler bisector on a deterministic cudagraph-specific failure and verify
  the bad stage is isolated.

`torch.cuda.Graph` replay:
- Not applicable except as the subject of a bisected failure.

Blockers:
- CI/perf-job access may be required for #183340/#183718.
- No multi-GPU requirement unless validating distributed trace artifacts.

### P3-2: Lower-priority runtime/correctness guards

PRs/clusters:
- #172951: skip fp32 attention fusion when TF32 is disabled.
- #181793/#181795: dynamic-shape tiling issue.
- #182139/#182178: raw stream name collisions.
- #177546: pad_mm padded strides not leaking to user-visible outputs.
- #175203, #182890, #172488: fusion/pass-ordering reliability.

Why it matters:
- These are worthwhile validation targets, but they are mostly correctness or
  reliability guards rather than top benchmark gaps.

Toggle or checkout:
- Current HEAD is usually enough for correctness validation.
- Pre/post required if the goal is to quantify performance or prove shipped
  impact for a PR.

Benchmark shape/model:
- #172951: fp32 attention with `(B,H,S,D)=(2,8,128,64)` and
  `torch.backends.cuda.matmul.allow_tf32=False/True`; report selected path,
  correctness, and direct/replay time.
- #181793/#181795: dynamic dim 0 with contiguous/transposed/NHWC base
  `(1,128,256,512)`; verify expected tiling and correctness.
- #182139/#182178: CUDA:1 user stream plus default/aux streams and Triton kernels
  over `(1024,1024)`; replay repeatedly and assert no stale reads/crashes.
- #177546: shape padding matmul shapes `(M,K,N)=(40,49,30)`, `(20,81,30)`,
  `(21,80,30)`, plus bmm with `(128,33,40)` inner dimensions; verify strides and
  runtime under `max_autotune`.
- #175203/#182890/#172488: compile the regression graphs from their tests and
  report compile success, fusion state, and direct/replay correctness.

`torch.cuda.Graph` replay:
- Applies to static CUDA runtime rows as a secondary steady-state check.
- Not useful for pure compile success or pass-ordering validation.

Blockers:
- Mostly single-GPU.
- Some cases require specific devices or multi-stream CUDA setup.

## Not queued for current performance benchmarking

- Closed-only/no-landed-proof PRs: do not credit as shipped impact unless new
  landed proof is found.
- #170729: reverted shortly after landing; benchmark only as historical context
  for joint-graph constant-folding compile-time ideas.
- #181617: PR body had strong test-infrastructure measurements, but the current
  upstream impact should be de-emphasized due revert evidence.
- #183638 and no-PR nested epilogue commits: keep as context until inventory or
  mainline landed proof is established.
- CI/lint/docs/build-only PRs such as #183632, #174706, #174973, #184821 are not
  priority benchmark gaps.

