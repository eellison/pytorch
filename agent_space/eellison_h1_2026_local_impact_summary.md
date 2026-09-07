# eellison H1 2026 local single-GPU impact summary

Generated: 2026-07-13

This is the final local-only synthesis for the H1 2026 eellison PR inventory and benchmark artifacts under `agent_space/`. It is scoped to local single-GPU PyTorch/Inductor impact. Distributed overlap, collectives, and multi-GPU remaining work are intentionally excluded from the remaining-work framing.

No source files outside `agent_space/` are part of this deliverable.

## Source baseline

Inventory source of truth: `agent_space/eellison_h1_2026_pr_inventory.md`.

- Inventory counts: 137 PR-numbered rows; 68 rows with landed or merged signal; 69 closed-only rows without landed proof; 0 H1 landed commits without a PR number in the inventory baseline.
- Local landed proof uses `origin/main`, not the detached/current `HEAD`; this checkout's `origin/main` snapshot ends at 2026-06-23.
- Closed-only rows are not credited as shipped impact unless superseded by a landed PR.
- #183638 and the no-PR sub-parent nested epilogue commits are useful benchmark context, but they are not inventory-credited landed PRs in the provided baseline.
- #170729 and #181617 are de-emphasized for current upstream impact because the source reports found revert evidence.

## Ranked local impact clusters

| Rank | Cluster | PRs or context | Local impact status |
| ---: | --- | --- | --- |
| 1 | Loop reindexing, loop ordering, and pointwise cat fusion | #176927, #179090, #179091, with #176345 and #182890 as related enablers | Strongest measured local runtime cluster. Loop reindexing naturally surfaced as a top result: RMSNorm/qknorm/reshape and RoPE cat patterns collapse from two kernels to one, with clear B200 direct and replay wins on representative shapes. |
| 2 | Nested reductions for norm plus grouped quantization | #182891, #182892, #182893, #182895, #182896, #183432, #182897, #182898; low-precision enablers #177922 and #172497 | Strong codegen and kernel-count evidence. B200 graph replay proves 2-or-3 kernel graphs become one nested kernel for RMSNorm/amax/NVFP4-style paths, but selected steady-state kernel-body times are mostly flat. |
| 3 | Compile-time symbolic fast paths and repro quality | #181275, #181276, #181277, #176345 | Highest reported user-facing compile-time win, especially #181275's reported wide-symbol backward compile drop from over 50 minutes to about 6 minutes. Attribution still needs parent-vs-landed compile benchmarks, not CUDA graph replay. |
| 4 | CUDA graph policy and replay reliability | #175271, #175275, #175276, #176620, #174103, #182524, #182531, #188117, #188078 | Current-head local probes validate behavior and counters for several reduce-overhead scenarios. PR-level speed attribution requires pre/post checkouts because the old behavior is not exposed by stable current-head toggles. |
| 5 | Autotuning, combo kernels, PDL, and benchmark controls | #174232, #175275, #175276, #175277, #175278, #175422, #182895, #184905 | Mixed local evidence. Combo kernels help simple independent pointwise fusion, but synthetic PDL rows regress on B200 by synchronized wall time. Custom-op CUDA graph benchmark APIs are active, while `aten.mm` max-autotune attribution needs pre/post. |
| 6 | Low-precision primitives and production pack paths | #172497, #177922, #183638 context only | #172497 shows a clear B200 fp32 primitive win for `cvt_e8m0_rceil`; #177922 inline asm HOP is graph-capturable. Production NVFP4/MXFP8 pack attribution remains pre/post and, for #183638, inventory proof is unresolved. |
| 7 | Local correctness, diagnostics, and maintenance that preserve performance work | #185951, #177546, #181793/#181795, #182139/#182178, #172951, #175203, #183340/#183718 local trace infrastructure | Important support work, but not broad first-order local runtime leaders. #185951 is the clearest remaining compile/logging overhead item. |

## Local B200 methodology

Common environment from the completed benchmark reports:

- Source HEAD: `d59fc87a47b0848573c67c25481a8eff3f31df98` (`[inductor] Fuse contiguous sub-parent epilogues`)
- Imported torch: `2.13.0a0+gite1067a5` from this checkout
- GPU: NVIDIA B200, capability `10.0`, CUDA runtime 12.8; visible GPUs: 8, but benchmark tensors ran on local `cuda:0`
- Triton: 3.6.0 in the nested benchmark report
- Fresh compile and Triton caches were rooted under `agent_space/`

Timing methodology:

- Nested reductions: compile each case twice with `torch.compile(..., fullgraph=True, dynamic=False)`, toggling `torch._inductor.config.patch({"triton.nested_reduction": False/True})`; warm up 20 calls, capture one compiled call in `torch.cuda.CUDAGraph`, then time 1000 graph replays with CUDA events.
- Loop/fusion/autotune slice: fresh caches, `fx_graph_cache=False`, `benchmark_kernel=True`, `triton.unique_kernel_names=True`; direct timing used CUDA events with 10 warmups and 5 reps of 50 iterations; graph replay captured one static compiled invocation and timed replay with CUDA events.
- Local remaining slice: reported CUDA-event direct timing, synchronized wall timing where needed, and external CUDA graph replay where valid. PDL rows use synchronized wall time as primary because CUDA events can undercount dependent-launch tail work.
- `mode="reduce-overhead"` CUDA graph policy rows measure Inductor's internal CUDA graph behavior. External recapture is often invalid when Inductor has already captured internally.

## What is now benchmarked

### Loop reindexing, loop ordering, pointwise cat

These are the strongest local runtime results.

| PRs | Case | Result |
| --- | --- | --- |
| #179090 | Outer sum plus pointwise, `x=(32, 2^20)` | Kernels `2->1`; direct `0.1080->0.0804 ms` (`1.34x`); graph replay `0.1048->0.0800 ms` (`1.31x`). |
| #179090 | Outer softmax, `x=(32, 2^20)` | Kernels `2->1`; direct `1.06x`; graph replay `1.12x`. |
| #176927 | RMSNorm reshape slice, bf16 `x=(16,8192)` from wider qkv | Kernels `2->1`; direct `0.0248->0.0203 ms` (`1.22x`); graph replay `0.00416->0.00230 ms` (`1.81x`). |
| #176927/#179090 | RMSNorm reshape transposed, bf16 `x=(16,8192)`, stride `(1,16)` | Kernels `2->1`; direct `1.20x`; replay nearly flat in the first slice. Larger local row had noisy direct timing but replay improved `6.26->4.23 us` (`1.48x`). |
| #179091 | QKNorm plus split RoPE cat, `B=4,H=8,S=128,D=64` | Kernels `2->1`; direct `0.0388->0.0232 ms` (`1.67x`); replay flat. Larger local shape `B=8,H=16,S=256,D=128` improved direct `28.46->12.46 us` (`2.29x`) but replay regressed `10.33->12.48 us`. |
| #179091 | QKNorm plus interleaved RoPE stack/flatten | Kernels `2->1`; direct `1.39x` on the smaller case and `1.33x` on the larger case; replay regressed on these small/static bodies. |

Interpretation: the local evidence supports fewer launches and less materialization as the core win. CUDA graph replay sometimes shows flat or negative body-time deltas for fused cat variants, so the best claim is shape-dependent runtime improvement plus robust kernel-count reduction.

### Nested reductions and low-precision-adjacent nested paths

All rows used current-head `triton.nested_reduction=False/True` with B200 graph replay timing.

| Case | Result |
| --- | --- |
| RMSNorm block amax/scale, `B=128,D=4096,G=16` | Kernels `2->1`; `codegen_nested_reduction 0->1`; `8.58->8.54 us` (`1.004x`). |
| RMSNorm NVFP4-style pack, `B=128,D=4096,G=16` | Kernels `3->1`; nested metric `0->1`; `8.54->8.51 us` (`1.004x`). #183638 remains context-only for inventory credit. |
| RMSNorm MXFP8-style path, `B=128,D=4096,G=32` | Kernels `3->2`; nested metric `0->1`; `14.08->14.18 us` (`0.993x`). This is an Inductor-expressible approximation, not the final production swizzled MXFP8 pack. |
| Half-resolution sub-parent epilogue, `B=128,D=4096,G=16` | Kernels `2->1`; `8.54->8.35 us` (`1.023x`). Useful current-HEAD context, not inventory PR credit without PR/mainline proof. |
| Weighted RMSNorm reduce-K, `B=64,K=16,D=4096` | Kernels `2->1`; runtime flat around `54.8 us`. |
| Chunk SwiGLU and chunk4 gated consumers, `B=128,D=1024` | Kernels `2->1`; runtime flat around `8.3 us`. |

Interpretation: the nested stack is benchmarked for codegen coverage and launch/materialization reduction. It does not yet show a large steady-state B200 kernel-body speedup on the selected static shapes.

### Combo kernels, PDL, and autotune surfaces

| PRs | Case | Result |
| --- | --- | --- |
| #174232 | Six independent pointwise ops, `6 * (2^22,)` | Kernels `6->1`; direct `0.0504->0.0338 ms` (`1.49x`); graph replay `0.0412->0.0329 ms` (`1.25x`). |
| #174232 | Mixed pointwise/reduction restriction | Regressed in the existing slice; useful negative coverage. |
| #174232 | PDL synthetic pointwise combos | `launch_pdl=True` and `gdc_launch` emitted, but B200 synchronized wall time regressed for `4096`, `65536`, and `1048576` element inputs. Current evidence does not prove the intended dependent-launch win. |
| #175271 | CUDA graph partition threshold, small matmul chain | Direct `0.0386->0.0318 ms` (`1.21x`). External graph replay was unavailable for one side because Inductor already captured internally. |
| #175275/#175276/#175277/#175278/#175422 | Custom-op autotune API path | API stable; graph replay around `0.0028 ms`; logs selected a faster decomposition than fallback in the first slice. This verifies behavior more than landed pre/post speedup. |
| #175278 current-head max-autotune `aten.mm` probe | bf16 `(64,256)x(256,128)` and `(1024,1024)x(1024,1024)` | Current toggles regressed direct time and failed external capture on max-autotune rows. Not clean attribution to the historical `aten.mm` autotune PR. |

### Low-precision primitives

| PRs | Case | Result |
| --- | --- | --- |
| #172497 | `cvt_e8m0_rceil` PTX lowering vs software fallback, `1<<20` elems | fp32 direct `23.09->4.24 us` (`5.45x`), graph replay `4.15->2.90 us` (`1.43x`), generated PTX contained `cvt.rp.satfinite.ue8m0x2.f32`. fp16 and bf16 were flat in this microbenchmark. |
| #177922 | `inline_asm_elementwise` compiled HOP, `1<<20` elems | f32 add, fp16 upcast double, and bf16 upcast double all graph-captured around `4.2 us`; generated code contained inline asm. No current-head off toggle for PR attribution. |

### CUDA graph policy current-head probes

These validate current behavior only.

| PRs | Case | Current-head observation |
| --- | --- | --- |
| #182524 | Self-overlap expand input, pointwise plus `mm` | Direct event `58.43 us`, wall `56.83 us`, rerecord count `0`, no `cudagraph_skips`. |
| #182531 | Fresh `nn.Parameter` pointer each call with low rerecord limit | Direct event `1540.81 us`, wall `4501.74 us`, rerecord count `0`, no `cudagraph_skips`. |
| #188117/#188078 | Retained live outputs smoke | Direct event `48.55 us`, wall `46.60 us`, no `cudagraph_skips`; the smoke did not force the stronger alias hazard needed to prove clone behavior. |
| #177783/#179039/#180599 | Host assert/copy deferral probe | Current generated wrapper is inspectable, but there is no old/new toggle for placement attribution. |

## What remains only pre/post-checkout measurable

These items should not be claimed as landed speedups from current-head toggles alone.

| Cluster | PRs and commits from the reports | Required comparison |
| --- | --- | --- |
| Wide-symbol compile-time fast paths | #181275 `f296c860884`, #181276 `b44a57da7e3`, #181277 `3565a492def`/`89ab27e972a`, #176345 `54028337cf5` | Parent vs landed checkouts on a wide-symbol after-AOT repro. Measure compile wall time, time in `sympy.gcd`, `statically_known_true/false`, call counts, generated code size, and preserved symbolic relationships. CUDA graph replay is not applicable. |
| CUDA graph policy and live-output behavior | #182524 `10efc0f4c57`, #182531 `7089ea8477d`, #188117 `d46712a5fb2`, #188078 `d877745fe08`, #176620, #174103 | Parent vs landed checkouts with `mode="reduce-overhead"` harnesses for self-overlap, parameter pointer churn, live-output cloning, saved activations, and graph opt-out annotations. Report graph ids, rerecords, skip reasons, clones/allocations, and steady direct timing. |
| Wrapper overhead and first-use deferral | #177783 `f0c3582ef2d`/`779c78b3d7d`, #179039 `ddaac926c33`/`55fc17f8653`, #180599 `13a6d0ddbc7` | Parent vs landed checkouts with CPU wrapper-entry-to-first-kernel instrumentation. CUDA graph replay misses the host placement win. |
| Tiling/coalescing attribution | #182895 `5387f64e4d2`, #183585 `a13f6966448` | Parent vs landed checkouts to separate block-size-floor propagation from coalescing-analysis caching. Current `triton.coalesce_tiling_analysis` exercises the surface but does not isolate either PR. |
| Custom-op and `aten.mm` autotune attribution | #175275 `dd3c11bbb3b`, #175276 `1bb77ba1b09`, #175277 `9e85a6ead69`, #175278 `b9b20336385`/`e8120f9062c`/`d7f39875e57`, #175422 `67a21876c88` | Commit-by-commit parent vs landed benchmarks for custom-op choices, CUDA graph benchmarking cleanup, scoped config patches, `aten.mm` registration, and fallback normalization. |
| Horizontal partition fusion skip | #184904 `eeccb459bcc` | Current HEAD can compare `CapabilityBasedPartitioner(skip_horizontal_fusion=False/True)`, but shipped impact needs parent vs landed because the old commit lacks the API. |
| GPU benchmark lock hooks | #184905 `d79ee01a97f` | Current HEAD can register a real local lock hook, but landed impact needs parent vs landed under 4-8 local autotuning workers contending on one GPU. This is single-GPU contention, not distributed overlap. |
| Low-precision primitive and production pack attribution | #172497 `b79269a4308`, #177922 `a9be19830db`/`1ae64875e7e`; #183638 context only | Parent vs landed checkouts for primitive/HOP attribution. Production NVFP4/MXFP8 pack needs the actual production pack/swizzle primitive. #183638 should not be credited until inventory/mainline proof is found. |
| Missing-op fallback log laziness | #185951 `067e9a9f1a1` | Parent vs landed compile/logging benchmark on large fallback/missing-op graphs. Measure formatting call counts and compile time with INFO disabled/enabled. |

## Local follow-up priority

1. Run pre/post wide-symbol compile benchmarks for #181275/#181276 using an after-AOT repro preserved by #181277. This is the highest-impact unmeasured local item.
2. Run parent-vs-landed CUDA graph policy attribution for #182524/#182531/#188117/#188078. Current-head probes passed, but old/new behavior is the point.
3. Instrument wrapper-entry-to-first-kernel latency for #177783/#179039/#180599. This is host overhead, so graph replay is secondary.
4. Attribute #182895 and #183585 separately with parent-vs-landed compile/runtime measurements.
5. Re-run custom-op and `aten.mm` autotune commit-by-commit; current-head toggles are too coarse.
6. Validate low-precision production pack paths with real NVFP4/MXFP8 pack/swizzle primitives and keep #183638 context-only until inventory proof is resolved.

Distributed and multi-GPU overlap benchmarks are intentionally not part of this local queue.

## Exact artifacts and harnesses

Primary synthesis inputs:

- `agent_space/eellison_h1_2026_impact_summary.md`
- `agent_space/eellison_h1_2026_local_remaining_benchmarks.md`
- `agent_space/eellison_h1_2026_nested_benchmarks.md`
- `agent_space/eellison_h1_2026_loop_fusion_autotune_benchmarks.md`
- `agent_space/eellison_h1_2026_remaining_benchmark_queue.md`
- `agent_space/eellison_h1_2026_inductor_runtime_impact.md`
- `agent_space/eellison_h1_2026_compile_debug_impact.md`
- `agent_space/eellison_h1_2026_nested_impact.md`
- `agent_space/eellison_h1_2026_pr_inventory.md`

Benchmark harnesses:

- `agent_space/bench_h1_nested_reduction_impact.py`
- `agent_space/eellison_h1_2026_nested_benchmark_worker.py`
- `agent_space/bench_h1_loop_fusion_autotune.py`
- `agent_space/bench_h1_local_remaining.py`

Raw benchmark outputs:

- `agent_space/eellison_h1_2026_nested_benchmarks.json`
- `agent_space/h1_loop_fusion_autotune_results.json`
- `agent_space/h1_loop_fusion_autotune_smoke.json`
- `agent_space/h1_local_remaining_results.json`
- `agent_space/h1_local_remaining_smoke.json`

Compile caches and generated code:

- `agent_space/.bench_cache_h1_2026_nested/`
- `agent_space/h1_loop_fusion_autotune_cache/`
- `agent_space/h1_local_remaining_cache/`
- `agent_space/captured_nested_reduction_core_round2/`

Inventory artifacts:

- `agent_space/build_eellison_h1_2026_pr_inventory.py`
- `agent_space/eellison_h1_2026_candidate_pr_meta.tsv`
- `agent_space/eellison_h1_2026_closed_prs.tsv`
- `agent_space/eellison_h1_2026_landed_commits.tsv`
- `agent_space/eellison_merged_prs_h1_2026.json`
- `agent_space/eellison_search_merged_prs_h1_2026.json`

