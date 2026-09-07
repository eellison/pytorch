# eellison H1 2026 local single-GPU impact summary v2

Generated: 2026-07-13

Scope: local single-GPU PyTorch/Inductor impact only. Distributed,
collectives, NCCL, multi-GPU overlap, and distributed trace-estimator work are
explicitly excluded from this summary and from the remaining-work queue below.

No source files outside `agent_space/` are part of this deliverable.

## Source baseline

Inventory source of truth: `agent_space/eellison_h1_2026_pr_inventory.md`.

- Inventory counts: 137 PR-numbered rows; 68 rows with landed or merged signal;
  69 closed-only rows without landed proof; 0 H1 landed commits without a PR
  number in the inventory baseline.
- Local landed proof uses `origin/main`, not detached/current `HEAD`; this
  checkout's `origin/main` snapshot ends at 2026-06-23.
- Closed-only rows are not credited as shipped impact unless superseded by a
  landed PR.
- #183638 and the no-PR sub-parent nested epilogue commits are useful current
  HEAD benchmark context, but they are not inventory-credited landed PRs in the
  provided baseline.
- #170729 and #181617 are de-emphasized for current upstream impact because the
  source reports found revert evidence.

## Bottom line

The strongest measured local runtime cluster is loop reindexing, loop ordering,
and pointwise-cat fusion (#176927, #179090, #179091). The focused toggle report
shows representative B200 cases collapsing from 2 kernels to 1 with direct
runtime wins up to 1.50x on normal pointwise-cat gates, 1.51x on the force-cat
sanity check, 1.43x on loop reindexing, and 1.49x CUDA graph replay on the
grouped-quant loop-ordering candidate-selection isolation.

Nested reductions remain important for codegen coverage and launch/materialized
intermediate removal. The measured B200 static-shape body times are mostly flat,
but the nested stack reliably turns 2-or-3 kernel RMSNorm/grouped-quant/NVFP4
style paths into one nested kernel when `triton.nested_reduction=True`.

#172497 now has focused SM100 evidence beyond the primitive microbenchmark:
current E8M0 PTX lowering and pattern replacement are equal to the fallback and
show encode-only speedups up to 2.000x, while full BF16 scale-generation rows
are flat-to-modestly faster because surrounding reduction work dominates.

Compile-time symbolic fast paths, CUDA graph policy changes, wrapper first-use
deferral, custom-op/autotune attribution, and production pack paths still need
parent-vs-landed checkouts for strict landed-impact claims.

## Local B200 methodology

Common environment from the completed local reports:

- Source HEAD: `d59fc87a47b0848573c67c25481a8eff3f31df98`
  (`[inductor] Fuse contiguous sub-parent epilogues`)
- Imported torch: `2.13.0a0+gite1067a5` from this checkout
- GPU: NVIDIA B200, capability `10.0`, CUDA runtime 12.8; visible GPUs: 8, but
  benchmark tensors ran on local `cuda:0`
- Fresh compile and Triton caches were rooted under `agent_space/`

Nested reductions were compiled twice with `torch.compile(..., fullgraph=True,
dynamic=False)`, toggling `torch._inductor.config.patch({"triton.nested_reduction":
False/True})`, then timed with external `torch.cuda.CUDAGraph` replay.

Loop reindexing/order/cat rows used fresh Inductor/Triton caches,
`fx_graph_cache=False`, `benchmark_kernel=True`, and
`triton.unique_kernel_names=True`. The harness checked eager-vs-compiled
correctness, generated-kernel counts, wrapper launch sites, generated Triton
definitions, and `num_loop_reordering`, then measured CUDA-event direct runtime
plus external CUDA graph replay where capture was valid. For #179090 candidate
selection, current HEAD has no config toggle to re-enable the old candidate
picker, so the grouped-quant row used a scratch benchmark-local monkey-patch to
emulate the prior largest-buffer candidate choice.

The focused `cvt_e8m0_rceil` report compared current Inductor E8M0
replacement/PTX lowering against a before-state compiled with
`torch._inductor.config.pattern_matcher=False`. Each callable was compiled
first, then timed with manual `torch.cuda.CUDAGraph` replay; numbers exclude
`torch.compile` overhead. Workloads were full MXFP8 scale generation from BF16
input and encode-only over the corresponding scale tensor shape.

## Loop reindexing, loop ordering, pointwise cat

Speedup is before/after, so values above 1.0 mean the enabled/current setting
was faster.

| PR / case | Toggle | Kernels | Direct median | CUDA graph replay median | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| #179090 outer sum + pointwise, x=(32, 2^20) | `loop_ordering_after_fusion False -> True` | 2 -> 1 | 108.07 us -> 80.58 us (1.34x) | 104.84 us -> 79.93 us (1.31x) | Broad loop-ordering toggle from test_loop_ordering; not candidate-selection-specific. |
| #179090 outer softmax, x=(32, 2^20) | `loop_ordering_after_fusion False -> True` | 2 -> 1 | 144.20 us -> 133.73 us (1.08x) | 150.07 us -> 133.20 us (1.13x) | Broad loop-ordering toggle from test_loop_ordering. |
| #179090 candidate selection grouped quant, x=(8,7168) bf16 | `scratch legacy candidate selection -> current HEAD` | 2 -> 1 | 58.99 us -> 52.36 us (1.13x) | 6.20 us -> 4.15 us (1.49x) | Monkey-patch emulates old largest-buffer candidate selection; closest isolation for #179090. |
| #176927 RMSNorm reshape slice, x=(16,8192) bf16 | `loop_reindexing_after_fusion False -> True, loop_ordering_after_fusion=False` | 2 -> 1 | 30.32 us -> 21.15 us (1.43x) | 4.20 us -> 3.04 us (1.38x) | Clean reindexing toggle isolation. |
| #176927 RMSNorm reshape transposed, x=(16,8192) bf16 stride=(1,16) | `loop_reindexing_after_fusion False -> True, loop_ordering_after_fusion=True` | 2 -> 1 | 27.40 us -> 21.45 us (1.28x) | 6.18 us -> 4.09 us (1.51x) | Combined with loop-ordering path. |
| #179091 QKNorm + split RoPE cat, B=4,H=8,S=128,D=64 | `max_*_pointwise_cat_inputs 0 -> 8` | 2 -> 1 | 32.80 us -> 21.89 us (1.50x) | 4.21 us -> 4.14 us (1.02x) | Normal current HEAD pointwise-cat gate. |
| #179091 QKNorm + interleaved RoPE stack/flatten | `max_*_pointwise_cat_inputs 0 -> 8` | 2 -> 1 | 29.85 us -> 24.00 us (1.24x) | 4.19 us -> 6.19 us (0.68x) | Normal current HEAD pointwise-cat gate. |
| #179091 split RoPE force pointwise cat | `force_pointwise_cat False -> True with max_* limits at 0` | 2 -> 1 | 32.80 us -> 21.71 us (1.51x) | 4.21 us -> 4.14 us (1.02x) | Sanity check of the explicit force toggle; not the PR heuristic. |

Interpretation: this is the clearest local runtime evidence. The robust claim is
fewer launches and less materialization, with direct runtime wins on the tested
static B200 shapes. CUDA graph replay remains shape-sensitive for pointwise-cat
variants, especially interleaved RoPE, so pure kernel-body speedup should not be
overstated.

## `cvt_e8m0_rceil` and E8M0 scale encoding

Implementation evidence from the focused report:

- `torch/_inductor/inductor_prims.py` defines
  `inductor_cvt_e8m0_rceil(Tensor input) -> Tensor` with an eager
  bit-manipulation fallback.
- `torch/_inductor/lowering.py` lowers the prim on SM100+ to
  `tl.inline_asm_elementwise` with PTX
  `cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;`, outputting `uint16` then
  casting to `uint8`.
- `torch/_inductor/fx_passes/misc_patterns.py` replaces the CUDA float32
  `ceil(log2(x))` E8M0 encode pattern with the prim on SM100+.
- `test/inductor/test_fp8.py::TestCvtE8M0Rceil` covers correctness, pattern
  matching, PTX codegen, and near-power-of-two behavior.

| Workload | Shape | Current ms | Fallback ms | Speedup | Kernels cur/fb | Current PTX | Fallback log2 | Equal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scale_from_bf16` | `1024x4096` | 0.0061 | 0.0061 | 1.000x | 1/1 | True | True | True |
| `encode_only` | `1024x128` | 0.0022 | 0.0040 | 1.826x | 1/1 | True | True | True |
| `scale_from_bf16` | `2048x4096` | 0.0061 | 0.0082 | 1.333x | 1/1 | True | True | True |
| `encode_only` | `2048x128` | 0.0030 | 0.0041 | 1.357x | 1/1 | True | True | True |
| `scale_from_bf16` | `4096x8192` | 0.0123 | 0.0123 | 1.001x | 1/1 | True | True | True |
| `encode_only` | `4096x256` | 0.0041 | 0.0041 | 1.001x | 1/1 | True | True | True |
| `scale_from_bf16` | `8192x8192` | 0.0298 | 0.0312 | 1.045x | 1/1 | True | True | True |
| `encode_only` | `8192x256` | 0.0041 | 0.0061 | 1.500x | 1/1 | True | True | True |
| `scale_from_bf16` | `8192x16384` | 0.0577 | 0.0605 | 1.049x | 1/1 | True | True | True |
| `encode_only` | `8192x512` | 0.0041 | 0.0082 | 2.000x | 1/1 | True | True | True |

Headline: current codegen removed fallback `log2` from all current rows, emitted
the SM100 PTX path, preserved equality, and improved encode-only rows by
1.357x-2.000x except the `4096x256` row, which was flat. End-to-end
`scale_from_bf16` rows were 1.000x-1.333x because the surrounding amax/reduction
work dominates the encode step.

## Nested reductions and NVFP4-style paths

All rows used current-head `triton.nested_reduction=False/True` with B200 CUDA
graph replay timing. `triton.nested_reduction` is not credited as enabled by
default; these rows measure the local effect when the config is patched on.

| Case | Result |
| --- | --- |
| RMSNorm block amax/scale, `B=128,D=4096,G=16` | Kernels `2->1`; `codegen_nested_reduction 0->1`; `8.58->8.54 us` (`1.004x`). |
| RMSNorm NVFP4-style pack, `B=128,D=4096,G=16` | Kernels `3->1`; nested metric `0->1`; `8.54->8.51 us` (`1.004x`). #183638 remains context-only for inventory credit. |
| RMSNorm MXFP8-style path, `B=128,D=4096,G=32` | Kernels `3->2`; nested metric `0->1`; `14.08->14.18 us` (`0.993x`). This is an Inductor-expressible approximation, not the final production swizzled MXFP8 pack. |
| Half-resolution sub-parent epilogue, `B=128,D=4096,G=16` | Kernels `2->1`; `8.54->8.35 us` (`1.023x`). Useful current-HEAD context, not inventory PR credit without PR/mainline proof. |
| Weighted RMSNorm reduce-K, `B=64,K=16,D=4096` | Kernels `2->1`; runtime flat around `54.82->54.88 us` (`0.999x`). |
| Chunk SwiGLU, `B=128,D=1024` | Kernels `2->1`; runtime flat around `8.35->8.38 us` (`0.996x`). |
| Chunk4 gated consumers, `B=128,D=1024` | Kernels `2->1`; runtime flat around `8.29->8.32 us` (`0.996x`). |

Interpretation: the nested stack proves codegen coverage, launch-count
reduction, and removal of intermediate materialization for grouped reduction and
NVFP4-style flows. It does not yet prove a broad steady-state B200 kernel-body
speedup on the selected static shapes.

## Other completed local probes

- Combo kernels without PDL helped simple independent pointwise fusion in the
  earlier local slice: 6 kernels collapsed to 1 and graph replay improved
  `0.0412->0.0329 ms` (`1.25x`). PDL-specific synthetic rows emitted
  `launch_pdl=True` and `gdc_launch`, but regressed by synchronized wall time on
  B200 for the tested simple pointwise combo shapes.
- Current-head CUDA graph policy probes for self-overlap, parameter pointer
  churn, and retained live outputs validate current behavior and counters, but
  they are not pre/post attribution.
- Current-head custom-op autotune API rows verify the CUDA graph benchmark path,
  but `aten.mm` max-autotune attribution remains too coarse from current toggles.
- #177922 `inline_asm_elementwise` HOP is locally graph-capturable and generated
  inline asm in the earlier local primitive rows, but it has no current-head
  feature-off toggle for strict shipped attribution.

## Remaining pre/post-only local work

No parent-vs-landed checkout or rebuild was performed in these targeted passes.
The following local items remain pre/post-only for strict landed-impact claims:

| Cluster | PRs and commits from the reports | Required comparison |
| --- | --- | --- |
| Wide-symbol compile-time fast paths | #181275 `f296c860884`, #181276 `b44a57da7e3`, #181277 `3565a492def`/`89ab27e972a`, #176345 `54028337cf5` | Parent vs landed checkouts on a wide-symbol after-AOT repro. Measure compile wall time, time in `sympy.gcd`, `statically_known_true/false`, call counts, generated code size, and preserved symbolic relationships. |
| CUDA graph policy and live-output behavior | #182524 `10efc0f4c57`, #182531 `7089ea8477d`, #188117 `d46712a5fb2`, #188078 `d877745fe08`, #176620, #174103 | Parent vs landed checkouts with `mode="reduce-overhead"` harnesses for self-overlap, parameter pointer churn, live-output cloning, saved activations, and graph opt-out annotations. |
| Wrapper overhead and first-use deferral | #177783 `f0c3582ef2d`/`779c78b3d7d`, #179039 `ddaac926c33`/`55fc17f8653`, #180599 `13a6d0ddbc7` | Parent vs landed checkouts with CPU wrapper-entry-to-first-kernel instrumentation. CUDA graph replay misses the host placement win. |
| Tiling/coalescing attribution | #182895 `5387f64e4d2`, #183585 `a13f6966448` | Parent vs landed checkouts to separate block-size-floor propagation from coalescing-analysis caching. |
| Custom-op and `aten.mm` autotune attribution | #175275 `dd3c11bbb3b`, #175276 `1bb77ba1b09`, #175277 `9e85a6ead69`, #175278 `b9b20336385`/`e8120f9062c`/`d7f39875e57`, #175422 `67a21876c88` | Commit-by-commit parent vs landed benchmarks for custom-op choices, CUDA graph benchmarking cleanup, scoped config patches, `aten.mm` registration, and fallback normalization. |
| Horizontal partition fusion skip | #184904 `eeccb459bcc` | Current HEAD can compare `CapabilityBasedPartitioner(skip_horizontal_fusion=False/True)`, but shipped impact needs parent vs landed because the old commit lacks the API. |
| GPU benchmark lock hooks | #184905 `d79ee01a97f` | Parent vs landed under 4-8 local autotuning workers contending on one GPU. This is single-GPU contention only, not distributed overlap. |
| Strict low-precision primitive/HOP attribution and production pack paths | #172497 `b79269a4308`, #177922 `a9be19830db`/`1ae64875e7e`; #183638 context only | Current-head fallback/toggle evidence is useful, but strict commit attribution still needs parent vs landed. Production NVFP4/MXFP8 pack throughput needs the actual production pack/swizzle primitive; #183638 should not be credited until inventory/mainline proof is found. |
| Missing-op fallback log laziness | #185951 `067e9a9f1a1` | Parent vs landed compile/logging benchmark on large fallback/missing-op graphs. Measure formatting call counts and compile time with INFO disabled/enabled. |

Distributed and multi-GPU overlap benchmarks are intentionally omitted.

## Primary artifacts

- `agent_space/eellison_h1_2026_pr_inventory.md`
- `agent_space/eellison_h1_2026_local_impact_summary.md`
- `agent_space/eellison_h1_2026_loop_reindexing_toggle_benchmarks.md`
- `agent_space/eellison_h1_2026_cvt_e8m0_rceil_benchmarks.md`
- `agent_space/eellison_h1_2026_nested_benchmarks.md`
- `agent_space/eellison_h1_2026_nested_impact.md`
- `agent_space/eellison_h1_2026_local_remaining_benchmarks.md`
- `agent_space/eellison_h1_2026_remaining_benchmark_queue.md`
