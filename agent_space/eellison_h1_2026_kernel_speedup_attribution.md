# H1 2026 Kernel Speedup Attribution Audit

Generated: 2026-07-13

Scope: local scratch artifacts under `agent_space/` only. No source files outside
`agent_space/` were modified. I audited the existing benchmark reports, raw JSON,
generated wrapper/code snippets, and cache source files, then added one scratch
capture pass for missing nested-reduction kernel names:

- Existing reports: `eellison_h1_2026_loop_reindexing_toggle_benchmarks.md`,
  `eellison_h1_2026_nested_benchmarks.md`,
  `eellison_h1_2026_cvt_e8m0_rceil_benchmarks.md`,
  `eellison_h1_2026_local_remaining_benchmarks.md`,
  `eellison_h1_2026_loop_fusion_autotune_benchmarks.md`.
- Existing raw data: `h1_loop_reindexing_toggle_results.json`,
  `eellison_h1_2026_nested_benchmarks.json`,
  `eellison_h1_2026_cvt_e8m0_rceil_benchmarks.json`,
  `h1_local_remaining_results.json`,
  `h1_loop_fusion_autotune_results.json`.
- Added scratch attribution capture:
  `capture_h1_nested_kernel_attribution.py`,
  `h1_nested_attribution_kernels.json`,
  `h1_nested_attribution_code/`.

## Bottom Line

The reports mostly correctly show which benchmark cases get faster, but the
kernel attribution is different by cluster:

- Loop reindexing, loop ordering, pointwise cat, combo kernels, and nested
  reductions are mostly not "the same old kernel got faster." They remove one
  or more old kernels and replace them with one fused kernel.
- `cvt_e8m0_rceil` is the clearest same-kernel-count generated-body speedup:
  fallback log2/bitwise encode kernels are replaced by a one-kernel SM100 PTX
  `cvt.rp.satfinite.ue8m0x2.f32` path.
- The coalescing/tiling induced-amax row is also same kernel count and same
  source name, with replay improvement from the generated body/config under the
  `triton.coalesce_tiling_analysis=True` surface.
- Nested reductions prove launch/materialization removal, but the measured B200
  body times are mostly flat; only the half-resolution sub-parent row has a
  meaningful local positive median (`1.023x`), and that row is context-only
  without PR inventory credit.
- PDL combo rows, max-autotune rows, custom-op API rows, and CUDA graph policy
  rows should not be described as kernel-body speedups.

## Loop Reindexing, Loop Ordering, Pointwise Cat

Primary artifacts:

- Raw: `agent_space/h1_loop_reindexing_toggle_results.json`
- Generated snippets: `agent_space/h1_loop_reindexing_toggle_code/`
- Larger-shape refinements: `agent_space/h1_local_remaining_results.json`

| PR(s) | Benchmark case | Before -> after config | Timing | Kernels | Kernel attribution |
| --- | --- | --- | --- | --- | --- |
| #179090 | outer sum + pointwise, `x=(32,2^20)` | `loop_ordering_after_fusion=False -> True` | direct `108.07 -> 80.58 us` (`1.34x`), graph `104.84 -> 79.93 us` (`1.31x`) | `2 -> 1` | Old `triton_per_fused_sum_0` plus `triton_poi_fused_add_1` disappeared. New fused kernel is `triton_per_fused_add_sum_0`. Snippets: `h1_loop_reindexing_toggle_code/loop_ordering_outer_sum_False/`, `.../loop_ordering_outer_sum_True/`. |
| #179090 | outer softmax, `x=(32,2^20)` | `loop_ordering_after_fusion=False -> True` | direct `144.20 -> 133.73 us` (`1.08x`), graph `150.07 -> 133.20 us` (`1.13x`) | `2 -> 1` | Old `triton_per_fused_prepare_softmax_online_0` plus `triton_poi_fused__softmax_1` disappeared. New fused kernel is `triton_per_fused__softmax_prepare_softmax_online_0`. Snippets: `h1_loop_reindexing_toggle_code/loop_ordering_outer_softmax_False/`, `.../loop_ordering_outer_softmax_True/`. |
| #179090, #176927 | grouped quant candidate selection, `x=(8,7168)` bf16 | scratch legacy candidate scoring -> current HEAD candidate scoring | direct `58.99 -> 52.36 us` (`1.13x`), graph `6.20 -> 4.15 us` (`1.49x`) | `2 -> 1` | Old `triton_per_fused__to_copy_abs_amax_clamp_div_opaque_gemm_..._squeeze_view_0` plus `triton_poi_fused__to_copy_clamp_div_opaque_gemm_..._squeeze_view_1` collapsed into the first fused reduction kernel. Snippets: `h1_loop_reindexing_toggle_code/loop_ordering_candidate_grouped_quant_legacy_True/`, `..._legacy_False/`. |
| #176927 | RMSNorm reshape slice, `x=(16,8192)` bf16 | `loop_reindexing_after_fusion=False -> True`, loop ordering disabled | direct `30.32 -> 21.15 us` (`1.43x`), graph `4.20 -> 3.04 us` (`1.38x`) | `2 -> 1` | Old reduction `triton_per_fused__to_copy__unsafe_view_clone_mean_pow_0` plus pointwise `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_view_1` disappeared. New fused kernel is `triton_per_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_view_0`. |
| #176927, #179090 | RMSNorm reshape transposed, `x=(16,8192)` bf16 stride `(1,16)` | `loop_reindexing_after_fusion=False -> True`, loop ordering enabled | direct `27.40 -> 21.45 us` (`1.28x`), graph `6.18 -> 4.09 us` (`1.51x`) | `2 -> 1` | Old `triton_red_fused__to_copy__unsafe_view_clone_mean_pow_0` plus `triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_view_1` disappeared. New fused kernel is `triton_red_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_view_0`. |
| #176927, #179090 | larger transposed RMSNorm, `x=(64,16384)` stride `(1,64)` | `loop_reindexing_after_fusion=False -> True` | direct event `8.35 -> 31.48 us` (`0.27x`), graph `6.26 -> 4.23 us` (`1.48x`) | `2 -> 1` | Same attribution as the smaller transposed row: old reduction plus pointwise kernels collapsed into `triton_red_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_view_0`. Direct timing is noisy/regressed, so only graph replay supports a speedup claim here. Source names are in `h1_local_remaining_cache/loop_reindex_rmsnorm_transposed_large_*`. |
| #179091 | QKNorm + split RoPE cat, `B=4,H=8,S=128,D=64` | `max_*_pointwise_cat_inputs=0 -> 8` | direct `32.80 -> 21.89 us` (`1.50x`), graph `4.21 -> 4.14 us` (`1.02x`) | `2 -> 1` | Old `triton_per_fused_mean_pow_0` plus `triton_poi_fused_add_mean_mul_pow_rsqrt_slice_sub_1` disappeared. New fused cat kernel is `triton_per_fused_add_cat_mean_mul_pow_rsqrt_slice_sub_0`. |
| #179091 | QKNorm + interleaved RoPE stack/flatten | `max_*_pointwise_cat_inputs=0 -> 8` | direct `29.85 -> 24.00 us` (`1.24x`), graph `4.19 -> 6.19 us` (`0.68x`) | `2 -> 1` | Old `triton_per_fused_mean_pow_0` plus `triton_poi_fused_add_mean_mul_pow_rsqrt_select_stack_sub_view_1` disappeared. New fused kernel is `triton_per_fused_add_mean_mul_pow_rsqrt_select_stack_sub_view_0`. Graph replay regressed, so this is a launch/materialization win in direct timing, not a faster body. |
| #179091 | larger split RoPE, `B=8,H=16,S=256,D=128` | `max_*_pointwise_cat_inputs=0 -> 8` | direct event `28.46 -> 12.46 us` (`2.29x`), graph `10.33 -> 12.48 us` (`0.83x`) | `2 -> 1` | Old `triton_per_fused_mean_pow_0` plus `triton_poi_fused_add_mean_mul_pow_rsqrt_slice_sub_1` collapsed into `triton_per_fused_add_cat_mean_mul_pow_rsqrt_slice_sub_0`. Graph replay says fused body is slower at this shape. |
| #179091 | larger interleaved RoPE | `max_*_pointwise_cat_inputs=0 -> 8` | direct event `30.55 -> 23.01 us` (`1.33x`), graph `14.52 -> 18.52 us` (`0.78x`) | `2 -> 1` | Old `triton_per_fused_mean_pow_0` plus `triton_poi_fused_add_mean_mul_pow_rsqrt_select_stack_sub_view_1` collapsed into `triton_red_fused_add_mean_mul_pow_rsqrt_select_stack_sub_view_0`. Graph replay again regressed. |

Audit verdict: the loop/cat reports are accurate if interpreted as fusion and
launch/materialization removal. They should not generally be described as the
same kernel becoming faster.

## Nested Reductions And Low-Precision Pack Paths

Primary artifacts:

- Timings: `agent_space/eellison_h1_2026_nested_benchmarks.json`
- Added attribution capture: `agent_space/h1_nested_attribution_kernels.json`
- Generated snippets: `agent_space/h1_nested_attribution_code/`

Credited landed nested stack in the inventory: #182896, #182897, #182898,
#183432, #184821. #183638 and the no-PR sub-parent commits are current-HEAD
context only in the local inventory baseline.

| PR(s) / context | Benchmark case | Timing | Kernels | Kernel attribution |
| --- | --- | --- | --- | --- |
| #182896/#182897/#182898/#183432 | RMSNorm block amax/scale, `B=128,D=4096,G=16` | graph `8.58 -> 8.54 us` (`1.004x`) | `2 -> 1` | Old `triton_red_fused__fused_rms_norm_0` plus `triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_view_1` disappeared. Current nested body is emitted as `triton_red_fused__fused_rms_norm_0`. Snippets: `h1_nested_attribution_code/rmsnorm_block_amax_scale_B128_D4096_G16_off/`, `..._on/`. |
| #183638 context, #177922 dependency | RMSNorm NVFP4-style pack, `B=128,D=4096,G=16` | graph `8.54 -> 8.51 us` (`1.004x`) | `3 -> 1` | Old RMSNorm reduction `triton_red_fused__fused_rms_norm_0`, scale kernel `triton_per_fused__fused_rms_norm__to_copy_abs_amax_clamp_div_view_1`, and pack/div kernel `triton_poi_fused__fused_rms_norm__to_copy_div_select_unsqueeze_view_2` disappeared into nested `triton_red_fused__fused_rms_norm_0`. This is count/materialization evidence, not a meaningful body speedup. |
| #182896/#182897/#182898/#183432 plus #172497-like e8m0 path | RMSNorm MXFP8-style, `B=128,D=4096,G=32` | graph `14.08 -> 14.18 us` (`0.993x`) | `3 -> 2` | Old `triton_per_fused__fused_rms_norm_abs_amax_clamp_div_view_1` disappeared into nested `triton_red_fused__fused_rms_norm_0`; payload conversion remained separate as `triton_poi_fused__fused_rms_norm__to_copy_div_unsqueeze_view_1`. Not a speedup. |
| no-PR sub-parent context | half-resolution sub-parent, `B=128,D=4096,G=16` | graph `8.54 -> 8.35 us` (`1.023x`) | `2 -> 1` | Old `triton_per_fused__to_copy_abs_amax_clamp_div_view_0` plus `triton_poi_fused__to_copy_add_div_select_sub_unsqueeze_view_1` disappeared. New fused kernel is `triton_per_fused__to_copy_abs_add_amax_clamp_div_select_sub_unsqueeze_view_0`. |
| #182898 | weighted RMSNorm reduce-K, `B=64,K=16,D=4096` | graph `54.82 -> 54.88 us` (`0.999x`) | `2 -> 1` | Old `triton_red_fused_mean_mul_view_0` plus `triton_per_fused_add_div_mean_mul_sqrt_sum_unsqueeze_view_1` disappeared into nested `triton_red_fused_mean_mul_view_0`. Flat runtime. |
| no-PR contiguous sub-parent context | chunk SwiGLU, `B=128,D=1024` | graph `8.35 -> 8.38 us` (`0.996x`) | `2 -> 1` | Old `triton_per_fused_add_mean_pow_0` plus `triton_poi_fused_add_mean_mul_pow_rsqrt_silu_split_1` disappeared. New fused kernel is `triton_per_fused_add_mean_mul_pow_rsqrt_silu_split_0`. Flat/slower runtime. |
| no-PR contiguous sub-parent context | chunk4 gated consumers, `B=128,D=1024` | graph `8.29 -> 8.32 us` (`0.996x`) | `2 -> 1` | Old `triton_per_fused_add_mean_pow_0` plus `triton_poi_fused_add_mean_mul_pow_rsqrt_silu_split_tanh_1` disappeared. New fused kernel is `triton_per_fused_add_mean_mul_pow_rsqrt_silu_split_tanh_0`. Flat/slower runtime. |

Audit verdict: nested reports correctly show kernel-count reductions. They do
not support a broad claim that nested kernels are faster on these B200 static
shapes; the strongest claim is removal of old launches and global intermediate
materialization.

## `cvt_e8m0_rceil`

Primary artifacts:

- Focused report/raw data:
  `agent_space/eellison_h1_2026_cvt_e8m0_rceil_benchmarks.md`,
  `agent_space/eellison_h1_2026_cvt_e8m0_rceil_benchmarks.json`
- Primitive source snippets from local remaining cache:
  `agent_space/h1_local_remaining_cache/cvt_e8m0_rceil_ptx_prim_float32/triton/45WVPKT37MK4NSGV2CLO3Q3NGGNGGPEH2OSNXDV4PE6ZJV4IWIGA/triton_poi_fused_inductor_cvt_e8m0_rceil_0.source`
  and
  `agent_space/h1_local_remaining_cache/cvt_e8m0_rceil_bitwise_fallback_float32/triton/KOYS3IKX5SHFT44A3B6GFWDUPNWDGC3F5SIDTOY7QUM35OR7XWLQ/triton_poi_fused___rshift____to_copy_add_bitwise_and_clamp_ne_view_0.source`.

PR: #172497, `[Inductor] Add cvt_e8m0_rceil prim with PTX lowering for SM100+`.

| Benchmark case | Before -> after mechanism | Timing | Kernels | Kernel attribution |
| --- | --- | --- | --- | --- |
| focused `encode_only`, `1024x128` | `pattern_matcher=False` fallback log2/ceil -> current pattern replacement/PTX | graph `0.0040 -> 0.0022 ms` (`1.826x`) | `1 -> 1` | Fallback `triton_poi_fused__to_copy_add_ceil_clamp_log2_view_0` disappeared. Current single kernel is `triton_poi_fused__to_copy_view_0` containing the inductor prim and `cvt.rp.satfinite.ue8m0x2.f32`. |
| focused `encode_only`, `2048x128` | same | graph `0.0041 -> 0.0030 ms` (`1.357x`) | `1 -> 1` | Same attribution as above. |
| focused `encode_only`, `8192x256` | same | graph `0.0061 -> 0.0041 ms` (`1.500x`) | `1 -> 1` | Same attribution as above. |
| focused `encode_only`, `8192x512` | same | graph `0.0082 -> 0.0041 ms` (`2.000x`) | `1 -> 1` | Same attribution as above. |
| focused `scale_from_bf16`, all tested shapes | full BF16 scale generation with fallback log2/ceil -> current PTX encode after surrounding amax/reduction | graph speedups `1.000x` to `1.333x` | `1 -> 1` | Same encode replacement is present, but amax/reduction work dominates; this is not a broad full-kernel speedup. |
| primitive local remaining, fp32 `1<<20` | bitwise fallback kernel -> PTX prim kernel | direct `23.09 -> 4.24 us` (`5.45x`), graph `4.15 -> 2.90 us` (`1.43x`) | `1 -> 1` | Old source `triton_poi_fused___rshift____to_copy_add_bitwise_and_clamp_ne_view_0`; new source `triton_poi_fused_inductor_cvt_e8m0_rceil_0` contains PTX and inline asm. |
| primitive local remaining, fp16/bf16 `1<<20` | same | fp16 direct `4.19 -> 4.20 us`, bf16 direct `22.29 -> 22.05 us` | `1 -> 1` | Same kernel replacement exists, but measured throughput is flat for these dtypes. |

Audit verdict: this cluster is a real same-kernel-count generated-body speedup,
especially for fp32/encode-only work. The report should keep end-to-end
`scale_from_bf16` claims modest because the encode step is only part of the
kernel.

## Combo Kernels And PDL

Primary artifacts:

- Raw: `agent_space/h1_loop_fusion_autotune_results.json`,
  `agent_space/h1_local_remaining_results.json`
- Source snippets:
  old six kernels under `agent_space/h1_loop_fusion_autotune_cache/combo_pointwise_False/triton/*/*.source`;
  new combined kernel
  `agent_space/h1_loop_fusion_autotune_cache/combo_pointwise_True/triton/DI4KIFYWBLNGHL24NV4J5ZJYPLE4E7TD53HKKTZBOGCY5TFFUZMQ/triton_poi_fused_0.source`.

PR: #174232, `Add PDL support to combo kernels`. The positive local row is
ordinary combo-kernel fusion, not PDL itself.

| Benchmark case | Before -> after config | Timing | Kernels | Kernel attribution |
| --- | --- | --- | --- | --- |
| six independent pointwise ops, `6 * [2^22]` | `combo_kernels=False -> True` | direct `50.35 -> 33.77 us` (`1.49x`), graph `41.22 -> 32.90 us` (`1.25x`) | `6 -> 1` | Old kernels `triton_poi_fused_mul_0`, `triton_poi_fused_add_1`, `triton_poi_fused_sin_2`, `triton_poi_fused_cos_3`, `triton_poi_fused_exp_4`, `triton_poi_fused_neg_5` disappeared. New fused combo source is `triton_poi_fused_0`. |
| PDL combo pointwise, `n=4096/65536/1048576` | `triton.enable_pdl=False -> True`, already one combo kernel | wall speedups `0.85x`, `0.38x`, `0.42x` in local remaining; graph/event rows also mixed/regressed | `1 -> 1` | Same source name `triton_poi_fused_0`. PDL true emits `gdc_launch`/`gdc_wait` sites (`8` each), but this synthetic B200 row does not get faster. |

Audit verdict: combo-kernel fusion has a valid kernel-disappearance speedup.
The PDL-specific rows are negative coverage and should not be summarized as a
speedup.

## Tiling And Coalescing Surface

Primary artifacts:

- Raw: `agent_space/h1_loop_fusion_autotune_results.json`
- Source snippets:
  `agent_space/h1_loop_fusion_autotune_cache_v3/tiling_induced_amax_coalesce_False/triton/AZGKRHNXUC6GWLDBHYOS6IPR5BSD5WLVR4Y7QOCQBZTQJQJI6KYA/triton_per_fused_amax_clone_view_0.source`
  and
  `agent_space/h1_loop_fusion_autotune_cache_v3/tiling_induced_amax_coalesce_True/triton/A2MOPSRESJ4D7YAUEAXXCZBVONTPFVVWBBAJRFLT5EYDWECBA6RQ/triton_per_fused_amax_clone_view_0.source`.

PRs: #182895 (`Thread block-size floors through Triton autotuning`) and #183585
(`Cache scheduler coalescing analysis`). The current-head toggle is a surface
measurement, not strict PR-by-PR landed attribution.

| Benchmark case | Before -> after config | Timing | Kernels | Kernel attribution |
| --- | --- | --- | --- | --- |
| permute clone view amax, bf16 `[1024,2048] -> [2048,1024]` | `triton.coalesce_tiling_analysis=False,max_tiles=2 -> True,max_tiles=3` | direct `20.42 -> 20.47 us` (`1.00x`), graph `12.33 -> 6.20 us` (`1.99x`) | `1 -> 1` | Same kernel name `triton_per_fused_amax_clone_view_0`; generated body/config differs between source snippets. This is same-kernel-count attribution, not launch removal. |
| contiguous + transposed add, `4096x4096` | same | direct `34.56 -> 34.65 us`, graph `34.03 -> 33.49 us` | `1 -> 1` | Same kernel name `triton_poi_fused_add_0`; effectively flat. |

Audit verdict: only the induced-amax graph replay row supports a speedup, and it
is a same-kernel-count generated-body/config result. Direct timing is flat.

## CUDA Graph Policy, Autotune, And Other Local Rows

These rows were useful probes, but they should not be included as "which kernels
got faster" without more qualification.

| PR(s) | Row | Kernel attribution |
| --- | --- | --- |
| #175271 | `cudagraph_min_partition_size 0 -> 10`, small matmul chain | Same generated Triton source `triton_poi_fused_add_cos_mul_sin_0`, count `1 -> 1`. Direct `38.65 -> 31.82 us`, but this is CUDA graph partition/capture policy; the `0` side failed external graph replay. Not a kernel-body speedup. |
| #175275/#175276/#175277/#175278/#175422 | custom-op autotune and `aten.mm` max-autotune rows | The current-head custom-op rows validate API path/cudagraph benchmarking. The `aten.mm` max-autotune rows generated many candidate kernels and regressed on tested shapes. Not clean shipped kernel-speedup attribution. |
| #177922 | inline_asm_elementwise HOP rows | Current HEAD emits graph-capturable inline asm, but there is no off toggle in the local reports. Treat as enabling evidence for NVFP4/cvt-style paths, not a measured before/after speedup by itself. |

## Overall Attribution Answer

The reports are directionally correct about faster benchmark cases, but the
concrete attribution should be stated this way:

- #176927/#179090/#179091: faster cases are from old reduction/pointwise/cat
  kernels disappearing and one fused kernel replacing them.
- #182896/#182897/#182898/#183432 nested stack: old dependent kernels disappear,
  but measured body times are mostly flat on the tested B200 shapes.
- #172497: same generated-kernel count; fallback log2/bitwise encode kernels are
  replaced by a PTX `cvt_e8m0_rceil` body, with clear encode-only speedups.
- #174232 combo fusion: six old pointwise kernels disappear into one combo
  kernel; PDL-specific enabled rows do not show a speedup.
- #182895/#183585 coalescing surface: same kernel name/count; one amax replay row
  gets faster from generated body/config changes, while direct timing is flat.
- CUDA graph policy/autotune API rows are not kernel attribution and should stay
  outside a "which kernels got faster" headline.
