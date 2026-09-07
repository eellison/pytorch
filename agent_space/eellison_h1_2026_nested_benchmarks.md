# H1 2026 eellison nested-reduction benchmarks

Generated: 2026-07-13T17:33:05

## Methodology

Each case was compiled twice with `torch.compile(..., fullgraph=True, dynamic=False)`: first with `torch._inductor.config.patch({'triton.nested_reduction': False})`, then with the same patch set to `True`. Compile caches were fresh per side and rooted under `agent_space/.bench_cache_h1_2026_nested/`.

Timing used 20 warmup calls, captured one compiled callable invocation inside `torch.cuda.CUDAGraph`, then measured 1000 graph replays with CUDA events. The reported times are GPU elapsed time in microseconds, so torch.compile and Python launch setup overhead are excluded.

## Environment

- GPU: NVIDIA B200 capability [10, 0]
- GPU memory: 182631 MiB
- SM count: 148
- torch: 2.13.0a0+gite1067a5
- CUDA runtime: 12.8
- Triton: 3.6.0
- nvidia-smi: NVIDIA B200, 10.0, 183359, 580.82.07 (8 GPUs listed)

## Results

| case | off med us | on med us | speedup | kernels off->on | nested off->on | IR nodes off->on |
| --- | --- | --- | --- | --- | --- | --- |
| `rmsnorm_block_amax_scale_B128_D4096_G16` | 8.58 | 8.54 | 1.004x | 2->1 | 0->1 | 3->3 |
| `rmsnorm_nvfp4_pack_B128_D4096_G16` | 8.54 | 8.51 | 1.004x | 3->1 | 0->1 | 4->4 |
| `rmsnorm_mxfp8_style_B128_D4096_G32` | 14.08 | 14.18 | 0.993x | 3->2 | 0->1 | 8->8 |
| `half_resolution_subparent_B128_D4096_G16` | 8.54 | 8.35 | 1.023x | 2->1 | 0->1 | 4->4 |
| `rmsnorm_weighted_reduce_B64_K16_D4096` | 54.82 | 54.88 | 0.999x | 2->1 | 0->1 | 2->2 |
| `rmsnorm_chunk_swiglu_B128_D1024` | 8.35 | 8.38 | 0.996x | 2->1 | 0->1 | 2->2 |
| `rmsnorm_chunk4_gating_B128_D1024` | 8.29 | 8.32 | 0.996x | 2->1 | 0->1 | 2->2 |

## Cases

### `rmsnorm_block_amax_scale_B128_D4096_G16`

- Status: ok
- Pattern: RMSNorm followed by per-16-value block amax and FP16 scale.
- Shapes: x=(128,4096) bf16, weight=(4096) bf16, group=16
- Outputs: `[{'shape': [128, 256], 'dtype': 'float16', 'numel': 32768}]`
- Off metrics: `{"codegen_nested_reduction": 0, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 2, "ir_nodes_pre_fusion": 3, "num_bytes_accessed": 0}`
- On metrics: `{"codegen_nested_reduction": 1, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 1, "ir_nodes_pre_fusion": 3, "num_bytes_accessed": 0}`
- Timing: off median 8.58 us (min 6.37 us), on median 8.54 us (min 6.21 us)

### `rmsnorm_nvfp4_pack_B128_D4096_G16`

- Status: ok
- Pattern: RMSNorm, block amax/FP8 scale, then NVFP4 e2m1x2 inline-asm packing.
- Shapes: x=(128,4096) bf16, weight=(4096) bf16, group=16
- Outputs: `[{'shape': [128, 2048], 'dtype': 'uint8', 'numel': 262144}, {'shape': [128, 256], 'dtype': 'float8_e4m3fn', 'numel': 32768}]`
- Off metrics: `{"codegen_nested_reduction": 0, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 3, "ir_nodes_pre_fusion": 4, "num_bytes_accessed": 0}`
- On metrics: `{"codegen_nested_reduction": 1, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 1, "ir_nodes_pre_fusion": 4, "num_bytes_accessed": 0}`
- Timing: off median 8.54 us (min 7.74 us), on median 8.51 us (min 7.81 us)

### `rmsnorm_mxfp8_style_B128_D4096_G32`

- Status: ok
- Pattern: RMSNorm, per-32-value amax, e8m0 scale, and FP8 payload conversion.
- Shapes: x=(128,4096) bf16, weight=(4096) bf16, group=32
- Outputs: `[{'shape': [128, 4096], 'dtype': 'float8_e4m3fn', 'numel': 524288}, {'shape': [128, 128], 'dtype': 'float8_e8m0fnu', 'numel': 16384}]`
- Off metrics: `{"codegen_nested_reduction": 0, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 3, "ir_nodes_pre_fusion": 8, "num_bytes_accessed": 0}`
- On metrics: `{"codegen_nested_reduction": 1, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 2, "ir_nodes_pre_fusion": 8, "num_bytes_accessed": 0}`
- Timing: off median 14.08 us (min 12.48 us), on median 14.18 us (min 12.80 us)
- Caveat: This is an Inductor-expressible MXFP8-style microbenchmark, not a full production swizzled MXFP8 packing path.

### `half_resolution_subparent_B128_D4096_G16`

- Status: ok
- Pattern: Standalone block amax/scale with even/odd half-resolution consumers.
- Shapes: x=(128,4096) bf16, group=16
- Outputs: `[{'shape': [128, 256, 8], 'dtype': 'float32', 'numel': 262144}, {'shape': [128, 256, 8], 'dtype': 'float32', 'numel': 262144}, {'shape': [128, 256], 'dtype': 'float32', 'numel': 32768}]`
- Off metrics: `{"codegen_nested_reduction": 0, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 2, "ir_nodes_pre_fusion": 4, "num_bytes_accessed": 0}`
- On metrics: `{"codegen_nested_reduction": 1, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 1, "ir_nodes_pre_fusion": 4, "num_bytes_accessed": 0}`
- Timing: off median 8.54 us (min 7.78 us), on median 8.35 us (min 7.78 us)

### `rmsnorm_weighted_reduce_B64_K16_D4096`

- Status: ok
- Pattern: RMSNorm over D followed by weighted K-axis reduction.
- Shapes: x=(64,16,4096) bf16, w=(64,16) bf16
- Outputs: `[{'shape': [64, 4096], 'dtype': 'bfloat16', 'numel': 262144}]`
- Off metrics: `{"codegen_nested_reduction": 0, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 2, "ir_nodes_pre_fusion": 2, "num_bytes_accessed": 0}`
- On metrics: `{"codegen_nested_reduction": 1, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 1, "ir_nodes_pre_fusion": 2, "num_bytes_accessed": 0}`
- Timing: off median 54.82 us (min 53.25 us), on median 54.88 us (min 53.22 us)

### `rmsnorm_chunk_swiglu_B128_D1024`

- Status: ok
- Pattern: Residual RMSNorm feeding chunk(2) SwiGLU consumer.
- Shapes: x/residual=(128,1024) bf16, weight=(1024) bf16
- Outputs: `[{'shape': [128, 512], 'dtype': 'bfloat16', 'numel': 65536}]`
- Off metrics: `{"codegen_nested_reduction": 0, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 2, "ir_nodes_pre_fusion": 2, "num_bytes_accessed": 0}`
- On metrics: `{"codegen_nested_reduction": 1, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 1, "ir_nodes_pre_fusion": 2, "num_bytes_accessed": 0}`
- Timing: off median 8.35 us (min 7.81 us), on median 8.38 us (min 7.84 us)

### `rmsnorm_chunk4_gating_B128_D1024`

- Status: ok
- Pattern: Residual RMSNorm feeding chunk(4) gated pointwise consumer.
- Shapes: x/residual=(128,1024) bf16, weight=(1024) bf16
- Outputs: `[{'shape': [128, 256], 'dtype': 'bfloat16', 'numel': 32768}]`
- Off metrics: `{"codegen_nested_reduction": 0, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 2, "ir_nodes_pre_fusion": 2, "num_bytes_accessed": 0}`
- On metrics: `{"codegen_nested_reduction": 1, "generated_cpp_vec_kernel_count": 0, "generated_kernel_count": 1, "ir_nodes_pre_fusion": 2, "num_bytes_accessed": 0}`
- Timing: off median 8.29 us (min 7.68 us), on median 8.32 us (min 7.74 us)

## Caveats And Proposed Benchmarks

- `rmsnorm_mxfp8_style_B128_D4096_G32` measures the nested-reduction piece of an MXFP8-like path: per-block amax, e8m0 scale, and FP8 payload conversion. It does not prove the final production swizzled MXFP8 memory layout. The exact follow-up benchmark should call the eventual production MXFP8 pack/swizzle primitive after `scale = amax.to(torch.float8_e8m0fnu)` and compare the same nested-reduction off/on toggle.
- The kernel counts are Inductor generated-kernel metrics collected during fresh compilation, not profiler launch counts. They are sufficient here to confirm the expected 2-or-more-kernel off path collapsing to one nested kernel when `codegen_nested_reduction` increments.
- `metrics.num_bytes_accessed` was zero for these generated reduction kernels, so the summary table reports `ir_nodes_pre_fusion`; the complete metric dictionaries are listed per case.
- The benchmarks are forward microbenchmarks with fixed static shapes. They do not cover backward, dynamic-shape recompilation behavior, or end-to-end model overlap effects.

Selected cases: `rmsnorm_block_amax_scale_B128_D4096_G16`, `rmsnorm_nvfp4_pack_B128_D4096_G16`, `rmsnorm_mxfp8_style_B128_D4096_G32`, `half_resolution_subparent_B128_D4096_G16`, `rmsnorm_weighted_reduce_B64_K16_D4096`, `rmsnorm_chunk_swiglu_B128_D1024`, `rmsnorm_chunk4_gating_B128_D1024`
