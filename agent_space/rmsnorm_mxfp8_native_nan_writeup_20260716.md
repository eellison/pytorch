# RMSNorm -> MXFP8 vs Transformer Engine

## Result

The production-shape gap was caused primarily by Inductor's NaN-propagating
min/max helpers, not by missing fusion. Replacing compare/select helper bodies
with Triton's native `PropagateNan.ALL` operations changes SM80+ codegen to
single `min.NaN` / `max.NaN` instructions while preserving NaN behavior.

At `4096x8192`, unchanged scheduler selection improves from 40.83 us to
30.59 us and beats fused Transformer Engine at 32.77 us. Forcing the existing
persistent-reduction path reaches 28.54 us, but persistent config selection is
not stable enough to include in this fix.

## Methodology

- GPU: NVIDIA B200, SM100
- CUDA runtime / driver: 12.8 / 580.82.07
- PyTorch source HEAD: `e552f86629fdb358e2e6ed784a30fc4b3abae3da`
- PyTorch binary commit: `e1067a5ad33609f9486adefbbd627e18281844c2`
- Transformer Engine: 2.17.0, tag commit `2e559f062497bef768dfbe9d7e45548fadeca80a`
- cuDNN for fused TE path: 9.24.0.43
- Timing: `triton.testing.do_bench` over CUDA graph replay
- Warmup / rep: 25 ms / 100 ms, three measurements per graph
- Cache policy: normal caches enabled; no fresh-cache context
- Clock preconditioning: untimed `torch.cuda._sleep(200_000_000)`
- Timed outputs: raw E4M3 payload plus E8M0 scales; no dequant or `q.float()`
- Correctness: dequantized outside timing against FP32 RMSNorm reference,
  `atol=0.05`, `rtol=0.05`; all rows passed with zero mismatches
- Input: BF16, epsilon `1e-5`, MXFP8 block size 32
- Compile mode: `max-autotune-no-cudagraphs`

## Compact Scales

TE uses its one-kernel cuDNN fused RMSNorm -> MXFP8 path. Both implementations
return compact E8M0 scales.

| Shape | TE fused | Inductor nested + asm | Speedup | Kernels TE / Inductor |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 8.06 us | 8.06 us | 1.00x | 1 / 1 |
| `1024x4096` | 12.16 us | 10.11 us | 1.20x | 1 / 1 |
| `256x8192` | 10.11 us | 8.06 us | 1.25x | 1 / 1 |
| `4096x8192` | 32.77 us | 30.59 us | 1.07x | 1 / 1 |

Inductor's geomean speedup is 1.13x.

### Compact Ablations

Kernel count is shown in parentheses.

| Shape | Nested + asm | Nested, no asm | No nested + asm | Neither |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 8.06 (1) | 8.06 (1) | 10.11 (2) | 10.11 (2) |
| `1024x4096` | 10.11 (1) | 10.11 (1) | 14.21 (2) | 14.21 (2) |
| `256x8192` | 8.06 (1) | 10.11 (1) | 12.16 (2) | 12.16 (2) |
| `4096x8192` | 30.59 (1) | 30.59 (1) | 38.82 (2) | 38.78 (2) |

At `4096x8192`, the prior nested + asm result was 40.83 us. The NaN helper
change accounts for a 1.33x improvement without changing scheduling.

## Persistent Ablation

Forcing the existing persistent-reduction choice at `4096x8192` gives one
input load and 28.54 us. Explicit reduced-domain scale realization and no-x
codegen did not improve it further. `multi_kernel=3` plus coordinate descent
measured 30.69 us with two generated choices, effectively selecting the normal
path. Persistent config selection should be treated as a separate follow-up.

## GEMM-Swizzled Scales

For this layout TE deliberately uses separate RMSNorm and quantize/swizzle
kernels. Inductor fuses the swizzle into its nested kernel.

| Shape | TE swizzled | Inductor nested + asm | Speedup | Kernels TE / Inductor |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 10.27 us | 8.16 us | 1.26x | 2 / 1 |
| `1024x4096` | 16.38 us | 10.24 us | 1.60x | 2 / 1 |
| `256x8192` | 14.34 us | 8.19 us | 1.75x | 2 / 1 |
| `4096x8192` | 53.22 us | 32.70 us | 1.63x | 2 / 1 |

Inductor's geomean speedup is 1.55x.

## NCU Findings

TE and fixed Inductor NCU reports collected exactly one post-warmup launch
using CUDA profiler start/stop. The old-helper baseline is a single filtered
launch of the matching selected config. NCU duration differs from graph replay
timing, but instruction and memory comparisons are directly aligned.

| Kernel | NCU time | Registers | SASS instructions | DRAM read / write |
| --- | ---: | ---: | ---: | ---: |
| TE `ln_tma_fwd_kernel` | 31.78 us | 126 | 20.05M | 57.7 / 36.1 MB |
| Inductor, old helper | 38.30 us | 40 | 25.90M | 59.0 / 42.8 MB |
| Inductor, native NaN helper | 27.23 us | 62 | 17.79M | 61.0 / 39.2 MB |

The old Inductor kernel executes 6.55M `FSETP` and 2.23M `FSEL`
instructions for NaN-aware max/clamp. The fixed kernel replaces that bulk with
2.56M `FMNMX` and 0.33M `FMNMX3` instructions. TE similarly relies on
`FMNMX`/`FMNMX3` rather than bulk compare/select sequences.

Generated PTX contains `max.NaN.f32` for the grouped amax and
`min.NaN.xorsign.abs.f32` for clamp. Triton compiles this form on SM70, SM75,
SM80, SM90, and SM100. SM70/75 automatically expand it to
`setp.nan + max/min + selp`; SM80+ use the native instruction.

## Verification

- `test/inductor/test_triton_helpers.py`: 8 tests passed
- `test/inductor/test_nested_reduction.py`: 244 tests passed
- Elementwise min/max checked for one-sided NaNs, infinities, and signed zero
- Min/max reductions checked for NaN propagation
- Full RMSNorm -> MXFP8 correctness passed on every benchmark row
- `test_torchinductor.py` could not start because installed torchvision lacks
  the `torchvision::nms` operator; no package changes were made

## Artifacts

- Compact JSON: `agent_space/rmsnorm_mxfp8_native_nan_compact_20260716.json`
- Swizzled JSON: `agent_space/rmsnorm_mxfp8_native_nan_swizzled_20260716.json`
- Persistent/multi-kernel JSON:
  `agent_space/rmsnorm_mxfp8_native_nan_multikernel3_coordesc_4096.json`
- Forced-persistent JSON:
  `agent_space/rmsnorm_mxfp8_forced_persistent_4096.json`
- TE NCU report: `agent_space/ncu_te_final_rmsnorm_mxfp8_4096x8192.ncu-rep`
- Inductor NCU report:
  `agent_space/ncu_inductor_native_nan_default_final_rmsnorm_mxfp8_4096x8192.ncu-rep`

## Stack Status

The native NaN helper change and its tests are currently uncommitted. NVFP4 and
half-resolution nested-reduction work is also present only as uncommitted
worktree changes; it is not part of the four committed patches at HEAD.
