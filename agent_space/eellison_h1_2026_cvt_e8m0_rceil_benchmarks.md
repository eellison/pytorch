# H1 2026 cvt_e8m0_rceil Benchmarks

## Scope

Local single-GPU B200/SM100 measurements for PR #172497, comparing the current Inductor e8m0 rceil pattern replacement/PTX lowering against a before-state compiled with `torch._inductor.config.pattern_matcher=False`.

## Environment

- Host: `devgpu005.snb3.facebook.com`
- GPU: `NVIDIA B200` cc=[10, 0] count=8
- CUDA: `12.8`
- Torch: `2.13.0a0+gite1067a5` from `/data/users/eellison/pytorch/torch/__init__.py`
- Git: `d59fc87a47b` (`detached`)
- Timestamp: `2026-07-13 18:02:05 PDT`

## Implementation Discovered

- Prim: `torch/_inductor/inductor_prims.py` defines `inductor_cvt_e8m0_rceil(Tensor input) -> Tensor` with an eager bit-manipulation fallback.
- Lowering: `torch/_inductor/lowering.py` lowers the prim on SM100+ to `tl.inline_asm_elementwise` with PTX `cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;`, outputting `uint16` then casting to `uint8`.
- Pattern: `torch/_inductor/fx_passes/misc_patterns.py` replaces the CUDA float32 `ceil(log2(x))` e8m0 encode pattern with the prim on SM100+.
- Tests: `test/inductor/test_fp8.py::TestCvtE8M0Rceil` covers correctness, pattern matching, PTX codegen, and near-power-of-two behavior.

## Timing Method

Each callable was compiled first, then timed with manual `torch.cuda.CUDAGraph` replay. The numbers below exclude `torch.compile` overhead. Workloads are full MXFP8 scale generation from bf16 input and encode-only over the corresponding scale tensor shape.

## Results

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

## Codegen Checks

- `scale_from_bf16` `m1024_k4096`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `encode_only` `m1024_k4096`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `scale_from_bf16` `m2048_k4096`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `encode_only` `m2048_k4096`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `scale_from_bf16` `m4096_k8192`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `encode_only` `m4096_k8192`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `scale_from_bf16` `m8192_k8192`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `encode_only` `m8192_k8192`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `scale_from_bf16` `m8192_k16384`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.
- `encode_only` `m8192_k16384`: current has prim=True, PTX=True, inline_asm=True, log2=False; fallback has PTX=False, log2=True.

Raw data: `agent_space/eellison_h1_2026_cvt_e8m0_rceil_benchmarks.json`.
