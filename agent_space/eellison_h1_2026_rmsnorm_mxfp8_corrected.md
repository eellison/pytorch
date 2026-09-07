# RMSNorm Grouped FP8/MXFP8 Corrected Benchmark

Generated: 2026-07-14

Methodology: each case was compiled once with `triton.nested_reduction=False`
and once with `triton.nested_reduction=True`. One compiled invocation was
captured in a `torch.cuda.CUDAGraph`, then `graph.replay` was timed with
`triton.testing.do_bench`. Timings are graph replay microseconds.

Environment: NVIDIA B200, capability `(10, 0)`, torch
`2.13.0a0+gite1067a5`.

| Case | Config | Old work | New work | Kernels | Graph replay |
| --- | --- | --- | --- | ---: | ---: |
| RMSNorm -> grouped FP8 quant, raw FP8 output, `B=128,D=4096,G=128` | default | RMSNorm + block amax/scale + full-res FP8 payload kernels | One nested RMSNorm/grouped-quant kernel | 2 -> 1 | 8.16 -> 8.16 us (1.00x) |
| RMSNorm -> grouped FP8 quant with explicit clamp, raw FP8 output, `B=128,D=4096,G=128` | default | RMSNorm + block amax/scale + full-res FP8 payload kernels | One nested RMSNorm/grouped-quant kernel | 2 -> 1 | 10.21 -> 10.21 us (1.00x) |
| Residual RMSNorm -> grouped FP8 quant with explicit clamp, raw FP8 output, `B=128,D=2048,G=128` | default | Residual RMSNorm + block amax/scale + full-res FP8 payload kernels | One nested residual-RMSNorm/grouped-quant kernel | 2 -> 1 | 8.16 -> 8.16 us (1.00x) |
| Hidden-state view RMSNorm -> grouped FP8 quant with explicit clamp, raw FP8 output, `tokens=128,view=32x128` | default | RMSNorm + block amax/scale + full-res FP8 payload kernels | One nested RMSNorm/grouped-quant kernel | 2 -> 1 | 10.21 -> 10.21 us (1.00x) |
| RMSNorm -> E8M0-scale MXFP8-style quant, raw FP8 output, `B=128,D=4096,G=32` | default | RMSNorm + E8M0 scale + FP8 payload kernels | Nested RMSNorm/E8M0 scale kernel + separate payload conversion | 3 -> 2 | 12.42 -> 14.30 us (0.87x) |
| RMSNorm -> E8M0-scale MXFP8-style quant, raw FP8 output, `B=128,D=4096,G=32` | coordinate descent | RMSNorm + E8M0 scale + FP8 payload kernels | Nested RMSNorm/E8M0 scale kernel + separate payload conversion | 3 -> 2 | 12.26 -> 12.26 us (1.00x) |

The grouped FP8 rows match the full-resolution epilogue pattern covered by
`test_producer_consumer_rmsnorm_quant` and
`test_producer_consumer_residual_rmsnorm_quant` in
`test/inductor/test_nested_reduction.py`, adapted so the timed compiled
functions return raw FP8 tensors and scales. There is no `q.float()` in the
timed path; correctness conversion happens after timing. On this B200 checkout,
the graph-replay runtime is flat even though the generated-kernel count drops.

The E8M0/MXFP8 rows are left as follow-up investigation. They use both
`inductor_prims.cvt_e8m0_rceil` and the local
`inline_asm_elementwise` HOP. Codegen inspection confirmed the PTX path emits
`cvt.rp.satfinite.ue8m0x2.f32`, but this checkout still materializes
`scale_e8.float()` before the payload conversion. That means the MXFP8
scale-output path is partial fusion, not a one-kernel RMSNorm -> MXFP8 payload
fusion. Coordinate descent removes the default regression but does not create a
stable graph-replay speedup.

For the clean swizzled RMSNorm -> MXFP8 comparison against Transformer Engine,
use `agent_space/eellison_h1_2026_rmsnorm_mxfp8_te_swizzled.md` and
`agent_space/rmsnorm_mxfp8_writeup_swizzled.json`.
