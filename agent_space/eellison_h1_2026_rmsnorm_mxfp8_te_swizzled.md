# RMSNorm -> MXFP8 Swizzled Benchmark

Generated from
`agent_space/rmsnorm_mxfp8_writeup_swizzled_after_index_simplify.json`.

Methodology: CUDA graph replay timed with `triton.testing.do_bench`. Timed
region returns raw FP8/MXFP8 payload plus E8M0 scales, with no dequant or
`q.float()` in the timed path. Compile mode is `max-autotune-no-cudagraphs`.

Environment: NVIDIA B200, CUDA 12.8, torch
`2.13.0a0+gite1067a5`, Transformer Engine `2.17.0`.

| Shape | TE | Inductor one-kernel nested + inline asm | Speedup vs TE | Kernels |
| --- | ---: | ---: | ---: | ---: |
| `128x4096` | 10.18 us | 8.06 us | 1.26x | 1 |
| `1024x4096` | 16.26 us | 12.16 us | 1.34x | 1 |
| `256x8192` | 14.21 us | 10.11 us | 1.41x | 1 |
| `4096x8192` | 53.15 us | 40.83 us | 1.30x | 1 |

Geomean speedup vs Transformer Engine:

| Variant | Shapes | Geomean |
| --- | --- | ---: |
| One-kernel nested + inline asm | all 4 measured shapes | 1.33x |

Geomean speedup vs prior PyTorch:

| Baseline | Shapes | Geomean |
| --- | --- | ---: |
| Prior PyTorch nested-off decomposition, same inline-asm E8M0 path | all 4 measured shapes | 1.37x |

Nested reduction is the main win for the smaller and mid-size rows: it reduces
the Inductor decomposition from the nested-off path to a single nested kernel
for the inline-asm path. This table intentionally reports the stricter
one-kernel nested+inline-asm comparison rather than picking the fastest ablation
per shape.
