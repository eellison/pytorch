# eellison H1 2026 Local Single-GPU Impact

Scope: eellison-authored PyTorch PRs closed or landed in H1 2026, restricted to
local single-GPU PyTorch/Inductor impact. Distributed, collectives, NCCL,
multi-GPU overlap, and distributed trace-estimator work are excluded.

Inventory: `agent_space/eellison_h1_2026_pr_inventory.md` found 137 H1 PR rows,
68 with landed/merged signal, and no H1 landed commits without a PR number.
Kernel timings below use CUDA graph replay unless explicitly labeled as host
wrapper timing.

## Grouped Quant Fusion

Representative PRs: #176927, #179090

Reindex pointwise loops and choose the write->read dependency that preserves the
important producer-consumer fusion candidate. In grouped quantization, this lets
the amax/scale reduction and quantize pointwise path fuse instead of launching a
separate quant kernel.

| Representative pattern | Old work | New work | Kernels | Graph replay |
| --- | --- | --- | ---: | ---: |
| Grouped quant candidate selection, bf16 `x=(8,7168)` | Grouped-quant amax/scale reduction + separate quant pointwise | Fused grouped-quant reduction stores scale and quant output | 2 -> 1 | 6.20 -> 4.15 us (1.49x) |

The isolated benchmark uses a scratch monkey-patch to emulate the prior
largest-buffer candidate picker. The raw loop-reindexing report also contains
synthetic RMSNorm-on-sliced/transposed-view rows for scheduler coverage, but
those are not the main user-facing grouped-quant story.

## Pointwise Cat Fusion

PR: #179091

Use pointwise cat when inputs recombine the same data, so RoPE/QKNorm-style
split branches can avoid materializing a cat-side pointwise kernel.

| Representative pattern | Old work | New work | Kernels | Graph replay |
| --- | --- | --- | ---: | ---: |
| QKNorm + split RoPE cat, `B=4,H=8,S=128,D=64` | QKNorm reduction + cat-side pointwise | One fused QKNorm/RoPE/cat kernel | 2 -> 1 | 4.21 -> 4.14 us (1.02x) |
| QKNorm + interleaved RoPE stack/flatten | QKNorm reduction + stack/flatten pointwise | One fused QKNorm/RoPE kernel | 2 -> 1 | 4.19 -> 6.19 us (0.68x) |

This is primarily a launch/materialization win. The split-RoPE case is flat
under graph replay, and the interleaved static microbenchmark is slower, so the
claim should be kernel-count/materialization improvement rather than a universal
kernel-body speedup.

## Nested Reductions

Representative PRs: #182891, #182892, #182893, #182896, #183432, #182897,
#182898, with tuning/codegen context from #182895.

Fuse a reduction whose result feeds another grouped/block-local reduction or
sub-parent epilogue, even when the two reductions operate over different logical
domains. This is the path for RMSNorm followed by grouped quant/block
amax-scale, grouped FP8 quant payload conversion, NVFP4-style packing, and
chunk/gating consumers.

| Representative pattern | Old work | New work | Kernels | Graph replay |
| --- | --- | --- | ---: | ---: |
| RMSNorm -> grouped quant scale, `B=128,D=4096,G=16` | RMSNorm kernel + block amax/scale kernel | Nested RMSNorm/block-scale kernel | 2 -> 1 | 8.58 -> 8.54 us (1.00x) |
| RMSNorm -> grouped FP8 quant, raw FP8 output, `B=128,D=4096,G=128` | RMSNorm + block amax/scale + full-res FP8 payload kernels | One nested RMSNorm/grouped-FP8 quant kernel | 2 -> 1 | 8.16 -> 8.16 us (1.00x) |
| Residual RMSNorm -> grouped FP8 quant, raw FP8 output, `B=128,D=2048,G=128` | Residual RMSNorm + block amax/scale + full-res FP8 payload kernels | One nested residual-RMSNorm/grouped-FP8 quant kernel | 2 -> 1 | 8.16 -> 8.16 us (1.00x) |
| Hidden-state view RMSNorm -> grouped FP8 quant, raw FP8 output, `tokens=128,view=32x128` | RMSNorm + block amax/scale + full-res FP8 payload kernels | One nested RMSNorm/grouped-FP8 quant kernel | 2 -> 1 | 10.21 -> 10.21 us (1.00x) |
| RMSNorm -> swizzled MXFP8 quant, raw FP8 output + E8M0 scales, `B=128,D=4096,block=32` | Transformer Engine baseline | One nested RMSNorm/MXFP8 kernel | n/a -> 1 | 10.18 -> 8.06 us (1.26x) |
| RMSNorm -> swizzled MXFP8 quant, raw FP8 output + E8M0 scales, `B=1024,D=4096,block=32` | Transformer Engine baseline | One nested RMSNorm/MXFP8 kernel | n/a -> 1 | 16.26 -> 12.16 us (1.34x) |
| RMSNorm -> swizzled MXFP8 quant, raw FP8 output + E8M0 scales, `B=256,D=8192,block=32` | Transformer Engine baseline | One nested RMSNorm/MXFP8 kernel | n/a -> 1 | 14.21 -> 10.11 us (1.41x) |
| RMSNorm -> swizzled MXFP8 quant, raw FP8 output + E8M0 scales, `B=4096,D=8192,block=32` | Transformer Engine baseline | One nested RMSNorm/MXFP8 kernel | n/a -> 1 | 53.15 -> 40.83 us (1.30x) |
| RMSNorm -> NVFP4-style pack, `B=128,D=4096,G=16` | RMSNorm + scale + pack/div kernels | Nested RMSNorm kernel with pack path | 3 -> 1 | 8.54 -> 8.51 us (1.00x) |
| Half-resolution sub-parent epilogue, `B=128,D=4096,G=16` | Block amax/scale + half-resolution consumers | One nested/sub-parent kernel | 2 -> 1 | 8.54 -> 8.35 us (1.02x) |
| RMSNorm -> weighted reduce-K, `B=64,K=16,D=4096` | RMSNorm kernel + K-reduction kernel | One nested reduction kernel | 2 -> 1 | 54.82 -> 54.88 us (1.00x) |
| Residual RMSNorm -> chunk(2) SwiGLU, `B=128,D=1024` | RMSNorm kernel + chunk/SwiGLU pointwise | One nested/consumer kernel | 2 -> 1 | 8.35 -> 8.38 us (1.00x) |

The grouped-FP8 and older nested microbenchmarks show reliable
launch/materialization removal but mostly flat graph-replay timings when
measured as `triton.testing.do_bench` over CUDA graph replay. The test-covered
full-resolution grouped FP8 quant path fuses to one kernel, including residual
and hidden-state-view variants.

The swizzled MXFP8 benchmark is the clean RMSNorm -> MXFP8 comparison: the timed
region returns raw FP8/MXFP8 payload plus E8M0 scales, with no dequant or
`q.float()` in the timed path. Nested reduction is the important win on the
smaller and mid-size shapes. Across all four measured shapes, the one-kernel
nested+inline-asm path is 1.33x geomean vs Transformer Engine and 1.37x
geomean vs the prior PyTorch nested-off decomposition with the same inline-asm
E8M0 path. This reports the stricter one-kernel path rather than selecting the
fastest ablation per shape.

## E8M0 Scale Encoding

PR: #172497

Replace software `ceil(log2(x))` E8M0 scale encoding with the SM100 PTX
conversion path, emitting `cvt.rp.satfinite.ue8m0x2.f32`.

| Representative pattern | Old work | New work | Kernels | Graph replay |
| --- | --- | --- | ---: | ---: |
| E8M0 encode-only, `1024x128` | Fallback log2/ceil encode kernel | PTX conversion kernel | 1 -> 1 | 4.0 -> 2.2 us (1.83x) |
| E8M0 encode-only, `8192x512` | Fallback log2/ceil encode kernel | PTX conversion kernel | 1 -> 1 | 8.2 -> 4.1 us (2.00x) |
| Full bf16 MXFP8 scale generation, `2048x4096` | Amax/reduction + fallback encode | Amax/reduction + PTX encode | 1 -> 1 | 8.2 -> 6.1 us (1.33x) |

The isolated encode rows show the PTX lowering most clearly. Full scale
generation is partly reduction-bound, so speedups are smaller.

## Deferred Assertions And Misaligned Copies

Landed PRs: #177783, #179039, #180599. Reference PR #178489 reported
HuggingFace geomean improvements of +6.3% to +9.9%, with smaller timm and
TorchBench gains, but #178489 itself was reverted; landed credit maps to the H1
deferral PRs above.

This is host-wrapper placement, not kernel-body speedup, so CUDA graph replay is
not the right measurement.

| Representative pattern | Old wrapper work | New wrapper work | Host timing |
| --- | --- | --- | ---: |
| Size/stride assertions before first matmul, 256 tail tensors first used later | 6 pre-first-mm `assert_size_stride` calls | 2 pre-first-mm asserts; tail asserts move to first use | 129.01 -> 69.16 us to first mm |
| `copy_if_misaligned`, 32 misaligned tail tensors first used later | 6 pre-first-mm copies | 2 pre-first-mm copies; tail copies move to consumer boundary | 218.18 -> 36.41 us to first mm |

## Backing Artifacts

- PR inventory: `agent_space/eellison_h1_2026_pr_inventory.md`
- Representative kernel table: `agent_space/eellison_h1_2026_representative_kernel_table.md`
- Loop/reindex/cat benchmarks: `agent_space/eellison_h1_2026_loop_reindexing_toggle_benchmarks.md`
- Nested reduction benchmarks: `agent_space/eellison_h1_2026_nested_benchmarks.md`
- Corrected RMSNorm grouped FP8/MXFP8 benchmark: `agent_space/eellison_h1_2026_rmsnorm_mxfp8_corrected.md`
- RMSNorm grouped FP8 quant variants: `agent_space/eellison_h1_2026_rmsnorm_quant_variants.json`
- RMSNorm MXFP8 vs Transformer Engine swizzled benchmark: `agent_space/eellison_h1_2026_rmsnorm_mxfp8_te_swizzled.md`
- E8M0 benchmarks: `agent_space/eellison_h1_2026_cvt_e8m0_rceil_benchmarks.md`
- Deferred wrapper benchmarks: `agent_space/eellison_h1_2026_deferred_assertions_benchmarks.md`
