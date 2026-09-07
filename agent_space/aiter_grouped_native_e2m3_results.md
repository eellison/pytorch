# Grouped MXFP6 with native E2M3x2 conversion

Date: 2026-08-26

This is a scratch-only experiment. No reviewed production worktree files were
changed.

## Result

The native Blackwell `cvt.rn.satfinite.e2m3x2.f32` instruction works inside the
`torch.compile` graph without fragmenting the final kernel, provided its HOP
returns one E2M3 code per full-resolution input element.

The working graph shape is:

1. Preserve scaled fields as `[rows, width / 32, 32]`.
2. Invoke `inline_asm_elementwise(..., pack=2)` on that full-resolution tensor.
   One PTX instruction converts two FP32 values, and the asm exposes the two
   resulting six-bit codes as two int32 output elements.
3. Feed the full-resolution code tensor to the existing no-realize 4-to-3 pack.
4. Flatten only the final packed output.

This preserves the deferred full-resolution source shape expected by the staged
planner. Generated code has one Triton kernel, one nested reduction, one inline
asm call site, and no intermediate store for the converted E2M3 values.

Prototype:

`/data/users/eellison/pytorch/agent_space/bench_aiter_vs_inductor_grouped_native_subagent.py`

The relevant implementation is lines 19-24 and 57-78. It is a graph helper
using `torch._higher_order_ops.inline_asm_elementwise`, not a hand-written whole
kernel.

## Why the shape matters

Two initially plausible placements fragmented:

- Returning two packed E2M3 codes per HOP element created rate-2 intermediate
  buffers. The reduction, conversion, and 4-to-3 pack became three kernels with
  no nested reduction.
- Making the HOP directly produce the three packed output lanes removed one
  intermediate, but its four direct reads crossed AITER's field permutation.
  The staged source projection correctly rejected those reads, leaving two
  kernels with no nested reduction.

The successful `pack=2` form applies the native instruction while keeping the
logical result at full resolution. The existing deferred-source machinery then
places conversion in the final reduction loop, and the already-supported
`(4,3)` epilogue consumes it.

## Correctness

- All 63,488 finite FP16 bit patterns converted bit-exactly to the extracted
  AITER E2M3 semantics.
- The only difference from the old Python test helper is `-0.0`: the helper
  canonicalizes it to code 0, while both the native instruction and AITER retain
  the sign and produce code 32.
- Full quantization was bit-exact for all four benchmark shapes.
- Additional wide-value, all-zero, and all-negative-zero inputs were bit-exact
  against `quantize_fp6_lastdim_triton`; each compiled as one kernel with one
  nested reduction.

Generated form for `8192x128`:

```text
kernels 1 nested 1 triton_defs 1 inline_asm 1 stores 4
tl.inline_asm_elementwise(... '=r,=r,f,f',
    [triton_helpers.inline_asm_pack(tmp53, 2)],
    dtype=tl.int32, is_pure=True, pack=2)
```

The four stores are the scale output and the three packed-byte output lanes.

## Performance

GPU: NVIDIA B200, CUDA 13.0. Coordinate descent was disabled. Times are CUDA
Graph medians using 100 calls per replay, 50 warmup iterations, and 300 samples.

| Shape | AITER | Grouped software | Grouped native | Native / AITER | Native speedup vs software |
|---|---:|---:|---:|---:|---:|
| 8192x128 | 4.711 us | 3.767 us | 2.498 us | 0.530x | 1.508x |
| 32768x128 | 12.892 us | 10.793 us | 5.417 us | 0.420x | 1.992x |
| 1024x1024 | 4.363 us | 3.768 us | 2.498 us | 0.573x | 1.508x |
| 128x12288 | 5.899 us | 5.162 us | 3.031 us | 0.514x | 1.703x |

Both Inductor variants were one kernel with `codegen_nested_reduction == 1` for
all four shapes. The native graph was 1.75x-2.38x faster than the extracted
AITER reference and 1.51x-1.99x faster than the software-conversion graph.

## Commands

```bash
source agent_space/env.sh
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 "$PY" \
  agent_space/bench_aiter_vs_inductor_grouped_native_subagent.py
```

Additional full-quantizer edge cases were run with the same environment for
`wide`, `zeros`, and `negative_zero` inputs. All reported exact packed and scale
outputs, `kernels=1`, and `nested=1`.

## Production impact

This experiment needs only the graph-level native conversion helper. It does
not require scheduler or codegen changes beyond the existing `(4,3)` staged
epilogue work, and it avoids materializing the converted value tensor.

The simple benchmark graph can still duplicate the scale expression across
field branches. The production-like DCN graph was checked separately and
generated one scale computation. See
`mxfp6_native_pack2_final_validation_20260826.md` for the final assessment.
