# D117896257 padded quantization benchmark

Date: 2026-08-31

This is an experimental overlay and is not part of the main-only benchmark
table in `REPORT.md`.

## Revision and protocol

- Base: PyTorch main `cdd22ade2948699188c3e2d0d80b5a396a8489ae`.
- D117896257 V0.3: `301421b0c8f930f35d3a824dd173df88039a1b93`.
- The isolated worktree also contains D117's prerequisite auxiliary-write and
  padded-scatter commits. Its rebased head is `7fe816090ad8a6ac9707cfb468105e8401a97e01`.
- NVIDIA B200, CUDA 13.0, BF16 inputs, coordinate-descent tuning.
- 20 warmups and 50 CUDA-event samples. Each sample replays an external CUDA
  graph containing 100 calls. Inductor's internal CUDAGraph integration is off.
- The scale layout is expressed in the ATen form D117 targets: valid values use
  `_unsafe_index_put` into final 128x4 blocked storage, followed by a scalar
  boolean `index_put` for zero-valued padding lanes.

Six focused D117 tests passed. Generated-source inspection confirms that every
main padding kernel loads and rewrites the output, while every D117 padding
kernel contains zero loads and one masked store.

## FP4 results with PDL on both sides

All times are microseconds. Inductor PDL is enabled with
`TORCHINDUCTOR_ENABLE_PDL=1`; generated metadata has `launch_pdl=True`, and the
producer kernels contain both `gdc_wait` and `gdc_launch`. D117 and both
FlashInfer variants were measured in dedicated uncontended runs.

| Format | Shape | D117, PDL off | D117, PDL on | FlashInfer, PDL on | D117-on / FI-on |
|---|---:|---:|---:|---:|---:|
| NVFP4 | 1x4096 | 4.584 | 3.882 | 2.621 | 1.48x |
| NVFP4 | 19x4096 | 2.765 | 2.204 | 2.693 | 0.82x |
| NVFP4 | 99x4096 | 2.956 | 2.260 | 2.801 | 0.81x |
| NVFP4 | 129x4096 | 2.915 | 2.436 | 2.782 | 0.88x |
| NVFP4 | 989x4096 | 5.868 | 5.199 | 4.290 | 1.21x |
| MXFP4 | 1x4096 | 3.044 | 2.253 | 2.441 | 0.92x |
| MXFP4 | 19x4096 | 2.693 | 2.109 | 2.595 | 0.81x |
| MXFP4 | 99x4096 | 2.882 | 2.648 | 2.639 | 1.00x |
| MXFP4 | 129x4096 | 2.915 | 2.700 | 2.661 | 1.01x |
| MXFP4 | 989x4096 | 5.697 | 5.108 | 4.972 | 1.03x |

PDL helps both implementations. D117 plus Inductor PDL beats PDL-enabled
FlashInfer by 12-19% at NVFP4 19/99/129 and by 8-19% at MXFP4 1/19. MXFP4
99/129 is within 1.4%. At 989 rows, NVFP4 remains 21% slower and MXFP4 is 2.7%
slower. NVFP4 at one row remains a separate four-kernel outlier, 48% slower.

Coordinate descent was not deterministic for two format/shape pairs. Repeated
runs selected different reduction blocks. A fixed `R0_BLOCK=1024`, eight-warp
comparison showed D117 improving the same graph by 3.5% at MXFP8 19x4096, 4.2%
at MXFP4 129x4096, and a noise-level 0.1% at NVFP4 989x4096. This confirms that
the apparent tuned regressions are selection effects rather than a regression
from the masked-write lowering.

## MXFP8

FlashInfer is a composed RMSNorm plus MXFP8 baseline. PDL is disabled here.

| Shape | D117 | FlashInfer composed | D117 / FlashInfer |
|---:|---:|---:|---:|
| 1x4096 | 2.795 | 4.126 | 0.68x |
| 19x4096 | 3.096 | 4.368 | 0.71x |
| 99x4096 | 2.890 | 4.728 | 0.61x |
| 129x4096 | 2.921 | 4.611 | 0.63x |
| 989x4096 | 5.724 | 7.455 | 0.77x |
| 129x4128 | 3.707 | 5.442 | 0.68x |

D117 remains 1.30-1.64x faster than composed FlashInfer MXFP8.

## One-kernel ceiling

To isolate the value of folding padding initialization into the producer, I
also measured a one-kernel variant with PDL enabled. It omits the padding
initialization while preserving and checking every logical scale value, so it
is an upper-bound experiment rather than a proposed implementation.

| Format | Shape | D117, 2 kernels | One-kernel ceiling | FlashInfer, PDL on |
|---|---:|---:|---:|---:|
| NVFP4 | 129x4096 | 2.436 | 2.059 | 2.794 |
| NVFP4 | 989x4096 | 5.199 | 4.745 | 4.273 |
| MXFP4 | 129x4096 | 2.700 | 2.225 | 2.676 |
| MXFP4 | 989x4096 | 5.108 | 4.701 | 4.990 |
| MXFP8 | 129x4096 | 2.303 | 1.992 | 3.740 |
| MXFP8 | 989x4096 | 5.063 | 4.598 | 6.418 |

Removing the second launch is worth 8.0-17.6% in these cases. It would make
the tested MXFP4 and MXFP8 shapes faster than FlashInfer, but 989x4096 NVFP4
would remain 11.1% slower. That residual is in the producer kernel, not padding.

## Conclusion

D117 works as intended: it removes the full output load/rewrite from the second
kernel. With PDL enabled symmetrically, two kernels are sufficient for MXFP8
and for most small and medium FP4 shapes. They are not sufficient for large
NVFP4. A one-kernel direct store would remove the remaining padding launch, but
the aligned 1024-row NVFP4 result also has an 11-15% producer-kernel gap, so
one-kernel padding alone does not close the entire large-shape difference.

Primary artifacts:

- `bench_d117_aten_padded_quant.py`
- `results/d117_aten_padded_quant_consolidated.json`
- `results/d117_aten_padded_quant_pdl_pair.json`
- `results/d117_aten_padded_quant_inductor_pdl_on.json`
