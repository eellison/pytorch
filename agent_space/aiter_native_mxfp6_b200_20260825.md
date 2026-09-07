# Native MXFP6 conversion benchmark on B200

Scratch implementation:

- `agent_space/aiter_mxfp6_reference.py`
- `agent_space/bench_aiter_ideal_mxfp6.py`

The native variant differs from the pinned AITER kernel only in conversion and
packing. It preserves AITER's 32-element input permutation, exponent/scale
calculation, dense 24-byte output layout, and three byte stores. Two native
`cvt.rn.satfinite.e2m3x2.f32` instructions convert each group of four values.

## Correctness

Both the dense three-store and masked-four-store native variants matched the
AITER software conversion byte-for-byte for packed values and scales on:

- random FP16 input;
- random FP16 input multiplied by 1000;
- all-zero input;
- all 65,536 FP16 bit patterns, including signed zeros, subnormals, infinities,
  and NaNs.

The inline assembly uses `$2, $1` operand order. The exhaustive FP16 comparison
confirmed that the low and high result bytes correspond to the expected first
and second values. No rounding or special-value discrepancy was observed.

## Method

NVIDIA B200. Each result is the median of 300 CUDA graph replay measurements,
with 100 kernel calls per replay and 100 warmup calls. `BLOCK_N`/warp sweeps
covered `(8,1)`, `(8,2)`, `(16,1)`, `(16,2)`, `(16,4)`, `(32,1)`, `(32,2)`,
`(32,4)`, `(64,2)`, `(64,4)`, `(64,8)`, `(128,4)`, and `(128,8)`.

`AITER` is the pinned kernel at its upstream `BLOCK_N=16`, one-warp launch.
`software fixed` is the prior dense three-store Triton kernel at the same
launch configuration. `software tuned` is its best result from the same sweep.

## Results

| Shape | AITER us | Software fixed us | Software tuned us (config, regs) | Native dense us (config, regs) | Speedup vs AITER / fixed / tuned |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192x128 | 4.711 | 4.547 | 3.953 (32, 4, 32) | 3.349 (16, 4, 25) | 1.407x / 1.358x / 1.180x |
| 32768x128 | 12.883 | 12.123 | 11.899 (32, 4, 32) | 9.216 (64, 4, 32) | 1.398x / 1.315x / 1.291x |
| 1024x1024 | 4.363 | 4.035 | 3.686 (16, 2, 32) | 2.949 (32, 4, 32) | 1.480x / 1.368x / 1.250x |
| 128x12288 | 5.899 | 5.366 | 4.998 (8, 1, 50) | 3.953 (32, 4, 32) | 1.492x / 1.357x / 1.264x |

All reported kernels had zero spills. The exact AITER baseline used 70 registers
for width 128 and 68 for wider rows. The fixed software dense baseline used 63
and 64 registers respectively.

At the identical `BLOCK_N=16`, one-warp launch, native dense conversion was
1.276x, 1.312x, 1.313x, and 1.350x faster than software dense in table order.
The remaining best-config gain comes from launch tuning rather than conversion.

### Native conversion with masked-four store

| Shape | Native masked-four us (config, regs) | Speedup vs AITER | Speedup vs fixed software dense | Speedup vs tuned software dense | Speedup vs software masked-four |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192x128 | 3.953 (16, 2, 38) | 1.192x | 1.150x | 1.000x | 1.259x |
| 32768x128 | 11.941 (64, 2, 123) | 1.079x | 1.015x | 0.997x | 1.142x |
| 1024x1024 | 3.564 (32, 4, 40) | 1.224x | 1.132x | 1.034x | 1.281x |
| 128x12288 | 4.936 (32, 4, 40) | 1.195x | 1.087x | 1.013x | 1.245x |

All masked-four results also had zero spills.

## Conclusion

Native E2M3 conversion is a material improvement when paired with the dense
three-store layout: 1.40x-1.49x faster than exact upstream AITER and
1.18x-1.29x faster than the best tuned software conversion.

The masked-four store consumes most of that gain. It is 18%-30% slower than the
native dense store and is only parity to 3% faster than the best tuned software
dense kernel. For this standalone kernel, native conversion plus dense stores is
the performance target; conversion alone does not make masked-four competitive.
