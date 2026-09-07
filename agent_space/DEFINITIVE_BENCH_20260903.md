# Definitive RMSNorm -> FP4 quant benchmark

Date: 2026-09-03. AI-assisted local report; not for GitHub without human review
and the `AI_POLICY.md` disclosure.

## Tree under test

- `wip_quant` = `origin/main` (`543a80bde59`) + `697a112840a`
  "[inductor] Remap epilogue index expressions into the reduced domain".
- Built natively (`torch 2.15.0a0+git697a112`, CUDA 13.0, B200). No overlay
  harness -- the previous numbers went through `agent_space/run_wt.py`.
- `torch/_inductor/decomposition.py` `aten._unsafe_index_put` exclusion applied,
  uncommitted.
- `test/inductor/test_nested_reduction.py`: 421 tests, OK (skipped=8).

## Method

One implementation per process, CUDA-graph capture/replay, coordinate-descent
tuning on. Cooldown matched to kernel duration (see below): 3.0 s / 20 samples /
20 calls per graph for the large shapes, 0.02 s / 50 samples / 100 calls per
graph for the small ones. Driver: `agent_space/definitive_bench_20260903.py`,
raw JSON in `agent_space/results_20260903/`.

## Results

Large shapes (3 s cooldown):

| shape | format | Inductor | FlashInfer | ratio | Inductor TB/s |
|---|---|---:|---:|---:|---:|
| 16000x8192 | NVFP4 | 92.04 us | 102.52 us | 0.898 | 3.65 |
| 16000x8192 | MXFP4 | 90.37 us | 105.69 us | 0.855 | 3.67 |
| 16384x8192 | NVFP4 | 93.41 us | 104.26 us | 0.896 | 3.68 |
| 16384x8192 | MXFP4 | 91.61 us | 107.99 us | 0.848 | 3.71 |

Small shapes (0.02 s cooldown, 100 calls per graph):

| shape | format | Inductor | FlashInfer | ratio |
|---|---|---:|---:|---:|
| 99x4096 | NVFP4 | 2.956 us | 2.999 us | 0.986 |
| 99x4096 | MXFP4 | 2.980 us | 3.282 us | 0.908 |

Inductor is 10-15% faster than FlashInfer at the large shapes and at parity to
9% faster at the small ones. This supersedes the earlier 0.94x figure, which was
measured through the overlay harness on an older toolkit.

Byte agreement with FlashInfer at 16000x8192: NVFP4 quant 0.9917 / scale 0.9841;
MXFP4 quant 0.9945 / scale 0.9981. These are the previously documented
reduction-order differences, not a regression. Topology is `kernel_count=2`,
`codegen_nested_reduction=1` at every shape.

## Cooldown is duration-dependent

The 3 s inter-sample cooldown that defeats B200 thermal throttling on ~100 us
kernels *inflates* few-microsecond kernels, because after 3 s idle the clocks
have dropped and 20 x 5 us of replay cannot ramp them back. The same 99x4096
NVFP4 point reads 5.08 us with the 3 s cooldown and 2.96 us without it, and the
ratio moves from 0.937 to 0.986. Absolute numbers from the two regimes are not
comparable.

## `tl.device_assert` on the swizzle scatter

The assert only ever existed on dynamic shapes.

| | `index_put` (check=True) | `_unsafe_index_put` |
|---|---|---|
| static | 0 asserts | 0 asserts |
| dynamic | 1 assert | 0 asserts |

On static shapes Inductor already proves the bound, so the decomposition
exclusion is a no-op there. The reason it works is that the term-wise interval
maximum of

```
off = (row//128)*(pcols//4)*512 + (col//4)*512 + (row%32)*16 + ((row//32)%4)*4 + (col%4)
```

collapses to exactly `prows*pcols - 1`, i.e. precisely `size - 1`, because
`512/(128*4) == 1`. No correlated reasoning is needed -- but there is also zero
slack, so any conservative widening in a single `ValueRanges` op would bring the
assert back.

On dynamic shapes the proof needs `prows = ceil(rows/128)*128` to cancel against
the `(3 + ks0) // 4` that appears in the emitted condition, which interval
arithmetic cannot recover. Measured cost of the assert there: 59.66 -> 55.26 us,
about 8%. That 8% is also the entire ceiling for an assert-after-store
transformation, which is therefore not worth building given the exclusion
already captures it for `_unsafe_index_put`.

## Zero-init toggle

At 16000x8192 NVFP4, `torch.zeros` -> `torch.empty` for the blocked-scale
destination drops the fill kernel (`kernel_count` 2 -> 1) and saves 0.96 us
(1.04%), consistent with the 1.13% measured previously. Valid only when
`rows % 128 == 0 and cols % 4 == 0`, where the swizzle map is a bijection.
Not applied.
