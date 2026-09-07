# Padded swizzle and quantization matrix

Source: `agent_space/padding_matrix_20260903/results_predicated_f2ac_core.jsonl`
Revisions: `f2ac1f47f60e316013441e64cb3637b8e5cc6c3c`
GPUs: NVIDIA B200
Successful cells: 88; failed cells: 0.

Every cell ran in a fresh process. Timings are CUDA-graph replay medians; each replay contains the number of calls recorded in the JSON result.

## Correctness

All padding lanes contain the required sentinel and all Inductor padding variants for a workload/shape produce byte-identical payload and scale outputs.

| workload | shape | tuning | cells | byte-identical | padding values |
| --- | --- | --- | --- | --- | --- |
| mxfp4_quant | 19x4096 | coordesc | 4 | yes | 13952 |
| mxfp4_quant | 19x4096 | default | 4 | yes | 13952 |
| mxfp4_quant | 128x4096 | coordesc | 4 | yes | 0 |
| mxfp4_quant | 128x4096 | default | 4 | yes | 0 |
| mxfp4_quant | 129x4128 | coordesc | 4 | yes | 17151 |
| mxfp4_quant | 129x4128 | default | 4 | yes | 17151 |
| mxfp4_quant | 989x4096 | coordesc | 4 | yes | 4480 |
| mxfp4_quant | 989x4096 | default | 4 | yes | 4480 |
| rmsnorm_mxfp4 | 19x4096 | coordesc | 4 | yes | 13952 |
| rmsnorm_mxfp4 | 19x4096 | default | 4 | yes | 13952 |
| rmsnorm_mxfp4 | 128x4096 | coordesc | 4 | yes | 0 |
| rmsnorm_mxfp4 | 128x4096 | default | 4 | yes | 0 |
| rmsnorm_mxfp4 | 129x4128 | coordesc | 4 | yes | 17151 |
| rmsnorm_mxfp4 | 129x4128 | default | 4 | yes | 17151 |
| rmsnorm_mxfp4 | 989x4096 | coordesc | 4 | yes | 4480 |
| rmsnorm_mxfp4 | 989x4096 | default | 4 | yes | 4480 |
| swizzle_mxfp4 | 19x4096 | default | 4 | yes | 13952 |
| swizzle_mxfp4 | 128x4096 | default | 4 | yes | 0 |
| swizzle_mxfp4 | 129x4128 | default | 4 | yes | 17151 |
| swizzle_mxfp4 | 989x4096 | default | 4 | yes | 4480 |

## Padding-only swizzle

Times are `us (generated kernels)`. Ratios above 1 mean the scatter form is slower.

| workload | shape | padding | F.pad | fill+scatter | fill/F.pad | sparse pad scatter | sparse/F.pad | experimental | experimental/F.pad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swizzle_mxfp4 | 19x4096 | 85.2% | 1.028 (1k) | 1.882 (2k) | 1.83x | 1.939 (2k) | 1.89x | 1.933 (2k) | 1.88x |
| swizzle_mxfp4 | 128x4096 | 0.0% | 1.080 (1k) | 1.971 (2k) | 1.83x | 1.217 (1k) | 1.13x | 1.672 (2k) | 1.55x |
| swizzle_mxfp4 | 129x4128 | 50.8% | 1.176 (1k) | 2.044 (2k) | 1.74x | 2.699 (3k) | 2.29x | 2.011 (2k) | 1.71x |
| swizzle_mxfp4 | 989x4096 | 3.4% | 1.295 (1k) | 2.259 (2k) | 1.74x | 2.333 (2k) | 1.80x | 2.349 (2k) | 1.81x |

## Full pipelines

| workload | shape | tuning | F.pad | fill+scatter | sparse pad scatter | experimental | best Inductor | FlashInfer | best/FI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mxfp4_quant | 19x4096 | coordesc | 4.551 (4k) | 3.321 (3k) | 3.234 (3k) | 3.398 (3k) | pad_scatter 3.234 | 1.468 (1k) | 2.20x |
| mxfp4_quant | 19x4096 | default | 4.532 (4k) | 3.309 (3k) | 3.421 (3k) | 3.350 (3k) | fill_scatter 3.309 | 1.468 (1k) | 2.25x |
| mxfp4_quant | 128x4096 | coordesc | 1.503 (1k) | 3.430 (3k) | 2.772 (2k) | 3.378 (3k) | fpad 1.503 | 1.571 (1k) | 0.96x |
| mxfp4_quant | 128x4096 | default | 1.675 (1k) | 3.657 (3k) | 3.120 (2k) | 3.148 (3k) | fpad 1.675 | 1.571 (1k) | 1.07x |
| mxfp4_quant | 129x4128 | coordesc | 2.752 (2k) | 4.020 (3k) | 5.065 (4k) | 4.073 (3k) | fpad 2.752 | 2.078 (1k) | 1.32x |
| mxfp4_quant | 129x4128 | default | 3.062 (2k) | 4.099 (3k) | 5.265 (4k) | 4.135 (3k) | fpad 3.062 | 2.078 (1k) | 1.47x |
| mxfp4_quant | 989x4096 | coordesc | 8.573 (4k) | 7.313 (3k) | 7.856 (3k) | 7.348 (3k) | fill_scatter 7.313 | 3.801 (1k) | 1.92x |
| mxfp4_quant | 989x4096 | default | 8.319 (4k) | 7.036 (3k) | 7.083 (3k) | 7.031 (3k) | predicated 7.031 | 3.801 (1k) | 1.85x |
| rmsnorm_mxfp4 | 19x4096 | coordesc | 3.841 (3k) | 2.672 (2k) | 2.733 (2k) | 3.102 (2k) | fill_scatter 2.672 | 2.817 (1k) | 0.95x |
| rmsnorm_mxfp4 | 19x4096 | default | 4.758 (3k) | 3.590 (2k) | 3.686 (2k) | 3.654 (2k) | fill_scatter 3.590 | 2.817 (1k) | 1.27x |
| rmsnorm_mxfp4 | 128x4096 | coordesc | 2.059 (1k) | 2.751 (2k) | 2.060 (1k) | 2.483 (2k) | fpad 2.059 | 2.883 (1k) | 0.71x |
| rmsnorm_mxfp4 | 128x4096 | default | 2.905 (1k) | 3.662 (2k) | 2.913 (1k) | 3.417 (2k) | fpad 2.905 | 2.883 (1k) | 1.01x |
| rmsnorm_mxfp4 | 129x4128 | coordesc | 3.973 (2k) | 3.596 (2k) | 4.517 (3k) | 3.854 (2k) | fill_scatter 3.596 | 3.217 (1k) | 1.12x |
| rmsnorm_mxfp4 | 129x4128 | default | 4.614 (2k) | 4.232 (2k) | 5.084 (3k) | 4.252 (2k) | fill_scatter 4.232 | 3.217 (1k) | 1.32x |
| rmsnorm_mxfp4 | 989x4096 | coordesc | 6.999 (3k) | 5.605 (2k) | 5.529 (2k) | 5.772 (2k) | pad_scatter 5.529 | 5.058 (1k) | 1.09x |
| rmsnorm_mxfp4 | 989x4096 | default | 7.356 (3k) | 6.193 (2k) | 6.237 (2k) | 6.259 (2k) | fill_scatter 6.193 | 5.058 (1k) | 1.22x |

## Padding mechanism ranges

- mxfp4_quant: fill+scatter / F.pad 0.73-2.28x (median 1.10x).
- rmsnorm_mxfp4: fill+scatter / F.pad 0.70-1.34x (median 0.87x).
