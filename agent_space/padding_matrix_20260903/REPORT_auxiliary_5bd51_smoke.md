# Padded swizzle and quantization matrix

Source: `agent_space/padding_matrix_20260903/results_auxiliary_5bd51_smoke.jsonl`
Revisions: `5bd51f56358846780bd3d0f6ab2faa6543be474d`
GPUs: NVIDIA B200
Successful cells: 42; failed cells: 0.

Every cell ran in a fresh process. Timings are CUDA-graph replay medians; each replay contains the number of calls recorded in the JSON result.

## Correctness

All padding lanes contain the required sentinel and all Inductor padding variants for a workload/shape produce byte-identical payload and scale outputs.

| workload | shape | tuning | cells | byte-identical | padding values |
| --- | --- | --- | --- | --- | --- |
| dcn_mxfp6 | 96x3072 | coordesc | 3 | yes | 3072 |
| dcn_mxfp6 | 96x3072 | default | 3 | yes | 3072 |
| dcn_mxfp6 | 97x3104 | coordesc | 3 | yes | 16191 |
| dcn_mxfp6 | 97x3104 | default | 3 | yes | 16191 |
| mxfp6_quant | 96x3072 | coordesc | 3 | yes | 3072 |
| mxfp6_quant | 96x3072 | default | 3 | yes | 3072 |
| mxfp6_quant | 97x3104 | coordesc | 3 | yes | 16191 |
| mxfp6_quant | 97x3104 | default | 3 | yes | 16191 |
| rmsnorm_mxfp6 | 96x3072 | coordesc | 3 | yes | 3072 |
| rmsnorm_mxfp6 | 96x3072 | default | 3 | yes | 3072 |
| rmsnorm_mxfp6 | 97x3104 | coordesc | 3 | yes | 16191 |
| rmsnorm_mxfp6 | 97x3104 | default | 3 | yes | 16191 |
| swizzle_dcn | 96x3072 | default | 3 | yes | 3072 |
| swizzle_dcn | 97x3104 | default | 3 | yes | 16191 |

## Padding-only swizzle

Times are `us (generated kernels)`. Ratios above 1 mean the scatter form is slower.

| workload | shape | padding | F.pad | fill+scatter | fill/F.pad | sparse pad scatter | sparse/F.pad | experimental | experimental/F.pad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swizzle_dcn | 96x3072 | 25.0% | 1.128 (1k) | 1.916 (2k) | 1.70x | - | - | 1.054 (1k) | 0.94x |
| swizzle_dcn | 97x3104 | 63.2% | 1.352 (1k) | 2.002 (2k) | 1.48x | - | - | 1.073 (1k) | 0.79x |

## Full pipelines

| workload | shape | tuning | F.pad | fill+scatter | sparse pad scatter | experimental | best Inductor | FlashInfer | best/FI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dcn_mxfp6 | 96x3072 | coordesc | 3.020 (2k) | 5.385 (4k) | - | 3.002 (2k) | auxiliary 3.002 | - | - |
| dcn_mxfp6 | 96x3072 | default | 3.456 (2k) | 5.471 (4k) | - | 2.965 (2k) | auxiliary 2.965 | - | - |
| dcn_mxfp6 | 97x3104 | coordesc | 3.193 (2k) | 5.533 (4k) | - | 3.449 (2k) | fpad 3.193 | - | - |
| dcn_mxfp6 | 97x3104 | default | 4.016 (2k) | 5.463 (4k) | - | 3.028 (2k) | auxiliary 3.028 | - | - |
| mxfp6_quant | 96x3072 | coordesc | 2.729 (2k) | 5.286 (4k) | - | 2.893 (2k) | fpad 2.729 | - | - |
| mxfp6_quant | 96x3072 | default | 3.374 (2k) | 5.395 (4k) | - | 2.867 (2k) | auxiliary 2.867 | - | - |
| mxfp6_quant | 97x3104 | coordesc | 2.920 (2k) | 5.714 (4k) | - | 2.788 (2k) | auxiliary 2.788 | - | - |
| mxfp6_quant | 97x3104 | default | 3.938 (2k) | 6.353 (4k) | - | 2.933 (2k) | auxiliary 2.933 | - | - |
| rmsnorm_mxfp6 | 96x3072 | coordesc | 3.656 (2k) | 3.952 (2k) | - | 3.653 (2k) | auxiliary 3.653 | - | - |
| rmsnorm_mxfp6 | 96x3072 | default | 4.796 (2k) | 3.940 (2k) | - | 4.275 (2k) | fill_scatter 3.940 | - | - |
| rmsnorm_mxfp6 | 97x3104 | coordesc | 3.902 (2k) | 4.374 (2k) | - | 3.766 (2k) | auxiliary 3.766 | - | - |
| rmsnorm_mxfp6 | 97x3104 | default | 5.728 (2k) | 4.376 (2k) | - | 4.706 (2k) | fill_scatter 4.376 | - | - |

## Padding mechanism ranges

- dcn_mxfp6: fill+scatter / F.pad 1.36-1.78x (median 1.66x).
- mxfp6_quant: fill+scatter / F.pad 1.60-1.96x (median 1.77x).
- rmsnorm_mxfp6: fill+scatter / F.pad 0.76-1.12x (median 0.95x).
