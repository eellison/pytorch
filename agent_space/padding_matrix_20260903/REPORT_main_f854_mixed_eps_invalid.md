# Padded swizzle and quantization matrix

Source: `agent_space/padding_matrix_20260903/results_main_f854_core.jsonl`
Revisions: `f8546ee64c3b9ed5cfdef3686d4f465d9af6c869`
GPUs: NVIDIA B200
Successful cells: 67; failed cells: 0.

Every cell ran in a fresh process. Timings are CUDA-graph replay medians; each replay contains the number of calls recorded in the JSON result.

## Correctness

All padding lanes contain the required sentinel and all Inductor padding variants for a workload/shape produce byte-identical payload and scale outputs.

| workload | shape | cells | byte-identical | padding values |
| --- | --- | --- | --- | --- |
| dcn_mxfp6 | 95x3072 | 3 | yes | 3168 |
| dcn_mxfp6 | 96x3072 | 2 | yes | 3072 |
| dcn_mxfp6 | 97x3104 | 1 | yes | 16191 |
| dcn_mxfp6 | 2048x3072 | 2 | yes | 73728 |
| mxfp4_quant | 19x4096 | 2 | yes | 13952 |
| mxfp4_quant | 128x4096 | 6 | yes | 0 |
| mxfp4_quant | 129x4128 | 1 | yes | 17151 |
| mxfp6_quant | 95x3072 | 4 | yes | 3168 |
| mxfp6_quant | 96x3072 | 1 | yes | 3072 |
| mxfp6_quant | 97x3104 | 2 | yes | 16191 |
| mxfp6_quant | 2048x3072 | 2 | yes | 73728 |
| rmsnorm_mxfp4 | 19x4096 | 4 | yes | 13952 |
| rmsnorm_mxfp4 | 128x4096 | 5 | yes | 0 |
| rmsnorm_mxfp4 | 129x4128 | 3 | yes | 17151 |
| rmsnorm_mxfp4 | 989x4096 | 4 | yes | 4480 |
| rmsnorm_mxfp6 | 95x3072 | 2 | yes | 3168 |
| rmsnorm_mxfp6 | 96x3072 | 4 | yes | 3072 |
| rmsnorm_mxfp6 | 2048x3072 | 3 | yes | 73728 |
| swizzle_dcn | 95x3072 | 2 | yes | 3168 |
| swizzle_dcn | 2048x3072 | 1 | yes | 73728 |
| swizzle_mxfp4 | 19x4096 | 2 | yes | 13952 |
| swizzle_mxfp4 | 128x4096 | 2 | yes | 0 |
| swizzle_mxfp4 | 129x4128 | 2 | yes | 17151 |
| swizzle_mxfp4 | 989x4096 | 3 | yes | 4480 |

## Padding-only swizzle

Times are `us (generated kernels)`. Ratios above 1 mean the scatter form is slower.

| workload | shape | padding | F.pad | fill+scatter | fill/F.pad | sparse pad scatter | sparse/F.pad | experimental | experimental/F.pad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swizzle_dcn | 95x3072 | 25.8% | 1.114 (1k) | 1.902 (2k) | 1.71x | - | - | - | - |
| swizzle_dcn | 2048x3072 | 27.3% | 1.313 (1k) | - | - | - | - | - | - |
| swizzle_mxfp4 | 19x4096 | 85.2% | 1.022 (1k) | 1.821 (2k) | 1.78x | - | - | - | - |
| swizzle_mxfp4 | 128x4096 | 0.0% | 1.081 (1k) | - | - | 1.132 (1k) | 1.05x | - | - |
| swizzle_mxfp4 | 129x4128 | 50.8% | 1.128 (1k) | 2.019 (2k) | 1.79x | - | - | - | - |
| swizzle_mxfp4 | 989x4096 | 3.4% | 1.251 (1k) | 2.260 (2k) | 1.81x | 2.390 (2k) | 1.91x | - | - |

## Full pipelines

| workload | shape | tuning | F.pad | fill+scatter | sparse pad scatter | experimental | best Inductor | FlashInfer | best/FI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dcn_mxfp6 | 95x3072 | coordesc | 3.145 (2k) | - | - | - | fpad 3.145 | - | - |
| dcn_mxfp6 | 95x3072 | default | 3.343 (2k) | 5.390 (4k) | - | - | fpad 3.343 | - | - |
| dcn_mxfp6 | 96x3072 | default | 3.442 (2k) | 5.470 (4k) | - | - | fpad 3.442 | - | - |
| dcn_mxfp6 | 97x3104 | default | 3.936 (2k) | - | - | - | fpad 3.936 | - | - |
| dcn_mxfp6 | 2048x3072 | coordesc | 9.597 (2k) | - | - | - | fpad 9.597 | - | - |
| dcn_mxfp6 | 2048x3072 | default | 10.667 (2k) | - | - | - | fpad 10.667 | - | - |
| mxfp4_quant | 19x4096 | coordesc | - | - | 3.345 (3k) | - | pad_scatter 3.345 | - | - |
| mxfp4_quant | 19x4096 | default | 4.541 (4k) | - | - | - | fpad 4.541 | - | - |
| mxfp4_quant | 128x4096 | coordesc | 1.556 (1k) | 3.667 (3k) | 3.183 (2k) | - | fpad 1.556 | 1.583 (1k) | 0.98x |
| mxfp4_quant | 128x4096 | default | 1.653 (1k) | 3.660 (3k) | 2.632 (2k) | - | fpad 1.653 | 1.583 (1k) | 1.04x |
| mxfp4_quant | 129x4128 | coordesc | - | - | 4.651 (4k) | - | pad_scatter 4.651 | - | - |
| mxfp6_quant | 95x3072 | coordesc | 2.953 (2k) | 5.433 (4k) | - | - | fpad 2.953 | - | - |
| mxfp6_quant | 95x3072 | default | 3.406 (2k) | 6.166 (4k) | - | - | fpad 3.406 | - | - |
| mxfp6_quant | 96x3072 | coordesc | - | 5.239 (4k) | - | - | fill_scatter 5.239 | - | - |
| mxfp6_quant | 97x3104 | coordesc | - | 5.688 (4k) | - | - | fill_scatter 5.688 | - | - |
| mxfp6_quant | 97x3104 | default | 3.860 (2k) | - | - | - | fpad 3.860 | - | - |
| mxfp6_quant | 2048x3072 | coordesc | 9.060 (2k) | 20.302 (4k) | - | - | fpad 9.060 | - | - |
| rmsnorm_mxfp4 | 19x4096 | coordesc | - | - | 2.972 (2k) | - | pad_scatter 2.972 | 2.743 (1k) | 1.08x |
| rmsnorm_mxfp4 | 19x4096 | default | 4.730 (3k) | 3.541 (2k) | 3.670 (2k) | - | fill_scatter 3.541 | 2.743 (1k) | 1.29x |
| rmsnorm_mxfp4 | 128x4096 | coordesc | 2.191 (1k) | - | 2.547 (1k) | - | fpad 2.191 | 2.898 (1k) | 0.76x |
| rmsnorm_mxfp4 | 128x4096 | default | 2.899 (1k) | 3.709 (2k) | 2.898 (1k) | - | pad_scatter 2.898 | 2.898 (1k) | 1.00x |
| rmsnorm_mxfp4 | 129x4128 | coordesc | - | 3.632 (2k) | 4.520 (3k) | - | fill_scatter 3.632 | - | - |
| rmsnorm_mxfp4 | 129x4128 | default | 4.710 (2k) | - | - | - | fpad 4.710 | - | - |
| rmsnorm_mxfp4 | 989x4096 | coordesc | - | 5.597 (2k) | 5.617 (2k) | - | fill_scatter 5.597 | - | - |
| rmsnorm_mxfp4 | 989x4096 | default | - | 6.196 (2k) | 6.225 (2k) | - | fill_scatter 6.196 | - | - |
| rmsnorm_mxfp6 | 95x3072 | coordesc | 4.184 (2k) | - | - | - | fpad 4.184 | - | - |
| rmsnorm_mxfp6 | 95x3072 | default | 4.642 (2k) | - | - | - | fpad 4.642 | - | - |
| rmsnorm_mxfp6 | 96x3072 | coordesc | 3.822 (2k) | 3.738 (2k) | - | - | fill_scatter 3.738 | - | - |
| rmsnorm_mxfp6 | 96x3072 | default | 4.728 (2k) | 3.907 (2k) | - | - | fill_scatter 3.907 | - | - |
| rmsnorm_mxfp6 | 2048x3072 | coordesc | 13.289 (2k) | 12.843 (2k) | - | - | fill_scatter 12.843 | - | - |
| rmsnorm_mxfp6 | 2048x3072 | default | - | 14.465 (2k) | - | - | fill_scatter 14.465 | - | - |

## Padding mechanism ranges

- dcn_mxfp6: fill+scatter / F.pad 1.59-1.61x (median 1.60x).
- mxfp4_quant: fill+scatter / F.pad 2.21-2.36x (median 2.29x).
- mxfp6_quant: fill+scatter / F.pad 1.81-2.24x (median 1.84x).
- rmsnorm_mxfp4: fill+scatter / F.pad 0.75-1.28x (median 1.01x).
- rmsnorm_mxfp6: fill+scatter / F.pad 0.83-0.98x (median 0.97x).
