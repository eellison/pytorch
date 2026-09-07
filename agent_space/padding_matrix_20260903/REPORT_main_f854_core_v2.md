# Padded swizzle and quantization matrix

Source: `agent_space/padding_matrix_20260903/results_main_f854_core_v2.jsonl`
Revisions: `f8546ee64c3b9ed5cfdef3686d4f465d9af6c869`
GPUs: NVIDIA B200
Successful cells: 208; failed cells: 0.

Every cell ran in a fresh process. Timings are CUDA-graph replay medians; each replay contains the number of calls recorded in the JSON result.

## Correctness

All padding lanes contain the required sentinel and all Inductor padding variants for a workload/shape produce byte-identical payload and scale outputs.

| workload | shape | tuning | cells | byte-identical | padding values |
| --- | --- | --- | --- | --- | --- |
| dcn_mxfp6 | 95x3072 | coordesc | 2 | yes | 3168 |
| dcn_mxfp6 | 95x3072 | default | 2 | yes | 3168 |
| dcn_mxfp6 | 96x3072 | coordesc | 2 | yes | 3072 |
| dcn_mxfp6 | 96x3072 | default | 2 | yes | 3072 |
| dcn_mxfp6 | 97x3104 | coordesc | 2 | yes | 16191 |
| dcn_mxfp6 | 97x3104 | default | 2 | yes | 16191 |
| dcn_mxfp6 | 2048x3072 | coordesc | 2 | yes | 73728 |
| dcn_mxfp6 | 2048x3072 | default | 2 | yes | 73728 |
| mxfp4_quant | 19x4096 | coordesc | 3 | yes | 13952 |
| mxfp4_quant | 19x4096 | default | 3 | yes | 13952 |
| mxfp4_quant | 128x4096 | coordesc | 3 | yes | 0 |
| mxfp4_quant | 128x4096 | default | 3 | yes | 0 |
| mxfp4_quant | 129x4128 | coordesc | 3 | yes | 17151 |
| mxfp4_quant | 129x4128 | default | 3 | yes | 17151 |
| mxfp4_quant | 989x4096 | coordesc | 3 | yes | 4480 |
| mxfp4_quant | 989x4096 | default | 3 | yes | 4480 |
| mxfp6_quant | 95x3072 | coordesc | 2 | yes | 3168 |
| mxfp6_quant | 95x3072 | default | 2 | yes | 3168 |
| mxfp6_quant | 96x3072 | coordesc | 2 | yes | 3072 |
| mxfp6_quant | 96x3072 | default | 2 | yes | 3072 |
| mxfp6_quant | 97x3104 | coordesc | 2 | yes | 16191 |
| mxfp6_quant | 97x3104 | default | 2 | yes | 16191 |
| mxfp6_quant | 2048x3072 | coordesc | 2 | yes | 73728 |
| mxfp6_quant | 2048x3072 | default | 2 | yes | 73728 |
| mxfp8_quant | 19x4096 | coordesc | 3 | yes | 13952 |
| mxfp8_quant | 19x4096 | default | 3 | yes | 13952 |
| mxfp8_quant | 128x4096 | coordesc | 3 | yes | 0 |
| mxfp8_quant | 128x4096 | default | 3 | yes | 0 |
| mxfp8_quant | 129x4128 | coordesc | 3 | yes | 17151 |
| mxfp8_quant | 129x4128 | default | 3 | yes | 17151 |
| mxfp8_quant | 989x4096 | coordesc | 3 | yes | 4480 |
| mxfp8_quant | 989x4096 | default | 3 | yes | 4480 |
| rmsnorm_mxfp4 | 19x4096 | coordesc | 3 | yes | 13952 |
| rmsnorm_mxfp4 | 19x4096 | default | 3 | yes | 13952 |
| rmsnorm_mxfp4 | 128x4096 | coordesc | 3 | yes | 0 |
| rmsnorm_mxfp4 | 128x4096 | default | 3 | yes | 0 |
| rmsnorm_mxfp4 | 129x4128 | coordesc | 3 | yes | 17151 |
| rmsnorm_mxfp4 | 129x4128 | default | 3 | yes | 17151 |
| rmsnorm_mxfp4 | 989x4096 | coordesc | 3 | yes | 4480 |
| rmsnorm_mxfp4 | 989x4096 | default | 3 | yes | 4480 |
| rmsnorm_mxfp6 | 95x3072 | coordesc | 2 | yes | 3168 |
| rmsnorm_mxfp6 | 95x3072 | default | 2 | yes | 3168 |
| rmsnorm_mxfp6 | 96x3072 | coordesc | 2 | yes | 3072 |
| rmsnorm_mxfp6 | 96x3072 | default | 2 | yes | 3072 |
| rmsnorm_mxfp6 | 97x3104 | coordesc | 2 | yes | 16191 |
| rmsnorm_mxfp6 | 97x3104 | default | 2 | yes | 16191 |
| rmsnorm_mxfp6 | 2048x3072 | coordesc | 2 | yes | 73728 |
| rmsnorm_mxfp6 | 2048x3072 | default | 2 | yes | 73728 |
| rmsnorm_mxfp8 | 19x4096 | coordesc | 3 | yes | 13952 |
| rmsnorm_mxfp8 | 19x4096 | default | 3 | yes | 13952 |
| rmsnorm_mxfp8 | 128x4096 | coordesc | 3 | yes | 0 |
| rmsnorm_mxfp8 | 128x4096 | default | 3 | yes | 0 |
| rmsnorm_mxfp8 | 129x4128 | coordesc | 3 | yes | 17151 |
| rmsnorm_mxfp8 | 129x4128 | default | 3 | yes | 17151 |
| rmsnorm_mxfp8 | 989x4096 | coordesc | 3 | yes | 4480 |
| rmsnorm_mxfp8 | 989x4096 | default | 3 | yes | 4480 |
| rmsnorm_nvfp4 | 19x4096 | coordesc | 3 | yes | 27904 |
| rmsnorm_nvfp4 | 19x4096 | default | 3 | yes | 27904 |
| rmsnorm_nvfp4 | 128x4096 | coordesc | 3 | yes | 0 |
| rmsnorm_nvfp4 | 128x4096 | default | 3 | yes | 0 |
| rmsnorm_nvfp4 | 129x4128 | coordesc | 3 | yes | 33278 |
| rmsnorm_nvfp4 | 129x4128 | default | 3 | yes | 33278 |
| rmsnorm_nvfp4 | 989x4096 | coordesc | 3 | yes | 8960 |
| rmsnorm_nvfp4 | 989x4096 | default | 3 | yes | 8960 |
| swizzle_dcn | 95x3072 | default | 2 | yes | 3168 |
| swizzle_dcn | 96x3072 | default | 2 | yes | 3072 |
| swizzle_dcn | 97x3104 | default | 2 | yes | 16191 |
| swizzle_dcn | 2048x3072 | default | 2 | yes | 73728 |
| swizzle_mxfp4 | 19x4096 | default | 3 | yes | 13952 |
| swizzle_mxfp4 | 128x4096 | default | 3 | yes | 0 |
| swizzle_mxfp4 | 129x4128 | default | 3 | yes | 17151 |
| swizzle_mxfp4 | 989x4096 | default | 3 | yes | 4480 |

## Padding-only swizzle

Times are `us (generated kernels)`. Ratios above 1 mean the scatter form is slower.

| workload | shape | padding | F.pad | fill+scatter | fill/F.pad | sparse pad scatter | sparse/F.pad | experimental | experimental/F.pad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swizzle_dcn | 95x3072 | 25.8% | 1.074 (1k) | 1.856 (2k) | 1.73x | - | - | - | - |
| swizzle_dcn | 96x3072 | 25.0% | 1.136 (1k) | 2.030 (2k) | 1.79x | - | - | - | - |
| swizzle_dcn | 97x3104 | 63.2% | 1.281 (1k) | 2.012 (2k) | 1.57x | - | - | - | - |
| swizzle_dcn | 2048x3072 | 27.3% | 1.311 (1k) | 3.779 (2k) | 2.88x | - | - | - | - |
| swizzle_mxfp4 | 19x4096 | 85.2% | 1.024 (1k) | 1.845 (2k) | 1.80x | 1.955 (2k) | 1.91x | - | - |
| swizzle_mxfp4 | 128x4096 | 0.0% | 1.079 (1k) | 1.963 (2k) | 1.82x | 1.221 (1k) | 1.13x | - | - |
| swizzle_mxfp4 | 129x4128 | 50.8% | 1.106 (1k) | 2.047 (2k) | 1.85x | 2.700 (3k) | 2.44x | - | - |
| swizzle_mxfp4 | 989x4096 | 3.4% | 1.292 (1k) | 2.365 (2k) | 1.83x | 2.287 (2k) | 1.77x | - | - |

## Full pipelines

| workload | shape | tuning | F.pad | fill+scatter | sparse pad scatter | experimental | best Inductor | FlashInfer | best/FI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dcn_mxfp6 | 95x3072 | coordesc | 2.948 (2k) | 5.273 (4k) | - | - | fpad 2.948 | - | - |
| dcn_mxfp6 | 95x3072 | default | 3.375 (2k) | 5.213 (4k) | - | - | fpad 3.375 | - | - |
| dcn_mxfp6 | 96x3072 | coordesc | 2.953 (2k) | 5.376 (4k) | - | - | fpad 2.953 | - | - |
| dcn_mxfp6 | 96x3072 | default | 3.441 (2k) | 5.214 (4k) | - | - | fpad 3.441 | - | - |
| dcn_mxfp6 | 97x3104 | coordesc | 3.210 (2k) | 5.525 (4k) | - | - | fpad 3.210 | - | - |
| dcn_mxfp6 | 97x3104 | default | 3.941 (2k) | 5.469 (4k) | - | - | fpad 3.941 | - | - |
| dcn_mxfp6 | 2048x3072 | coordesc | 9.516 (2k) | 20.853 (4k) | - | - | fpad 9.516 | - | - |
| dcn_mxfp6 | 2048x3072 | default | 10.688 (2k) | 23.404 (4k) | - | - | fpad 10.688 | - | - |
| mxfp4_quant | 19x4096 | coordesc | 4.532 (4k) | 4.032 (3k) | 3.346 (3k) | - | pad_scatter 3.346 | 1.439 (1k) | 2.32x |
| mxfp4_quant | 19x4096 | default | 4.576 (4k) | 3.979 (3k) | 3.444 (3k) | - | pad_scatter 3.444 | 1.439 (1k) | 2.39x |
| mxfp4_quant | 128x4096 | coordesc | 1.506 (1k) | 3.715 (3k) | 2.794 (2k) | - | fpad 1.506 | 1.568 (1k) | 0.96x |
| mxfp4_quant | 128x4096 | default | 1.653 (1k) | 3.980 (3k) | 3.124 (2k) | - | fpad 1.653 | 1.568 (1k) | 1.05x |
| mxfp4_quant | 129x4128 | coordesc | 3.093 (2k) | 3.614 (3k) | 4.445 (4k) | - | fpad 3.093 | 2.076 (1k) | 1.49x |
| mxfp4_quant | 129x4128 | default | 3.014 (2k) | 4.108 (3k) | 4.334 (4k) | - | fpad 3.014 | 2.076 (1k) | 1.45x |
| mxfp4_quant | 989x4096 | coordesc | 7.769 (4k) | 6.567 (3k) | 21.269 (3k) | - | fill_scatter 6.567 | 3.808 (1k) | 1.72x |
| mxfp4_quant | 989x4096 | default | 8.264 (4k) | 7.032 (3k) | 7.075 (3k) | - | fill_scatter 7.032 | 3.808 (1k) | 1.85x |
| mxfp6_quant | 95x3072 | coordesc | 3.227 (2k) | 5.221 (4k) | - | - | fpad 3.227 | - | - |
| mxfp6_quant | 95x3072 | default | 3.342 (2k) | 5.436 (4k) | - | - | fpad 3.342 | - | - |
| mxfp6_quant | 96x3072 | coordesc | 2.747 (2k) | 5.563 (4k) | - | - | fpad 2.747 | - | - |
| mxfp6_quant | 96x3072 | default | 3.408 (2k) | 5.316 (4k) | - | - | fpad 3.408 | - | - |
| mxfp6_quant | 97x3104 | coordesc | 2.946 (2k) | 5.416 (4k) | - | - | fpad 2.946 | - | - |
| mxfp6_quant | 97x3104 | default | 3.860 (2k) | 5.471 (4k) | - | - | fpad 3.860 | - | - |
| mxfp6_quant | 2048x3072 | coordesc | 8.954 (2k) | 20.385 (4k) | - | - | fpad 8.954 | - | - |
| mxfp6_quant | 2048x3072 | default | 10.036 (2k) | 23.009 (4k) | - | - | fpad 10.036 | - | - |
| mxfp8_quant | 19x4096 | coordesc | 3.309 (3k) | 2.783 (2k) | 2.253 (2k) | - | pad_scatter 2.253 | 2.308 (1k) | 0.98x |
| mxfp8_quant | 19x4096 | default | 3.357 (3k) | 2.171 (2k) | 2.961 (2k) | - | fill_scatter 2.171 | 2.308 (1k) | 0.94x |
| mxfp8_quant | 128x4096 | coordesc | 1.564 (1k) | 2.632 (2k) | 2.900 (1k) | - | fpad 1.564 | 2.296 (1k) | 0.68x |
| mxfp8_quant | 128x4096 | default | 1.513 (1k) | 2.303 (2k) | 1.808 (1k) | - | fpad 1.513 | 2.296 (1k) | 0.66x |
| mxfp8_quant | 129x4128 | coordesc | 2.784 (2k) | 3.059 (2k) | 3.611 (3k) | - | fpad 2.784 | 3.176 (1k) | 0.88x |
| mxfp8_quant | 129x4128 | default | 2.899 (2k) | 2.690 (2k) | 3.567 (3k) | - | fill_scatter 2.690 | 3.176 (1k) | 0.85x |
| mxfp8_quant | 989x4096 | coordesc | 5.744 (3k) | 4.428 (2k) | 4.505 (2k) | - | fill_scatter 4.428 | 3.488 (1k) | 1.27x |
| mxfp8_quant | 989x4096 | default | 5.782 (3k) | 4.698 (2k) | 4.753 (2k) | - | fill_scatter 4.698 | 3.488 (1k) | 1.35x |
| rmsnorm_mxfp4 | 19x4096 | coordesc | 3.854 (3k) | 2.610 (2k) | 2.715 (2k) | - | fill_scatter 2.610 | 2.690 (1k) | 0.97x |
| rmsnorm_mxfp4 | 19x4096 | default | 4.744 (3k) | 3.549 (2k) | 3.665 (2k) | - | fill_scatter 3.549 | 2.690 (1k) | 1.32x |
| rmsnorm_mxfp4 | 128x4096 | coordesc | 2.064 (1k) | 3.196 (2k) | 2.061 (1k) | - | pad_scatter 2.061 | 2.897 (1k) | 0.71x |
| rmsnorm_mxfp4 | 128x4096 | default | 2.899 (1k) | 3.679 (2k) | 2.897 (1k) | - | pad_scatter 2.897 | 2.897 (1k) | 1.00x |
| rmsnorm_mxfp4 | 129x4128 | coordesc | 3.945 (2k) | 3.625 (2k) | 4.220 (3k) | - | fill_scatter 3.625 | 3.225 (1k) | 1.12x |
| rmsnorm_mxfp4 | 129x4128 | default | 4.603 (2k) | 4.250 (2k) | 4.968 (3k) | - | fill_scatter 4.250 | 3.225 (1k) | 1.32x |
| rmsnorm_mxfp4 | 989x4096 | coordesc | 6.889 (3k) | 5.796 (2k) | 5.406 (2k) | - | pad_scatter 5.406 | 5.063 (1k) | 1.07x |
| rmsnorm_mxfp4 | 989x4096 | default | 7.270 (3k) | 6.256 (2k) | 6.235 (2k) | - | pad_scatter 6.235 | 5.063 (1k) | 1.23x |
| rmsnorm_mxfp6 | 95x3072 | coordesc | 4.168 (2k) | 3.916 (2k) | - | - | fill_scatter 3.916 | - | - |
| rmsnorm_mxfp6 | 95x3072 | default | 4.649 (2k) | 3.935 (2k) | - | - | fill_scatter 3.935 | - | - |
| rmsnorm_mxfp6 | 96x3072 | coordesc | 4.211 (2k) | 3.904 (2k) | - | - | fill_scatter 3.904 | - | - |
| rmsnorm_mxfp6 | 96x3072 | default | 4.727 (2k) | 3.893 (2k) | - | - | fill_scatter 3.893 | - | - |
| rmsnorm_mxfp6 | 97x3104 | coordesc | 4.057 (2k) | 4.383 (2k) | - | - | fpad 4.057 | - | - |
| rmsnorm_mxfp6 | 97x3104 | default | 5.688 (2k) | 4.396 (2k) | - | - | fill_scatter 4.396 | - | - |
| rmsnorm_mxfp6 | 2048x3072 | coordesc | 13.658 (2k) | 14.456 (2k) | - | - | fpad 13.658 | - | - |
| rmsnorm_mxfp6 | 2048x3072 | default | 16.014 (2k) | 14.443 (2k) | - | - | fill_scatter 14.443 | - | - |
| rmsnorm_mxfp8 | 19x4096 | coordesc | 3.874 (3k) | 2.692 (2k) | 3.142 (2k) | - | fill_scatter 2.692 | 4.211 (2k) | 0.64x |
| rmsnorm_mxfp8 | 19x4096 | default | 4.870 (3k) | 3.672 (2k) | 3.801 (2k) | - | fill_scatter 3.672 | 4.211 (2k) | 0.87x |
| rmsnorm_mxfp8 | 128x4096 | coordesc | 2.034 (1k) | 3.255 (2k) | 2.047 (1k) | - | fpad 2.034 | 4.523 (2k) | 0.45x |
| rmsnorm_mxfp8 | 128x4096 | default | 2.962 (1k) | 3.749 (2k) | 2.967 (1k) | - | fpad 2.962 | 4.523 (2k) | 0.65x |
| rmsnorm_mxfp8 | 129x4128 | coordesc | 3.628 (2k) | 3.350 (2k) | 3.969 (3k) | - | fill_scatter 3.350 | 5.420 (2k) | 0.62x |
| rmsnorm_mxfp8 | 129x4128 | default | 4.600 (2k) | 4.236 (2k) | 5.035 (3k) | - | fill_scatter 4.236 | 5.420 (2k) | 0.78x |
| rmsnorm_mxfp8 | 989x4096 | coordesc | 6.766 (3k) | 5.706 (2k) | 5.571 (2k) | - | pad_scatter 5.571 | 7.499 (2k) | 0.74x |
| rmsnorm_mxfp8 | 989x4096 | default | 8.615 (3k) | 7.441 (2k) | 7.520 (2k) | - | fill_scatter 7.441 | 7.499 (2k) | 0.99x |
| rmsnorm_nvfp4 | 19x4096 | coordesc | 3.838 (3k) | 3.587 (2k) | 3.206 (2k) | - | pad_scatter 3.206 | 2.766 (1k) | 1.16x |
| rmsnorm_nvfp4 | 19x4096 | default | 4.757 (3k) | 3.592 (2k) | 3.704 (2k) | - | fill_scatter 3.592 | 2.766 (1k) | 1.30x |
| rmsnorm_nvfp4 | 128x4096 | coordesc | 2.128 (1k) | 2.829 (2k) | 2.391 (1k) | - | fpad 2.128 | 2.940 (1k) | 0.72x |
| rmsnorm_nvfp4 | 128x4096 | default | 2.884 (1k) | 3.692 (2k) | 2.914 (1k) | - | fpad 2.884 | 2.940 (1k) | 0.98x |
| rmsnorm_nvfp4 | 129x4128 | coordesc | 3.987 (2k) | 3.590 (2k) | 4.590 (3k) | - | fill_scatter 3.590 | 3.378 (1k) | 1.06x |
| rmsnorm_nvfp4 | 129x4128 | default | 4.685 (2k) | 4.189 (2k) | 5.088 (3k) | - | fill_scatter 4.189 | 3.378 (1k) | 1.24x |
| rmsnorm_nvfp4 | 989x4096 | coordesc | 7.020 (3k) | 5.732 (2k) | 5.882 (2k) | - | fill_scatter 5.732 | 4.376 (1k) | 1.31x |
| rmsnorm_nvfp4 | 989x4096 | default | 7.494 (3k) | 6.254 (2k) | 6.353 (2k) | - | fill_scatter 6.254 | 4.376 (1k) | 1.43x |

## Padding mechanism ranges

- dcn_mxfp6: fill+scatter / F.pad 1.39-2.19x (median 1.75x).
- mxfp4_quant: fill+scatter / F.pad 0.85-2.47x (median 1.03x).
- mxfp6_quant: fill+scatter / F.pad 1.42-2.29x (median 1.73x).
- mxfp8_quant: fill+scatter / F.pad 0.65-1.68x (median 0.88x).
- rmsnorm_mxfp4: fill+scatter / F.pad 0.68-1.55x (median 0.89x).
- rmsnorm_mxfp6: fill+scatter / F.pad 0.77-1.08x (median 0.91x).
- rmsnorm_mxfp8: fill+scatter / F.pad 0.69-1.60x (median 0.89x).
- rmsnorm_nvfp4: fill+scatter / F.pad 0.76-1.33x (median 0.90x).
