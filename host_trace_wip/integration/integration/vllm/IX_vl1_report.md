> **UNSOUND CONFIG (open _C ops harvested): every V-stock result in this file up to 2026-10-12 08:45 PDT.** Those runs harvested open vLLM _C ops (rms_norm, fused_add_rms_norm, rotary_embedding, silu_and_mul, reshape_and_cache_flash; on NVFP4 also scaled_fp4_quant / static_scaled_fp8_quant / cooperative_topk) and open ATen ops through the attention / rng / reduce families. The user ruled that only cuBLAS is harvested ("no extra harvest"). The sound reruns use ARMV_SOUND=1 (FAMILIES=blas, no extern list, the run fails on any non-cuBLAS learned key). Default and default-nc numbers are not affected.

Provisional, pre-core-done (learn-cost follow-ups, attention fixes, entry binding pending). InferenceX client (fixed_seq.py client_argv via `infx.bench fixed-seq point`) and result conversion (build_result), fresh server per point.
Deviations from InferenceX: Qwen3-8B is not an InferenceX model (no recipe): server args = vLLM defaults + --attention-backend FLASHINFER, --no-enable-prefix-caching, --max-model-len isl+osl+256 and --max-num-seqs max(conc,16) (the qwen3.827b PRs' rules), --seed 0; local: fixed KV pool (--kv-cache-memory-bytes 120 GiB, util 0.3 for the startup check) because GPUs are shared; ours adds --enforce-eager, the worker extension + endpoint plugin, and an HTTP range warm-up after install (startup); no chat template for 1k1k either; one run per point; profiler window after each point on a separate burst (not in the numbers).

| sku (line) | shape | conc | steps | new keys | learns (share of run) | traces | eager steps | redispatches (host s, share of run) | replayed / FULL | tok/s/GPU total | output tok/s/GPU | TPOT p50 / p90 ms | ITL p50 / p90 ms | TTFT p50 / p90 ms | E2E p50 / p90 ms | KV tokens | rc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen38bnc4-bf16-gb300-vllm (?) | 1k1k | 1 | 11050 | - | - | - | - | - | FULL 1.00 | 459.0 | 230.3 | 4.22 / 4.32 | 4.22 / 4.38 | 46.1 / 48.3 | 3871 / 4376 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 1k1k | 1 | 11050 | - | - | - | - | - | FULL 1.00 | 488.6 | 245.2 | 4.01 / 4.01 | 4.00 / 4.18 | 70.6 / 73.0 | 3697 / 4053 | 873808 | 0 |
| qwen38bpl-bf16-gb300-vllm (?) | 1k1k | 1 | 11050 | - | - | - | - | - | FULL 1.00 | 280.4 | 140.7 | 5.28 / 9.20 | 3.97 / 4.17 | 97.0 / 2878.5 | 6659 / 10733 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 1k1k | 1 | 11050 | 0 | 121 (0%) | 4 | 2 | 0 (0.1 s = 0%) | 1.00 | 296.2 | 148.6 | 5.07 / 12.40 | 3.90 / 4.10 | 62.8 / 100.6 | 4860 / 11249 | 873808 | 0 |
| qwen38bnc4-bf16-gb300-vllm (?) | 1k1k | 4 | 11248 | - | - | - | - | - | FULL 1.00 | 1734.2 | 862.9 | 4.42 / 4.46 | 4.33 / 4.54 | 49.9 / 63.8 | 4139 / 4508 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 1k1k | 4 | 11250 | - | - | - | - | - | FULL 1.00 | 1834.5 | 912.8 | 4.16 / 4.22 | 4.01 / 4.18 | 71.0 / 84.4 | 3917 / 4249 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 1k1k | 4 | 11246 | 0 | 440 (1%) | 9 | 3 | 0 (0.1 s = 0%) | 1.00 | 1472.4 | 732.6 | 4.41 / 6.29 | 4.25 / 4.42 | 64.0 / 1957.0 | 4270 / 6608 | 873808 | 0 |
| qwen38bnc4-bf16-gb300-vllm (?) | 1k1k | 16 | 11098 | - | - | - | - | - | FULL 0.99 | 6070.5 | 3019.2 | 5.12 / 5.18 | 4.68 / 4.98 | 50.8 / 93.5 | 4746 / 5218 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 1k1k | 16 | 11099 | - | - | - | - | - | FULL 0.99 | 5988.3 | 2978.3 | 5.22 / 5.33 | 4.40 / 4.66 | 74.2 / 138.2 | 4809 / 5340 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 1k1k | 16 | 11100 | 0 | 1375 (3%) | 9 | 3 | 1 (0.4 s = 1%) | 1.00 | 4829.7 | 2402.1 | 5.22 / 7.14 | 4.62 / 4.94 | 68.0 / 436.4 | 4957 / 8238 | 873808 | 0 |
| qwen38bnc4-bf16-gb300-vllm (?) | 1k1k | 64 | 11717 | - | - | - | - | - | FULL 0.96 | 16138.6 | 8067.5 | 7.64 / 8.21 | 6.01 / 6.94 | 86.1 / 143.9 | 7135 / 7940 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 1k1k | 64 | 11732 | - | - | - | - | - | FULL 0.96 | 12990.2 | 6493.6 | 9.20 / 10.98 | 5.71 / 6.84 | 152.2 / 424.9 | 8779 / 10666 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 1k1k | 64 | 11718 | 0 | 3454 (7%) | 8 | 3 | 2 (0.5 s = 0%) | 1.00 | 13144.9 | 6570.9 | 7.81 / 8.89 | 6.22 / 7.65 | 87.5 / 1044.5 | 7481 / 10191 | 873808 | 0 |
| qwen38bnc4-bf16-gb300-vllm (?) | 8k1k | 1 | 11141 | - | - | - | - | - | FULL 1.00 | 1934.2 | 216.1 | 4.48 / 4.52 | 4.48 / 4.64 | 122.8 / 131.0 | 4175 / 4666 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 8k1k | 1 | 11141 | - | - | - | - | - | FULL 1.00 | 2051.6 | 229.2 | 4.24 / 4.25 | 4.24 / 4.37 | 118.6 / 129.3 | 3968 / 4336 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 8k1k | 1 | 11141 | 0 | 121 (1%) | 2 | 2 | 3 (1.0 s = 2%) | 1.00 | 2040.7 | 228.0 | 4.13 / 4.21 | 4.14 / 4.29 | 175.3 / 221.8 | 4088 / 4364 | 873808 | 0 |
| qwen38bnc4-bf16-gb300-vllm (?) | 8k1k | 4 | 11565 | - | - | - | - | - | FULL 1.00 | 6696.0 | 744.9 | 5.11 / 5.18 | 4.86 / 5.08 | 125.1 / 174.3 | 4817 / 5255 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 8k1k | 4 | 11566 | - | - | - | - | - | FULL 1.00 | 6786.9 | 755.0 | 4.88 / 5.51 | 4.63 / 4.89 | 123.2 / 243.9 | 4737 / 5267 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 8k1k | 4 | 11566 | 0 | 495 (2%) | 6 | 3 | 4 (1.3 s = 2%) | 1.00 | 6571.2 | 731.0 | 4.93 / 5.43 | 4.55 / 4.80 | 171.6 / 786.8 | 4811 / 5609 | 873808 | 0 |
| qwen38bnc4-bf16-gb300-vllm (?) | 8k1k | 16 | 11759 | - | - | - | - | - | FULL 0.99 | 16671.0 | 1847.3 | 8.20 / 8.48 | 6.83 / 10.72 | 198.1 / 343.5 | 7670 / 8584 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 8k1k | 16 | 11698 | - | - | - | - | - | FULL 0.99 | 17167.8 | 1902.3 | 7.73 / 9.04 | 6.49 / 8.05 | 176.3 / 421.0 | 7388 / 8681 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 8k1k | 16 | 11683 | 0 | 1771 (6%) | 10 | 3 | 4 (1.4 s = 1%) | 1.00 | 14453.6 | 1601.5 | 8.65 / 9.76 | 6.72 / 7.31 | 188.6 / 1630.3 | 8321 / 10150 | 873808 | 0 |
| qwen38bnc4-bf16-gb300-vllm (?) | 8k1k | 64 | 11579 | - | - | - | - | - | FULL 0.95 | 28562.8 | 3168.8 | 19.72 / 20.44 | 14.60 / 23.93 | 192.3 / 482.7 | 18470 / 20684 | 873808 | 0 |
| qwen38bp4-bf16-gb300-vllm (?) | 8k1k | 64 | 11571 | - | - | - | - | - | FULL 0.95 | 29285.5 | 3249.0 | 19.24 / 19.96 | 14.32 / 18.59 | 201.3 / 572.6 | 17971 / 20322 | 873808 | 0 |
| qwen38bvs4-bf16-gb300-vllm (?) | 8k1k | 64 | 11579 | 0 | 5764 (11%) | 10 | 3 | 6 (1.6 s = 1%) | 1.00 | 24055.0 | 2668.7 | 22.41 / 23.71 | 14.74 / 19.06 | 271.1 / 1583.7 | 21118 / 24519 | 873808 | 0 |
