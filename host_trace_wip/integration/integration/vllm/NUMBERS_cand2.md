> **UNSOUND CONFIG (open _C ops harvested): every V-stock result in this file up to 2026-10-12 08:45 PDT.** Those runs harvested open vLLM _C ops (rms_norm, fused_add_rms_norm, rotary_embedding, silu_and_mul, reshape_and_cache_flash; on NVFP4 also scaled_fp4_quant / static_scaled_fp8_quant / cooperative_topk) and open ATen ops through the attention / rng / reduce families. The user ruled that only cuBLAS is harvested ("no extra harvest"). The sound reruns use ARMV_SOUND=1 (FAMILIES=blas, no extern list, the run fails on any non-cuBLAS learned key). Default and default-nc numbers are not affected.

# vLLM arm V on int6 candidate 2: data points (accumulating; regenerate with `bash mk_numbers.sh`)

Setup: Qwen3-8B, vLLM v0.29.0 (Model Runner V2, FLASHINFER/trtllm-gen, in-process engine), max_model_len 4096, prefix caching off.
- Lines: cpp = candidate2/src_cpp on int6/build_cpp7/install (C++ entry); py = candidate2/src_py on install_land_int5b_frozen.
- Variants: closed = vLLM _C ops + trtllm attention as harvested externs. open = ARMV_TRACED_IMPLS=1 (Python launch ports of the
  five _C ops) + ARMV_TRTLLM_FORK=1 (the FlashInfer trtllm fork). In both, the GEMMs stay cuBLAS (harvested): no open GEMM is wired for vLLM.
  eager and default run the same kernels in both variants, so they were run once (stock launcher).
- default = vLLM's shipped FULL_AND_PIECEWISE cudagraphs + torch.compile (padded batches, its own graphs); eager = enforce_eager.
- Adapter: integration/vllm/armV_cand2 (changes in OVERLAY.md). V runs exact sizes (ARMV_PAD=0): under enforce_eager, vLLM's
  dispatch returns cg_mode NONE with the unpadded num_tokens/num_reqs, and the adapter's _same_inputs accepts only unpadded batches.
  So today's V never pads. ARMV_PAD=1 emulates the default's padding for the A/B.
- Correctness/counts: GPU 1 (gpu1.lock). util 0.6 on an empty GPU, 0.16 when a foreign job holds memory (counts do not
  depend on the KV size). Timing: GPU 0 under one gpu0.lock hold per job, CUDA_VISIBLE_DEVICES=0, taskset 0-71, a GPU idle check
  before each run, util 0.6 (no foreign GPU 0 memory). Load and CPU% at each run start: logs/timing_idle.log.
- Caveat: the host was CPU-saturated by other users during these runs (load avg 75-200; the 0-71 range had 35-40 busy cores).
  Host-bound step times are noisy; see the per-round JSONs. Startup is hit hardest: `import vllm` took up to 330 s, 210 s of it
  in transformers.utils.import_utils (logs/importtime.txt), and model load took 8-89 s. Imports are listed separately, and
  "ready - imports" is the comparable column.
- Raw: out/*.json (c2_* step 1, t_* timing, s_* startup), logs/*.log.

### counts

| run | checks bitwise | traces | variants | boundaries (serving) | segments | eager first calls | keys harvested | bound hits | model fwds init/bench | json |
|---|---|---|---|---|---|---|---|---|---|---|
| c2_attn_vstock_Vcheck | 188/188 | 4 | 4 | {0: 4} | {1: 4} | 3 | 187 | - | 7/199 | out/c2_attn_vstock_Vcheck.json |
| c2_cpp_VP2_Vcheck | 127/127 | 7 | 7 | {0: 4, 1: 3} | {1: 4, 2: 3} | 5 | 140 | 0 | 7/273 | out/c2_cpp_VP2_Vcheck.json |
| c2_cpp_VP3_Vcheck | 127/127 | 7 | 7 | {0: 4, 1: 3} | {1: 4, 2: 3} | 5 | 140 | 0 | 7/273 | out/c2_cpp_VP3_Vcheck.json |
| c2_cpp_VP4_Vcheck | 127/127 | 7 | 7 | {0: 7} | {1: 7} | 5 | 146 | 0 | 7/273 | out/c2_cpp_VP4_Vcheck.json |
| c2_cpp_VP_Vcheck | 127/127 | 7 | 7 | {0: 4, 1: 3} | {1: 4, 2: 3} | 5 | 140 | 0 | 7/273 | out/c2_cpp_VP_Vcheck.json |
| c2_cpp_closed_Vcheck | 228/228 | 7 | 7 | {0: 7} | {1: 7} | 5 | 269 | 0 | 7/475 | out/c2_cpp_closed_Vcheck.json |
| c2_cpp_closed_maxseq_Vcheck | 23/23 | 1 | 1 | {0: 1} | {1: 1} | 5 | 24 | 0 | 7/155 | out/c2_cpp_closed_maxseq_Vcheck.json |
| c2_cpp_closed_pad_Vcheck | 281/297 | 9 | 9 | {0: 9} | {1: 9} | 5 | 285 | 0 | 7/617 | out/c2_cpp_closed_pad_Vcheck.json |
| c2_cpp_open_Vcheck | 193/228 | 7 | 11 | {0: 11} | {1: 11} | 5 | 88 | 0 | 7/475 | out/c2_cpp_open_Vcheck.json |
| c2_fast_attn_vstock_Vcheck | 188/188 | 4 | 4 | {0: 4} | {1: 4} | 3 | 187 | - | 7/199 | out/c2_fast_attn_vstock_Vcheck.json |
| c2_fast_attnlc_vstock_Vcheck | 188/188 | 4 | 4 | {0: 4} | {1: 4} | 3 | 187 | - | 7/199 | out/c2_fast_attnlc_vstock_Vcheck.json |
| c2_fork_sym_Vcheck | 228/228 | 8 | 9 | {0: 9} | {1: 9} | 5 | 242 | - | 7/249 | out/c2_fork_sym_Vcheck.json |
| c2_pinned_VPpin_Vcheck | 127/127 | 7 | 7 | {0: 7} | {1: 7} | 5 | 146 | 0 | 7/273 | out/c2_pinned_VPpin_Vcheck.json |
| c2_py_closed_Vcheck | 228/228 | 7 | 7 | {0: 7} | {1: 7} | 5 | 269 | - | 7/475 | out/c2_py_closed_Vcheck.json |
| c2_stockfast_attnlc_Vcheck | 188/188 | 4 | 4 | {0: 4} | {1: 4} | 3 | 187 | - | 7/199 | out/c2_stockfast_attnlc_Vcheck.json |
| c2_stockfast_c4_Vcheck | 188/188 | 4 | 4 | {0: 4} | {1: 4} | 3 | 187 | - | 7/199 | out/c2_stockfast_c4_Vcheck.json |
| c2_vstock_Vcheck | 188/188 | 4 | 4 | {0: 4} | {1: 4} | 3 | 187 | - | 7/199 | out/c2_vstock_Vcheck.json |
| c2_vstock_nodirect_Vcheck | 109/109 | 6 | 6 | {36: 5, 72: 1} | {37: 5, 73: 1} | 3 | 110 | - | 7/124 | out/c2_vstock_nodirect_Vcheck.json |

| range run | checks bitwise | traces dec/pre/mix | variants | boundaries/variant | relowers | keys learned | eager entry | first-call median ms dec/pre/mix | json |
|---|---|---|---|---|---|---|---|---|---|
| c2_attn_vstock_range | 958/958 | 3/0/5 = 8 | 8 | {'0': 8} | 0 | 5995 | 3 | None/None/None | out/c2_attn_vstock_range.json |
| c2_cpp_closed_maxseq_range | 543/543 | 3/1/4 = 8 | 6 | {'0': 6} | 0 | 5976 | 10 | 164.63/None/89.13 | out/c2_cpp_closed_maxseq_range.json |
| c2_cpp_closed_range | 969/969 | 4/5/9 = 18 | 15 | {'0': 15} | 0 | 6619 | 10 | 189.14/154.37/142.75 | out/c2_cpp_closed_range.json |
| c2_cpp_open_range | 960/969 | 5/5/9 = 19 | 20 | {'0': 20} | 0 | 2180 | 10 | 162.35/100.79/110.74 | out/c2_cpp_open_range.json |
| c2_fork_model_range | 969/969 | 5/5/9 = 19 | 20 | {'0': 20} | 0 | 5995 | 10 | 190.27/128.21/139.94 | out/c2_fork_model_range.json |
| c2_fork_sym_range | 969/969 | 4/5/10 = 19 | 19 | {'0': 19} | 0 | 5995 | 10 | 125.86/392.82/101.03 | out/c2_fork_sym_range.json |
| c2_pinned_norejit_range | 969/969 | 4/5/9 = 18 | 15 | {'0': 15} | 0 | 6619 | 10 | 201.33/144.3/148.08 | out/c2_pinned_norejit_range.json |
| c2_pinned_rejit_range | 969/969 | 4/5/9 = 18 | 15 | {'0': 15} | 0 | 6619 | 10 | 196.48/157.5/153.25 | out/c2_pinned_rejit_range.json |
| c2_py_closed_range | 969/969 | 4/5/9 = 18 | 15 | {'0': 15} | 0 | 6619 | 10 | 125.91/119.34/130.82 | out/c2_py_closed_range.json |
| c2_stockfast_attnlc_range | 958/958 | 3/0/5 = 8 | 8 | {'0': 8} | 0 | 5995 | 3 | None/None/None | out/c2_stockfast_attnlc_range.json |
| c2_stockfast_c4_range | 958/958 | 3/0/5 = 8 | 8 | {'0': 8} | 0 | 5995 | 3 | None/None/None | out/c2_stockfast_c4_range.json |
| c2_vstock_range | 958/958 | 3/0/6 = 9 | 9 | {'0': 9} | 0 | 5995 | 3 | None/None/None | out/c2_vstock_range.json |

### stock

Arm V-stock (the default reported arm) next to the shipped default and default-nc (vLLM with compile off and its own full decode graphs;
prefill and mixed run eager). Pinned build (land/core/pinned).

Counts (V-stock; c2_vstock_* on the pinned build, c2_attn_vstock_* on CAND_LINE=attn = pinned + A1 + F1 (choice() present but unused)):
- c2_vstock_Vcheck: 188/188 bitwise vs stock eager, 4 traces, 4 variants, redispatches prefill/decode/mixed 4/1/15, model forwards init/bench 7/199 (out/c2_vstock_Vcheck.json)
- c2_vstock_range: 958/958 bitwise, 9 traces (out/c2_vstock_range.json)
- c2_attn_vstock_Vcheck: 188/188 bitwise vs stock eager, 4 traces, 4 variants, redispatches prefill/decode/mixed 0/1/0, model forwards init/bench 7/199 (out/c2_attn_vstock_Vcheck.json)
- c2_attn_vstock_range: 958/958 bitwise, 8 traces (out/c2_attn_vstock_range.json)
- q35_vstock_Vcheck: 163/163 bitwise vs stock eager, 25 traces, 4 variants, redispatches prefill/decode/mixed 0/0/0, model forwards init/bench 0/0 (out/q35_vstock_Vcheck.json)

Greedy tokens (e2e_tokens.py: seed-0 random 128-token prompts, 32 tokens, 2 repeats per bs): sequences equal to eager per bs.
| arm | bs 1 | bs 8 | bs 64 | bs 128 | json |
|---|---|---|---|---|---|
| default-nc | 1/1 | 5/8 | 64/64 | 128/128 | out/e2e_p_fullnc.json |
| default-nc (FULL requested) | 1/1 | 5/8 | 64/64 | 128/128 | out/e2e_p_fullnc_full.json |
| V-stock | 1/1 | 8/8 | 64/64 | 128/128 | out/e2e_p_vstock.json |
| V-stock, attn line | 1/1 | 8/8 | 64/64 | 128/128 | out/e2e_a_vstock.json |
| default (int6 run) | 1/1 | 7/8 | 52/64 | 89/128 | out/e2e_default.json |
| template V (int6 run) | 1/1 | 5/8 | 64/64 | 128/128 | out/e2e_V.json |

Set g3 (InferenceX-shaped single steps; V-stock attention = FlashInfer trtllm fork (F1), CAND_LINE=attn = A1 + F1 (choice() present but unused); max_model_len 16384, util 0.9; 4 measured steps per mixed row, 16 per decode row): load-gated rounds (rounds_gpu0.sh, one gpu0.lock hold per round, arm order rotated, CPUs 0-71).
Per-round load gate (logs/rounds_g3.log; per-run load in logs/timing_idle.log):
- 19:46:33 round 1 load_gate ok load1 88.97 busy_cores0-71 23.1 (1236 s wait)
- 20:04:56 round 1 done rc=0 load 130.61 125.83 122.17
- 20:31:21 round 2 load_gate ok load1 87.97 busy_cores0-71 22.2 (1585 s wait)
- 20:48:16 round 2 done rc=0 load 125.46 133.98 133.95
- 21:06:50 round 3 load_gate ok load1 81.84 busy_cores0-71 20.0 (1114 s wait)
- 21:43:06 round 3 done rc=0 load 121.68 98.87 106.85
- 21:46:39 round 4 load_gate ok load1 84.66 busy_cores0-71 21.5 (213 s wait)
- 22:02:01 round 4 done rc=0 load 154.00 151.56 132.09
- 22:13:07 round 5 load_gate ok load1 80.07 busy_cores0-71 16.5 (665 s wait)
- 00:52:47 round 5 done rc=0 load 80.56 132.02 149.83
- 00:52:57 round 6 load_gate ok load1 80.56 busy_cores0-71 22.9 (10 s wait)
- 01:24:29 round 6 done rc=0 load 166.23 184.11 183.13

Step ms (execute_model, synced unless async), median over rounds [min-max]; async = (wall(48 tok) - wall(16 tok)) / 32 under async scheduling.
| workload | default | default-nc | V-stock | default / V-stock | default-nc / V-stock | decoder KV min/med/max |
|---|---|---|---|---|---|---|
| decode synced 1 | 4.98 [4.68-5.27] | 4.84 [4.77-5.02] | 5.16 [4.98-5.36] | 0.96 | 0.94 | - |
| decode synced 8 | 4.94 [4.82-5.20] | 4.89 [4.84-5.03] | 5.38 [5.26-5.58] | 0.92 | 0.91 | - |
| decode async 1 | 5.87 [5.62-6.11] | 5.84 [5.63-6.18] | 5.83 [5.41-9.70] | 1.01 | 1.00 | - |
| decode async 8 | 5.92 [5.83-6.05] | 5.90 [5.61-6.14] | 6.12 [5.70-9.51] | 0.97 | 0.96 | - |
| mixed 8192-token chunk + 32 decodes | 94.49 [94.09-94.84] | 98.08 [97.46-145.29] | 97.34 [96.87-97.78] | 0.97 | 1.01 | 4109/6061/8012 |
| mixed 8192-token chunk + 64 decodes | 98.50 [97.52-98.88] | 101.85 [101.45-116.45] | 101.15 [100.90-101.56] | 0.97 | 1.01 | 4121/6068/8012 |
| mixed 8192-token chunk + 128 decodes | 105.88 [105.45-106.88] | 109.14 [108.47-110.60] | 108.44 [107.86-109.55] | 0.98 | 1.01 | 4145/6082/8012 |
| mixed 4096-token chunk + 64 decodes | 57.54 [55.26-65.91] | 56.35 [55.67-58.56] | 55.52 [55.27-56.12] | 1.04 | 1.02 | 4121/6068/8012 |
| decode bs 64 | 14.43 [14.28-15.01] | 14.49 [14.32-15.02] | 15.18 [14.84-15.68] | 0.95 | 0.95 | 6029/7025/8021 |
| decode bs 128 | 23.74 [23.61-24.02] | 23.45 [23.28-24.00] | 24.45 [23.77-24.81] | 0.97 | 0.96 | 6056/7040/8021 |
| decode bs 256 | 37.48 [37.10-37.65] | 37.05 [36.85-37.54] | 37.55 [37.36-37.92] | 1.00 | 0.99 | 5096/6062/7021 |
| KV tokens (k) | 1677 | 1678 | 1684 | | | |
| bench peak over init MiB | 1536 | 1696 | 1696 | | | |
| init s (engine ready; V-stock traces at first call, not at init) | 65 | 52 | 32 | | | |
| model forwards at init | 506 | 174 | 7 | | | |
rounds: {'default': 6, 'default-nc': 6, 'V-stock': 6} | json: out/t_g3_default_r*.json; out/t_g3_fullnc_r*.json; out/t_g3_vstock_r*.json

Set g4 (1k1k-shaped and 8k1k prompt-tail mixed steps; V-stock attention = FlashInfer trtllm fork (F1), CAND_LINE=attn = A1 + F1 (choice() present but unused); max_model_len 16384, util 0.9; 4 measured steps per row): load-gated rounds (rounds_gpu0.sh, one gpu0.lock hold per round, arm order rotated, CPUs 0-71).
Per-round load gate (logs/rounds_g4.log; per-run load in logs/timing_idle.log):
- 01:39:36 round 1 load_gate ok load1 80.32 busy_cores0-71 18.1 (866 s wait)
- 01:49:18 round 1 done rc=0 load 309.70 371.06 274.20
- 13:43:16 round 2 load_gate ok load1 24.27 busy_cores0-71 9.3 (10 s wait)
- 13:50:49 round 2 done rc=0 load 71.60 64.33 41.24
- 13:50:59 round 3 load_gate ok load1 71.60 busy_cores0-71 11.7 (10 s wait)
- 14:24:50 round 3 done rc=0 load 51.24 51.93 51.73
- 14:25:00 round 4 load_gate ok load1 51.24 busy_cores0-71 9.9 (10 s wait)
- 14:34:01 round 4 done rc=0 load 76.55 64.43 57.10
- 14:34:11 round 5 load_gate ok load1 76.55 busy_cores0-71 8.1 (10 s wait)
- 14:43:48 round 5 done rc=0 load 138.18 78.78 62.40
- 14:48:03 round 6 load_gate ok load1 75.71 busy_cores0-71 16.9 (255 s wait)
- 14:54:53 round 6 done rc=0 load 84.27 78.12 72.56

Step ms (execute_model, synced unless async), median over rounds [min-max]; async = (wall(48 tok) - wall(16 tok)) / 32 under async scheduling.
| workload | default | default-nc | V-stock | default / V-stock | default-nc / V-stock | decoder KV min/med/max |
|---|---|---|---|---|---|---|
| 1k1k-shaped mixed: 1024-token chunk + 64 decodes | 59.23 [52.51-62.61] | 48.81 [33.30-59.44] | 17.40 [16.83-18.59] | 3.40 | 2.81 | 1032/1546/2060 |
| 1k1k-shaped mixed: 1024-token chunk + 128 decodes | 59.08 [51.61-159.64] | 51.45 [34.14-60.31] | 20.16 [19.10-21.18] | 2.93 | 2.55 | 1038/1550/2060 |
| 1k1k-shaped mixed: 2048-token chunk + 64 decodes | 58.46 [54.32-274.14] | 49.87 [34.17-51.05] | 28.21 [27.19-29.42] | 2.07 | 1.77 | 1032/1546/2060 |
| 1k1k-shaped mixed: 2048-token chunk + 128 decodes | 66.41 [59.49-70.27] | 47.60 [32.00-54.18] | 30.19 [29.66-31.76] | 2.20 | 1.58 | 1038/1550/2060 |
| 8k1k prompt-tail mixed: 1024-token chunk + 64 decodes | 58.02 [52.95-64.16] | 47.93 [34.22-51.99] | 25.37 [24.54-51.90] | 2.29 | 1.89 | 6029/7021/8012 |
| 8k1k prompt-tail mixed: 1024-token chunk + 128 decodes | 60.50 [55.51-71.81] | 47.80 [35.12-60.43] | 33.89 [33.55-35.21] | 1.79 | 1.41 | 6056/7035/8012 |
| 8k1k prompt-tail mixed: 2048-token chunk + 64 decodes | 60.89 [50.94-68.17] | 45.29 [34.88-53.06] | 35.48 [34.78-36.24] | 1.72 | 1.28 | 6029/7021/8012 |
| 8k1k prompt-tail mixed: 2048-token chunk + 128 decodes | 62.86 [55.46-71.68] | 47.91 [44.44-52.70] | 44.40 [43.91-45.06] | 1.42 | 1.08 | 6056/7035/8012 |
| KV tokens (k) | 1677 | 1678 | 1684 | | | |
| bench peak over init MiB | 1536 | 1696 | 1696 | | | |
| init s (engine ready; V-stock traces at first call, not at init) | 57 | 31 | 17 | | | |
| model forwards at init | 506 | 174 | 7 | | | |
rounds: {'default': 6, 'default-nc': 6, 'V-stock': 6} | json: out/t_g4_default_r*.json; out/t_g4_fullnc_r*.json; out/t_g4_vstock_r*.json

Set q1 (Qwen3.5-9B (GDN hybrid; recipe args --trust-remote-code --language-model-only, vLLM's own attention backend); V-stock attention = TRT-LLM C++ launcher path (CAND_LINE=trtcpp: attn snap s5 Python on the pinned install, ARMV_TRTLLM_CPP=1); max_model_len 16384, util 0.9): load-gated rounds (rounds_gpu0.sh, one gpu0.lock hold per round, arm order rotated, CPUs 0-71).
Per-round load gate (logs/rounds_q1.log; per-run load in logs/timing_idle.log):
- 15:59:58 round 1 load_gate ok load1 57.56 busy_cores0-71 16.0 (52 s wait)
- 16:18:38 round 1 done rc=0 load 38.74 43.78 51.80
- 16:18:48 round 2 load_gate ok load1 38.74 busy_cores0-71 13.1 (10 s wait)
- 17:05:30 round 2 done rc=0 load 37.72 46.27 49.43
- 17:05:40 round 3 load_gate ok load1 37.72 busy_cores0-71 12.1 (10 s wait)
- 17:17:36 round 3 done rc=0 load 40.09 42.12 47.03
- 17:17:48 round 4 load_gate ok load1 40.09 busy_cores0-71 9.4 (12 s wait)
- 17:29:16 round 4 done rc=0 load 52.93 65.06 56.80
- 17:29:26 round 5 load_gate ok load1 52.93 busy_cores0-71 13.4 (10 s wait)
- 17:42:26 round 5 done rc=0 load 69.07 89.11 82.33
- 17:42:36 round 6 load_gate ok load1 69.07 busy_cores0-71 13.0 (10 s wait)
- 17:55:00 round 6 done rc=0 load 77.04 94.86 93.65

Step ms (execute_model, synced unless async), median over rounds [min-max]; async = (wall(48 tok) - wall(16 tok)) / 32 under async scheduling.
| workload | default | default-nc | V-stock | default / V-stock | default-nc / V-stock | decoder KV min/med/max |
|---|---|---|---|---|---|---|
| decode synced 1 | 5.02 [4.74-5.20] | 6.33 [5.85-6.80] | 6.55 [6.23-7.54] | 0.77 | 0.97 | - |
| decode synced 8 | 5.35 [5.07-5.44] | 6.98 [6.65-7.38] | 7.15 [6.68-7.64] | 0.75 | 0.98 | - |
| decode async 1 | 5.96 [5.73-6.51] | 7.52 [7.04-9.23] | 6.99 [6.66-7.72] | 0.85 | 1.08 | - |
| decode async 8 | 6.45 [5.92-7.22] | 8.42 [6.59-10.58] | 8.60 [7.55-9.38] | 0.75 | 0.98 | - |
| 1k1k-shaped mixed: 1024-token chunk + 64 decodes | 86.02 [70.01-101.09] | 100.41 [66.17-117.29] | 112.06 [82.66-124.82] | 0.77 | 0.90 | 1032/1546/2060 |
| 1k1k-shaped mixed: 1024-token chunk + 128 decodes | 94.00 [72.70-95.04] | 93.95 [66.22-119.97] | 96.15 [74.46-121.62] | 0.98 | 0.98 | 1038/1550/2060 |
| 1k1k-shaped mixed: 2048-token chunk + 64 decodes | 92.61 [76.88-97.49] | 94.77 [65.92-120.65] | 98.85 [74.72-122.07] | 0.94 | 0.96 | 1032/1546/2060 |
| 1k1k-shaped mixed: 2048-token chunk + 128 decodes | 92.56 [74.37-97.05] | 93.70 [70.40-112.92] | 97.88 [77.88-119.51] | 0.95 | 0.96 | 1038/1550/2060 |
| 8k1k prompt-tail mixed: 1024-token chunk + 64 decodes | 83.75 [74.49-96.12] | 91.50 [66.21-104.48] | 113.25 [88.73-146.19] | 0.74 | 0.81 | 6029/7021/8012 |
| 8k1k prompt-tail mixed: 1024-token chunk + 128 decodes | 92.86 [75.34-99.85] | 98.01 [69.87-106.72] | 103.59 [76.68-120.22] | 0.90 | 0.95 | 6056/7035/8012 |
| 8k1k prompt-tail mixed: 2048-token chunk + 64 decodes | 91.33 [74.92-94.74] | 95.14 [70.43-105.75] | 102.82 [68.62-123.76] | 0.89 | 0.93 | 6029/7021/8012 |
| 8k1k prompt-tail mixed: 2048-token chunk + 128 decodes | 97.21 [78.19-103.64] | 99.11 [66.94-115.35] | 99.84 [75.38-120.85] | 0.97 | 0.99 | 6056/7035/8012 |
| 8k1k mixed: 4096-token chunk + 64 decodes | 98.24 [75.31-114.86] | 99.82 [73.03-107.34] | 97.80 [80.02-120.35] | 1.00 | 1.02 | 4121/6068/8012 |
| 8k1k mixed: 8192-token chunk + 32 decodes | 100.27 [98.37-107.28] | 124.68 [121.85-125.97] | 128.33 [126.40-131.28] | 0.78 | 0.97 | 4109/6061/8012 |
| 8k1k mixed: 8192-token chunk + 64 decodes | 104.44 [100.78-108.56] | 126.47 [124.64-128.25] | 130.78 [127.88-132.02] | 0.80 | 0.97 | 4121/6068/8012 |
| 8k1k mixed: 8192-token chunk + 128 decodes | 107.44 [103.42-110.67] | 130.75 [128.32-131.68] | 134.83 [132.13-138.07] | 0.80 | 0.97 | 4145/6082/8012 |
| decode bs 64 | 8.87 [8.63-8.97] | 10.91 [10.46-11.30] | 10.68 [10.29-11.16] | 0.83 | 1.02 | 6029/7025/8021 |
| decode bs 128 | 12.71 [12.32-13.06] | 14.32 [14.05-14.75] | 14.84 [14.54-14.99] | 0.86 | 0.97 | 6056/7040/8021 |
| decode bs 256 | 19.63 [19.38-19.81] | 21.77 [21.09-22.07] | 22.15 [21.71-22.27] | 0.89 | 0.98 | 6112/7070/8021 |
| KV tokens (k) | 7476 | 7476 | 7504 | | | |
| bench peak over init MiB | 1714 | 1883 | 1882 | | | |
| init s (engine ready; V-stock traces at first call, not at init) | 77 | 46 | 32 | | | |
| model forwards at init | 0 | 0 | 0 | | | |
rounds: {'default': 6, 'default-nc': 6, 'V-stock': 6} | json: out/t_q1_default_r*.json; out/t_q1_fullnc_r*.json; out/t_q1_vstock_r*.json

#### Diagnostics appendix: small-step microbenchmarks (not InferenceX shapes)
prefill 64/512 and mixed 8x128 are too small to stand for serving; kept as raw data only.

Set g1 (pinned build; V-stock attention = FlashInfer trtllm fork (before F1); mixed before the attention fix, 1 measured mixed step per round): load-gated rounds (rounds_gpu0.sh, one gpu0.lock hold per round, arm order rotated, CPUs 0-71).
Per-round load gate (logs/rounds_g1.log; per-run load in logs/timing_idle.log):
- 17:45:22 round 1 load_gate ok load1 88.84 busy_cores0-71 22.4 (298 s wait)
- 18:02:23 round 1 done rc=0 load 63.24 75.62 88.96
- 18:02:34 round 2 load_gate ok load1 63.24 busy_cores0-71 18.1 (11 s wait)
- 18:16:08 round 2 done rc=0 load 161.41 136.54 112.12
- 18:20:21 round 3 load_gate ok load1 62.46 busy_cores0-71 22.7 (253 s wait)
- 18:26:16 round 3 done rc=0 load 99.71 97.09 101.31
- 18:27:07 round 4 load_gate ok load1 88.52 busy_cores0-71 18.4 (51 s wait)
- 18:35:45 round 4 done rc=0 load 56.03 75.95 90.49
- 18:35:57 round 5 load_gate ok load1 56.03 busy_cores0-71 18.4 (12 s wait)
- 18:52:10 round 5 done rc=0 load 60.23 59.75 71.60
- 18:53:42 round 6 load_gate ok load1 64.47 busy_cores0-71 22.6 (92 s wait)
- 19:11:26 round 6 done rc=0 load 118.06 111.76 102.61

Step ms (execute_model, synced unless async), median over rounds [min-max]; async = (wall(48 tok) - wall(16 tok)) / 32 under async scheduling.
| workload | default | default-nc | V-stock | default / V-stock | default-nc / V-stock |
|---|---|---|---|---|---|
| decode synced 1 | 4.60 [4.56-4.86] | 4.90 [4.81-5.35] | 5.38 [4.99-6.31] | 0.86 | 0.91 |
| decode synced 8 | 4.75 [4.53-4.99] | 5.13 [4.88-5.60] | 5.18 [4.98-5.72] | 0.92 | 0.99 |
| decode synced 64 | 5.44 [5.07-5.78] | 5.48 [5.29-5.95] | 6.08 [5.84-6.49] | 0.89 | 0.90 |
| decode synced 128 | 6.11 [5.76-6.48] | 6.66 [6.11-6.96] | 6.90 [6.52-7.06] | 0.88 | 0.96 |
| decode async 1 | 5.65 [5.36-6.08] | 6.24 [5.65-10.06] | 5.96 [5.52-6.47] | 0.95 | 1.05 |
| decode async 8 | 5.89 [5.65-6.48] | 6.72 [5.73-6.82] | 6.09 [5.73-6.75] | 0.97 | 1.10 |
| decode async 64 | 7.31 [6.93-8.42] | 8.30 [6.67-9.31] | 8.23 [7.43-12.74] | 0.89 | 1.01 |
| decode async 128 | 9.18 [8.40-9.91] | 9.82 [8.57-10.58] | 8.77 [8.29-9.54] | 1.05 | 1.12 |
| prefill 64 | 40.96 [37.84-45.63] | 43.79 [35.12-56.77] | 6.58 [6.17-7.18] | 6.22 | 6.65 |
| prefill 512 | 40.34 [36.98-45.29] | 36.92 [34.75-56.85] | 9.70 [8.98-10.00] | 4.16 | 3.81 |
| mixed 8x128 (pre-fix: known tile-choice pin) | 28.11 [24.38-34.10] | 42.24 [33.54-70.58] | 323.70 [287.56-394.28] | 0.09 | 0.13 |
| KV tokens (k) | 1073 | 1074 | 1080 | | |
| bench peak over init MiB | 1536 | 1696 | 1696 | | |
| init s (engine ready; V-stock traces at first call, not at init) | 61 | 29 | 18 | | |
| model forwards at init | 506 | 174 | 7 | | |
rounds: {'default': 6, 'default-nc': 6, 'V-stock': 6} | json: out/t_g1_default_r*.json; out/t_g1_fullnc_r*.json; out/t_g1_vstock_r*.json

Earlier ungated runs (10-08 afternoon; load avg 83-155, up to 20% round-to-round spread).
Step ms (execute_model, synced unless async), median over rounds [min-max]; async = (wall(48 tok) - wall(16 tok)) / 32 under async scheduling.
| workload | default | default-nc | V-stock | default / V-stock | default-nc / V-stock |
|---|---|---|---|---|---|
| decode synced 1 | 4.53 [4.44-4.67] | 4.85 [4.78-4.97] | 5.17 [5.05-5.33] | 0.88 | 0.94 |
| decode synced 8 | 4.61 [4.53-4.81] | 5.00 [4.72-5.35] | 5.06 [5.02-5.79] | 0.91 | 0.99 |
| decode synced 64 | 5.12 [5.11-5.37] | 5.49 [5.22-5.77] | 6.02 [5.91-6.52] | 0.85 | 0.91 |
| decode synced 128 | 5.93 [5.82-6.14] | 6.40 [6.19-6.69] | 6.97 [6.78-7.29] | 0.85 | 0.92 |
| decode async 1 | 5.38 [5.16-5.56] | 5.86 [5.45-6.08] | 5.73 [5.48-6.21] | 0.94 | 1.02 |
| decode async 8 | 5.55 [5.34-6.07] | 6.18 [5.72-6.42] | 5.97 [5.79-6.65] | 0.93 | 1.04 |
| decode async 64 | 6.68 [6.39-6.95] | 8.05 [7.34-8.66] | 7.81 [7.53-8.74] | 0.86 | 1.03 |
| decode async 128 | 8.57 [8.41-8.88] | 9.32 [8.34-10.27] | 8.98 [8.03-10.61] | 0.95 | 1.04 |
| prefill 64 | 37.47 [35.89-66.85] | 31.82 [27.60-35.72] | 6.72 [6.68-7.27] | 5.58 | 4.74 |
| prefill 512 | 37.92 [36.37-65.11] | 33.09 [27.69-36.10] | 9.58 [9.25-9.87] | 3.96 | 3.46 |
| mixed 8x128 | 26.49 [24.84-34.25] | 35.05 [29.41-40.14] | 342.22 [295.99-375.24] | 0.08 | 0.10 |
| KV tokens (k) | 1073 | 1074 | 1080 | | |
| bench peak over init MiB | 1536 | 1696 | 1696 | | |
| init s (engine ready; V-stock traces at first call, not at init) | 67 | 34 | 24 | | |
| model forwards at init | 506 | 174 | 7 | | |
rounds: {'default': 4, 'default-nc': 4, 'V-stock': 4} | json: out/t_dnc_default_r*.json; out/t_pdefault_r*.json; out/t_dnc_fullnc_r*.json; out/t_dnc_fullnc_full_r*.json; out/t_dnc_vstock_r*.json; out/t_vstock_r*.json

### timing

Step ms (execute_model, synced; median over steps, then median over rounds); speedup = default / arm.
| workload | eager | default | Vcpp_closed | Vcpp_open | Vpy_closed | Vpy_open | Vcpp_closed_pad | Vcpp_open_pad | Vcpp_closed x | Vcpp_open x | Vpy_closed x | Vpy_open x | Vcpp_closed_pad x | Vcpp_open_pad x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 | 32.22 | 4.66 | 4.36 | 4.29 | 4.44 | 4.67 | 4.18 | 4.38 | 1.07 | 1.09 | 1.05 | 1.00 | 1.11 | 1.06 |
| decode 5 | 36.58 | 4.91 | 4.32 | 4.56 | 4.69 | 4.82 | 4.35 | 4.47 | 1.14 | 1.08 | 1.05 | 1.02 | 1.13 | 1.10 |
| decode 8 | 31.60 | 4.76 | 4.42 | 4.59 | 4.68 | 4.78 | 4.24 | 4.62 | 1.08 | 1.04 | 1.02 | 1.00 | 1.12 | 1.03 |
| decode 64 | 32.63 | 5.25 | 5.01 | 5.04 | 5.20 | 5.28 | 4.86 | 5.00 | 1.05 | 1.04 | 1.01 | 1.00 | 1.08 | 1.05 |
| decode 100 | 37.48 | 5.87 | 5.67 | 5.57 | 6.17 | 5.96 | 5.51 | 5.68 | 1.04 | 1.05 | 0.95 | 0.99 | 1.07 | 1.03 |
| prefill 64 | 40.20 | 35.89 | 5.27 | 5.28 | 5.53 | 5.45 | 5.02 | 5.06 | 6.80 | 6.80 | 6.49 | 6.59 | 7.15 | 7.09 |
| prefill 100 | 34.15 | 34.34 | 5.26 | 5.53 | 5.74 | 5.50 | 5.35 | 5.25 | 6.53 | 6.21 | 5.99 | 6.25 | 6.42 | 6.53 |
| prefill 300 | 33.92 | 42.44 | 6.42 | 6.49 | 7.23 | 6.83 | 6.57 | 7.80 | 6.61 | 6.54 | 5.87 | 6.21 | 6.46 | 5.44 |
| prefill 512 | 33.48 | 32.51 | 8.36 | 8.04 | 8.96 | 7.96 | 8.15 | 7.53 | 3.89 | 4.04 | 3.63 | 4.08 | 3.99 | 4.32 |
| prefill 2048 | 35.18 | 36.80 | 22.44 | 22.24 | 22.44 | 22.41 | 23.72 | 21.91 | 1.64 | 1.66 | 1.64 | 1.64 | 1.55 | 1.68 |
| prefill 4x128 | 33.63 | 47.16 | 8.29 | 8.11 | 8.33 | 8.23 | 7.70 | 8.15 | 5.69 | 5.81 | 5.66 | 5.73 | 6.13 | 5.79 |
| prefill 8x128 | 38.68 | 34.84 | 12.40 | 12.49 | 12.86 | 12.69 | 12.68 | 20.30 | 2.81 | 2.79 | 2.71 | 2.75 | 2.75 | 1.72 |
| mixed 8x128 | 46.46 | 35.62 | 6.25 | 6.38 | 6.21 | 6.62 | 5.86 | 6.25 | 5.70 | 5.58 | 5.73 | 5.38 | 6.08 | 5.70 |
| mixed 32x512 | 42.81 | 36.63 | 9.18 | 9.27 | 9.27 | 9.88 | 8.78 | 9.31 | 3.99 | 3.95 | 3.95 | 3.71 | 4.17 | 3.93 |
| mixed 4x2048 | 45.74 | 44.69 | 23.45 | 23.31 | 23.57 | 23.81 | 22.86 | 23.42 | 1.91 | 1.92 | 1.90 | 1.88 | 1.95 | 1.91 |
| mixed 64x64 | 40.37 | 39.98 | 5.99 | 5.88 | 6.04 | 7.03 | 5.49 | 6.09 | 6.67 | 6.80 | 6.62 | 5.69 | 7.28 | 6.57 |
| mixed 128x16 | 41.47 | 41.35 | 6.73 | 6.76 | 7.11 | 7.91 | 6.40 | 6.88 | 6.14 | 6.12 | 5.82 | 5.23 | 6.46 | 6.01 |
rounds: {'eager': 3, 'default': 3, 'Vcpp_closed': 3, 'Vcpp_open': 3, 'Vpy_closed': 3, 'Vpy_open': 3, 'Vcpp_closed_pad': 2, 'Vcpp_open_pad': 2}

### vp

Decode step, arm VP (forward + sampler + post_update in one entry; VP_UVA stopgap: staging and token outputs via pinned UVA views).
synced = execute_model synced step ms; async = (wall(48 tok) - wall(16 tok)) / 32 under vLLM's async scheduling; pre-launch = execute_model entry -> graph launch (us, launch shim);
outside = GPU ops outside the graph per decode step (profile, bs 1 and 128 for default/V; all bs for VAf/VP).
| bs | default synced | VAf synced | V synced | VP synced | default async | VAf async | V async | VP async | default pre-launch | VAf pre-launch | V pre-launch | VP pre-launch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.49 | 4.30 | 4.29 | 4.56 | 5.79 | 5.09 | 4.98 | 4.54 | 901 | 570 | 320 | 379 |
| 8 | 4.66 | 4.34 | 4.31 | 4.58 | 5.60 | 5.33 | 5.26 | 4.82 | 928 | 692 | 377 | 403 |
| 64 | 5.05 | 4.81 | 4.97 | 5.24 | 6.75 | 6.63 | 6.46 | 6.03 | 939 | 701 | 354 | 606 |
| 128 | 5.84 | 5.62 | 5.79 | 6.06 | 8.64 | 8.30 | 8.17 | 7.99 | 952 | 830 | 536 | 547 |
- prof_default4 bs1: graph launches: 1 (542 kernels inside); outside graphs: 21 (kernels 16, memcpy 4, memset 1)
- prof_default4 bs128: graph launches: 1 (470 kernels inside); outside graphs: 21 (kernels 16, memcpy 4, memset 1)
- prof_VAf bs1: graph launches: 1 (542 kernels inside); outside graphs: 17 (kernels 14, memcpy 3, memset 0)
- prof_VAf bs8: graph launches: 1 (542 kernels inside); outside graphs: 17 (kernels 14, memcpy 3, memset 0)
- prof_VAf bs64: graph launches: 1 (470 kernels inside); outside graphs: 17 (kernels 14, memcpy 3, memset 0)
- prof_VAf bs128: graph launches: 1 (470 kernels inside); outside graphs: 17 (kernels 14, memcpy 3, memset 0)
- prof_V4 bs1: graph launches: 1 (548 kernels inside); outside graphs: 13 (kernels 10, memcpy 3, memset 0)
- prof_V4 bs128: graph launches: 1 (476 kernels inside); outside graphs: 13 (kernels 10, memcpy 3, memset 0)
- prof_VP bs1: graph launches: 1 (562 kernels inside); outside graphs: 0 (kernels 0, memcpy 0, memset 0)
- prof_VP bs8: graph launches: 1 (562 kernels inside); outside graphs: 0 (kernels 0, memcpy 0, memset 0)
- prof_VP bs64: graph launches: 1 (490 kernels inside); outside graphs: 0 (kernels 0, memcpy 0, memset 0)
- prof_VP bs128: graph launches: 1 (490 kernels inside); outside graphs: 0 (kernels 0, memcpy 0, memset 0)
Correctness: c2_cpp_VP4_Vcheck 127/127 bitwise, 108/108 sampled tokens equal to vLLM's sampler on eager logits, 0 boundaries;
e2e greedy tokens VP == V at bs 1/8/64/128 (e2e_VP4 vs e2e_V); VAf == default (e2e_VAf vs e2e_default).

### sweep

One step of exactly T tokens (nd decoding requests + prefill prompts <= 4000 tokens filling T - nd); engine max_num_batched_tokens 32768,
max_num_seqs 512 in every arm. Step ms, median of reps 3-6, median over rounds. Kernel-bound where speedup -> 1.
| T | nd | default | V cpp closed | speedup | eager |
|---|---|---|---|---|---|
| 1024 | 0 | 15.97 | 11.94 | 1.34 | nan |
| 2048 | 0 | 43.22 | 21.94 | 1.97 | 30.52 |
| 4096 | 0 | 43.73 | 41.39 | 1.06 | nan |
| 8192 | 0 | 79.27 | 81.77 | 0.97 | nan |
| 16384 | 0 | 154.10 | 160.73 | 0.96 | 161.48 |
| 32768 | 0 | 318.18 | 329.76 | 0.96 | nan |
| 1024 | 64 | 25.56 | 12.67 | 2.02 | nan |
| 2048 | 64 | 51.25 | 22.69 | 2.26 | nan |
| 4096 | 64 | 53.85 | 41.78 | 1.29 | nan |
| 8192 | 64 | 80.25 | 82.45 | 0.97 | nan |
| 16384 | 64 | 156.58 | 161.18 | 0.97 | nan |
| 32768 | 64 | 320.66 | 332.51 | 0.96 | nan |
| 1024 | 256 | 25.08 | 13.54 | 1.85 | nan |
| 2048 | 256 | 53.27 | 23.25 | 2.29 | nan |
| 4096 | 256 | 54.12 | 43.71 | 1.24 | nan |
| 8192 | 256 | 82.71 | 84.81 | 0.98 | nan |
| 16384 | 256 | 157.28 | 162.30 | 0.97 | nan |
| 32768 | 256 | 321.28 | 332.08 | 0.97 | nan |

### shrink

Fluctuating decode batch: N requests, one finishing per step (bs N..1). Mean step ms over the pass (pass 2 = settled), and pass-2 median.
| run | 64 pass1 mean | 64 pass2 mean | 64 pass2 median | 128 pass1 mean | 128 pass2 mean | 128 pass2 median | json |
|---|---|---|---|---|---|---|---|
| _Vcpp_closed_r1 | 107.15 | 7.33 | 6.02 | 52.12 | 5.93 | 6.03 | out/t_shrink_Vcpp_closed_r1.json |
| _Vcpp_closed_r2 | 122.17 | 7.47 | 6.35 | 56.94 | 6.69 | 6.73 | out/t_shrink_Vcpp_closed_r2.json |
| _Vcpp_open_r1 | 156.29 | 8.49 | 5.52 | 29.94 | 5.81 | 5.87 | out/t_shrink_Vcpp_open_r1.json |
| _Vcpp_open_r2 | 153.47 | 8.44 | 5.61 | 29.94 | 5.88 | 5.80 | out/t_shrink_Vcpp_open_r2.json |
| _Vpy_closed_r1 | 104.06 | 6.60 | 5.75 | 35.15 | 6.14 | 6.13 | out/t_shrink_Vpy_closed_r1.json |
| _Vpy_closed_r2 | 103.24 | 6.80 | 5.88 | 35.33 | 6.17 | 6.20 | out/t_shrink_Vpy_closed_r2.json |
| _default_r1 | 6.53 | 4.67 | 4.70 | 5.05 | 4.92 | 4.88 | out/t_shrink_default_r1.json |
| _default_r2 | 4.68 | 4.67 | 4.75 | 4.94 | 4.90 | 4.88 | out/t_shrink_default_r2.json |
| _eager_r1 | 31.49 | 30.43 | 30.61 | 31.94 | 31.27 | 31.35 | out/t_shrink_eager_r1.json |
| _eager_r2 | 35.51 | 33.81 | 35.05 | 36.54 | 35.14 | 35.94 | out/t_shrink_eager_r2.json |

### memory

| arm | util | KV tokens | after init MiB | reserved after init | reserved end | bench peak over init MiB | V graph growth (reserved end - after init) | default capture GiB | json |
|---|---|---|---|---|---|---|---|---|---|
| eager_r1 | 0.6 | 1080192 | 169608 | 172854 | 172856 | 1696 | 2 |  | out/t_eager_r1.json |
| eager_r2 | 0.6 | 1080192 | 169608 | 172854 | 172856 | 1696 | 2 |  | out/t_eager_r2.json |
| default_r1 | 0.6 | 1084032 | 170156 | 172706 | 173474 | 1536 | 768 | 1.00, 0.33 | out/t_default_r1.json |
| default_r2 | 0.6 | 1072528 | 168538 | 171090 | 171858 | 1536 | 768 | 1.00, 0.33 | out/t_default_r2.json |
| Vcpp_closed_r1 | 0.6 | 1080192 | 169609 | 172854 | 174332 | 1696 | 1478 |  | out/t_Vcpp_closed_r1.json |
| Vcpp_closed_r2 | 0.6 | 1080192 | 169609 | 172854 | 174332 | 1696 | 1478 |  | out/t_Vcpp_closed_r2.json |
| Vcpp_open_r1 | 0.6 | 1091712 | 171228 | 174474 | 176258 | 1696 | 1784 |  | out/t_Vcpp_open_r1.json |
| Vcpp_open_r2 | 0.6 | 1080192 | 169608 | 172854 | 174638 | 1696 | 1784 |  | out/t_Vcpp_open_r2.json |
| Vpy_closed_r1 | 0.6 | 1087712 | 170667 | 173912 | 175390 | 1696 | 1478 |  | out/t_Vpy_closed_r1.json |
| Vpy_closed_r2 | 0.6 | 1080192 | 169609 | 172854 | 174332 | 1696 | 1478 |  | out/t_Vpy_closed_r2.json |
| Vpy_open_r1 | 0.6 | 1080192 | 169608 | 172854 | 174638 | 1696 | 1784 |  | out/t_Vpy_open_r1.json |
| Vpy_open_r2 | 0.6 | 1080192 | 169608 | 172854 | 174638 | 1696 | 1784 |  | out/t_Vpy_open_r2.json |
| Vcpp_closed_pad_r1 | 0.6 | 1080192 | 169609 | 172854 | 174332 | 1696 | 1478 |  | out/t_Vcpp_closed_pad_r1.json |
| Vcpp_closed_pad_r2 | 0.6 | 1080192 | 169609 | 172854 | 174332 | 1696 | 1478 |  | out/t_Vcpp_closed_pad_r2.json |
| Vcpp_open_pad_r1 | 0.6 | 1080192 | 169608 | 172854 | 174642 | 1696 | 1788 |  | out/t_Vcpp_open_pad_r1.json |
| Vcpp_open_pad_r2 | 0.6 | 1080192 | 169608 | 172854 | 174642 | 1696 | 1788 |  | out/t_Vcpp_open_pad_r2.json |

### startup

| run | imports s | LLM() s (after imports) | model load s | compile s | capture s (GiB) | V install s | V warm-up s | ready - imports s | warm-up traces / harvests / restored | probes first-call s (traces, harvests) | json |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Vcpp_closed_prewarm | 321.7 | 169.4 | 64.5 | - | - | 5.0 | 22.0 | 196.4 | (8, 132, 0) | decode5 0.20 (0,24); decode48 1.00 (0,24); decode200 2.18 (0,24); prefill300 0.09 (0,12); prefill3000 0.12 (0,12); prefill4x700 0.12 (0,12); mixed16x700 0.27 (0,25); mixed64x40 0.19 (0,14) | out/s_Vcpp_closed_prewarm.json |
| Vcpp_closed_r1 | 127.7 | 94.9 | 43.7 | - | - | 1.2 | 37.1 | 133.2 | (8, 132, 0) | decode5 0.18 (0,24); decode48 0.82 (0,24); decode200 1.43 (0,24); prefill300 0.09 (0,12); prefill3000 0.11 (0,12); prefill4x700 2.31 (0,12); mixed16x700 2.79 (0,25); mixed64x40 0.18 (0,14) | out/s_Vcpp_closed_r1.json |
| Vcpp_closed_r2 | 46.4 | 59.5 | 22.4 | - | - | 0.5 | 15.0 | 75.0 | (8, 132, 0) | decode5 0.18 (0,24); decode48 0.64 (0,24); decode200 1.37 (0,24); prefill300 0.08 (0,12); prefill3000 0.11 (0,12); prefill4x700 0.11 (0,12); mixed16x700 0.23 (0,25); mixed64x40 0.17 (0,14) | out/s_Vcpp_closed_r2.json |
| Vcpp_closed_r3 | 13.7 | 21.2 | 8.7 | - | - | 0.4 | 16.0 | 37.6 | (8, 132, 0) | decode5 0.17 (0,24); decode48 0.81 (0,24); decode200 1.57 (0,24); prefill300 0.08 (0,12); prefill3000 0.11 (0,12); prefill4x700 0.11 (0,12); mixed16x700 0.21 (0,25); mixed64x40 0.17 (0,14) | out/s_Vcpp_closed_r3.json |
| Vcpp_closed_warm_r1 | 55.1 | 33.9 | 8.9 | - | - | 0.5 | 14.8 | 49.3 | (8, 0, 132) | decode5 0.10 (0,0); decode48 0.10 (0,0); decode200 0.17 (0,0); prefill300 0.09 (0,0); prefill3000 0.08 (0,0); prefill4x700 0.07 (0,0); mixed16x700 0.18 (0,0); mixed64x40 0.12 (0,0) | out/s_Vcpp_closed_warm_r1.json |
| Vcpp_closed_warm_r2 | 33.2 | 21.6 | 8.0 | - | - | 0.3 | 17.7 | 39.5 | (8, 0, 132) | decode5 0.10 (0,0); decode48 0.12 (0,0); decode200 0.20 (0,0); prefill300 0.05 (0,0); prefill3000 0.10 (0,0); prefill4x700 0.63 (0,0); mixed16x700 0.28 (0,0); mixed64x40 0.23 (0,0) | out/s_Vcpp_closed_warm_r2.json |
| Vcpp_closed_warm_r3 | 11.7 | 18.0 | 7.4 | - | - | 0.3 | 14.2 | 32.5 | (8, 0, 132) | decode5 0.10 (0,0); decode48 0.11 (0,0); decode200 0.18 (0,0); prefill300 0.05 (0,0); prefill3000 0.07 (0,0); prefill4x700 0.07 (0,0); mixed16x700 0.16 (0,0); mixed64x40 0.12 (0,0) | out/s_Vcpp_closed_warm_r3.json |
| Vcpp_open_prewarm | 368.5 | 58.2 | 12.6 | - | - | 1.9 | 22.3 | 82.3 | (8, 44, 0) | decode5 1.75 (0,8); decode48 0.68 (0,8); decode200 1.28 (0,8); prefill300 0.10 (0,4); prefill3000 0.07 (0,4); prefill4x700 0.07 (0,4); mixed16x700 0.38 (0,8); mixed64x40 0.15 (0,4) | out/s_Vcpp_open_prewarm.json |
| Vcpp_open_r1 | 350.8 | 117.1 | 36.3 | - | - | 2.9 | 22.5 | 142.5 | (8, 44, 0) | decode5 1.71 (0,8); decode48 0.78 (0,8); decode200 1.72 (0,8); prefill300 0.14 (0,4); prefill3000 0.13 (0,4); prefill4x700 0.07 (0,4); mixed16x700 0.42 (0,8); mixed64x40 0.18 (0,4) | out/s_Vcpp_open_r1.json |
| Vcpp_open_r2 | 46.7 | 44.8 | 12.6 | - | - | 0.2 | 18.0 | 63.0 | (8, 44, 0) | decode5 1.60 (0,8); decode48 1.05 (0,8); decode200 1.79 (0,8); prefill300 0.11 (0,4); prefill3000 0.07 (0,4); prefill4x700 0.07 (0,4); mixed16x700 0.41 (0,8); mixed64x40 0.17 (0,4) | out/s_Vcpp_open_r2.json |
| Vcpp_open_r3 | 13.0 | 19.2 | 6.9 | - | - | 0.2 | 16.8 | 36.1 | (8, 44, 0) | decode5 1.58 (0,8); decode48 0.99 (0,8); decode200 1.53 (0,8); prefill300 0.10 (0,4); prefill3000 0.07 (0,4); prefill4x700 0.07 (0,4); mixed16x700 0.39 (0,8); mixed64x40 0.17 (0,4) | out/s_Vcpp_open_r3.json |
| Vcpp_open_warm_r1 | 96.8 | 46.1 | 23.1 | - | - | 0.2 | 17.2 | 63.5 | (8, 0, 44) | decode5 1.58 (0,0); decode48 0.95 (0,0); decode200 1.79 (0,0); prefill300 0.09 (0,0); prefill3000 0.06 (0,0); prefill4x700 0.06 (0,0); mixed16x700 0.41 (0,0); mixed64x40 0.15 (0,0) | out/s_Vcpp_open_warm_r1.json |
| Vcpp_open_warm_r2 | 102.6 | 521.3 | 88.9 | - | - | 0.9 | 22.4 | 544.7 | (8, 0, 44) | decode5 2.20 (0,0); decode48 1.50 (0,0); decode200 2.72 (0,0); prefill300 0.15 (0,0); prefill3000 0.08 (0,0); prefill4x700 0.08 (0,0); mixed16x700 0.61 (0,0); mixed64x40 0.21 (0,0) | out/s_Vcpp_open_warm_r2.json |
| Vcpp_open_warm_r3 | 12.2 | 19.1 | 7.1 | - | - | 0.1 | 16.6 | 35.8 | (8, 0, 44) | decode5 1.41 (0,0); decode48 0.79 (0,0); decode200 1.43 (0,0); prefill300 0.09 (0,0); prefill3000 0.06 (0,0); prefill4x700 0.06 (0,0); mixed16x700 0.40 (0,0); mixed64x40 0.13 (0,0) | out/s_Vcpp_open_warm_r3.json |
| Vpy_closed_prewarm | 325.5 | 63.6 | 10.0 | - | - | 0.4 | 20.8 | 84.8 | (8, 132, 0) | decode5 0.16 (0,24); decode48 0.67 (0,24); decode200 1.46 (0,24); prefill300 0.08 (0,12); prefill3000 0.11 (0,12); prefill4x700 0.11 (0,12); mixed16x700 0.22 (0,25); mixed64x40 0.16 (0,14) | out/s_Vpy_closed_prewarm.json |
| Vpy_closed_r1 | 246.6 | 89.0 | 21.1 | - | - | 0.7 | 18.1 | 107.8 | (8, 132, 0) | decode5 0.18 (0,24); decode48 0.76 (0,24); decode200 1.68 (0,24); prefill300 0.09 (0,12); prefill3000 0.11 (0,12); prefill4x700 0.12 (0,12); mixed16x700 0.23 (0,25); mixed64x40 0.16 (0,14) | out/s_Vpy_closed_r1.json |
| Vpy_closed_r2 | 74.1 | 76.2 | 26.6 | - | - | 0.8 | 17.0 | 94.0 | (8, 132, 0) | decode5 0.18 (0,24); decode48 0.87 (0,24); decode200 2.00 (0,24); prefill300 0.09 (0,12); prefill3000 0.11 (0,12); prefill4x700 0.11 (0,12); mixed16x700 0.24 (0,25); mixed64x40 0.17 (0,14) | out/s_Vpy_closed_r2.json |
| Vpy_closed_r3 | 18.8 | 27.4 | 7.6 | - | - | 0.3 | 14.8 | 42.4 | (8, 132, 0) | decode5 0.21 (0,24); decode48 0.60 (0,24); decode200 1.27 (0,24); prefill300 0.11 (0,12); prefill3000 0.14 (0,12); prefill4x700 0.14 (0,12); mixed16x700 0.24 (0,25); mixed64x40 0.16 (0,14) | out/s_Vpy_closed_r3.json |
| default_prewarm | 342.2 | 231.1 | 79.9 | 2.47 | 56 (1.0) + 20 (0.33) | - | - | 231.1 | - | decode5 0.05; decode48 0.04; decode200 0.11; prefill300 0.03; prefill3000 0.07; prefill4x700 0.07; mixed16x700 0.08; mixed64x40 0.09 | out/s_default_prewarm.json |
| default_r1 | 475.7 | 173.9 | 34.4 | 0.78 | 45 (1.0) + 22 (0.33) | - | - | 173.9 | - | decode5 0.05; decode48 0.04; decode200 0.11; prefill300 0.02; prefill3000 0.06; prefill4x700 0.07; mixed16x700 0.08; mixed64x40 0.08 | out/s_default_r1.json |
| default_r2 | 117.8 | 103.0 | 10.5 | 2.16 | 18 (1.0) + 19 (0.33) | - | - | 103.0 | - | decode5 0.05; decode48 0.04; decode200 0.10; prefill300 0.02; prefill3000 0.06; prefill4x700 0.06; mixed16x700 0.07; mixed64x40 0.08 | out/s_default_r2.json |
| default_r3 | 13.3 | 57.9 | 8.2 | 0.34 | 18 (1.0) + 18 (0.33) | - | - | 57.9 | - | decode5 0.05; decode48 0.04; decode200 0.11; prefill300 0.02; prefill3000 0.06; prefill4x700 0.06; mixed16x700 0.08; mixed64x40 0.09 | out/s_default_r3.json |
| eager_prewarm | 247.4 | 107.0 | 44.3 | - | - | - | - | 107.0 | - | decode5 0.14; decode48 0.13; decode200 0.18; prefill300 0.05; prefill3000 0.05; prefill4x700 0.05; mixed16x700 0.18; mixed64x40 0.19 | out/s_eager_prewarm.json |
| eager_r1 | 387.9 | 181.7 | 68.4 | - | - | - | - | 181.7 | - | decode5 0.12; decode48 0.13; decode200 0.16; prefill300 0.04; prefill3000 0.04; prefill4x700 0.05; mixed16x700 0.16; mixed64x40 0.18 | out/s_eager_r1.json |
| eager_r2 | 95.7 | 119.4 | 71.4 | - | - | - | - | 119.4 | - | decode5 0.11; decode48 0.12; decode200 0.15; prefill300 0.04; prefill3000 0.04; prefill4x700 0.05; mixed16x700 0.23; mixed64x40 0.16 | out/s_eager_r2.json |
| eager_r3 | 12.2 | 20.5 | 8.3 | - | - | - | - | 20.5 | - | decode5 0.13; decode48 0.13; decode200 0.17; prefill300 0.04; prefill3000 0.05; prefill4x700 0.05; mixed16x700 0.18; mixed64x40 0.18 | out/s_eager_r3.json |

### pad

Step ms, ARMV_PAD=0 (exact sizes) vs 1 (vLLM-default padding); interleaved rounds. pad/nopad > 1 = padding costs.
| workload | Vcpp_closed_ab | Vcpp_closed_pad | Vcpp_open_ab | Vcpp_open_pad | closed pad/nopad | open pad/nopad |
|---|---|---|---|---|---|---|
| decode 1 | 4.22 | 4.18 | 4.25 | 4.38 | 0.99 | 1.03 |
| decode 5 | 4.42 | 4.35 | 4.38 | 4.47 | 0.98 | 1.02 |
| decode 8 | 4.30 | 4.24 | 4.53 | 4.62 | 0.99 | 1.02 |
| decode 64 | 4.99 | 4.86 | 4.93 | 5.00 | 0.97 | 1.01 |
| decode 100 | 5.61 | 5.51 | 5.54 | 5.68 | 0.98 | 1.02 |
| prefill 64 | 5.25 | 5.02 | 8.44 | 5.06 | 0.96 | 0.60 |
| prefill 100 | 5.15 | 5.35 | 5.11 | 5.25 | 1.04 | 1.03 |
| prefill 300 | 6.36 | 6.57 | 6.33 | 7.80 | 1.03 | 1.23 |
| prefill 512 | 7.79 | 8.15 | 8.00 | 7.53 | 1.05 | 0.94 |
| prefill 2048 | 21.93 | 23.72 | 22.30 | 21.91 | 1.08 | 0.98 |
| prefill 4x128 | 7.84 | 7.70 | 7.72 | 8.15 | 0.98 | 1.06 |
| prefill 8x128 | 11.93 | 12.68 | 12.10 | 20.30 | 1.06 | 1.68 |
| mixed 8x128 | 6.21 | 5.86 | 6.19 | 6.25 | 0.94 | 1.01 |
| mixed 32x512 | 8.83 | 8.78 | 8.99 | 9.31 | 0.99 | 1.04 |
| mixed 4x2048 | 23.03 | 22.86 | 23.17 | 23.42 | 0.99 | 1.01 |
| mixed 64x64 | 5.77 | 5.49 | 5.85 | 6.09 | 0.95 | 1.04 |
| mixed 128x16 | 6.30 | 6.40 | 6.57 | 6.88 | 1.02 | 1.05 |
- t_Vcpp_closed_pad_r1.json: padded rows {'tokens': 488, 'reqs': 416}, kernels per serving variant {'prefill/T2/fam': [442, 442, 442], 'decode/T1/fam': [547, 547], 'decode/T0/fam': [547], 'decode/T2/fam': [475, 475, 439], 'mixed/T2/fam': [514]}
- t_Vcpp_open_pad_r1.json: padded rows {'tokens': 488, 'reqs': 416}, kernels per serving variant {'prefill/T2/fam': [442, 442, 442], 'decode/T1/fam': [547, 547, 547, 475], 'decode/T0/fam': [547], 'decode/T2/fam': [475, 475], 'mixed/T2/fam': [514, 478, 478]}

### notes

- Padded-mode correctness (out/c2_cpp_closed_pad_Vcheck.json, ARMV_PAD=1, real rows only). 281/297 are bitwise against V's
  reference eager, which uses vLLM's metadata with the max_seq_len=max_model_len policy. The other 16 are decode steps with
  9 or 5 real rows right after a mixed step (max_abs 0.5-0.75). All 16 are bitwise against stock vLLM eager
  (bitwise_vs_stock): in those padded decode steps the replay used vLLM's own max_seq_len, not the pinned policy. So every
  forward matches one valid eager reference. The pin is skipped only in padded mode; unpadded V is 228/228.
- Startup, open variant: with 0 traces and 0 harvests (the warm load), decode first calls still take 1.6-2.2 s at bs 5 and
  ~1.8 s at bs 200, against 0.10-0.20 s for closed-warm. That is per-shape work in the trtllm fork's path, not ours to cache yet.
- KV capacity at util 0.6 moves by +-11k tokens between runs of the same arm (1,072,528-1,091,712), so compare it within a round.
  In round 1, default had 1,084,032 and eager/V had 1,080,192: default got 3,840 more tokens, not fewer. The overlay's
  "~7,664 more for V" does not reproduce here.
- default prefill (T 64-512) runs ~31-42 ms in both rounds. The forward itself is ~30 ms (the fwd field), the same as the
  overlay's s5 (33 ms) and eager (35-58 ms). Its prefill is host-bound by the per-layer work between piecewise graphs; V
  replays the whole step.
- Mixed spec names are NDxPL: ND decoding requests plus one PL-token prompt added every 2 steps. So "32x512" steps
  carry 32 + 512 = 544 tokens (shape [33, 544]), not 16k. Its 4.3x sits between prefill 512 (3.8x) and the small
  mixed rows. "4x2048" steps carry 4 + 2048 = 2052 tokens (1.9x, next to prefill 2048 at 1.6x). The tokens-per-step
  sweep (t_sweep_*) covers 1k-32k tokens per step with 0/64/256 decoders.
- Open variant (traced impls + trtllm fork) on candidate 2: NaN outputs on decode at bs 4-8 (c2_cpp_open_Vcheck 193/228;
  c2_cpp_open_range 960/969, all 9 failures decode bs 4/5). Fork only (diag_fork_only): 92/92 bitwise. The ports
  (ARMV_TRACED_IMPLS=1), or the ports combined with the fork, produce the NaNs; the impls-only and both diag runs are pending.
  Open decode timing rows are therefore not valid timings of a correct run.
- GPU 0 contention, 10-07 14:13-14:56: a foreign job (pid 2587156, 50-67 GB) shared GPU 0 during t_Vcpp_open_r3 and
  t_Vpy_closed_r3. Those runs are quarantined as *.json.contended and re-queued (timing_r3b_c2.sh). Since then
  gpu0_job.sh waits for GPU 0 to have no compute process before it takes the lock, and a run that still finds one is
  skipped (SKIPPED_CONTENDED in logs/timing_idle.log).
- default-nc (10-08, pinned build, t_dnc_* / e2e_p_*; table in "stock"). vLLM with compile off and its own full graphs
  (compilation_config mode 0, cudagraph_mode FULL_DECODE_ONLY, the recipe's dgx_station_gb300 override). Requesting `FULL`
  is not accepted: vLLM logs "CUDAGraphMode.FULL is not supported with FlashInferBackend backend (support:
  AttentionCGSupport.UNIFORM_BATCH); setting cudagraph_mode=FULL_DECODE_ONLY", so the FULL runs are pooled as extra rounds.
  Correctness: its greedy tokens equal template V's at every bs and differ from eager at bs 8 (3/8 sequences, first divergence at token
  1, 4 and 19; the same in both repeats). Its captured decode graphs use the same fixed metadata policy (max_seq_len) that template V
  pinned. V-stock equals eager at every bs. default-nc runs the same kernels as eager and V-stock (stock _C ops, no Inductor), so
  default-nc vs V-stock is the same-kernel, both-graphed comparison. Decode synced: V-stock 0.91-0.99x of default-nc (bs 1: +0.32 ms;
  default-nc is itself +0.32 ms over default at bs 1, which is the Inductor-fusion GPU gap seen in PROFILE_decode). Decode async: 1.02-1.04x.
  Prefill: default-nc runs eager (28-36 ms); V-stock is 3.5-4.7x faster. Mixed 8x128: V-stock 0.10x, caused by the fork's max_kv pin for
  context calls (CORE_TASKS), not by the graphs. The host was CPU-loaded (load avg 83-155) and round-to-round spread is up to 20%
  (min-max in the table). Read decode differences under ~10% as noise.
- default's mixed 8x128 step (about 27 ms, V-stock 7.1 ms post-fix) is real host time, not a harness artifact (bench/drive_prof.py,
  out/pm_{default,vstock,fullnc}.json, traces out/prof_mixed/, analysis prof_mixed_an.py; attn line, 10-08 20:05, load avg 123-174).
  - Request mix: identical in all three arms for every measured step: 8 decoders at KV 259..273 (1 token each) + one 128-token prompt
    from KV 0 = 136 tokens, 9 requests. No Dynamo/Inductor counter moved during the measured steps (no recompiles or guard failures)
    and there was no first-use JIT. Scheduling (chunked prefill, 16384-token budget) is the same.
  - Dispatch: default ran cudagraph mode PIECEWISE at 136 tokens; 136 is itself a capture size (max capture 1024), so there is no
    padding. FULL is not available for a mixed batch: FULL_AND_PIECEWISE gives FULL only to uniform decode. V-stock and default-nc
    dispatch NONE (enforce_eager / compile off for non-uniform batches) at 9 requests.
  - GPU kernel time per step is about 5 ms in every arm: default 5.03 ms (Inductor fusions), default-nc 5.30, V-stock 5.44. All three
    run the same cuBLAS (nvjet) GEMMs and the same trtllm-gen context and decode kernels. That matches the coordinator's decode-cost
    estimate.
  - Where default's 27 ms goes: host. A PIECEWISE step is 37 graph replays with the 36 attention layers run eagerly between them.
    Under the profiler the forward was 45 ms (the profiler inflates Python about 1.7x). Of that, vllm::unified_attention_with_output
    took 34 ms (36 calls, about 0.95 ms each). Inside it:
    - FlashInfer's Python wrappers for the two trtllm calls per layer (decode + context: 72 per step);
    - an aten::zeros per call (73 per step) for the multi-CTA KV counter buffer
      (flashinfer/utils.py _resolve_trtllm_gen_multi_ctas_kv_counter_buffer);
    - the api_logging wrapper, which reads os.environ on every call for FLASHINFER_TRACE_DUMP;
    - vLLM's metadata split into decode and prefill parts.
    unified_kv_cache_update took 4.8 ms, and the 37 graph replays about 2 ms.
    Unprofiled, that is about 22 ms of host on top of about 5 ms of GPU. V-stock replays the whole step as one graph (1 launch,
    28 kernel launches outside it), so it is about GPU time plus launch.
- Scope: this is the cost of vLLM's PIECEWISE path at small mixed steps. At 4K-8K-token chunks the GPU time (55-105 ms) covers the
  same host work, which is why the g3 InferenceX rows show parity.
- The decode bs 64/128 gap at long KV (V-stock about 1.0-1.4 ms slower than default-nc) comes from host time in the replay entry
  before the graph launch. It is not a kernel or dispatch difference (bench/drive_ix3.py, prof_dec128_gpu0_c2.sh; attn line, decode
  bs 128 at KV 6056-8021; traces out/prof_dec128/, prof_dec_cmp.py, prof_dec_host.py). Unprofiled step in that job: V-stock 24.29 ms,
  default-nc 23.25 ms.
  - Same GPU work: per step GPU busy is 22.2 ms (V-stock) and 22.1 ms (default-nc), with 530 vs 531 kernels (the extra one is one
    memcpy). The kernel names are identical: the trtllm-gen decode kernel is the same Persistent (not MultiCtasKv) variant at about
    18.0 ms per step in both arms, and every GEMM picks the same cuBLAS nvjet kernel (e.g. nvjet_sm103_tst_176x128_64x5_4x1_v_bz_TNN,
    0.86 ms). Per-kernel differences are within 0.03 ms per kernel type.
  - The difference is the host time from the model forward call to cudaGraphLaunch, profiled: V-stock 782 us vs default-nc 106 us
    (run_fullgraph -> replay). Of V-stock's 782 us:
    - about 150 us is the adapter's Python key/form checks (_stock_form, _form_ok);
    - 108 us is 36 cudaGraphExecMemsetNodeSetParams calls per replay;
    - the remaining ~520 us is the C++ entry (argument binding, guard evaluation, node parameter updates), which the profiler does not
      break down.
    The GPU sits idle for that time: the GPU span grows by about 2 ms profiled while busy time is unchanged. execute_model ->
    forward also measured +0.56 ms profiled, but the V-stock run started at load 230 vs 103 and vLLM's scheduler and prepare_inputs
    (the same code in both arms) were slower by the same proportion, so that part is host load, not V-stock.
  - Implication: at bs 128 the per-replay entry cost is about 0.7-0.8 ms of host before launch (about 3% of a 23 ms step). Async
    scheduling hides part of it in serving.
- Pre-launch host time after the integration fixes (decode bs 128, KV 6056-8021; forward call -> cudaGraphLaunch, profiled):
  | V-stock run | forward -> launch | breakdown | unprofiled step |
  |---|---|---|---|
  | original adapter, attn line (prof_dec128) | 770-799 us | adapter checks ~150 us, 36 memset updates 108 us, C++ entry ~520 us | 24.29 ms |
  | armV_fast, attn line (prof_dec128c) | 692-871 us | adapter ~45 us, memset updates 109 us, C++ entry ~550 us | 24.52 ms |
  | armV_fast, attn+lc (prof_dec128b) | 458-490 us | adapter ~45 us, memset updates 0 (L1c), C++ entry ~420 us | 23.89 ms |
  default-nc's vLLM replay takes 29-41 us (unprofiled steps 23.25 / 23.37 ms). GPU busy is the same in every run (14.8 ms per
  annotated window). Both fixes are bitwise: c2_fast_attn_vstock_Vcheck and c2_fast_attnlc_vstock_Vcheck are 188/188 with the same
  traces, variants and redispatches as before.
- Capture limit under vLLM defaults and under InferenceX's batch rule (vLLM 0.29 source, land/scratch/vllm/site/vllm):
  - config/vllm.py:2003-2013 (no explicit max_cudagraph_capture_size):
    ```
    default_max_graph_size = (
        1024 if current_platform.is_device_capability_family(100) else 512
    )
    decode_query_len = self.uniform_decode_query_len
    max_num_seqs = self.scheduler_config.max_num_seqs
    max_cudagraph_capture_size = min(
        max_num_seqs * decode_query_len * 2, default_max_graph_size
    )
    ```
    The default capture list (config/vllm.py:2120-2132) is [1, 2, 4] + range(8, min(max+1, 256), 8) + range(256, max+1, 16).
  - Runtime dispatch, v1/worker/gpu/cudagraph_utils.py. _init_candidates (:306-311) maps every token count 0..(largest captured
    size) to the next captured size, for FULL and PIECEWISE. dispatch (:419-436) returns cg_mode=CUDAGraphMode.NONE for any
    (num_tokens, loras) with no candidate, i.e. above the largest captured size.
  - Our harness (vLLM defaults on GB300, sm_103 = family 100; max_num_seqs 1024, no spec decode): max capture 1024. So a mixed step
    of 1088+ tokens runs NONE: torch.compile'd code, no CUDA graph.
  - InferenceX's per-point rule, reproduced by serve/ix/ix_run.py: `args["max-num-seqs"] = max(conc, 16)`. That gives max capture
    = 2 x max(conc, 16): 32 tokens at conc 1-16, 128 at conc 64, 256 at conc 128. In InferenceX's setup every step that carries a
    prefill chunk larger than that (any real 1k or 8k prompt chunk) runs without CUDA graphs: compiled, piecewise graphs unused.
    Only decode steps (one token per request) stay graphed. The InferenceX vLLM speed-bench script's
    --max-cudagraph-capture-size 512 caps it at 512 as well.
- The g4 1024-token chunk + 64 decodes row, dispatch log and profile (bench/drive_ix4.py, prof_ix1k_gpu0_c2 on GPU 3 under the weekend
  rule; out/px_{default,fullnc,vstock}.json, trace out/prof_ix1k/):
  - Every measured step is 1088 tokens (65 requests: 64 decoders at KV 1032-2060 plus the 1024-token chunk). default dispatched
    cg_mode NONE at 1088 tokens (max capture 1024, no padding), with no Dynamo/Inductor counter change. default-nc and V-stock also
    dispatch NONE (V-stock replays its own graph inside the forward).
  - default's step has 13.9 ms of GPU kernels: the same nvjet GEMMs and trtllm-gen context + decode kernels, plus Inductor's fused
    Triton kernels. The step is host-bound. Under the profiler the forward is 109 ms:
    - 37 compiled pieces (piecewise_backend -> standalone_compile -> AOT runtime wrapper -> CompiledFxGraph), about 64 ms. Inside:
      181 Triton launches through the static launcher (18 ms), 144 mm calls (9 ms), a torch.cuda.set_device per piece (6.7 ms),
      empty_strided / reinterpret_tensor;
    - 36 eager attention calls between the pieces, about 36 ms. As in the small mixed step: FlashInfer's Python wrappers, an
      aten::zeros per call for the multi-CTA counter, and the api_logging os.environ read.
    Unprofiled (g4 r1): default 60.4 ms, default-nc (eager, vLLM _C kernels, no compiled wrappers) 33.3 ms, V-stock 16.9 ms. At this
    size, compiled code without graphs costs more host time per layer than eager does, and the GPU work (about 14 ms) is far below
    either.
