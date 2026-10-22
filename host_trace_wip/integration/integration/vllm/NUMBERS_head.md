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
