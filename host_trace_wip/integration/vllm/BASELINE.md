# vLLM baseline on GB300 (no host trace): Qwen3-8B, TP=1

## Setup (2026-10-03)

- **Software:** vLLM v0.29.0, built from source into `vllm/site`. torch is 2.15.0a0+gita0fd720 (`install_land_int5b_frozen` via `sglang/src_sgl1`), launched through `python_vllm.sh`. FlashInfer 0.6.18.
- **Hardware:** GPU 2 (GB300, sm_103), bf16, held under `gpu2.lock` for each run, `taskset -c 72-107`. The host is shared with other agents.
- **Engine:** `LLM(..., attention_backend="FLASHINFER", enable_prefix_caching=False, gpu_memory_utilization=0.6, max_model_len=4096)`, in process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), Model Runner V2 (v0.29's default).
- **Attention:** trtllm-gen in both arms (log: `decode_backend=trtllm-gen`, "Using TRTLLM prefill attention").
- **Driver:** `bench/drive.py`, mirroring `sglang/bench/drive.py`.
  - Decode: bs requests with 256-token prompts, 24 decode steps, steps[4:] (n=20), context about 256.
  - Prefill: one request of T tokens, 7 reps, reps[2:] (n=5).
  - **step** = sync-bracketed V2 `GPUModelRunner.execute_model`: input prep, metadata and forward, but not sampling. This is the closest match to SGLang's `ModelRunner.forward`.
  - **fwd** = the forward alone (`Qwen3ForCausalLM.forward`, `run_pw_graph` or `run_fullgraph`), also sync-bracketed.
- **Arms:**
  - eager: `enforce_eager=True` (mode NONE, `_C` custom ops).
  - default: FULL_AND_PIECEWISE with VLLM_COMPILE (Inductor) and `custom_ops=none`. Capture sizes run 1..1024: FULL for uniform decode, PIECEWISE for the rest, split at `unified_attention_with_output` and `unified_kv_cache_update` (72 splits).

## Step time, ms (run 2; median, with min in parentheses)

| case | eager | default | default speedup vs eager |
|---|---|---|---|
| decode bs=1 | 33.1 (28.8) | 4.67 (4.62) | 7.1x |
| decode bs=8 | 32.7 (29.0) | 4.69 (4.67) | 7.0x |
| decode bs=64 | 32.9 (29.3) | 5.11 (5.06) | 6.4x |
| prefill T=64 | 33.8 (29.4) | 19.0 (18.8) | 1.8x |
| prefill T=512 | 37.6 (35.5) | 19.4 (18.8) | 1.9x |
| prefill T=2048 | 32.7 (30.3) | 55.3 (54.7) | 0.59x |

Forward-only ms (run 2):

| case | eager | default |
|---|---|---|
| decode bs=1 / 8 / 64 | 32.1 / 31.7 / 31.7 | 3.67 / 3.71 / 4.10 |
| prefill 64 / 512 / 2048 | 31.8 / 35.5 / 30.2 | 17.0 / 17.4 / 53.0 |

Run 1 (`bench/out/run1_maxseqs128`) had `max_num_seqs=128`, which caps capture at 256. Its step ms:
- eager: decode 27.8 / 28.8 / 34.3; prefill 30.5 / 34.5 / 34.2.
- default: decode 4.50 / 4.55 / 4.97; prefill 18.1 / 51.6 (uncaptured) / 52.5 (uncaptured).

Eager rows move by up to about 20% between runs; they are host-bound on a shared host. Default decode moves by about 3%.

**Memory:** the default arm's graphs cost 1.00 + 0.33 GiB at capture. At a fixed `gpu_memory_utilization` that leaves 1,072,256 KV tokens vs 1,080,192 for eager (-7,936 tokens, about 1.1 GiB). Init takes 66.5 s (torch.compile 20 s, capture 30 s) vs 15.6 s for eager.

## Observations

- **Eager is host-bound** at about 28-34 ms per step regardless of batch size or T up to 2048. This matches SGLang arm D on the same torch build (26 ms pinned, 35-46 ms unpinned; `sglang/BASELINE.md`).
- **Default decode is close to SGLang** (4.67 vs 3.8-3.9 ms at bs=1). The 1 ms step-minus-fwd gap is V2's input prep and FlashInfer metadata build.
- **Default prefill at T <= 1024 is PIECEWISE-captured** but still about 19 ms flat: piece replays interleaved with 72 eager split ops (attention and KV write) per step. SGLang's BCG prefill is about 10 ms over the same span.
- **Default prefill at T=2048 is above the 1024 capture limit** and runs the Inductor pieces with no graph. It is about 20 ms slower than eager. The same happened at T=512 in run 1, with capture capped at 256. Not investigated; likely the per-piece host cost of the compiled wrapper on this host.

## Why default prefill is 19 ms at T=64 (Phase 4 probe, `armV/probe_default.py`, log `armV/out/probe_default.log`)

Default arm, GPU 2. Timers sit on `FlashInferImpl.forward`, `do_kv_cache_update` and `CUDAGraphWrapper.__call__`, plus a Kineto table. Per forward (n=5):

| T | forward wall | forward GPU | eager attention host (36 calls) | eager KV write host (36) | piece replays |
|---|---|---|---|---|---|
| 64 | 22.4 ms | 20.8 ms | 12.6 ms | 2.7 ms | 37, all replays |
| 512 | 22.0 ms | 21.9 ms | 13.3 ms | 3.5 ms | 37, all replays |

- There is no recompile and no eager fallback for the model. Every step is one `run_pw_graph`: 37 piece-graph replays (the breakable-graph path is off), with the 72 split ops run eagerly between them.
- The cost is the host time of those eager split ops. The GPU waits on the host: GPU time from the first piece to the last roughly equals host time.
  - Attention takes about 350 us of host time per layer. Kineto attributes 466 us per call to `vllm::unified_attention_with_output`, which covers FlashInfer's trtllm context Python wrapper, the TVM-FFI call, and an `aten::zeros` counter buffer per call (37 memsets).
  - The KV write takes about 75-100 us per layer (`unified_kv_cache_update` -> `reshape_and_cache_flash`).
  - The 37 `cudaGraphLaunch` calls add about 1.1 ms.
- The GPU work itself is about 4-6 ms (Kineto self CUDA total: 4.1 ms at T=64, 6.4 ms at T=512). That is why default prefill is flat in T up to 1024.
- At T=2048, above `max_cudagraph_capture_size` 1024, the same split ops run between the Inductor pieces with no graph at all, which gives 55-65 ms.
