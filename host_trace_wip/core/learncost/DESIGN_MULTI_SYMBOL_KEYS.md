# Keys that still learn while serving: trtllm context / decode (design, no code)

Learncost lane, 2026-10-12. The user's framing: "or offline, or warmup, capture individual kernels idk". These are the
routes for the keys that still learn once T-only keys are warmed. A fourth route, the TRT-LLM C++ launcher (READY in
land/core/trtllm_cpp), is compared too.

## What still learns (vLLM Qwen3-8B, GB300, conc 16, 48 requests, 6.5-8K prompts, budget 16384)
- After a warm-up of the mixed entry over T in [32, 16384] (out/wu_v_warm.json), the T-only keys never learn again:
  mm, rms_norm, rotary, reshape_and_cache, fused_add_rms_norm, silu.
- Still learning: 33 trtllm_context keys over 33 new mixed steps (one per step), and 4 trtllm_decode keys.
- trtllm_context's key (operand sizes plus scalars) depends on:
  - P = prefill tokens = T - decode tokens (query rows);
  - np = number of prefills (block table rows [np, 640], seq_lens [np], cu_seqlens [np + 1] x2);
  - max_q_len, the longest prefill chunk, which sets the grid's CTA count over Q.
  - max_kv is pinned to max_model_len by ARMV_PREFILL_MAX_KV=model.
  - It does not depend on the decode count, except through P.
- trtllm_decode's key depends only on nd (number of decodes). max_seq_len is pinned by ARMV_DECODE_MAX_SEQ=model.
- Mixed-step mix in this workload: np = 1 in 25 of 35 steps, np = 2/3/4 in 5/4/1. All 35 mixed steps had distinct
  (P, np), so every step is a new context key.

## Size of the declared space

| keys | symbols | distinct keys |
|---|---|---|
| decode | nd in [1, max_num_seqs] | <= max_num_seqs (64) |
| context, np = 1 | P in [1, budget]; max_q_len = P | <= budget (16384) |
| context, np = 2 | P in [2, budget], max_q_len in [ceil(P/2), P-1] | ~budget^2 / 4, about 6.7e7 |
| context, np >= 3 | adds the next-longest chunk only through P | same order as np = 2, for each np |

- At 4.1 ms per learn, np = 1 plus decode is about 67 s cold. That is tractable, and close to the T-only warm-up's
  cost per key.
- np >= 2 is not tractable to enumerate: 6.7e7 keys is about 76 hours per np value. So warm-up cannot reach zero
  runtime learns for attention under this key. Here 10 of 35 mixed steps (29%) would still learn.
- W1 (warm_up) can declare nd and P as symbols only if the adapter passes P (or T and nd) as ints that move with
  them. max_q_len cannot be declared as a function of two symbols (noted in W1).

## Per-learn cost today (trtllm_context, the 29 post-warm-up steps with exactly one learn; instr.py timers)

| part | ms per learn |
|---|---|
| whole learn (H.learn), median | 4.11 |
| - the capture itself (C.HarvestCapture, 1 capture: the sibling route) | 0.31 |
| - verify launches plus digests (H._launch, C.Digests) | 0.70 |
| - constant-address scan (H._address_hits) | 0.27 |
| - Python in H._harvest (sibling adopt, slot match, bookkeeping) | 2.65 |
| - operand setup in R._learn | 0.56 |

- Outside the learn, each new key also costs binding_row (about 3 ms at 422 calls), form instantiation (2.9 ms in 24%
  of steps) and _call_slow's own 4 ms: roughly 10 ms of host work per new mixed step.
- The step itself is GPU-bound: median 114 ms at T of about 7-16K.
- So the capture is 7.5% of a learn, and a learn is about 4% of a new mixed step. Capturing only the closed kernel
  launch could cut the 4.1 ms to roughly 0.5-1 ms (the capture plus a lean C++ adopt). It would not touch the ~10 ms
  around it.

## Routes

| route | runtime learns left | start-up cost | host / disk cost | soundness | maintenance |
|---|---|---|---|---|---|
| 1. Offline (HarvestProvider.save/load) | only keys never seen in an earlier run; the cache fills over traffic | load about 4.5 s per 100k keys, plus one plain capture per kernel function | about 4 KB/key in memory (compact), 2.9 KB/key on disk | only as complete as the key; the restore check is per function name, not per key (gaps below) | key-completeness list to maintain |
| 2. Warm-up (W1) | context np >= 2 (29% of mixed steps here); 0 for decode and np = 1 | about 67 s for decode plus np = 1 | as 1 in memory | same as runtime learns (each key verified) | adapter declares the symbols |
| 3. Capture only the kernel | all of them, each about 3 ms cheaper | 0 | 0 | same as today | C++ adopt path |
| 4. TRT-LLM C++ launcher (land/core/trtllm_cpp, READY) | 0: no key. Tile, split and CGA choices are guards, and the grid is evaluated per call. A guard flip redispatches the op (context sweep max_kv 300..2000: 1 trace, 0 redispatches; split boundaries: a few redispatches per region, not per value) | 0 | 0 | exact by construction: the guards are the launcher's own branches, byte-for-byte check against stock | 6 patched FlashInfer 0.6.18 files with a drift test; an extension built per torch install; not yet on a candidate-4-or-later core (QUESTIONS 2026-10-12 b) |

Gaps in route 1's key: library identity is path plus size (use the ELF build-id or a content hash). Missing env vars:
NVIDIA_TF32_OVERRIDE, CUBLAS_WORKSPACE_CONFIG, CUBLASLT_WORKSPACE_SIZE, CUBLAS_EMULATION_STRATEGY. Also missing: an SM
limit under MPS or green contexts. The microbench's load run restored the mm keys without captures. It found the lib
rms_norm stand-in stale every time (53 times) and relearned it; that verification is the safety net.

## Recommendation
- Attention keys (trtllm context/decode): route 4. It is the only route that leaves zero runtime learns for np >= 2,
  and it removes the per-key host work around the learn (binding rows, forms), not only the learn. Blocker: port it
  onto a candidate-4-or-later core, which is the attention lane's item. Until then, warm up decode (nd) and context
  np = 1 (P) through W1. That leaves about 29% of mixed steps with one 4 ms learn here.
- T-only keys (GEMMs, externs): W1 warm-up, as measured: 0 runtime learns, 726 s cold and 61 s hot for vLLM today
  (the hot pass drops further with the key-only evaluation).
- Offline: worth it as a start-up accelerator for the cold warm-up (726 s down to seconds of load), once the key gaps
  are closed. It does not solve np >= 2 either.
- Route 3 is low value for attention: it saves about 3 ms per new mixed step, which is about 3% of a GPU-bound step.
