## 8B cold start, decode-only graphs (like-for-like with vLLM CUDA graphs FULL_DECODE_ONLY, compile off)

Harness: serve/repro/run_startup.sh's run step (Qwen3-8B, max_model_len 16384, gpu_memory_utilization 0.9 with profiled KV, vLLM default max_num_seqs; DRIVE_STARTUP: engine ready = process start -> LLM() + our install with cuBLAS bindings loaded + our minimal warm-up; then 8 requests at once (their TTFT) and a C=32 stream of 800 steps, seed 7, 110 requests). Same as STARTUP_BREAKDOWN_8B.
Ours: vLLM's eager forward (V, armV_stock_fast_m4, VPS, GC freeze + refreeze), ARMV_EAGER_KINDS=prefill,mixed (prefill and mixed steps run vLLM's eager forward, as FULL_DECODE_ONLY does), bindings_startup_vrs.pkl (the c5h offline learn, ~431 s one-time, not in engine ready), minimal warm-up v2 = 4 decoders + an 8-request burst (its prefills run eagerly) so only decode forms are traced.
GB300 gb300-box GPU 3, CPUs 108-143, 2026-10-18 12:57-13:11 PDT, 3 rotated rounds, in-session fullnc. Talk-set fullnc (STARTUP_BREAKDOWN_8B, GPU 0, 10-15): 32.1 s.

| phase | fullnc r1 | fullnc r2 | fullnc r3 | vrsd r1 | vrsd r2 | vrsd r3 | vrsd20 r1 | vrsd20 r2 | vrsd20 r3 | fullnc median | vrsd median | vrsd20 median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weight load | 3.0 | 3.1 | 3.1 | 3.2 | 3.2 | 3.2 | 3.2 | 3.4 | 3.2 | 3.1 | 3.2 | 3.2 |
| model init incl. weights | 5.1 | 5.0 | 5.2 | 4.8 | 4.9 | 4.9 | 5.0 | 5.1 | 5.0 | 5.1 | 4.9 | 5.0 |
| CUDA graph capture (memory profiling + final) | 6.0 | 6.0 | 8.0 | 0 | 0 | 0 | 0 | 0 | 0 | 6.0 | 0 | 0 |
| vLLM init engine total (profile, KV, warmup, capture) | 10.2 | 9.5 | 11.2 | 3.6 | 3.3 | 3.2 | 3.3 | 3.2 | 3.3 | 10.2 | 3.3 | 3.3 |
| LLM() total | 20.1 | 19.0 | 20.7 | 13.1 | 12.5 | 12.4 | 12.6 | 12.6 | 12.6 | 20.1 | 12.5 | 12.6 |
| ours: bindings load | - | - | - | 1.8 | 1.8 | 2.0 | 1.8 | 1.8 | 2.0 |  | 1.8 | 1.8 |
| ours: minimal warm-up (decode forms) | - | - | - | 3.5 | 3.4 | 3.6 | 2.4 | 2.3 | 2.5 |  | 3.5 | 2.4 |
| other: pre-LLM imports / observers + ours' install (ready - LLM() - bindings - warm-up) | 14.4 | 10.2 | 11.1 | 16.0 | 14.2 | 15.5 | 14.5 | 13.5 | 14.4 | 11.1 | 15.5 | 14.4 |
| ENGINE READY (process start) | 34.4 | 29.2 | 31.8 | 34.4 | 32.0 | 33.5 | 31.3 | 30.2 | 31.5 | 31.8 | 33.5 | 31.3 |
| TTFT first 8 (median ms) | 69.2 | 68.4 | 125.2 | 70.2 | 70.8 | 72.4 | 70.5 | 71.5 | 74.7 | 69.2 | 70.8 | 71.5 |
| ready to steady (s) | 34.4 | 29.2 | 31.8 | 35.6 | 33.1 | 34.7 | 32.4 | 31.4 | 32.8 | 31.8 | 34.7 | 32.4 |
| stream: steps with trace / learn / new binding (of 800) | 0 | 0 | 0 | 3 | 3 | 3 | 2 | 2 | 2 | 0 | 3 | 2 |
| stream: events (traces, harvests, bindings) | (0, 0, 0) | (0, 0, 0) | (0, 0, 0) | (0, 0, 15) | (0, 0, 15) | (0, 0, 15) | (0, 0, 49) | (0, 0, 49) | (0, 0, 49) |  |  |  |
| ours: prefill/mixed calls run eagerly (policy) / decode replays | - | - | - | {'eager_kind_policy': 86, 'decode': 731, 'vps_sampled': 731} | {'eager_kind_policy': 86, 'decode': 731, 'vps_sampled': 731} | {'eager_kind_policy': 86, 'decode': 731, 'vps_sampled': 731} | {'eager_kind_policy': 86, 'decode': 731, 'vps_sampled': 731} | {'eager_kind_policy': 86, 'decode': 731, 'vps_sampled': 731} | {'eager_kind_policy': 86, 'decode': 731, 'vps_sampled': 731} |  |  |  |
| PSI full s in stream / flagged | 0.60 / True | 0.01 / False | 0.00 / False | 0.00 / False | 0.00 / False | 0.00 / True | 0.00 / True | 0.01 / True | 0.62 / True |  |  |  |

Arms: fullnc = vLLM CUDA graphs (FULL_DECODE_ONLY, compile off), c5h; vrsd = ours decode-only (c5h); vrsd20 = ours decode-only (ms20).
