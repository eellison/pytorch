## 8B startup per phase (existing logs, gb300-box GPU 0, Qwen3-8B; serve/out/startup_table.md runs)

| phase | fullnc r3 | breakable (bnc) r3 | default r1 | ours vcsl r5 | ours vrs r5 |
|---|---|---|---|---|---|
| weight load | 3.7 | 3.3 | 5.1 | 4.0 | 3.6 |
| model init incl. weights | 6.1 | 5.3 | 7.2 | 6.1 | 5.4 |
| torch.compile | 0.0 | 0.0 | 0.3 | 0.3 | 0.0 |
| KV profiling / cache init (minus a capture inside it) | 3.0 | 2.0 | 3.0 | 2.0 | 1.0 |
| CUDA graph capture inside memory profiling | 1.0 | 39.0 | 6.0 | 0.0 | 0.0 |
| FlashInfer autotune [Loaded 0 configs; Saved 0 configs; Using FlashInfer autotune cache file] | 0.2 | 0.2 | 0.2 | 0.0 | 0.2 |
| CUDA graph capture | 5.0 | 45.0 | 13.0 | 0.0 | 0.0 |
| vLLM init engine total (profile, KV, warmup, capture) | 10.4 | 88.3 | 24.0 | 4.2 | 3.5 |
| LLM() total | 21.6 | 99.5 | 36.1 | 15.4 | 13.9 |
| ours: bindings load | - | - | - | 1.9 | 2.3 |
| ours: minimal warm-up | 0.0 | 0.0 | 0.0 |  |  |
| other: pre-LLM imports / observers + ours' install | 10.5 | 10.2 | 10.3 | 16.6 | 17.3 |
| ENGINE READY (process start) | 32.1 | 109.7 | 46.4 | 43.2 | 48.9 |
| TTFT first 8 (median ms) | 69.1 | 69.8 | 67.0 | 84.8 | 88.9 |
| ready to steady (last trace / learn / redispatch step; 800 quiet steps after) | 32.1 | 109.7 | 46.4 | 43.2 | 48.9 |
| ours: minimal warm-up (adapter: 5 traces) |  |  |  | 11.1 | 17.7 |
| offline cuBLAS learn (seed + learn_range, not in engine ready) |  |  |  | 432.7 | 430.6 |

### Per-arm configuration (from each run's log and serve/repro/run_startup.sh; checked 10-15)
| arm | head line | KV | max_num_seqs | cudagraph mode / capture sizes | compile | stream |
|---|---|---|---|---|---|---|
| fullnc r3 | c5h (CAND_LINE=c5h) | gpu_memory_utilization 0.9, KV from memory profiling | vLLM default | FULL_DECODE_ONLY; default size list, max 1024 | off (mode NONE) | 8 at once then C=32, 800 steps, seed 7 (110 requests) |
| breakable (bnc) r3 | c5h | same | default | FULL_AND_PIECEWISE (VLLM_USE_BREAKABLE_CUDAGRAPH=1); same list, max 1024 | off | same |
| default r1 | c5h | same | default | FULL_AND_PIECEWISE; same list, max 1024 | on (vLLM default; AOT cache hit, 0.26 s) | same |
| ours vcsl r5 | c5h + stock fast adapter | same | default | NONE (ours replays) | on (Vc: vLLM's compiled forward) | same |
| ours vrs r5 | c5h + stock fast adapter | same | default | NONE, enforce_eager | off (V: eager forward) | same |

All five share the KV setting (0.9 utilization, profiled), max_num_seqs, the capture-size set where they capture, the stream
and the head line: no rerun needed. The graph arms capture inside vLLM's CUDA-graph memory profiling as well (bnc 39 s
there + 45 s final; that is vLLM's default with a utilization-sized KV, not a config difference). The 27B table instead
fixes KV at 64 GiB (kv_cache_memory_bytes), which skips memory profiling, and max_num_seqs 128.
