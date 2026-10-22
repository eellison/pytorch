# Open attention backends for vLLM host tracing (STOPPED EARLY, 2026-10-08)

Written by the coordinator from the lane's final message (the lane couldn't write files). Stopped per the user: Triton-only isn't a realistic kernel choice; attention stays on FlashInfer TRT-LLM (vLLM's default) via the real C++ launcher.

Done: Qwen3-8B (bf16, head 128, bf16 KV), pinned build, vLLM 0.29.0, V-stock --check (140 forwards) + eager attention profiles. Not done: R1, fp8-KV acceptance, default-vs-V-stock step times, FA4's exact eager reason.

| backend | V-stock traces / variants / eager steps per variant | bitwise vs stock eager | decode attn us bs1/8/64 (x trtllm) | prefill 512/2048 | mixed |
|---|---|---|---|---|---|
| FlashInfer trtllm-gen (ref) | 4 / 4 / 0 (earlier) | 188/188 | 261 / 416 / 1731 | 413 / 1486 | 495 |
| FLASH_ATTN (FA4, CuTe DSL) | 6 / 6 / 36 (every attention call eager) | 140/140 | 434 (1.66x) / 574 (1.38x) / 2445 (1.41x) | 481 (1.17x) / 1374 (0.92x) | 475 (0.96x) |
| TRITON_ATTN | 10 / 10 / 0 | 140/140 | 421 (1.61x) / 640 (1.54x) / 2515 (1.45x) | 1122 (2.72x) / 11750 (7.91x) | 581 (1.17x) |
| FLEX_ATTENTION | 93 traces, 0 variants, all eager | 140/140 | 11.9x / 65x / 32x | 7.5x / 24x | 16x |

Constraints (vLLM source): FA4 refuses fp8 KV with head 256 and needs 128-token blocks; Flex has no fp8 KV; Triton lists fp8 KV.

Findings worth keeping:
- Triton: a variant per request count from `len(seqused_k)` (ops/triton_unified_attention.py:926; Python len() pins) and an expand guard on `_k_scale.expand((num_reqs, ...))` (backends/triton_attn.py:633).
- FA4: vLLM calls cute.compile(..., "--enable-tvm-ffi") and invokes the TVM-FFI function with tensors, a Python float, Nones and an AuxData(None, None) NamedTuple; our CuTe route leaves every call eager (likely None/NamedTuple/float binding gap, unconfirmed; repro_fa4.py not run). Split choice num_splits_heuristic (cute/interface.py:331) retraces at graph level.
- Flex: declines on unpinned CPU metadata tensor (query_start_loc_cpu); block mask closure references untracked tensors.

Files: drive_attn.py, armV_open/, repro_fa4.py, prof_attn.py, vsum.py, out/ (pe_q8_*.json, q8_*_V_check.json), prof/, logs/.
