# V-stock env for the generic adapter (no FlashInfer fork; CAND_FORK/ARMV_TRTLLM_FORK added only for the FLASHINFER baseline)
VS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_GENERIC=1 ARMV_NO_REJIT=1)
Q8=/data/eellison/models/Qwen3-8B
R1=/data/eellison/models/Qwen3.8-27B-NVFP4
CHECKW=(--batch-size 1 8 64 --prefill-lens 64 512 2048 --prefill-reqs 4 8 --mixed 4 --no-host-probe)
R1KW_FP8='{"language_model_only": true, "kv_cache_dtype": "fp8", "max_model_len": 4096, "max_num_seqs": 64, "gdn_prefill_backend": "triton"}'
R1KW_BF16='{"language_model_only": true, "kv_cache_dtype": "auto", "max_model_len": 4096, "max_num_seqs": 64, "gdn_prefill_backend": "triton"}'
PROFW=(--batch-size 1 8 64 --decode-len 1024 --prefill-lens 512 2048 --mixed 4 --no-host-probe --profile-kinds)
