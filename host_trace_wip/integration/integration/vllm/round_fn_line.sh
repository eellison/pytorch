#!/usr/bin/env bash
# fn (serve lane, 10-12): one round (R) of Qwen3.8-Flash-Next NVFP4 controlled steps, candidate 4 (CAND_LINE=c4), arms rotated:
# fullnc = the recipe's single-GB300 default (-cc.mode none, FULL_DECODE_ONLY) and V-stock (serve/armV_stock_m, pinned host args,
# prefill / mixed eager by configuration (ARMV_EAGER_KINDS: CORE_TASKS (ix)), decode replayed). Recipe sizes: max_num_seqs 32,
# max_num_batched_tokens 8192, indexer KV fp8, no FlashInfer autotune; bf16 KV (v0.29 rejects fp8 KV for QSA); vLLM's default
# FLASHINFER_TRTLLM MoE (cubins in serve/fi_ws_ht); fixed 30 GiB KV pool.
source $(dirname $0)/timing_lib.sh
export UTIL_CAP=0.9 FLASHINFER_NO_DOWNLOAD=1 INTEG_KV_BYTES=32212254720
FI=FLASHINFER_WORKSPACE_BASE=$PWD/serve/fi_ws_ht
VS=(CAND_LINE=c4 $FI ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_EAGER_KINDS=prefill,mixed ARMV_CUTE_OBSERVE=1
    ARMV_EXTRA_EXTERN=scaled_fp4_quant.out,static_scaled_fp8_quant.default,cooperative_topk.default,persistent_topk.default INTEG_ARMV=$PWD/serve/armV_stock_m)
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false, "kv_cache_dtype": "auto"}'
W=(--model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 8 --prefill-lens 64 --mixed 0 --async-decode 1 8
   --ix-mixed 1024x16@1024-2048 2048x16@6000-8000 --ix-decode 16:6000-8000 32:6000-8000 --ix-reps 6 --llm-kw "$KW")
arms=(fullnc vstock); k=$(( (R - 1) % 2 )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    fullnc) trun t_${SET}_fullnc_r$R CAND_LINE=c4 $FI -- vllm/bench/drive_fn.py --arm fullnc "${W[@]}" --out vllm/out/t_${SET}_fullnc_r$R.json ;;
    vstock) trun t_${SET}_vstock_r$R "${VS[@]}" -- vllm/bench/drive_fn.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vstock_r$R.json ;;
  esac
done
echo ALLDONE
