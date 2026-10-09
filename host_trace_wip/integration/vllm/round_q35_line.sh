#!/usr/bin/env bash
# One round (R) of the Qwen3.5-9B set SET (recipe args: --trust-remote-code --language-model-only, vLLM's own attention backend) on CAND_LINE=$STOCK_LINE (default attn), arms rotated.
# Rows as g3 + g4 (KV per token is ~32 KiB here: 8 full-attention layers x 4 KV heads x 256, so bs 256 runs at KV 6000-8000).
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-trtcpp}; export UTIL_CAP=0.9 FLASHINFER_NO_DOWNLOAD=1  # cubins fetched once by q35_check_gpu0.sh step 1
VS=(CAND_LINE=$L ARMV_TRTLLM_CPP=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
W=(--model /data/eellison/models/Qwen3.5-9B --trust-remote-code --language-model-only --max-model-len 16384 --batch-size 1 8 --prefill-lens 64 --mixed 0 --async-decode 1 8
   --ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000
              4096x64@4096-8000 8192x32@4096-8000 8192x64@4096-8000 8192x128@4096-8000
   --ix-decode 64:6000-8000 128:6000-8000 256:6000-8000 --ix-reps 6)
arms=(default fullnc vstock); k=$(( (R - 1) % 3 )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    default) trun t_${SET}_default_r$R CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_q35.py --arm default "${W[@]}" --out vllm/out/t_${SET}_default_r$R.json ;;
    fullnc) trun t_${SET}_fullnc_r$R CAND_LINE=$L -- vllm/bench/drive_q35.py --arm fullnc "${W[@]}" --out vllm/out/t_${SET}_fullnc_r$R.json ;;
    vstock) trun t_${SET}_vstock_r$R "${VS[@]}" -- vllm/bench/drive_q35.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vstock_r$R.json ;;
  esac
done
echo ALLDONE
