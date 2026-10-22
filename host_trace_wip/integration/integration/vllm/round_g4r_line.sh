#!/usr/bin/env bash
# g4r (serve lane, 10-12): g4s + a 4th arm, V-registered = ARMV_SOUND + ARMV_PORTS with vllm/armV_stock_fast_m2 ("vLLM custom ops via
# registered launchers: the real compiled kernels, launch config traced, each launch checked against the real op's"). Arms rotated;
# ARMS="vreg" runs only that arm (reuse g4s's columns, labelled so). Check first: serve/repro/run_q8p2_check.sh (179/179, 0 eager steps).
# One round (R) of the follow-up set SET (g4): 1k1k-shaped and 8k1k prompt-tail mixed steps on CAND_LINE=$STOCK_LINE (default attn), arms rotated.
# 1k1k-shaped: chunk 1024/2048 + 64/128 decoders at KV 1024-2048; 8k1k prompt-tail: the same at KV 6000-8000. max_model_len 16384, util 0.9.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c4}; export UTIL_CAP=0.9
VS=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m)
VR=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m2)
W=(--max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0
   --ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000 --ix-reps 6)
arms=(${ARMS:-default fullnc vstock vreg}); k=$(( (R - 1) % ${#arms[@]} )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    default) trun t_${SET}_default_r$R CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_ix2.py --arm default "${W[@]}" --out vllm/out/t_${SET}_default_r$R.json ;;
    fullnc) trun t_${SET}_fullnc_r$R CAND_LINE=$L -- vllm/bench/drive_ix2.py --arm fullnc "${W[@]}" --out vllm/out/t_${SET}_fullnc_r$R.json ;;
    vstock) trun t_${SET}_vstock_r$R "${VS[@]}" -- vllm/bench/drive_ix2.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vstock_r$R.json ;;
    vreg) trun t_${SET}_vreg_r$R "${VR[@]}" -- vllm/bench/drive_ix2.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vreg_r$R.json ;;
  esac
done
echo ALLDONE
