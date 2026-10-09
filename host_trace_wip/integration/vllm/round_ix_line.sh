#!/usr/bin/env bash
# One round (R) of the InferenceX-shaped set SET on CAND_LINE=$STOCK_LINE (default attn): default, default-nc, V-stock, arm order rotated.
# max_model_len 16384, util 0.9 (bs 256 at KV ~6K needs ~1.55M KV tokens). Rows: decode bs 1/8 (synced + async, short KV),
# mixed = one 8192-token chunk + 32/64/128 decoders, one 4096-token chunk + 64 decoders (decoder KV spread 4096-8000),
# pure decode bs 64/128 at KV 6000-8000 and bs 256 at KV 5000-7000 (capacity). prefill 64 only because drive needs one prefill row.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-attn}; export UTIL_CAP=0.9
VS=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
W=(--max-model-len 16384 --batch-size 1 8 --prefill-lens 64 --mixed 0 --async-decode 1 8 --ix-mixed 8192x32 8192x64 8192x128 4096x64
   --ix-decode 64:6000-8000 128:6000-8000 256:5000-7000 --ix-reps 6)
arms=(default fullnc vstock); k=$(( (R - 1) % 3 )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    default) trun t_${SET}_default_r$R CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_ix.py --arm default "${W[@]}" --out vllm/out/t_${SET}_default_r$R.json ;;
    fullnc) trun t_${SET}_fullnc_r$R CAND_LINE=$L -- vllm/bench/drive_ix.py --arm fullnc "${W[@]}" --out vllm/out/t_${SET}_fullnc_r$R.json ;;
    vstock) trun t_${SET}_vstock_r$R "${VS[@]}" -- vllm/bench/drive_ix.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vstock_r$R.json ;;
  esac
done
echo ALLDONE
