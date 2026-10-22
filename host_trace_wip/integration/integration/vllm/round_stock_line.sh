#!/usr/bin/env bash
# One round (R) of the stock-comparison set SET on CAND_LINE=$STOCK_LINE (default pinned): default, default-nc, V-stock; the arm order rotates by round so
# no arm always runs first/last in a hold. Workload: decode bs 1/8/64/128 synced + async, prefill 64/512, mixed 8x128 (--mixed 8: 8 mixed steps, the last 5 measured).
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-pinned}
VS=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
W=(--batch-size 1 8 64 128 --prefill-lens 64 512 --mixed 8 --async-decode 1 8 64 128)
arms=(default fullnc vstock); k=$(( (R - 1) % 3 )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    default) trun t_${SET}_default_r$R CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${W[@]}" --out vllm/out/t_${SET}_default_r$R.json ;;
    fullnc) trun t_${SET}_fullnc_r$R CAND_LINE=$L -- vllm/bench/drive.py --arm fullnc "${W[@]}" --out vllm/out/t_${SET}_fullnc_r$R.json ;;
    vstock) trun t_${SET}_vstock_r$R "${VS[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vstock_r$R.json ;;
  esac
done
echo ALLDONE
