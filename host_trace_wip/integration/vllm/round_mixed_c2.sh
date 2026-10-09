#!/usr/bin/env bash
# One round (R) of the mixed-only set SET (g2) after the attention lane's fix: default, default-nc, V-stock on the fix's build
# (G2_LINE: a python_vllm_cand2.sh CAND_LINE; G2_ENV: extra env for V-stock, e.g. the fix's switch). Arm order rotates by round.
# Mixed 8x128 (--mixed 4) plus the smallest decode/prefill rows drive.py requires (bs 1, prefill 64).
source $(dirname $0)/timing_lib.sh
L=${G2_LINE:?}
VS=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ${G2_ENV:-})
W=(--batch-size 1 --prefill-lens 64 --mixed 4)
arms=(default fullnc vstock); k=$(( (R - 1) % 3 )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    default) trun t_${SET}_default_r$R CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${W[@]}" --out vllm/out/t_${SET}_default_r$R.json ;;
    fullnc) trun t_${SET}_fullnc_r$R CAND_LINE=$L -- vllm/bench/drive.py --arm fullnc "${W[@]}" --out vllm/out/t_${SET}_fullnc_r$R.json ;;
    vstock) trun t_${SET}_vstock_r$R "${VS[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vstock_r$R.json ;;
  esac
done
echo ALLDONE
