#!/usr/bin/env bash
# mml (serve lane, 10-13): max_model_len A/B for vLLM's arms (model default 40960 vs our 16384): decode at ~1K / ~8K context,
# bs 1 / 8 / 64, 128 measured steps per point, full-step metric; arms default16 / default40 / fullnc16 / fullnc40.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5h}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
W=(--batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --dec-ctx 1100 8300 --dec-ctx-sizes 1 8 64 --dec-ctx-steps 128)
arms=(${ARMS:-default16 default40 fullnc16 fullnc40}); k=$(( (R - 1) % ${#arms[@]} )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  t=t_${SET}_${a}_r$R; mml=${a//[a-z]/}; mml=$([ "$mml" = 40 ] && echo 40960 || echo 16384)
  case $a in
    default*) trun $t CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_sweep.py --arm default --max-model-len $mml "${W[@]}" --out vllm/out/$t.json ;;
    fullnc*) trun $t CAND_LINE=$L -- vllm/bench/drive_sweep.py --arm fullnc --max-model-len $mml "${W[@]}" --out vllm/out/$t.json ;;
  esac
done
echo ALLDONE
