#!/usr/bin/env bash
# capsz (serve lane, 10-13): "what if vLLM raises the capture size?" g4f's workload and full-step metric (DRIVE_STEP_WITH_SAMPLE=1)
# on c5g, vLLM arms with the capture range extended past g4's mixed shapes (1,088-2,176 tokens):
#   dc2304 / dc4096: default (compile + FULL_AND_PIECEWISE), max_cudagraph_capture_size 2304 / 4096;
#   bnc2304 / bnc4096: no compile, FULL_AND_PIECEWISE through vLLM's breakable CUDA graphs (VLLM_USE_BREAKABLE_CUDAGRAPH=1, the
#     only way vLLM graphs a mixed step without torch.compile);
#   fnc2304: no compile, cudagraph_mode FULL requested (vLLM downgrades it to FULL_DECODE_ONLY with FlashInfer: support UNIFORM_BATCH).
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5g}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
W=(--max-model-len 16384 --batch-size 1 64 --prefill-lens 64 --mixed 0
   --ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000 --ix-reps 6)
arms=(${ARMS:-dc2304 bnc2304 dc4096 bnc4096}); k=$(( (R - 1) % ${#arms[@]} )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
cc() { echo "{\"max_cudagraph_capture_size\": $1${2:+, \"cudagraph_mode\": \"$2\"}}"; }
for a in "${arms[@]}"; do
  t=t_${SET}_${a}_r$R
  case $a in
    dc*) trun $t CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 DRIVE_CC_JSON="$(cc ${a#dc})" -- vllm/bench/drive_ix2.py --arm default "${W[@]}" --out vllm/out/$t.json ;;
    bnc*) trun $t CAND_LINE=$L VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON="$(cc ${a#bnc} FULL_AND_PIECEWISE)" -- vllm/bench/drive_ix2.py --arm fullnc_full "${W[@]}" --out vllm/out/$t.json ;;
    fnc*) trun $t CAND_LINE=$L DRIVE_CC_JSON="$(cc ${a#fnc} FULL)" -- vllm/bench/drive_ix2.py --arm fullnc_full "${W[@]}" --out vllm/out/$t.json ;;
  esac
done
echo ALLDONE
