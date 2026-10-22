#!/usr/bin/env bash
# g4f (serve lane, 10-12): the full-step set. Every arm's step = execute_model + sample_tokens, synced (DRIVE_STEP_WITH_SAMPLE=1),
# g4's shapes (decode bs 1 and 64 plus the 8 mixed shapes), candidate 4 + the trtllm fork, arms rotated:
#   default (compile + vLLM graphs), fullnc (no compile, FULL decode graphs), vreg (V-registered, eager forward, sampling eager),
#   vc (V-registered around the compiled forward, sampling eager), vcs (vc + sampling and request-state writes in the step),
#   vrs (vreg + sampling and request-state writes in the step). ARMV_SOUND everywhere for our arms.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c4}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
VR=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4)
VC=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1)
W=(--max-model-len 16384 --batch-size 1 64 --prefill-lens 64 --mixed 0
   --ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000 --ix-reps 6)
arms=(${ARMS:-default fullnc vreg vc vcs vrs}); k=$(( (R - 1) % ${#arms[@]} )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    default) trun t_${SET}_default_r$R CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_ix2.py --arm default "${W[@]}" --out vllm/out/t_${SET}_default_r$R.json ;;
    fullnc) trun t_${SET}_fullnc_r$R CAND_LINE=$L -- vllm/bench/drive_ix2.py --arm fullnc "${W[@]}" --out vllm/out/t_${SET}_fullnc_r$R.json ;;
    vreg) trun t_${SET}_vreg_r$R "${VR[@]}" -- vllm/bench/drive_ix2.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vreg_r$R.json ;;
    vrs) trun t_${SET}_vrs_r$R "${VR[@]}" ARMV_VPS=1 -- vllm/bench/drive_ix2.py --arm V "${W[@]}" --out vllm/out/t_${SET}_vrs_r$R.json ;;
    vc) trun t_${SET}_vc_r$R "${VC[@]}" -- vllm/bench/drive_ix2.py --arm Vc "${W[@]}" --out vllm/out/t_${SET}_vc_r$R.json ;;
    vcsm) trun t_${SET}_vcsm_r$R "${VC[@]}" ARMV_VPS=1 ARMV_DECODE_MAX_SEQ=model -- vllm/bench/drive_ix2.py --arm Vc "${W[@]}" --out vllm/out/t_${SET}_vcsm_r$R.json ;;  # 10-13 option: vLLM's FULL-graph max_seq_len (max_model_len) at decode
    vcs) trun t_${SET}_vcs_r$R "${VC[@]}" ARMV_VPS=1 -- vllm/bench/drive_ix2.py --arm Vc "${W[@]}" --out vllm/out/t_${SET}_vcs_r$R.json ;;
  esac
done
echo ALLDONE
