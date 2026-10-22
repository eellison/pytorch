#!/usr/bin/env bash
# swp (serve lane, 10-13): the slide sweeps, full-step metric (DRIVE_STEP_WITH_SAMPLE=1), c5g, Qwen3-8B, one engine per arm-run:
# decode every batch size 1-128 (fixed batch per point, 36 steps, first 4 dropped), prefill 64-16000 (log-spaced), mixed chunk
# 256-4096 x decodes {16, 64, 128} at short KV (1024-2048), alternating decode 37<->50, 63<->65, 97<->100.
# Arms: default, fullnc, vcs (Vc + sampling, ARMV_SOUND, DRIVE_PHASES for patch counters; cuBLAS learned cold: diagnostic), vcsl
# (vcs with warm GEMMs: cuBLAS bindings preloaded from vllm/out/bindings_<SET>.pkl, which the "learn" step writes by running this
# same workload once with ARMV_BINDINGS_SAVE; the headline "ours"). Outputs vllm/out/t_<SET>_<arm>_r<R>.json.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5g}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
VC=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1)
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64 128; do MIX+=(${p}x${nd}@1024-2048); done; done
W=(--max-model-len 16384 --batch-size $(seq 1 128) --decode-tokens 36 --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0
   --ix-mixed "${MIX[@]}" --alt-decode 37:50@1024-2048 63:65@1024-2048 97:100@1024-2048
   --ix-decode 37:1024-2048 50:1024-2048 63:1024-2048 65:1024-2048 97:1024-2048 100:1024-2048 --ix-reps 6)  # fixed rows at the alternating rows' KV
arms=(${ARMS:-default fullnc vcs}); k=$(( (R - 1) % ${#arms[@]} )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  t=t_${SET}_${a}_r$R
  case $a in
    default) trun $t CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_sweep.py --arm default "${W[@]}" --out vllm/out/$t.json ;;
    fullnc) trun $t CAND_LINE=$L -- vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/$t.json ;;
    vcs) trun $t "${VC[@]}" ARMV_VPS=1 -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;
    vcsl) trun $t "${VC[@]}" ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_${SET}.pkl -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;
    learn) RERUN=1 trun t_${SET}_learn "${VC[@]}" ARMV_VPS=1 ARMV_BINDINGS_SAVE=$PWD/vllm/out/bindings_${SET}.pkl -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/t_${SET}_learn.json ;;
  esac
done
echo ALLDONE
