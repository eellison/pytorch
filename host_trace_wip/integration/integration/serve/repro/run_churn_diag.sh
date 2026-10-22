#!/usr/bin/env bash
# Churn diagnostic (serve lane, 10-13, low priority, GPUs 1-3 only): fixed-batch decode next to shape churn, full-step metric, c5g.
# Rows: decode bs 1 / 64 (fixed: the patch-skip best case), alternating decode 37<->50 and 63<->65 (every step a new shape),
# shrink 64 (a request finishes every step; pass 1 first sight, pass 2 replay), mixed 1024x64. Ours with the entry phase clocks
# (DRIVE_PHASES: patch us, nodes set per step). vLLM rows report padded tokens per step. Outputs serve/out/churn/<arm>_<HHMM>.json.
#   GPU=2 ARMS="vcs default fullnc" bash serve/repro/run_churn_diag.sh
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
GPU=${GPU:-2}; O=serve/out/churn; mkdir -p $O
BASE=(CAND_LINE=c5g CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4)
W=(--max-model-len 16384 --batch-size 1 64 --prefill-lens 64 --mixed 0 --ix-mixed 1024x64@1024-2048 --alt-decode 37:50 63:65 --shrink 64 --ix-reps 6)
for a in ${ARMS:-vcs default fullnc}; do
  case $GPU in 1|2|3) ;; *) echo "GPUs 1-3 only"; exit 1 ;; esac
  while ! bash ../eager/gpu_idle_ok.sh $GPU; do echo "$(date +%T) gpu$GPU foreign: waiting"; sleep 60; done
  case $a in
    vcs) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1); ARM=Vc ;;
    vcsp) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 ARMV_PAD=1); ARM=Vc ;;
    fullnc) E=(CAND_LINE=c5g); ARM=fullnc ;;
    default) E=(CAND_LINE=c5g HOSTTRACE_TORCH_DISABLE_CACHES=0); ARM=default ;;
  esac
  read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | tr -d ,)
  UTIL=$(python3 -c "print(min(${UTIL_CAP:-0.85}, round(($total - $used - 8192) / $total, 3)))")
  t=${a}_$(date +%H%M%S)
  echo "$(date +%T) start $a -> $O/$t.json"
  env CUDA_VISIBLE_DEVICES=$GPU DRIVE_STEP_WITH_SAMPLE=1 DRIVE_PHASES=1 INTEG_GPU_UTIL=$UTIL "${E[@]}" timeout 1800 flock -x ../eager/gpu$GPU.lock bash python_vllm_cand2.sh vllm/bench/drive_ix2.py --arm $ARM "${W[@]}" --out $O/$t.json > $O/$t.log 2>&1
  echo "$(date +%T) end $a rc=$?"
done
echo CHURN_DONE
