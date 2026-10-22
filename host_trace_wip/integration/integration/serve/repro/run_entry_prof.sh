#!/usr/bin/env bash
# Entry profile (serve lane, 10-13): the C++ replay entry's phase clocks + patch counters per full step (DRIVE_PHASES=1) and perf
# samples of a long decode at bs 1 and 64 (DRIVE_PERF_DECODE), g4f's arm configs on c5g. GPU $GPU under gpu$GPU.lock.
#   GPU=2 ARMS="vcs vrs fullnc" bash serve/repro/run_entry_prof.sh
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
GPU=${GPU:-2}; O=serve/out/entry_prof; mkdir -p $O
BASE=(CAND_LINE=c5g CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4)
W=(--max-model-len 16384 --batch-size 1 64 --prefill-lens 64 --mixed 0 --ix-mixed 1024x64@1024-2048 --ix-reps 6)
COMMON=(CUDA_VISIBLE_DEVICES=$GPU DRIVE_STEP_WITH_SAMPLE=1 DRIVE_PHASES=1 DRIVE_PERF_DECODE=1,64 INTEG_GPU_UTIL=${UTIL:-0.85})
for a in ${ARMS:-vcs vrs fullnc}; do
  while ! bash ../eager/gpu_idle_ok.sh $GPU; do echo "$(date +%T) gpu$GPU foreign: waiting"; sleep 60; done
  case $a in
    vcs) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1); ARM=Vc ;;
    vcs_ch) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 ARMV_CHANGED=1); ARM=Vc ;;
    vcsp) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 ARMV_PERSIST=1); ARM=Vc ;;
    vcsp_ch) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 ARMV_PERSIST=1 ARMV_CHANGED=1); ARM=Vc ;;
    vcsm_ch) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 ARMV_CHANGED=1 ARMV_DECODE_MAX_SEQ=model); ARM=Vc ;;
    vcsm) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 ARMV_DECODE_MAX_SEQ=model); ARM=Vc ;;
    vc) E=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1); ARM=Vc ;;
    vrs) E=("${BASE[@]}" ARMV_VPS=1); ARM=V ;;
    vreg) E=("${BASE[@]}"); ARM=V ;;
    fullnc) E=(CAND_LINE=c5g); ARM=fullnc ;;
    default) E=(CAND_LINE=c5g HOSTTRACE_TORCH_DISABLE_CACHES=0); ARM=default ;;
  esac
  t=$a; [ -e $O/$t.json ] && t=${a}_$(date +%H%M)  # never overwrite a reported output
  echo "$(date +%T) start $a -> $O/$t.json"
  env "${COMMON[@]}" "${E[@]}" DRIVE_PERF_OUT=$PWD/$O/$t timeout 1800 flock -x ../eager/gpu$GPU.lock bash python_vllm_cand2.sh vllm/bench/drive_ix2.py --arm $ARM "${W[@]}" --out $O/$t.json > $O/$t.log 2>&1
  echo "$(date +%T) end $a rc=$?"
done
echo ENTRY_PROF_DONE
