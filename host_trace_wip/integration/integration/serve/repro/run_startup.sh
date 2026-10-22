#!/usr/bin/env bash
# Fair startup (main 10-14), gb300-box GPU 0 after cbh4. STEP=seed|learn|run.
#  seed  ARM=vcsl|vrs: a short process (minimal warm-up: the decode / prefill / mixed traces, 20 stream steps) saving its harvested
#        cuBLAS keys (ARMV_BINDINGS_SAVE) -> vllm/out/startup_seed_<arm>.pkl
#  learn ARM=vcsl|vrs: learnfast learn_range.sh over the warm range (token counts 1-16384, sampled-row GEMMs 1-128) from the seed ->
#        vllm/out/bindings_startup_<arm>.pkl; its wall is the one-time offline cost (fresh processes)
#  run   ARM=default|fullnc|bnc|vcsl|vrs R=1|2: engine ready (ours: bindings loaded + minimal warm-up only) + the request stream
#        (DRIVE_STARTUP) -> vllm/out/t_startup_<arm>_r<R>.json
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
: ${STEP:?} ${ARM:?}
I=$PWD
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0)
BASE=(CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 ARMV_VPS=1 DRIVE_GC_FREEZE=1 ARMV_REFREEZE=1)
case $ARM in
  vcsl) OURS=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1); A=Vc ;;
  vrs) OURS=("${BASE[@]}"); A=V ;;
esac
case $STEP in
  seed) t=$(date +%s.%N)
        env "${OURS[@]}" DRIVE_STARTUP=1 DRIVE_MIN_WARM=1 DRIVE_STARTUP_STEPS=20 ARMV_BINDINGS_SAVE=$PWD/vllm/out/startup_seed_$ARM.pkl \
            bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm $A "${W[@]}" --out vllm/out/t_startup_seed_$ARM.json > serve/logs/startup_seed_$ARM.log 2>&1
        echo "seed $ARM rc=$? wall $(python3 -c "print(round($(date +%s.%N) - $t, 1))") s" ;;
  learn) t=$(date +%s.%N)
        L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
        (cd $L && LF_GPU=0 CPU0=0 N=8 PY=./py.sh bash learn_range.sh $I/vllm/out/startup_seed_$ARM.pkl $I/vllm/out/bindings_startup_$ARM.pkl 16384 128) > serve/logs/startup_learn_$ARM.log 2>&1
        echo "learn $ARM rc=$? wall $(python3 -c "print(round($(date +%s.%N) - $t, 1))") s" ;;
  run) : ${R:?}
       pw=0; p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory)  # 10-14: host memory pressure gate (all lanes)
       while python3 -c "import sys; sys.exit(not ($p60 >= 2))" && [ $pw -lt 3600 ]; do sleep 30; pw=$((pw + 30)); p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory); done
       echo "$(date +%T) PSI_GATE startup_${ARM}_r$R waited ${pw}s full_avg60 $p60" >> vllm/logs/timing_idle.log
       case $ARM in
         default) env CAND_LINE=c5h HOSTTRACE_TORCH_DISABLE_CACHES=0 DRIVE_STARTUP=1 bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm default "${W[@]}" --out vllm/out/t_startup_default_r$R.json ;;
         fullnc) env CAND_LINE=c5h DRIVE_STARTUP=1 bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm fullnc "${W[@]}" --out vllm/out/t_startup_fullnc_r$R.json ;;
         bnc) env CAND_LINE=c5h VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' DRIVE_STARTUP=1 bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm fullnc_full "${W[@]}" --out vllm/out/t_startup_bnc_r$R.json ;;
         vcsl|vrs) MWB=0; [ "$R" -ge 2 ] && MWB=8  # r2+: minimal warm-up v2 (+ an 8-request batched prefill and bs-8 decodes); r1 = v1
                   V3=; [ "$R" -ge 4 ] && V3=1  # r4+: v3 (+ a multi-prefill mixed step at the burst's decode count)
                   env "${OURS[@]}" DRIVE_STARTUP=1 DRIVE_MIN_WARM=1 DRIVE_MIN_WARM_BURST=$MWB DRIVE_MIN_WARM_V3=$V3 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_startup_$ARM.pkl bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm $A "${W[@]}" --out vllm/out/t_startup_${ARM}_r$R.json ;;
       esac > serve/logs/startup_${ARM}_r$R.log 2>&1
       echo "run $ARM r$R rc=$?" ;;
esac
