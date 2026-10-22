#!/usr/bin/env bash
# q_nsysc.sh: nsys hang probe C on step_ms24 (core agent 10-18), 27B decode warm-up + 5 steps, untimed, GPU 0:
#  C: node trace with mm_fp4 redispatches refused (DIAG27_REFUSE_REDO_OPS=flashinfer_mm_fp4: those calls retrace).
#  nvidia-smi -q -d PERFORMANCE,CLOCK + py-spy dump on a stall. Waits for q_wip27.done.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_nsysab.done ]; do sleep 60; done
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
INST=$I/build_cpp_ms12c/install EXT=$L/torch_ext_ms12c
mkdir -p $L/tmp_nsys $L/out/nsys
probe() {  # probe TAG GRAPHTRACE [env...]
  local tag=$1 gt=$2; shift 2
  gates; echo "$(date +%T) nsys$tag ($gt $*)"
  ./nsys_watch.sh $tag $L/out/startup27_h3t_vt_rw_sec_rnsys$tag.diag.jsonl &
  local w=$!
  TMPDIR=$L/tmp_nsys PY_PREFIX="$NSYS profile --trace=cuda --cuda-graph-trace=$gt --duration=600 --kill=sigkill --force-overwrite=true -o $L/out/nsys/ms24_decode_$tag" \
    env "$@" UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms24 HEAD3_INSTALL=$INST HEAD3_EXT=$EXT HOLD_S=900 RUN_TIMEOUT=850 RANGE=1 SECTIONS=1 ARMV_MEMORY=run_buffer \
    MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=5 DRIVE_RANGE_PHASES=decode ./gpulock.sh 0 43200 ./startup27.sh run vt nsys$tag > logs/rw_vtnsys$tag.log 2>&1
  echo "$(date +%T) nsys$tag rc=$? steps_done=$(grep -c STARTUP logs/rw_vtnsys$tag.log) cuda_err $(grep -ciE 'CUDA error|illegal instruction|illegal memory' logs/rw_vtnsys$tag.log) stalled: $(head -1 logs/nsyswatch_$tag.txt)"
  wait $w 2>/dev/null
  echo "leftover: $(ps -eo pid,etime,args | grep -E "nsys|rnsys$tag" | grep -v grep | cut -c1-120)"
}
probe 24c node DIAG27_REFUSE_REDO_OPS=flashinfer_mm_fp4
touch logs/q_nsysc.done
