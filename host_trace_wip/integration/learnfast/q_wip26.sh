#!/usr/bin/env bash
# q_wip26.sh: snap/wip26 (= wip25 + template-guard trim under pins) on build_cpp_ms12c: n27wf --check b first (GDN templates; bitwise
# + tpl confirmations), then rs5-style w26_r1 with DIAG27_RDCOST (pair with wip25's w25_r1); then the nsys retry on step_ms24
# (coordinator 10-18: 27B decode, node-level CUDA graph trace, --duration 600 --kill, TMPDIR on /data, untimed). GPU 0.
export LANE_GPU=0 LANE_CPUS=0-35
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_gpu.sh
cd $L
while [ ! -e logs/q_wip25.done ]; do sleep 60; done
export SNAP=wip26 INST=$I/build_cpp_ms12c/install
EXT=$L/torch_ext_ms12c
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
source <(sed -n '/^chk() {/,/^}/p;/^startup() {/,/^}/p' q_xvr.sh | sed "s|\$L/torch_ext_ms16|$EXT|g; s/gpulock.sh 3/gpulock.sh 0/g")
chk b w26chk_b
grep -q "confirmation mismatch" logs/w26chk_b.log && { echo "TPL CONFIRMATION MISMATCH: stop"; exit 1; }
startup w26_r1 DIAG27_GC_TIME=1 DIAG27_GCU_LOG=1 DIAG27_RDCOST=1
touch logs/q_wip26.done
# nsys retry (untimed): step_ms24, decode phase only, 5 stream steps, node-level graph trace
NSYS=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
mkdir -p $L/tmp_nsys $L/out/nsys
gates; echo "$(date +%T) nsys24"
TMPDIR=$L/tmp_nsys PY_PREFIX="$NSYS profile --trace=cuda --cuda-graph-trace=node --duration=600 --kill=sigkill --force-overwrite=true -o $L/out/nsys/ms24_decode" \
  env UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/step_ms24 HEAD3_INSTALL=$INST HEAD3_EXT=$EXT HOLD_S=900 RUN_TIMEOUT=850 RANGE=1 SECTIONS=1 ARMV_MEMORY=run_buffer \
  MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=5 DRIVE_RANGE_PHASES=decode ./gpulock.sh 0 43200 ./startup27.sh run vt nsys24 > logs/rw_vtnsys24.log 2>&1
echo "$(date +%T) nsys24 rc=$? $(ls -la out/nsys | grep ms24) cuda_err $(grep -ciE 'CUDA error|illegal instruction|illegal memory' logs/rw_vtnsys24.log)"
echo "leftover: $(ps -eo pid,etime,args | grep -E 'nsys|startup27.sh run vt nsys24' | grep -v grep | cut -c1-120)"
touch logs/q_nsys24.done
