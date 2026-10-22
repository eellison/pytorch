#!/usr/bin/env bash
# q_xvr.sh: cross-variant reuse prototype (xvr.py) on the head (step_ms16 / build_cpp_ms11), GPU 3. Gates before each lock;
# any CUDA error or non-bitwise row stops the chain.
#  1-2. n27wf --check, prefill part (a) and mixed part (b), the startup config (lift, closure, templates + keyout, sections,
#       run_buffer) + DIAG27_XVR=1: bitwise vs eager, XVR hit counts in the log
#  3-5. rs5-style cold start: x16_r1 (xvr on), b16_r3 (off; gc_untrack timer), x16_r2 (on)
source /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast/lib_g3.sh
cd $L
SNAP=${SNAP:-step_ms16} INST=${INST:-$I/build_cpp_ms11/install}
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
chk() {  # chk PART TAG [extra env...]
  local part=$1 tag=$2; shift 2
  rm -f out/$tag.json  # a stepped-off run must not read an older result
  gates; echo "$(date +%T) check $part $tag"
  HOLD_S=2700 ./gpulock.sh 3 43200 env $LIFT "$@" HEAD=1 HEAD_WT=$I/snap/$SNAP HEAD_INSTALL=$INST HEAD_EXT=$L/torch_ext_ms16 ARM=V GDN=triton PART=$part INTEG_GPU_UTIL=0.6 RUN_TIMEOUT=2600 ./run27wf.sh $tag chk > logs/$tag.log 2>&1
  echo "$(date +%T) check $part rc=$? $(python3 -c "
import json,collections
d=json.load(open('out/$tag.json')); ch=d['V']['checks']
print(len(ch), dict(collections.Counter((x['bitwise'], x.get('state_bitwise')) for x in ch)))" 2>&1 | tail -1) $(grep -h '^XVR' logs/$tag.log | cut -c1-600)"
  cudaerr logs/$tag.log && { echo "CUDA ERROR: stop"; exit 1; }
  python3 -c "
import json,sys
d=json.load(open('out/$tag.json')); ch=d['V']['checks']
sys.exit(0 if ch and all(x['bitwise'] and x.get('state_bitwise', True) for x in ch) else 1)" || { echo "NOT BITWISE: stop"; exit 1; }
}
startup() {  # startup TAG [extra env...]
  local tag=$1; shift
  gates; echo "$(date +%T) $tag"
  p0=$(awk '/^full/{split($5,a,"="); print a[2]}' /proc/pressure/memory)
  env "$@" UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/$SNAP HEAD3_INSTALL=$INST HEAD3_EXT=$L/torch_ext_ms16 HOLD_S=2700 RUN_TIMEOUT=2600 \
    RANGE=1 SECTIONS=1 ARMV_MEMORY=run_buffer MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 DRIVE_STARTUP_STEPS=${STEPS:-800} \
    ./gpulock.sh 3 43200 ./startup27.sh run vt $tag > logs/rw_vt$tag.log 2>&1
  rc=$?
  p1=$(awk '/^full/{split($5,a,"="); print a[2]}' /proc/pressure/memory)
  echo "$(date +%T) $tag rc=$rc psi_full_s_during $(python3 -c "print(round(($p1 - $p0) / 1e6, 1))") $(grep -h '^XVR' logs/rw_vt$tag.log | cut -c1-400)"
  cudaerr logs/rw_vt$tag.log && { echo "CUDA ERROR: stop"; exit 1; }
}
[ "${SKIP_CHK:-0}" = 1 ] || { chk a xvrchk_a DIAG27_XVR=1; chk b xvrchk_b DIAG27_XVR=1; }
startup x16_r1 DIAG27_XVR=1
startup b16_r3
startup x16_r2 DIAG27_XVR=1
touch logs/q_xvr.done
