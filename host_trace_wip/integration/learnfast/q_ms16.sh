#!/usr/bin/env bash
# q_ms16.sh SNAP INSTALL: when step_ms16 lands (coordinator 10-16), GPU 0 untimed, gates before each lock; any CUDA error stops:
#  1. eager-order warm-up (rs5-style edge phases + 50 steps, sections on, ARMV_MEMORY=eager, cProfile of _call_slow): section_places
#  2. split A/B: decode bs 1/8/32, alternating 9:16, mixed 8x512, split off then on (DRIVE_SPLIT_AB; after a decode + mixed warm-up)
#  then bitwise --check of the A/B shapes (PART=ab) split off and on BEFORE the timing (coordinator: TMA-descriptor bug suspected);
#  after it, n27wf prefill (a) / mixed (b) with the split on. Non-bitwise rows or a CUDA error stop the chain.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
SNAP=${1:?snap} INST=${2:?install}
cd $L
gate() { w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done; }
cudaerr() { grep -qiE "CUDA error|illegal instruction|illegal memory access|cudaErrorIllegal|Xid" "$@"; }
export UTIL=0.6 HEAD=3 HEAD_WT=$I/snap/$SNAP LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150 RANGE=1 SECTIONS=1 MEMO=0 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1
export HEAD3_INSTALL=$INST HEAD3_EXT=$L/torch_ext_ms16
gate; echo "$(date +%T) eager warm-up"
ARMV_MEMORY=eager DRIVE_STARTUP_STEPS=50 DIAG27_PROFILE_SLOW=$L/out/prof_slow_${SNAP}_eager.pstats ./gpulock.sh 0 3600 ./startup27.sh run vt profslow16 > logs/rw_vtprofslow16.log 2>&1
echo "$(date +%T) rc=$?"; cudaerr logs/rw_vtprofslow16.log && { echo "CUDA ERROR: stop"; exit 1; }
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_SECTIONS_ON=1 PROV_PIN_ATEN=1 ARMV_PIN_UNPINNED=1 ARMV_MEMORY=run_buffer"
chk() {  # chk PART SPLIT
  gate; echo "$(date +%T) check $1 split=$2"
  ./gpulock.sh 0 3600 env $LIFT DIAG27_SPLIT=$2 HEAD=1 HEAD_WT=$I/snap/$SNAP HEAD_INSTALL=$INST HEAD_EXT=$L/torch_ext_ms16 ARM=V GDN=triton PART=$1 INTEG_GPU_UTIL=0.6 LF_GPU=0 LF_CPUS=0-35 RUN_TIMEOUT=1120 ./run27wf.sh splitchk_$1_s$2 chk > logs/splitchk_$1_s$2.log 2>&1
  echo "$(date +%T) check $1 split=$2 rc=$? $(python3 -c "
import json,collections
d=json.load(open('out/splitchk_$1_s$2.json')); ch=d['V']['checks']
print(len(ch), dict(collections.Counter((x['bitwise'], x.get('state_bitwise')) for x in ch)))" 2>&1 | tail -1)"
  cudaerr logs/splitchk_$1_s$2.log && { echo "CUDA ERROR: stop"; exit 1; }
  python3 -c "
import json,sys
d=json.load(open('out/splitchk_$1_s$2.json')); ch=d['V']['checks']
sys.exit(0 if ch and all(x['bitwise'] and x.get('state_bitwise', True) for x in ch) else 1)" || { echo "NOT BITWISE: stop"; exit 1; }
}
chk ab 0
chk ab 1
gate; echo "$(date +%T) split A/B"
ARMV_MEMORY=run_buffer DRIVE_SPLIT_AB=1 DRIVE_RANGE_PHASES=decode,mixed ./gpulock.sh 0 3600 ./startup27.sh run vt splitab > logs/rw_vtsplitab.log 2>&1
echo "$(date +%T) rc=$? $(grep -c SPLIT_AB logs/rw_vtsplitab.log)"; cudaerr logs/rw_vtsplitab.log && { echo "CUDA ERROR: stop"; exit 1; }
chk a 1
chk b 1
touch logs/q_ms16.done
