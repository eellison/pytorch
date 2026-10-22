#!/usr/bin/env bash
# q_secchk.sh: RD2 / sections-on bitwise check (coordinator 10-15): 27B n27wf prefill (a) and mixed (b) parts, --check (every
# forward vs eager, twice), ours V on the Delta 3 snap with lifted state + fla overlay + cross-layer templates + refusal memo +
# sections on, Triton GDN. GPU 3, untimed.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_TPL=1 TPL_LAYER_KEYOUT=1 DIAG27_REFUSAL_MEMO=1 DIAG27_SECTIONS_ON=1"
for part in ${PARTS:-a b}; do
  echo "$(date +%T) part $part"
  ./gpulock.sh 3 3600 env $LIFT HEAD=1 HEAD_WT=$L/sec_snap ARM=V GDN=triton PART=$part INTEG_GPU_UTIL=0.6 LF_GPU=3 LF_CPUS=108-143 RUN_TIMEOUT=1120 ./run27wf.sh secchk_$part chk > logs/secchk_$part.log 2>&1
  echo "$(date +%T) part $part rc=$?"
done
