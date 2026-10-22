#!/usr/bin/env bash
# q_rd2.sh: RD2 under Delta 1 (coordinator 10-15, splicing fix item 2): Qwen3.5-9B learn range 60..62 (q9t's crash case: a GDN
# section made at 60, replayed at 61) with lifted per-step state + sections on (Delta 3 snap), under compute-sanitizer memcheck.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1 DIAG27_SECTIONS_ON=1"
M9="MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 LF_GPU=3 LF_CPUS=108-143 R27_WT_OVERRIDE=$L/sec_snap"
SAN=""
[ "${MEMCHECK:-1}" = 1 ] && SAN="/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --report-api-errors no --target-processes all --print-limit 50 --show-backtrace device --log-file $L/logs/${TAG:-rd2}_memcheck.%p.log"
HOLD_S=1150 ./gpulock.sh 3 3600 env SANITIZE="$SAN" $LIFT $M9 LEARN_TIMEOUT=1100 ./run27h.sh ${TAG:-rd2} ${LO:-60} ${HI:-62} > logs/${TAG:-rd2}.log 2>&1
echo "$(date +%T) rc=$?"
