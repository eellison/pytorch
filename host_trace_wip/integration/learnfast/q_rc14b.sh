#!/usr/bin/env bash
# 10-14 GPU 3, lowest priority (gpulock.sh yields to any other lane's waiter): mrc_chk (MiMo --check, full stand-ins) ->
# rc9l / rc9lc (9B, LAZY static prefix)
cd $(dirname $0)
RC="DIAG27_REDOCTX=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143"
./gpulock.sh 3 36000 env $RC REDOCTX_LAZY=0 RUN_TIMEOUT=1120 ./mimorc.sh mrc_chk chk > logs/mrc_chk.log 2>&1
echo done >> logs/mrc_chk.log
M9="ARMV_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 DIAG27_REDOCTX_LAZY=1 DIAG27_SECTIONDIAG=1 DIAG27_OPSPAN=1 $RC"
./gpulock.sh 3 36000 env $M9 LEARN_TIMEOUT=900 ./run27m.sh rc9l 60 140 > logs/rc9l.log 2>&1
echo done >> logs/rc9l.log
./gpulock.sh 3 36000 env $M9 CHECK_TIMEOUT=1100 ./run27checkm.sh rc9lc "60 61 63 64 65 70 80 100 128 129 140" > logs/rc9lc.log 2>&1
echo done >> logs/rc9lc.log
