#!/usr/bin/env bash
# 10-14 GPU 3 chain (each step through gpulock.sh): rc27a (27B learn 1..400, redo_context) -> rcmc (9B memcheck 60..62,
# redo_context) -> mrc_dc / mrc_chk (MiMo; after the MiMo lane's vc_long1, and not while its diag_dc8k / vc_long1 is queued).
cd $(dirname $0)
M=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/trtllm_moe_cpp
RC="DIAG27_REDOCTX=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143"
./gpulock.sh 3 7200 env ARMV_SECTIONS=1 DIAG27_OPSPAN=1 $RC LEARN_TIMEOUT=1100 ./run27.sh rc27a 1 400 > logs/rc27a.log 2>&1
echo done >> logs/rc27a.log
M9="ARMV_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 DIAG27_OPSPAN=1 $RC"
./gpulock.sh 3 7200 env SANITIZE="/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --report-api-errors no --target-processes all --print-limit 50 --show-backtrace device --log-file $PWD/logs/rcmc_memcheck.%p.log" $M9 LEARN_TIMEOUT=1100 ./run27s.sh rcmc 60 62 > logs/rcmc.log 2>&1
echo done >> logs/rcmc.log
until grep -q "vc_long1 rc=" $M/logs/vc_long1.log 2>/dev/null; do sleep 60; done
ahead() { ps -eo args | grep -v grep | grep -q "gpu3.sh \(diag_dc8k\|vc_long1\) "; }
for job in "mrc_dc dc" "mrc_chk chk"; do
  set -- $job
  while ahead; do sleep 60; done
  ./gpulock.sh 3 14400 env $RC RUN_TIMEOUT=1120 ./mimorc.sh $1 $2 > logs/$1.log 2>&1
  echo done >> logs/$1.log
done
