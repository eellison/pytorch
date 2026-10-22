#!/usr/bin/env bash
# 27B --check with closure + redo_context (cc27a's flags), c27s's prefill lengths; lowest GPU 3 priority
cd $(dirname $0)
CC="DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_FNRESOLVE=1 DIAG27_OPSPAN=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143"
./gpulock.sh 3 36000 env ARMV_SECTIONS=1 $CC CHECK_TIMEOUT=1100 ./run27check.sh cc27c "2 3 5 7 9 17 33 65 100 129 257 513" > logs/cc27c.log 2>&1
echo done >> logs/cc27c.log
