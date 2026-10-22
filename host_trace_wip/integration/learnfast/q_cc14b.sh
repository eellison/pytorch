#!/usr/bin/env bash
# after ccm_chk: ccm_chk2 = ccm_chk with the MiMo adapter's interim section refusal lifted (ARMV_REFUSE_SECTIONS=0 via mimorc.sh)
cd $(dirname $0)
while ! grep -q "^done" logs/ccm_chk.log 2>/dev/null; do sleep 30; done
CC="DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_FNRESOLVE=1 DIAG27_OPSPAN=1 DIAG27_SECTIONDIAG=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143"
./gpulock.sh 3 36000 env $CC REDOCTX_LAZY=1 RUN_TIMEOUT=1120 ./mimorc.sh ccm_chk2 chk > logs/ccm_chk2.log 2>&1
echo done >> logs/ccm_chk2.log
