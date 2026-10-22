#!/usr/bin/env bash
# criterion + closure, the 11 points, --check, section memo OFF (the memo keyed by (variant, nodes) failed equality:
# ccm_chk2m 22 traces vs ccm_chk2r 8); every failing selector per attempt logged (unselected_ops)
cd $(dirname $0)
G="LF_GPU=3 LF_CPUS=108-143"
PC="DIAG27_PROV=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 $G"
./gpulock.sh 3 36000 env $PC RUN_TIMEOUT=1120 ./mimorc.sh mpc_m3 chk > logs/mpc_m3.log 2>&1
echo done >> logs/mpc_m3.log
