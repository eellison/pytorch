#!/usr/bin/env bash
# 10-15 Delta 1 (lifted per-step state, the user's design): MiMo 11 decode points --check, sections refused (reported config)
cd $(dirname $0)
G=${G:-0}; CPUS=${CPUS:-0-35}; TAG=${TAG:-lm1}
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 ARMV_REFUSE_SECTIONS=1 LF_GPU=$G LF_CPUS=$CPUS"
[ "${BASE:-0}" = 1 ] && LIFT="LF_GPU=$G LF_CPUS=$CPUS"
STEP_OFF=1 ./gpulock.sh $G 36000 env $LIFT REDOCTX_LAZY=1 RUN_TIMEOUT=1120 ./mimorc.sh $TAG chk > logs/$TAG.log 2>&1
echo done >> logs/$TAG.log
