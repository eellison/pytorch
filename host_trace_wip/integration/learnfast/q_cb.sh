#!/usr/bin/env bash
# MiMo cb prewarm (validate_fix config, build_fix): Delta 1 vs baseline, C=16 and C=64 each in its own process, GPU 0
cd $(dirname $0)
G=0; CPUS=0-35
LIFT="DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1"
for spec in "cbl16 16 1" "cbb16 16 0" "cbl64 64 1" "cbb64 64 0"; do
  set -- $spec
  E="LF_GPU=$G LF_CPUS=$CPUS"; [ "$3" = 1 ] && E="$LIFT $E"
  STEP_OFF=1 ./gpulock.sh $G 36000 env $E RUN_TIMEOUT=1150 ./cbrc.sh $1 $2 > logs/$1.log 2>&1
  echo done >> logs/$1.log
done
