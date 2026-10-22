#!/usr/bin/env bash
cd $(dirname $0)
while ! grep -q "^done" logs/rc9c.log 2>/dev/null; do sleep 30; done
flock -w 7200 -x ../../scratch/eager/gpu3.lock timeout 1150 env ARMV_SECTIONS=1 DIAG27_REDOCTX=1 DIAG27_OPSPAN=1 DIAG27_FNRESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 LF_GPU=3 LF_CPUS=108-143 LEARN_TIMEOUT=1100 ./run27.sh rc27a 1 400 > logs/rc27a.log 2>&1
echo done >> logs/rc27a.log
