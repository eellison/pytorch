#!/usr/bin/env bash
# 10-14 Flash-Next decode ladder (1K), baseline vs criterion + closure (prov_redo), --check, chunks of 32 bs (the 20 min
# hold); lowest GPU 3 priority (gpulock.sh yields to any other lane's waiter)
cd $(dirname $0)
G="LF_GPU=3 LF_CPUS=108-143"
PC="DIAG27_PROV=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 $G"
run() { tag=$1; shift; ./gpulock.sh 3 72000 env "$@" > logs/$tag.log 2>&1; echo done >> logs/$tag.log; }
run fnp_c1 $PC BS="$(seq -s ' ' 1 32)" RUN_TIMEOUT=1120 ./fnrc.sh fnp_c1 chk
run fnb_c1 $G BS="$(seq -s ' ' 1 32)" RUN_TIMEOUT=1120 ./fnrc.sh fnb_c1 chk
run fnp_c2 $PC BS="$(seq -s ' ' 33 64)" RUN_TIMEOUT=1120 ./fnrc.sh fnp_c2 chk
run fnp_c3 $PC BS="$(seq -s ' ' 65 96)" RUN_TIMEOUT=1120 ./fnrc.sh fnp_c3 chk
run fnp_c4 $PC BS="$(seq -s ' ' 97 128)" RUN_TIMEOUT=1120 ./fnrc.sh fnp_c4 chk
run fnb_c4 $G BS="$(seq -s ' ' 97 128)" RUN_TIMEOUT=1120 ./fnrc.sh fnb_c4 chk
# c1 again (both arms' first runs spent their holds on FlashInfer JIT rebuilds into fncache; now prebuilt on the CPU)
run fnp_c1b $PC BS="$(seq -s ' ' 1 32)" RUN_TIMEOUT=1120 ./fnrc.sh fnp_c1b chk
run fnb_c1b $G BS="$(seq -s ' ' 1 32)" RUN_TIMEOUT=1120 ./fnrc.sh fnb_c1b chk
# c2 again: fnp_c2's confirmation compared keyed-site nodes too (fixed in prov_redo.confirm)
run fnp_c2b $PC BS="$(seq -s ' ' 33 64)" RUN_TIMEOUT=1120 ./fnrc.sh fnp_c2b chk
