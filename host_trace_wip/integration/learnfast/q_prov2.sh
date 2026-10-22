#!/usr/bin/env bash
# 10-14 provenance criterion, round 2 (lowest GPU 3 priority; sections refused as reported configs):
#   mpc_m: criterion + closure, --check, bs 1 2 3 (moe_forward's confirmation after the aliased-declared tie-break)
#   mb_chk: baseline (adapters as shipped), --check, the 11 points, same harness (traces, warm-up seconds)
#   ccm_chk2m: diagnostic: ccm_chk2 + section memo (equality vs ccm_chk2)
cd $(dirname $0)
G="LF_GPU=3 LF_CPUS=108-143"
PC="DIAG27_PROV=1 PROV_SECTION_MEMO=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 $G"
run() { tag=$1; shift; ./gpulock.sh 3 36000 env "$@" > logs/$tag.log 2>&1; echo done >> logs/$tag.log; }
run mpc_m $PC BS="1 2 3" RUN_TIMEOUT=1000 ./mimorc.sh mpc_m chk
run mb_chk $G RUN_TIMEOUT=1120 ./mimorc.sh mb_chk chk
CC="DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_FNRESOLVE=1 DIAG27_OPSPAN=1 DIAG27_SECTIONDIAG=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_SECTION_MEMO=1 $G"
run ccm_chk2m $CC REDOCTX_LAZY=1 RUN_TIMEOUT=1120 ./mimorc.sh ccm_chk2m chk
# memo-off twin of ccm_chk2m on the same code (the closure tie-break changed after ccm_chk2)
CC0="DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_FNRESOLVE=1 DIAG27_OPSPAN=1 DIAG27_SECTIONDIAG=1 DIAG27_SET=redo_resolves_arguments=1 $G"
run ccm_chk2r $CC0 REDOCTX_LAZY=1 RUN_TIMEOUT=1120 ./mimorc.sh ccm_chk2r chk
# moe_forward's confirmation with guards on declared inputs' symbols accepted (static-prefix contract)
run mpc_m2 $PC RUN_TIMEOUT=1120 ./mimorc.sh mpc_m2 chk
