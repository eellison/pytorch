#!/usr/bin/env bash
# 10-14 provenance-criterion opt-in (prov_redo.py), lowest GPU 3 priority. Sections stay refused (reported configs):
# MiMo adapter's ARMV_REFUSE_SECTIONS default, r27 adapter's no_sections. Section memo on (pure optimization; equality
# check ccm_chk2m vs ccm_chk2 is diagnostic, sections on).
cd $(dirname $0)
G="LF_GPU=3 LF_CPUS=108-143"
P="DIAG27_PROV=1 PROV_SECTION_MEMO=1 $G"
PC="DIAG27_PROV=1 PROV_SECTION_MEMO=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_SET=redo_resolves_arguments=1 $G"
run() { tag=$1; shift; ./gpulock.sh 3 36000 env "$@" > logs/$tag.log 2>&1; echo done >> logs/$tag.log; }
run mpc_chk $PC RUN_TIMEOUT=1120 ./mimorc.sh mpc_chk chk
run mp_chk $P RUN_TIMEOUT=1120 ./mimorc.sh mp_chk chk
run mpc_dc $PC RUN_TIMEOUT=1120 ./mimorc.sh mpc_dc dc
run mp_dc $P RUN_TIMEOUT=1120 ./mimorc.sh mp_dc dc
run mb_dc $G RUN_TIMEOUT=1120 ./mimorc.sh mb_dc dc
run pc27a ARMV_SECTIONS=1 $PC LEARN_TIMEOUT=1100 ./run27.sh pc27a 1 400
run p27a ARMV_SECTIONS=1 $P LEARN_TIMEOUT=1100 ./run27.sh p27a 1 400
run pc27c ARMV_SECTIONS=1 $PC CHECK_TIMEOUT=1100 ./run27check.sh pc27c "2 3 5 7 9 17 33 65 100 129 257 513"
CC="DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_CUTE_RESOLVE=1 DIAG27_FNRESOLVE=1 DIAG27_OPSPAN=1 DIAG27_SECTIONDIAG=1 DIAG27_SET=redo_resolves_arguments=1 DIAG27_SECTION_MEMO=1 $G"
run ccm_chk2m $CC REDOCTX_LAZY=1 RUN_TIMEOUT=1120 ./mimorc.sh ccm_chk2m chk
