#!/usr/bin/env bash
# IMA localization: bs 5 then 6, VP on, each hybrid call synced and named (ARMV_DIAG_PATHS); then the same with ARMV_VP_STOCK=0 (the VP rewrite, diagnosis only)
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
IMA_BS="5 6" IMA_ENV="ARMV_DIAG_PATHS=1" IMA_TAG=ima_paths bash $R/run_ima_c12.sh > $R/logs/ima_paths.log 2>&1; grep -E "^decode|illegal|rc=" $R/logs/ima_paths.log; grep "^PATH" $R/logs/ima_paths.log | tail -6
echo ---
IMA_BS="4 6" IMA_TAG=ima_4to6 bash $R/run_ima_c12.sh 2>&1 | grep -E "^decode|illegal|rc="
