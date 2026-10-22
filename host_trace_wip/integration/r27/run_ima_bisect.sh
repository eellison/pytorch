#!/usr/bin/env bash
# IMA bisect (c5h Vc+VP decode): bs 6 alone; bs 5 6 with VP off (Vc); bs 5 6 with VP on and --check
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
IMA_BS="6" IMA_TAG=ima_bs6 bash $R/run_ima_c12.sh 2>&1 | grep -E "^decode|illegal|rc=" ; echo ---
IMA_BS="5 6" IMA_ENV="ARMV_VP=0" IMA_TAG=ima_novp bash $R/run_ima_c12.sh 2>&1 | grep -E "^decode|illegal|rc="; echo ---
IMA_BS="5 6" IMA_CHECK=1 IMA_TAG=ima_check bash $R/run_ima_c12.sh > $R/logs/ima_check.log 2>&1; grep -E "^decode|illegal|rc=" $R/logs/ima_check.log
