#!/usr/bin/env bash
# IMA: bs 5 then 6 with --check and per-call / per-check prints (are bs-5 replays already wrong?)
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
IMA_BS="5 6" IMA_CHECK=1 IMA_ENV="ARMV_DIAG_PATHS=1" IMA_TAG=ima_check2 bash $R/run_ima_c12.sh > $R/logs/ima_check2.log 2>&1
grep -E "^decode|illegal|rc=|^CHECK|^PATH decode" $R/logs/ima_check2.log | tail -40
