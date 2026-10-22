#!/usr/bin/env bash
# The learn-range stall: a single-request prefill of 1952 tokens retraced for 25+ min (c5h, Vc+VP). Prefill 1951 then 1952, with every
# HostTraceReplay note (capture failures that make ops eager and retrace) printed (ARMV_DIAG_NOTES), each call's path printed.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
IMA_BS="1" IMA_PL="1951 1952" IMA_ENV="ARMV_DIAG_NOTES=1 ARMV_DIAG_PATHS=1" IMA_TAG=p1952 IMA_UTIL=0.5 bash $R/run_ima_c12.sh > $R/logs/p1952.log 2>&1
grep -E "^HTNOTE|^PATH prefill|rc=" $R/logs/p1952.log | head -80
