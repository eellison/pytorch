#!/usr/bin/env bash
# Standalone repros of the range sweep's retrace causes (memcpy fold refusal, full-width slice) on candidate 3 (step246)
source $(dirname $0)/timing_lib.sh
CAND_LINE=c3 timeout 600 bash python_vllm_cand2.sh vllm/probe/retrace_class_repro.py > vllm/logs/retrace_class_repro.log 2>&1
echo "rc=$?"
echo ALLDONE
