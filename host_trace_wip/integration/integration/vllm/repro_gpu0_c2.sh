#!/usr/bin/env bash
# run_buffer liveness repro (standalone torch, no vLLM) on the cpp candidate 2 line: run_buffer, then eager.
source $(dirname $0)/timing_lib.sh
for m in run_buffer eager; do
  echo "$(date +%T) repro $m"
  CAND_LINE=cpp timeout 900 bash python_vllm_cand2.sh vllm/probe/runbuffer_nan_repro.py $m > vllm/logs/runbuffer_repro_$m.log 2>&1
  echo "rc=$?"
done
echo ALLDONE
