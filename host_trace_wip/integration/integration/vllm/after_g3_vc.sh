#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q AFTER_DONE vllm/logs/after_check_g3.out 2>/dev/null || ! grep -q ALLDONE vllm/logs/q_prof_mixed_gpu0_c2.log 2>/dev/null; do sleep 60; done
JOBLIST="vc_gpu0_c2" bash vllm/queue_gpu0.sh > vllm/logs/queue_vc.log 2>&1
echo VC_DONE
