#!/usr/bin/env bash
# Weekend: non-timing jobs on GPU 3 (gpu3_job.sh): the 1K-chunk dispatch profile -> Qwen3.5-9B fetch + check (trtcpp) -> Vc diagnostic.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
JOBLIST="prof_ix1k_gpu0_c2;q35_check_gpu0" bash vllm/queue_gpu3.sh > vllm/logs/queue3_a.log 2>&1
echo Q35_CHECK_DONE
JOBLIST="vc_gpu0_c2" bash vllm/queue_gpu3.sh > vllm/logs/queue3_vc.log 2>&1
echo VC_DONE
