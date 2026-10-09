#!/usr/bin/env bash
# diag_both again (its first run failed at vLLM init: a foreign job freed GPU 1 memory during profiling).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q "^ALLDONE" vllm/logs/diag_open3_c2.log; do sleep 60; done
bash vllm/gpu1_run.sh diag_both CAND_LINE=cpp CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_TRACED_IMPLS=1 -- vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/diag_both.json
echo ALLDONE
