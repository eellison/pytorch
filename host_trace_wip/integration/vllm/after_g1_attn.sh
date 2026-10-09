#!/usr/bin/env bash
# After g1: V-stock bitwise counts on CAND_LINE=attn (one hold), then g2 = 6 load-gated rounds of round_stock_line on CAND_LINE=attn
# (decode + prefill + mixed, all post-fix), then NUMBERS with both sets.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q ALLDONE vllm/logs/rounds_g1.out; do sleep 60; done
JOBLIST="attn_check_gpu0" bash vllm/queue_gpu0.sh > vllm/logs/queue_attn_check.log 2>&1
STOCK_LINE=attn SET=g2 ROUND_SCRIPT=round_stock_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_g2.out 2>&1
STOCK_SETS=g1,g2 bash vllm/mk_numbers.sh
echo AFTER_DONE
