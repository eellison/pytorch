#!/usr/bin/env bash
# After the attn-line V-stock checks (queue pid $1): g3 = 6 load-gated rounds of round_ix_line on CAND_LINE=attn, then NUMBERS.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 30; done
STOCK_LINE=attn SET=g3 ROUND_SCRIPT=round_ix_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_g3.out 2>&1
STOCK_SETS=g1,g3 bash vllm/mk_numbers.sh
echo AFTER_DONE
