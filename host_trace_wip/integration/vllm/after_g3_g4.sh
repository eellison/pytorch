#!/usr/bin/env bash
# After g3 (after_check_g3.sh done): g4 = 6 gated rounds of round_ix2_line (1k1k-shaped and 8k1k prompt-tail mixed) on attn, NUMBERS,
# then (after the mixed profile too) the Vc diagnostic job.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q AFTER_DONE vllm/logs/after_check_g3.out 2>/dev/null; do sleep 60; done
STOCK_LINE=attn SET=g4 ROUND_SCRIPT=round_ix2_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_g4.out 2>&1
STOCK_MAIN_SETS=g3,g4 STOCK_SETS=g1 bash vllm/mk_numbers.sh
echo G4_DONE
while ! grep -q ALLDONE vllm/logs/q_prof_mixed_gpu0_c2.log 2>/dev/null; do sleep 60; done
JOBLIST="vc_gpu0_c2" bash vllm/queue_gpu0.sh > vllm/logs/queue_vc.log 2>&1
echo VC_DONE
