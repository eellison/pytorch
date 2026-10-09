#!/usr/bin/env bash
# After g3: g4 -> NUMBERS -> Qwen3.5-9B check (cubin fetch once, then NO_DOWNLOAD) -> q1 = 6 gated Qwen3.5 rounds -> NUMBERS -> Vc diagnostic.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q AFTER_DONE vllm/logs/after_check_g3.out 2>/dev/null; do sleep 60; done
STOCK_LINE=attn SET=g4 ROUND_SCRIPT=round_ix2_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_g4.out 2>&1
STOCK_MAIN_SETS=g3,g4 STOCK_SETS=g1 bash vllm/mk_numbers.sh
echo G4_DONE
JOBLIST="q35_check_gpu0" bash vllm/queue_gpu0.sh > vllm/logs/queue_q35_check.log 2>&1
echo Q35_CHECK_DONE
if [ -f vllm/out/q35_vstock_Vcheck.json ] && [ -f vllm/out/q35_default_smoke.json ]; then
  STOCK_LINE=trtcpp SET=q1 ROUND_SCRIPT=round_q35_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_q1.out 2>&1
  STOCK_MAIN_SETS=g3,g4,q1 STOCK_SETS=g1 bash vllm/mk_numbers.sh
  echo Q1_DONE
else
  echo "Q35 check incomplete: q1 skipped"
fi
while ! grep -q ALLDONE vllm/logs/q_prof_mixed_gpu0_c2.log 2>/dev/null; do sleep 60; done
JOBLIST="vc_gpu0_c2" bash vllm/queue_gpu0.sh > vllm/logs/queue_vc.log 2>&1
echo VC_DONE
