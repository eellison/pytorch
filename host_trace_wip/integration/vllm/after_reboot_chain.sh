#!/usr/bin/env bash
# Restart after the 10-09 ~11:45 reboot: g3 had finished (6/6 rounds), g4 had round 1. g4 rounds 2-6 -> NUMBERS -> Qwen3.5-9B
# (cubin fetch once + check, C++ path) -> q1 (6 gated rounds, trtcpp) -> NUMBERS -> Vc diagnostic. The attn-line armV_fast profile
# re-run (fast_prof2) had finished before the reboot.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
ROUND_START=2 STOCK_LINE=attn SET=g4 ROUND_SCRIPT=round_ix2_line bash vllm/rounds_gpu0.sh >> vllm/logs/rounds_g4.out 2>&1
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
JOBLIST="vc_gpu0_c2" bash vllm/queue_gpu0.sh > vllm/logs/queue_vc.log 2>&1
echo VC_DONE
