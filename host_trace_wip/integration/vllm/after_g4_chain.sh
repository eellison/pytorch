#!/usr/bin/env bash
# After g4 rounds 2-6 (rounds_gpu0 pid $1, started by after_reboot_chain.sh): NUMBERS -> default-at-1K-chunk profile (one hold) ->
# Qwen3.5-9B (fetch once + check, trtcpp) -> q1 -> NUMBERS -> Vc diagnostic.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 30; done
STOCK_MAIN_SETS=g3,g4 STOCK_SETS=g1 bash vllm/mk_numbers.sh
echo G4_DONE
JOBLIST="prof_ix1k_gpu0_c2" bash vllm/queue_gpu0.sh > vllm/logs/queue_prof_ix1k.log 2>&1
echo PROF_IX1K_DONE
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
