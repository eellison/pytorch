#!/usr/bin/env bash
# Weekend: timing on GPU 0 only. After g4 rounds (pid $1): NUMBERS -> (once the GPU 3 Qwen3.5 V-stock check q35_vcheck2 is done) q1,
# 6 gated rounds on trtcpp with vllm/armV_stock_fast_cpp -> NUMBERS. Prints Q1_DONE / "q1 skipped" (after_q1_sweep.sh waits on it).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 30; done
STOCK_MAIN_SETS=g3,g4 STOCK_SETS=g1 bash vllm/mk_numbers.sh
echo G4_DONE
while ! grep -q ALLDONE vllm/logs/q3_q35_vcheck2.log 2>/dev/null; do sleep 60; done
if [ -f vllm/out/q35_vstock_Vcheck.json ] && [ -f vllm/out/q35_default_smoke.json ]; then
  STOCK_LINE=trtcpp SET=q1 ROUND_SCRIPT=round_q35_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_q1.out 2>&1
  STOCK_MAIN_SETS=g3,g4,q1 STOCK_SETS=g1 bash vllm/mk_numbers.sh
  echo Q1_DONE
else
  echo "Q35 check incomplete: q1 skipped"
fi
