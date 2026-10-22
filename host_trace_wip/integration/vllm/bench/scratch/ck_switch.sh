#!/usr/bin/env bash
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
while ! grep -q done bench/logs/tests_chain.log 2>/dev/null; do sleep 20; done
for cfg in on off; do
  if [ $cfg = on ]; then e="ARMV_LEND_GROUPS=1 ARMV_LEARNER_LEARNS=1"; else e="ARMV_LEND_GROUPS=0 ARMV_LEARNER_LEARNS=0"; fi
  env $e CUDA_VISIBLE_DEVICES=2 timeout 3000 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock taskset -c 108-143 bash python_vllm_cpp.sh bench/rangesweep.py --check --out bench/out/range/sw_$cfg.json \
    --dec-bs 1 2 3 8 33 --pre-lens 7 16 17 32 100 257 --pre-multi 2x8 3x500 --mixed-nd 1 3 --mixed-lens 1 17 42 129 500 > bench/logs/range_sw_$cfg.log 2>&1
  echo "$cfg rc=$?" >> bench/logs/ck_switch.log
done
echo done >> bench/logs/ck_switch.log
