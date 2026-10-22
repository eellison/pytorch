#!/usr/bin/env bash
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock
cd $V
for t in replay opaque capture; do
  for arm in cur pre; do
    if [ $arm = cur ]; then sh=python_vllm_cpp.sh; d=src_vllmcpp; else sh=bench/scratch/py_pre.sh; d=bench/scratch/wt_pre; fi
    CUDA_VISIBLE_DEVICES=2 timeout 2400 flock $L taskset -c 108-143 bash $sh $d/test/test_cuda_host_trace_$t.py > bench/logs/t_${t}_$arm.log 2>&1
    echo "$t $arm rc=$? $(grep -E '^(Ran|OK|FAILED)' bench/logs/t_${t}_$arm.log | tr '\n' ' ')" >> bench/logs/tests_chain.log
  done
done
echo done >> bench/logs/tests_chain.log
