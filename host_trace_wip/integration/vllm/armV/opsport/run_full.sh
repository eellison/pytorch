#!/usr/bin/env bash
# The full unit pass, one GPU lock hold per step (GPU 2 lane): test_trace over T 1..8192 (on), the per-T xcheck, then
# test_trace on the default T list on and off. Logs armV/opsport/logs/full_{all,xcheck,onoff}.log.
O=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/armV/opsport
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock
cd $O
run() { CUDA_VISIBLE_DEVICES=2 flock $L taskset -c 108-143 timeout 7000 bash $V/python_vllm_cpp.sh "$@"; }
TS=all run $O/test_trace.py > $O/logs/full_all.log 2>&1
run $O/xcheck_params.py > $O/logs/full_xcheck.log 2>&1
run $O/test_trace.py off > $O/logs/full_onoff.log 2>&1
echo FULLDONE > $O/logs/full_done
