#!/usr/bin/env bash
# One GPU lock hold (GPU=0 or 2, default 2: the vLLM lane): test_trace (on and off) then the per-T xcheck. Log in armV/opsport/logs/unit_TAG.log.
# TS=all for the 1..8192 guard regions; SKIP_XCHECK=1 skips the xcheck.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
O=$V/armV/opsport
TAG=${TAG:-$(date +%H%M)}
GPU=${GPU:-2}
CPUS=$([ "$GPU" = 0 ] && echo 0-71 || echo 108-143)
cd $O
CUDA_VISIBLE_DEVICES=$GPU flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu$GPU.lock taskset -c $CPUS bash -c "
  echo start \$(date -Is)
  timeout 3000 bash $V/python_vllm_cpp.sh $O/test_trace.py $* off
  [ -n \"\$SKIP_XCHECK\" ] || timeout 3600 bash $V/python_vllm_cpp.sh $O/xcheck_params.py $*
  echo end \$(date -Is)
" > $O/logs/unit_$TAG.log 2>&1
