#!/usr/bin/env bash
# report-only (10-14): vLLM breakable's PIECEWISE prefill (Qwen GDN captured inside its segments) vs default-nc's eager prefill,
# one-request prefills of lengths 33-80, top-20 logprobs (Triton GDN: deterministic). The KID records show breakable replaying the
# GDN decode kernels (fused_recurrent, causal_conv1d_update) for prefill / mixed steps <= 1024 tokens. GPU 1, one hold per run.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=1
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=$S TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
for arm in breakable fullnc; do
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start brk $arm"
  [ $arm = breakable ] && B=1 || B=0
  env VLLM_USE_BREAKABLE_CUDAGRAPH=$B CUDA_VISIBLE_DEVICES=$G timeout 1200 taskset -c 72-107 bash $R/python_r27.sh $R/bench/brk_bitwise.py $R/out/brk_bitwise_$arm.json $arm > $R/logs/brk2_$arm.log 2>&1
  echo "$(date +%T) end brk $arm rc=$?"
  exec 9>&-
done
python3 $R/bench/brk_bitwise.py --compare $R/out/brk_bitwise_breakable.json $R/out/brk_bitwise_fullnc.json
