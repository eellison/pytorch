#!/usr/bin/env bash
# bench/run_arms.sh [arms...]: GPU 2 under gpu2.lock, one lock hold per arm. TAG (default qwen3_8b) prefixes outputs; arm V
# (armV/adapter.py) runs through python_vllm_v.sh; arm Vcpp is arm V through python_vllm_cpp.sh (install_sgl_cpp). DRIVE_ARGS are passed to drive.py (e.g. --stamps).
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
TAG=${TAG:-qwen3_8b}
for a in ${@:-eager default}; do
  echo "=== $a $(date)"
  [ $a = default ] && export HOSTTRACE_TORCH_DISABLE_CACHES=0  # vLLM AOT compile requires Inductor caches
  CUDA_VISIBLE_DEVICES=2 flock /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock taskset -c ${CPUS:-108-143} \
    timeout 2400 bash $(case $a in V) echo python_vllm_v.sh;; Vcpp) echo python_vllm_cpp.sh;; *) echo python_vllm.sh;; esac) bench/drive.py --arm ${a/Vcpp/V} ${DRIVE_ARGS:-} --out bench/out/${TAG}_$a.json > bench/logs/${TAG}_$a.log 2>&1
  echo "rc=$? $(date)"
done
