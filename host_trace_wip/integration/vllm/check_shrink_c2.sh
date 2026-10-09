#!/usr/bin/env bash
# GPU 1 (gpu1.lock): the fluctuating-batch workload with --check (V cpp closed), after run_c2s1.sh.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while pgrep -f "vllm/run_c2s1.sh" > /dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock env CAND_LINE=cpp bash -c '
  read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | tr -d ,)
  u=$(python3 -c "print(0.6 if $used < 2048 else 0.16)"); echo "gpu1 used $used util $u"
  INTEG_GPU_UTIL=$u exec timeout 3600 taskset -c 72-107 bash python_vllm_cand2.sh vllm/bench/drive.py --arm V --check --batch-size 1 --prefill-lens 64 --mixed 0 --shrink 64 128 --out vllm/out/c2_cpp_closed_shrink_Vcheck.json' > vllm/logs/c2_cpp_closed_shrink_Vcheck.log 2>&1
echo "$(date +%T) done rc=$?"
