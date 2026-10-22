#!/usr/bin/env bash
# q_gpu0.sh: 27B startup breakdown on GPU 0 (gpu0.lock, load gate before the lock): vLLM breakable+gdnbreak and fullnc on
# HEAD=2 (step_ms12 / build_cpp_ms9), Triton GDN line, max_num_seqs 128, KV 64 GiB, the 800-step stream.
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
cd $L
export HEAD=2 LF_GPU=0 LF_CPUS=0-35 HOLD_S=1150
for arm in brk fullnc; do
  w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge 60 ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
  echo "$(date +%T) $arm load $(cut -d' ' -f1 /proc/loadavg)"
  ./gpulock.sh 0 3600 ./startup27.sh run $arm 1 > logs/st27h2_${arm}_r1.log 2>&1
  echo "$(date +%T) $arm rc=$?"
done
