# lib_g3.sh (source): GPU 3 lane helpers after the 10-17 reboot. GPU 3 / CPUs 108-143 / gpu3.lock (exclusive while we run).
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
CB=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/cpubusy.py
export LF_GPU=3 LF_CPUS=108-143
gates() {  # own-core (< 0.30 busy on 108-143) and PSI full avg60 (< 2%) gates, before the lock; up to 30 / 60 min
  local lw=0 pw=0 own p60
  while own=$(python3 $CB 10 108 143 | cut -d' ' -f1); python3 -c "import sys; sys.exit(not ($own / 36 >= 0.30))" && [ $lw -lt 1800 ]; do sleep 20; lw=$((lw + 30)); done
  p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory)
  while python3 -c "import sys; sys.exit(not ($p60 >= 2))" && [ $pw -lt 3600 ]; do sleep 30; pw=$((pw + 30)); p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory); done
  echo "$(date +%T) GATES own_busy $(python3 -c "print(round($own / 36, 3))") (waited ${lw}s) psi_full_avg60 $p60 (waited ${pw}s)"
}
cudaerr() { grep -qiE "CUDA error|illegal instruction|illegal memory access|cudaErrorIllegal|Xid|misaligned address" "$@"; }
