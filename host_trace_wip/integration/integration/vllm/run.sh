#!/usr/bin/env bash
# arm V on the int6 lines: drive.py (--check, then plain V and eager for memory) and the rs4 range sweep, even line first.
# GPU 1 only, gpu1.lock held per command. INTEG_GPU_UTIL: a foreign job holds ~250 GB of every GPU (overlay runs used 0.6).
#   setsid nohup bash vllm/run.sh [lines...] > vllm/logs/run.log 2>&1 &
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
cd $G
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.08}
run() {  # tag line script args...
  local tag=$1 line=$2; shift 2
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=1 flock -x $LOCK bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1; exec timeout 3600 taskset -c 72-107 "$@"' _ \
    bash python_vllm_int6_$line.sh "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$?"
}
for line in ${@:-even odd}; do
  run ${line}_Vcheck $line vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --out vllm/out/${line}_Vcheck.json
  run ${line}_V $line vllm/bench/drive.py --arm V --prefill-reqs 4 8 --out vllm/out/${line}_V.json
  run ${line}_eager $line vllm/bench/drive.py --arm eager --prefill-reqs 4 8 --out vllm/out/${line}_eager.json
  ARMV_STATIC_SHAPES=1 run ${line}_range $line vllm/bench/rangesweep.py --out vllm/out/${line}_range.json
done
echo ALLDONE
