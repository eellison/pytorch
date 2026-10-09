#!/usr/bin/env bash
# Round 2: odd line drives + range, then the group-budget diagnostic ranges on both lines. Ranges at util 0.072 (the
# even default range at 0.08 hit 20 capture OOMs when the foreign job grew).
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
cd $G
run() {  # tag line script args...
  local tag=$1 line=$2; shift 2
  echo "$(date +%T) start $tag util $INTEG_GPU_UTIL"
  CUDA_VISIBLE_DEVICES=1 flock -x $LOCK bash -c 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1; exec timeout 3600 taskset -c 72-107 "$@"' _ \
    bash python_vllm_int6_$line.sh "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$?"
}
export INTEG_GPU_UTIL=0.08
run odd_Vcheck odd vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --out vllm/out/odd_Vcheck.json
run odd_V odd vllm/bench/drive.py --arm V --prefill-reqs 4 8 --out vllm/out/odd_V.json
run odd_eager odd vllm/bench/drive.py --arm eager --prefill-reqs 4 8 --out vllm/out/odd_eager.json
export INTEG_GPU_UTIL=0.072 ARMV_STATIC_SHAPES=1
run even_range_grp even vllm/bench/group_budget.py vllm/bench/rangesweep.py --out vllm/out/even_range_grp.json
run odd_range odd vllm/bench/rangesweep.py --out vllm/out/odd_range.json
run odd_range_grp odd vllm/bench/group_budget.py vllm/bench/rangesweep.py --out vllm/out/odd_range_grp.json
run even_range2 even vllm/bench/rangesweep.py --out vllm/out/even_range2.json
echo ALLDONE
