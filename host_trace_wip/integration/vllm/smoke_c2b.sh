#!/usr/bin/env bash
# Smoke for the new driver pieces (GPU 1, gpu1.lock per command): ARMV_PAD=1 with --check, startup.py save/load, default startup.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
cd $G
run() {
  local tag=$1; shift; local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=1 flock -x $LOCK env "${envs[@]}" INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.5} timeout 2400 taskset -c 72-107 bash python_vllm_cand2.sh "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$?"
}
while pgrep -f "vllm/smoke_c2.sh" >/dev/null; do sleep 30; done
run smoke_pad2 CAND_LINE=cpp ARMV_PAD=1 -- vllm/bench/drive.py --arm V --check --batch-size 5 64 --prefill-lens 100 600 --mixed 2 --mixed-spec 4x100 --out vllm/out/smoke_pad2.json
echo ALLDONE
