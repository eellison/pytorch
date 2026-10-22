#!/usr/bin/env bash
# GPU 1 (gpu1.lock per command): arm VP --check, then end-to-end greedy tokens for V / VP / default / VAf.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
g1() {
  local tag=$1; shift; local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "$(date +%T) start $tag"
  CUDA_VISIBLE_DEVICES=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock env "${envs[@]}" bash -c '
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | tr -d ,)
    u=$(python3 -c "print(0.6 if $used < 2048 else 0.16)"); echo "gpu1 used $used util $u"
    INTEG_GPU_UTIL=$u exec timeout 2400 taskset -c 72-107 bash python_vllm_cand2.sh "$@"' _ "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$?"
}
g1 c2_cpp_VP_Vcheck CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/drive.py --arm V --check --batch-size 1 8 64 128 --prefill-lens 64 --mixed 2 --out vllm/out/c2_cpp_VP_Vcheck.json
g1 e2e_VP CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_VP.json
g1 e2e_V CAND_LINE=cpp -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_V.json
g1 e2e_VAf CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/e2e_tokens.py --arm VAf --out vllm/out/e2e_VAf.json
g1 e2e_default CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/e2e_tokens.py --arm default --out vllm/out/e2e_default.json
echo ALLDONE
