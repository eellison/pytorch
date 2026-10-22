#!/usr/bin/env bash
# After vp_check_c2.sh: VP with the index-select sampler tail (no eager gather): --check and end-to-end tokens.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q "^ALLDONE" vllm/logs/vp_check_c2b.log; do sleep 60; done
g1() {
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock env CAND_LINE=cpp ARMV_VP=1 INTEG_GPU_UTIL=0.16 \
    timeout 2400 taskset -c 72-107 bash python_vllm_cand2.sh "$@" > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) $tag rc=$?"
}
g1 c2_cpp_VP2_Vcheck vllm/bench/drive.py --arm V --check --batch-size 1 8 64 128 --prefill-lens 64 --mixed 2 --out vllm/out/c2_cpp_VP2_Vcheck.json
g1 e2e_VP2 vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_VP2.json
echo ALLDONE
