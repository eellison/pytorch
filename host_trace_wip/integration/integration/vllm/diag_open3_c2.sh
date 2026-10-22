#!/usr/bin/env bash
# After diag_open2: bisect the ports under the fork: the KV writer alone, and the four others without it.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until grep -q "^ALLDONE" vllm/logs/diag_open2_c2.log; do sleep 60; done
d() {
  local tag=$1 names=$2
  CUDA_VISIBLE_DEVICES=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock env CAND_LINE=cpp CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_TRACED_IMPLS=1 \
    ARMV_TRACED_IMPLS_NAMES=$names INTEG_GPU_UTIL=0.16 timeout 2400 taskset -c 72-107 bash python_vllm_cand2.sh vllm/bench/drive.py --arm V --check --batch-size 5 8 64 \
    --prefill-lens 64 --mixed 0 --out vllm/out/$tag.json > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) $tag rc=$?"
}
d diag_fork_cache reshape_and_cache_flash
d diag_fork_nocache rms_norm,fused_add_rms_norm,rotary_embedding,silu_and_mul
echo ALLDONE
