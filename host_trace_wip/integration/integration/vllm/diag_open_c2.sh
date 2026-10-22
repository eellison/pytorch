#!/usr/bin/env bash
# GPU 1: which half of the open variant gives the NaN decode outputs at bs 5/8 (c2_cpp_open_Vcheck: 193/228).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
d() {
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=1 flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock env "$@" CAND_LINE=cpp bash -c '
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | tr -d ,)
    u=$(python3 -c "print(0.6 if $used < 2048 else 0.16)"); echo "gpu1 used $used util $u"
    INTEG_GPU_UTIL=$u exec timeout 2400 taskset -c 72-107 bash python_vllm_cand2.sh vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/'$tag'.json' > vllm/logs/$tag.log 2>&1
  echo "$(date +%T) $tag rc=$?"
}
d diag_fork_only CAND_FORK=1 ARMV_TRTLLM_FORK=1
d diag_impls_only ARMV_TRACED_IMPLS=1
d diag_both CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_TRACED_IMPLS=1
echo ALLDONE
