#!/usr/bin/env bash
# Out-of-graph breakdown (one profiled full step per class: decode bs 1, decode bs 64, mixed 1024x64) for each arm on c5g, warm
# caches, idle correctness GPU; traces in serve/out/prof_bd/, tables via serve/ix/prof_breakdown.py.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.5 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_PROFILE_DIR=$PWD/serve/out/prof_bd DRIVE_PROFILE_CLASSES=1:1,64:64,65:1088 DRIVE_PROFILE_AT=6
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 250000 ]; do sleep 60; done
L=c5g
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4)
W=(--max-model-len 16384 --batch-size 1 64 --prefill-lens 64 --mixed 0 --ix-mixed 1024x64@1024-2048 --ix-reps 8)
run() {  # tag arm ENV...
  tag=$1 arm=$2; shift 2; echo "== $tag"
  env DRIVE_PROFILE_TAG=$tag "$@" timeout 1500 bash python_vllm_cand2.sh vllm/bench/drive_ix2.py --arm $arm "${W[@]}" --out serve/out/prof_bd_$tag.json > serve/logs/prof_bd_$tag.log 2>&1
  echo "$tag rc=$?"
}
run default default CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0
run fullnc fullnc CAND_LINE=$L
run vreg V "${BASE[@]}"
run vc Vc "${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1
run vcs Vc "${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1
ls serve/out/prof_bd
python3 serve/ix/prof_breakdown.py serve/out/prof_bd > serve/out/prof_bd_tables.md; head -60 serve/out/prof_bd_tables.md
