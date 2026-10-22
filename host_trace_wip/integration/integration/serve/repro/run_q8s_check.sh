#!/usr/bin/env bash
# q8s: Qwen3-8B V-stock --check under ARMV_SOUND=1 (only cuBLAS harvested; open vLLM _C ops as eager steps) on candidate 4 + the
# trtllm fork (F1), armV_stock_fast_m; g4's mixed shapes (KV 1024-2048 and 6000-8000) plus long-KV decode.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.6
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 200000 ]; do sleep 60; done
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 8 64 --prefill-lens 64 512 --mixed 0
   --ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000
   --ix-decode 8:6000-8000 64:6000-8000 --ix-reps 3)
env CAND_LINE=c4 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_HOST_ARGS=pinned \
    ARMV_SOUND=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm V --check "${W[@]}" --out serve/out/q8s_check.json
echo "q8s rc=$?"
python3 - <<'PY'
import json, collections
d = json.load(open("serve/out/q8s_check.json")); V = d["V"]
print("Q8S", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "learned", V["learned_by_op"], "unsound", V.get("unsound_learned"))
ch = V["checks"]
print("Q8S_CHECKS", len(ch), sum(bool(c.get("bitwise")) for c in ch), dict(collections.Counter((c.get("kind"), str(c.get("shape")), c.get("bitwise")) for c in ch)))
for k, e in V["entries"].items():
    print("Q8S_ENTRY", k, {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [(pv["eager_calls"], pv["segments"]) for pv in e["per_variant"]][:8], list(collections.Counter(op for pv in e["per_variant"] for op in pv["eager_ops"]).items())[:8])
PY
