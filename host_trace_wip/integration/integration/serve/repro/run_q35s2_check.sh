#!/usr/bin/env bash
# q35s2 (control: ARMV_CHECK_TWICE=1, eager vs eager per check; q35s had prefill/mixed 16384-token rows not bitwise): Qwen3.5-9B V-stock --check under ARMV_SOUND=1 (only cuBLAS harvested; open vLLM _C ops as eager steps) with pinned host args
# and armV_stock_fast_cpp_m, covering long KV (decode and mixed at 6000-8000) as well as short.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.6
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 200000 ]; do sleep 60; done
W=(--model /data/eellison/models/Qwen3.5-9B --trust-remote-code --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0
   --ix-mixed 8192x32@4096-8000 2048x64@6000-8000 --ix-reps 3)
env CAND_LINE=trtcpp ARMV_TRTLLM_CPP=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_HOST_ARGS=pinned \
    ARMV_SOUND=1 ARMV_CHECK_TWICE=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_cpp_m bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm V --check "${W[@]}" --out serve/out/q35s2_check.json
echo "q35s2 rc=$?"
python3 - <<'PY'
import json, collections
d = json.load(open("serve/out/q35s2_check.json")); V = d["V"]
print("Q35S2", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "learned", V["learned_by_op"], "unsound", V.get("unsound_learned"))
ch = V["checks"]
print("Q35S2_TWICE", collections.Counter((c["kind"], str(c["shape"]), c["bitwise"], c.get("eager_twice_bitwise")) for c in ch))
print("Q35S2_CHECKS", len(ch), sum(bool(c.get("bitwise")) for c in ch), dict(collections.Counter((c.get("kind"), str(c.get("shape")), c.get("bitwise")) for c in ch)))
for k, e in V["entries"].items():
    print("Q35S2_ENTRY", k, {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [(pv["eager_calls"], pv["segments"]) for pv in e["per_variant"]][:8], list(collections.Counter(op for pv in e["per_variant"] for op in pv["eager_ops"]).items())[:8])
PY
