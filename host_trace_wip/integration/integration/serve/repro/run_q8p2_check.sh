#!/usr/bin/env bash
# q8p2: Qwen3-8B V-stock --check, ARMV_SOUND + ARMV_PORTS=1 with vllm/armV_stock_fast_m2 (ports installed and registered at the
# _C ops and the vLLM IR ops; a declined port = an eager step, logged VLLM_PORT_DECLINE). Sized to co-run beside another user's job
# (fixed 28 GiB KV, util 0.15): decode bs 1/8/64 short, bs 8 at KV 6000-8000, mixed at 1-2k (64 decoders) and 6-8k (16 decoders).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.15 INTEG_KV_BYTES=30064771072
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 52000 ]; do sleep 60; done
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 8 64 --prefill-lens 64 512 --mixed 0
   --ix-mixed 1024x64@1024-2048 2048x64@1024-2048 1024x16@6000-8000 2048x16@6000-8000 --ix-decode 8:6000-8000 --ix-reps 3)
env CAND_LINE=c4 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m2 bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm V --check "${W[@]}" --out serve/out/q8p2_check.json 2>&1 | grep -v "^INFO\|^(EngineCore\|WARNING" | grep "VLLM_PORT_DECLINE\|Error\|Traceback" | sort | uniq -c | head -20
echo "q8p2 rc=${PIPESTATUS[0]}"
python3 - <<'PY'
import json, collections
d = json.load(open("serve/out/q8p2_check.json")); V = d["V"]
print("Q8P2", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "learned", V["learned_by_op"], "unsound", V.get("unsound_learned"))
ch = V["checks"]
print("Q8P2_CHECKS", len(ch), sum(bool(c.get("bitwise")) for c in ch), dict(collections.Counter((c.get("kind"), str(c.get("shape")), c.get("bitwise")) for c in ch)))
for k, e in V["entries"].items():
    print("Q8P2_ENTRY", k[:12], {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [(pv["eager_calls"], pv["segments"]) for pv in e["per_variant"]][:8], e.get("eager_reasons", [])[:6])
PY
