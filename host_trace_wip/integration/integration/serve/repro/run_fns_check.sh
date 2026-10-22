#!/usr/bin/env bash
# fns: Flash-Next V-stock --check under ARMV_SOUND=1 (only cuBLAS harvested: top-k, quant, KV writes and every other open op traced or
# eager), armV_stock_m, prefill/mixed eager (CORE_TASKS (ix)); decode at short KV (bs 1, 16) and long KV (bs 8 at 6000-8000).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_KV_BYTES=32212254720 INTEG_GPU_UTIL=0.85 FLASHINFER_WORKSPACE_BASE=$PWD/serve/fi_ws_ht
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 240000 ]; do sleep 60; done
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false, "kv_cache_dtype": "auto"}'
W=(--model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 16 --prefill-lens 64 --mixed 0
   --ix-decode 8:6000-8000 --ix-reps 4 --llm-kw "$KW")
env CAND_LINE=c4 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 \
    ARMV_HOST_ARGS=pinned ARMV_EAGER_KINDS=prefill,mixed ARMV_CUTE_OBSERVE=1 ARMV_SOUND=1 INTEG_ARMV=$PWD/serve/armV_stock_m \
    bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm V --check "${W[@]}" --out serve/out/fns_check.json
echo "fns rc=$?"
python3 - <<'PY'
import json, collections
d = json.load(open("serve/out/fns_check.json")); V = d["V"]
print("FNS", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "learned", V["learned_by_op"], "unsound", V.get("unsound_learned"))
ch = V["checks"]
print("FNS_CHECKS", len(ch), sum(bool(c.get("bitwise")) for c in ch), dict(collections.Counter((str(c.get("shape")), c.get("bitwise")) for c in ch)))
for k, e in V["entries"].items():
    print("FNS_ENTRY", k, {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [(pv["eager_calls"], pv["segments"]) for pv in e["per_variant"]][:8], collections.Counter(op[:60] for pv in e["per_variant"][:1] for op in pv["eager_ops"] for _ in range(pv["eager_ops"][op])).most_common(12))
PY
