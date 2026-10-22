#!/usr/bin/env bash
# fnsmoke2: where the fnsmoke decode_16 (KV 6000-8000) mismatch comes from: decode bs 16 at short KV (--batch-size 16, 256-token
# prompts) vs bs 8 at long KV (--ix-decode 8:6000-8000); checks are told apart by batch (shape[0]).
# small workload, so the GPU 0 rounds do not discover a driver problem. fullnc arm: startup + a few steps.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_KV_BYTES=32212254720 INTEG_GPU_UTIL=0.85 FLASHINFER_WORKSPACE_BASE=$PWD/serve/fi_ws_ht
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 240000 ]; do sleep 60; done
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false, "kv_cache_dtype": "auto"}'
W=(--model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --language-model-only --max-model-len 16384 --batch-size 16 --prefill-lens 64 --mixed 0
   --ix-decode 8:6000-8000 --ix-reps 4 --llm-kw "$KW")
env CAND_LINE=c4 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 \
    ARMV_HOST_ARGS=pinned ARMV_EAGER_KINDS=prefill,mixed ARMV_CUTE_OBSERVE=1 \
    ARMV_EXTRA_EXTERN=scaled_fp4_quant.out,static_scaled_fp8_quant.default,cooperative_topk.default,persistent_topk.default INTEG_ARMV=$PWD/serve/armV_stock_m \
    bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm V --check "${W[@]}" --out serve/out/fn_smoke2_V.json
echo "fn_smoke2_V rc=$?"
python3 - <<'PY'
import json
d = json.load(open("serve/out/fn_smoke2_V.json")); V = d["V"]
print("FNSMOKE2", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "init_s", round(d["init_s"], 1))
ch = V.get("checks") or []
print("FNSMOKE2_CHECKS", len(ch), sum(bool(c.get("bitwise")) for c in ch))
import collections
print("FNSMOKE2_BY_SHAPE", dict(collections.Counter((c["shape"], c["bitwise"], c.get("nan")) for c in ch)))
for k, e in V["entries"].items():
    print("FNSMOKE2_ENTRY", k, {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [pv["eager_calls"] for pv in e["per_variant"]], e["declines"][:2])
print("FNSMOKE2_ROWS", {k: (round(v["step_ms"], 2) if isinstance(v, dict) and "step_ms" in v else v) for k, v in d.get("ix", {}).items()})
PY
