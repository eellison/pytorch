#!/usr/bin/env bash
# Vc smoke (Qwen3-8B): the sound registered-launcher hook (armV_stock_fast_m3 + vllm_ports, ARMV_SOUND + ARMV_PORTS) around vLLM's
# compiled forward (compile ON, cudagraph_mode NONE), --check on a small workload; INTEG_TRAP_NUMEL prints the Python stack of a
# numel() on a symbolic tensor (the 10-09 escaping error).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.15 INTEG_KV_BYTES=17179869184
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 60000 ]; do sleep 60; done
env CAND_LINE=c4 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m3 HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_TRAP_NUMEL=1 INTEG_KEEP_TRACEBACKS=1 ARMV_INDUCTOR_HOSTS=1 TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0 ${VC_ENV:-} \
    timeout 1200 bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm Vc --check --model /data/eellison/models/Qwen3-8B --max-model-len 8192 --batch-size 1 8 --prefill-lens 64 512 --mixed 0 --out serve/out/vc_smoke.json > serve/logs/vc_smoke_run.log 2>&1
echo "vc_smoke rc=$?"
grep -A12 "TRAP_NUMEL\|^CEXC" serve/logs/vc_smoke_run.log | head -60
grep -n "Error\|Traceback" serve/logs/vc_smoke_run.log | grep -v triton_kernels | head -10
python3 - <<'PY'
import json, collections
try:
    d = json.load(open("serve/out/vc_smoke.json"))
except Exception as e:
    print("VCS no json", e); raise SystemExit
V = d["V"]; ch = V["checks"]
print("VCS", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "learned", V["learned_by_op"], "unsound", V.get("unsound_learned"))
print("VCS_CHECKS", len(ch), sum(c["bitwise"] for c in ch), dict(collections.Counter((c.get("kind"), str(c.get("shape")), c["bitwise"]) for c in ch)))
for k, e in V["entries"].items():
    print("VCS_ENTRY", k[:12], {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [(pv["eager_calls"], pv["segments"]) for pv in e["per_variant"]][:6], e.get("eager_reasons", [])[:6], e.get("declines", [])[:3])
PY
