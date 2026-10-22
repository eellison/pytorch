#!/usr/bin/env bash
# q8vc: Qwen3-8B Vc --check with FULL g4s coverage (as q8p2f): the sound registered-launcher hook (armV_stock_fast_m3 + vllm_ports,
# ARMV_SOUND + ARMV_PORTS) around vLLM's compiled forward (compile ON, cudagraph_mode NONE); Inductor's Triton launches recorded
# through vllm_ports/inductor_hosts.py (ARMV_INDUCTOR_HOSTS=1, non-static launcher). Rows with NaN references fail.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.65 INTEG_KV_BYTES=150323855360
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 185000 ]; do sleep 60; done
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 8 64 --prefill-lens 64 512 --mixed 0
   --ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000
   --ix-decode 8:6000-8000 64:6000-8000 --ix-reps 3)
env CAND_LINE=c4 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m3 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0 \
    bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm Vc --check "${W[@]}" --out serve/out/q8vc_check.json 2>&1 | grep "VLLM_PORT_DECLINE\|Error\|Traceback" | grep -v triton_kernels | sort | uniq -c | head -20
echo "q8vc rc=${PIPESTATUS[0]}"
python3 - <<'PY'
import json, collections
d = json.load(open("serve/out/q8vc_check.json")); V = d["V"]
ch = V["checks"]
print("Q8VC", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "learned", V["learned_by_op"], "unsound", V.get("unsound_learned"))
clean = [c for c in ch if not c.get("nan")]
print("Q8VC_CHECKS", len(ch), "clean (no NaN ref)", len(clean), "bitwise among clean", sum(c["bitwise"] for c in clean), "bitwise all", sum(c["bitwise"] for c in ch),
      dict(collections.Counter((c.get("kind"), str(c.get("shape")), c["bitwise"]) for c in ch)))
for k, e in V["entries"].items():
    print("Q8VC_ENTRY", k[:12], {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [(pv["kernels"], pv["eager_calls"], pv["segments"]) for pv in e["per_variant"]][:8], e.get("eager_reasons", [])[:6], e.get("declines", [])[:3])
PY
