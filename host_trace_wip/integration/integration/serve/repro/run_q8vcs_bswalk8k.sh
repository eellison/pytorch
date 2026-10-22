#!/usr/bin/env bash
# Outputs: serve/out/q8vcs${TAG}_check.json, serve/logs/q8vcs${TAG}_run.log (TAG required, unique per config; never reuse a reported name).
# bswalk8k (10-14): decode at ~8K context for bs 1..BSW_MAX (decoders with 8000-8600-token prompts, ix-decode), every step checked bitwise vs vLLM eager: the multi-CTA (split-KV) region where the counter memset size matters.
# q8vcs (Vc + ARMV_VPS: forward + vLLM sampling + request-state writes in one step, armV_stock_fast_m4): Qwen3-8B Vc --check with FULL g4s coverage (as q8p2f): the sound registered-launcher hook (armV_stock_fast_m4 + vllm_ports,
# ARMV_SOUND + ARMV_PORTS) around vLLM's compiled forward (compile ON, cudagraph_mode NONE); Inductor's Triton launches recorded
# through vllm_ports/inductor_hosts.py (ARMV_INDUCTOR_HOSTS=1, non-static launcher). Rows with NaN references fail.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.65 INTEG_KV_BYTES=150323855360
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 185000 ]; do sleep 60; done
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 4 --prefill-lens 64 --mixed 0 --ix-decode $(for b in $(seq 1 ${BSW_MAX:-128}); do echo -n "$b:${BSW_LO:-8000}-${BSW_HI:-8600} "; done) --ix-reps ${BSW_REPS:-2})
env CAND_LINE=${VC_LINE:-c4} CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 ARMV_VPS=1 ARMV_VPS_POST=${VPS_POST:-1} ARMV_VPS_STAGE=${VPS_STAGE:-full} ARMV_PERSIST=${PERSIST:-0} CUDA_LAUNCH_BLOCKING=${CLB:-0} INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=${STATIC_LAUNCHER:-1} \
    bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm Vc --check "${W[@]}" --out serve/out/q8vcs${TAG:?TAG required: unique output name}_check.json 2>&1 | tee serve/logs/q8vcs${TAG}_run.log | grep "VLLM_PORT_DECLINE\|Error\|Traceback" | grep -v triton_kernels | sort | uniq -c | head -20
echo "q8vcs rc=${PIPESTATUS[0]}"
TAG=$TAG python3 - <<'PY'
import json, collections, os
d = json.load(open(f"serve/out/q8vcs{os.environ['TAG']}_check.json")); V = d["V"]
ch = V["checks"]
print("Q8VCS", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")}, "learned", V["learned_by_op"], "unsound", V.get("unsound_learned"))
clean = [c for c in ch if not c.get("nan")]
print("Q8VCS_CHECKS", len(ch), "clean (no NaN ref)", len(clean), "bitwise among clean", sum(c["bitwise"] for c in clean), "bitwise all", sum(c["bitwise"] for c in ch),
      dict(collections.Counter((c.get("kind"), str(c.get("shape")), c["bitwise"]) for c in ch)))
for k, e in V["entries"].items():
    print("Q8VCS_REASONS", k[:30], e.get("eager_reasons", [])[:8])
    print("Q8VCS_ENTRY", k[:12], {x: e[x] for x in ("traces", "replays", "eager", "variants")}, [(pv["kernels"], pv["eager_calls"], pv["segments"]) for pv in e["per_variant"]][:8], e.get("eager_reasons", [])[:6], e.get("declines", [])[:3])
sub = [c for c in ch if c["kind"].endswith("+sample")]
print("Q8VCS_SAMPLE_ROWS", len(sub), "bitwise", sum(c["bitwise"] for c in sub), "hidden", sum(c["hidden_bitwise"] for c in sub), "tokens", sum(c["tokens_bitwise"] for c in sub), "state", sum(c["state_bitwise"] for c in sub), "nan", sum(c["nan"] for c in sub))
print("Q8VCS_CALLS", V["calls"])
PY
