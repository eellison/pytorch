#!/usr/bin/env bash
# bswalk8k (10-14): correctness at long KV, the multi-CTA (split-KV) region: decode at ~8K context for every bs 1-128 (ix-decode,
# decoders with 8000-8600-token prompts) and mixed steps with 64 / 128 decodes at 6000-8000 KV, every step checked bitwise vs vLLM
# eager (hidden, sampled tokens, every written state tensor); both learnfast helpers, actual max_seq_len. ARM=Vc (compiled forward,
# vcsl config) or ARM=V (eager forward, vrs config).  TAG required.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
: ${TAG:?} ${ARM:?}
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.65 INTEG_KV_BYTES=214748364800  # 200 GiB: bs 128 x 8.6K tokens fits
EXTRA=(); [ "$ARM" = Vc ] && EXTRA=(HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1)
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 10 --prefill-lens 64 --mixed 0
   --ix-decode $(for b in $(seq 1 ${BSW_MAX:-128}); do echo -n "$b:8000-8600 "; done)
   --ix-mixed 1024x16@6000-8000 1024x64@6000-8000 1024x128@6000-8000 2048x16@6000-8000 2048x64@6000-8000 2048x128@6000-8000 --ix-reps ${BSW_REPS:-2})
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 ARMV_VPS=1 ARMV_CHECKS_JSONL=serve/out/bswalk8k${TAG}_checks.jsonl INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 "${EXTRA[@]}" \
    bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm $ARM --check "${W[@]}" --out serve/out/bswalk8k${TAG}_check.json > serve/logs/bswalk8k${TAG}_run.log 2>&1
echo "bswalk8k$TAG rc=$?"
TAG=$TAG python3 - <<'PY'
import json, collections, os
t = os.environ["TAG"]
if os.path.exists(f"serve/out/bswalk8k{t}_check.json"):
    V = json.load(open(f"serve/out/bswalk8k{t}_check.json"))["V"]; ch = V["checks"]
else:  # crashed before the final JSON: the rows flushed per check (no summary fields)
    ch = [json.loads(l) for l in open(f"serve/out/bswalk8k{t}_checks.jsonl")]; V = {"entries": {}}; print("BSWALK8K partial (from jsonl)")
bad = [c for c in ch if not c["bitwise"]]
print("BSWALK8K", t, "checks", len(ch), "bitwise", len(ch) - len(bad), "nan", sum(c.get("nan", 0) for c in ch), "sections", V.get("sections_refused"),
      {k: e.get("sections") for k, e in V["entries"].items()}, "lib_redo", V.get("lib_redo_refused"), "learned", V.get("learned_by_op"))
print("BSWALK8K_BAD", [(c.get("kind"), c.get("shape"), c.get("step")) for c in bad][:30])
print("BSWALK8K_SHAPES", dict(collections.Counter((c.get("kind"), str(c.get("shape"))[:16], c["bitwise"]) for c in ch).most_common(12)))
PY
