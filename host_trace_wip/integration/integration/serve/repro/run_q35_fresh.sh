#!/usr/bin/env bash
# Fresh-process control (coordinator): two separate processes, same seed and request schedule, stock eager only (no adapter, no
# check hook), Qwen3.5-9B with q35s3's workload; per-row digests of every forward >= 2048 tokens compared step by step.
# If these agree while the in-process eager-twice check disagrees, the check's state restore is incomplete.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.12} INTEG_KV_BYTES=${INTEG_KV_BYTES:-21474836480}
echo "GPU apps at start: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i ${GPU_IDX:-1} | tr '\n' ' ')"
W=(--model /data/eellison/models/Qwen3.5-9B --trust-remote-code --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0
   --ix-mixed 8192x32@4096-8000 2048x64@6000-8000 --ix-reps 3)
for r in a b; do
  CAND_LINE=c4 DRIVE_DIGEST=serve/out/q35_fresh_$r.json timeout 900 bash python_vllm_cand2.sh vllm/bench/drive_fn.py --arm eager "${W[@]}" --out serve/out/q35_fresh_$r.drive.json > serve/logs/q35_fresh_$r.log 2>&1
  echo "fresh $r rc=$?"
done
python3 - <<'PY'
import json
a = json.load(open("serve/out/q35_fresh_a.json")); b = json.load(open("serve/out/q35_fresh_b.json"))
print("FRESH steps", len(a), len(b))
same_sched = all((x["tokens"], x["ids"], x["pos"]) == (y["tokens"], y["ids"], y["pos"]) for x, y in zip(a, b))
print("FRESH same schedule", same_sched)
bad = []
for i, (x, y) in enumerate(zip(a, b)):
    d = [j for j, (p, q) in enumerate(zip(x["rows"], y["rows"])) if p != q]
    if d:
        bad.append((i, x["tokens"], len(d), d[0], d[-1]))
print("FRESH steps with differing rows", len(bad), "of", min(len(a), len(b)), bad[:12])
PY
