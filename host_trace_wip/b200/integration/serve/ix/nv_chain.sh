#!/usr/bin/env bash
# GPU 0 chain (coordinator 10-12): nv1 27B repeat 1 -> q1p (Qwen3.5-9B rounds, V-stock with ARMV_HOST_ARGS=pinned +
# armV_stock_fast_cpp_m; 3 rounds, arms rotated) if the GPU 1 check q35p is bitwise -> the rest of nv1 (Flash-Next, repeats).
#   setsid nohup bash serve/ix/nv_chain.sh > serve/logs/nv_chain.log 2>&1 &
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while pgrep -f "ix_run.py .*--tag nv1_" > /dev/null; do sleep 30; done  # the point the stopped nv_sweep left running
PART=27r1 bash serve/ix/nv_sweep.sh nv1 >> serve/logs/nv1.log 2>&1
echo "$(date +%T) 27B r1 done"
while ! grep -q "q35p done" serve/logs/q35p.log 2>/dev/null; do sleep 60; done
ok=$(python3 - <<'PY'
import json
try:
    V = json.load(open("vllm/out/q35p_vstock_Vcheck.json"))["V"]
    ch = V.get("checks") or []
    print(int(len(ch) > 0 and all(c.get("bitwise") for c in ch)))
except Exception as e:
    print(0)
PY
)
if [ "$ok" = 1 ]; then
  echo "$(date +%T) q35p bitwise: q1p rounds"
  ROUNDS=3 STOCK_LINE=trtcpp SET=q1p ROUND_SCRIPT=round_q35p_line bash vllm/rounds_gpu0.sh > vllm/logs/rounds_q1p.out 2>&1
  echo "$(date +%T) q1p done"
  python3 - <<'PY'
import glob, json
for p in sorted(glob.glob("vllm/out/t_q1p_vstock_r*.json")):
    V = json.load(open(p))["V"]
    cpu = sum(int(s.split("x ", 1)[0]) for s in V.get("decline_sites", []) if "is on cpu; only CUDA tensors" in s)
    print("Q1P", p, {k: V[k] for k in ("traces", "replays", "eager", "variants")}, "cpu_arg_declines", cpu, "MISCONFIGURED" if cpu else "")
PY
else
  echo "$(date +%T) q35p check not bitwise (or missing): q1p skipped"
fi
PART=rest bash serve/ix/nv_sweep.sh nv1 >> serve/logs/nv1.log 2>&1
echo "$(date +%T) NV_CHAIN_DONE"
