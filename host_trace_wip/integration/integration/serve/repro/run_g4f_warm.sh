#!/usr/bin/env bash
# g4f cache warm-up (correctness GPU, untimed): one drive_ix2 run per compiled arm on STOCK_LINE (default c5g) with the g4f workload,
# so the timed rounds load vLLM's compile cache and Inductor's autotuned configs (a cold untuned autotuner under a trace retries).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export HT_GPU=${GPU_IDX:-1} R=1 SET=g4fwarm STOCK_LINE=${STOCK_LINE:-c5g} ARMS="default vc vcs"
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 250000 ]; do sleep 60; done
bash vllm/round_g4f_line.sh
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("vllm/out/t_g4fwarm_*_r1.json")):
    d = json.load(open(f)); V = d.get("V")
    print("G4FWARM", f, {k: V[k] for k in ("traces", "replays", "eager", "variants")} if V else "", {k: round(v["step_ms"], 2) for k, v in d.get("ix", {}).items() if v.get("step_ms")})
PY
