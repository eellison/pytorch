#!/usr/bin/env bash
# q35p check (GPU 1, inside a gpu1.lock hold): vllm/q35_vcheck2.sh's Qwen3.5-9B V-stock --check with ARMV_HOST_ARGS=pinned and
# vllm/armV_stock_fast_cpp_m (the q1p round's V-stock config): mixed / prefill should now trace; every check bitwise.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export HT_GPU=${GPU_IDX:-1}
source vllm/timing_lib.sh
export UTIL_CAP=0.6 FLASHINFER_NO_DOWNLOAD=1
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i ${GPU_IDX:-1} | tr -d , | awk '{print $1 - $2}') -ge 200000 ]; do sleep 60; done
M=/data/eellison/models/Qwen3.5-9B
VS=(CAND_LINE=trtcpp ARMV_TRTLLM_CPP=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_HOST_ARGS=pinned FLASHINFER_NO_DOWNLOAD=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_cpp_m)
C=(--model $M --trust-remote-code --language-model-only --max-model-len 16384)
trun q35p_vstock_Vcheck "${VS[@]}" -- vllm/bench/drive_q35.py --arm V --check "${C[@]}" --batch-size 1 8 64 --prefill-lens 64 512 --prefill-reqs 4 8 --mixed 4 --mixed-spec 32x512 64x64 --out vllm/out/q35p_vstock_Vcheck.json
python3 - <<'PY'
import json
d = json.load(open("vllm/out/q35p_vstock_Vcheck.json"))
V = d["V"]
print("Q35P", {k: V[k] for k in ("traces", "replays", "eager", "variants", "calls")})
for k, e in V["entries"].items():
    print("Q35P_ENTRY", k, {x: e[x] for x in ("traces", "replays", "eager", "variants")}, e["declines"][:3])
ch = V.get("checks") or []
print("Q35P_CHECKS", len(ch), sum(bool(c.get("bitwise")) for c in ch))
PY
echo "q35p done"
