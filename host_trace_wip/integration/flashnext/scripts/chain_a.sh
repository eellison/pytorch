#!/usr/bin/env bash
# chain_a.sh PREBUILD_PID: after the CPU-only FlashInfer JIT prebuild, census first (unblocks the MoE lane), then 3 stock processes.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
while kill -0 $1 2>/dev/null; do sleep 15; done
grep -q "^moe " logs/prebuild_fi.log || { echo "prebuild did not finish the moe module"; exit 1; }
FN_ALLOW_DOWNLOAD=1 INTEG_GPU_UTIL=0.88 CMD_TIMEOUT=1200 bash gpu3.sh fn_census bash python_fn.sh scripts/fn_census.py --out $F/out/census
echo "$(date +%T) fn_census rc=$?"
bash scripts/stock3.sh fn_s
