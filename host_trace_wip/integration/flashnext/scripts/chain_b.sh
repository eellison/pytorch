#!/usr/bin/env bash
# chain_b.sh WAIT_PID: vLLM's default config (compile + FULL_AND_PIECEWISE cudagraphs; no enforce_eager), quality only:
# does the default load, compile, capture and generate on one GB300, and its resolved compile / cudagraph config.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
while kill -0 $1 2>/dev/null; do sleep 15; done
HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_GPU_UTIL=0.88 CMD_TIMEOUT=1200 bash gpu3.sh fn_qdef bash python_fn.sh scripts/stock_ref.py \
  --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --out out/fn_qdef.json --llm-kw '{"language_model_only": true}' \
  --enforce-eager 0 --decode-bs --long-decode --prefill-lens --mixed \
  --quality data/ppl_passage_pytorch_README.md data/ppl_passage_lane_status_20261013.md
echo "$(date +%T) fn_qdef rc=$?"
