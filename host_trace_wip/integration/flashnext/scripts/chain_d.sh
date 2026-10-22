#!/usr/bin/env bash
# chain_d.sh: vLLM default compile + graphs with the recipe's --max-num-batched-tokens 8192 (the default 16384 left -26.5 GiB
# for KV at util 0.88: fn_qdef), quality only.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_GPU_UTIL=0.88 CMD_TIMEOUT=1200 bash gpu3.sh fn_qdef2 bash python_fn.sh scripts/stock_ref.py \
  --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --out out/fn_qdef2.json \
  --llm-kw '{"language_model_only": true, "max_num_batched_tokens": 8192}' \
  --enforce-eager 0 --decode-bs --long-decode --prefill-lens --mixed \
  --quality data/ppl_passage_pytorch_README.md data/ppl_passage_lane_status_20261013.md
echo "$(date +%T) fn_qdef2 rc=$?"
