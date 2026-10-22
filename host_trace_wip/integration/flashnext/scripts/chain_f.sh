#!/usr/bin/env bash
# chain_e.sh: vLLM default compile + graphs with a fixed 40 GiB KV pool (16 GiB gave 432 Mamba blocks < max_num_seqs 1024: fn_qdef3) (kv_cache_memory_bytes: skips vLLM's memory
# profiling, as every timing arm's fixed pool does): fn_qdef / fn_qdef2 failed at the profiling estimate (-26.5 / -25.7 GiB).
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_GPU_UTIL=0.88 CMD_TIMEOUT=1200 bash gpu3.sh fn_qdef4 bash python_fn.sh scripts/stock_ref.py \
  --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --out out/fn_qdef4.json \
  --llm-kw '{"language_model_only": true, "kv_cache_memory_bytes": 42949672960}' \
  --enforce-eager 0 --decode-bs 1 8 64 --long-decode --prefill-lens 64 512 2048 --mixed \
  --quality data/ppl_passage_pytorch_README.md data/ppl_passage_lane_status_20261013.md
echo "$(date +%T) fn_qdef4 rc=$?"
