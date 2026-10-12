#!/usr/bin/env bash
# 3 rounds of bin/timing_round.sh (one stock warm-up run first, untimed, so the stock arms' JIT builds are not in round 1),
# then tools/timing_table.py. CUDA_VISIBLE_DEVICES=N bash bin/timing_all.sh
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; source "$B/env.sh"
mkdir -p "$WORK/timing"
NO_OVERLAY=1 FLASHINFER_WORKSPACE_BASE=$WORK/fi_ws_stock FLASHINFER_CUBIN_DIR=$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/cubins \
  HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_KV_BYTES=$(( 24 << 30 )) bash "$B/bin/python_27b.sh" "$B/bench/drive_fn.py" --arm default \
  --language-model-only --max-model-len 16384 --batch-size 1 8 --prefill-lens 64 --mixed 0 --llm-kw '{"kv_cache_dtype": "fp8"}' \
  --out "$WORK/timing/warm_stock.json" > "$WORK/timing/warm_stock.log" 2>&1
for r in 1 2 3; do bash "$B/bin/timing_round.sh" $r; done
python "$B/tools/timing_table.py" "$WORK/timing" | tee "$WORK/timing/TABLE.txt"
