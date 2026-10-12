#!/usr/bin/env bash
# run_check.sh TAG [extra drive_fn args]: the 27B V-stock --check (every replayed / traced step is compared with vLLM's own eager
# forward on the same metadata, recurrent GDN state included; ARMV_CHECK_TWICE=1 also runs eager twice as a control).
# Then tools/summ.py and tools/verdict.py. Run on one idle GPU: CUDA_VISIBLE_DEVICES=N bash bin/run_check.sh c1
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; source "$B/env.sh"; source "$B/bin/vstock_env.sh"
TAG=$1; shift
export FLASHINFER_NO_DOWNLOAD=1 ARMV_CHECK_TWICE=${ARMV_CHECK_TWICE:-1}
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.5} INTEG_KV_BYTES=$(( ${KV_GIB:-24} << 30 ))
mkdir -p "$WORK/logs" "$WORK/out"
set +e
bash "$B/bin/python_27b.sh" "$B/bench/drive_fn.py" --arm V --check $(cat "$B/bin/check_shapes.txt") "$@" --out "$WORK/out/$TAG.json" > "$WORK/logs/$TAG.log" 2>&1
echo "$TAG rc=$?"
set -e
python "$B/tools/summ.py" "$WORK/out/$TAG.json" | tee "$WORK/out/$TAG.summary.txt"
python "$B/tools/verdict.py" "$WORK/out/$TAG.json" | tee "$WORK/out/$TAG.verdict.txt"
