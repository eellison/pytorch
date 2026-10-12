#!/usr/bin/env bash
# One-time warm-up (network allowed): vLLM eager over the check's shapes, so FlashInfer JIT-builds its modules (fmha_gen, the
# NVFP4 quant / GEMM modules) for this arch and downloads every trtllm-gen cubin those shapes select into
# $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/cubins. ARMV_CUTE_OBSERVE=1 so the CuTe DSL mm_fp4 kernels compile under the
# host trace's observation (their persisted copies carry what the trace needs); never compile them in this workspace without it.
set -euo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; source "$B/env.sh"
unset FLASHINFER_NO_DOWNLOAD
export ARMV_CUTE_OBSERVE=1 INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.5} INTEG_KV_BYTES=$(( ${KV_GIB:-24} << 30 ))
mkdir -p "$WORK/logs" "$WORK/out"
bash "$B/bin/python_27b.sh" "$B/bench/drive_fn.py" --arm eager $(cat "$B/bin/check_shapes.txt") --out "$WORK/out/warm_eager.json" 2>&1 | tee "$WORK/logs/warm_eager.log"
