#!/usr/bin/env bash
# python_27b.sh <script> [args]: python with the bundle's environment. The FlashInfer overlay (bin/make_overlay.sh) goes ahead of
# the installed flashinfer unless NO_OVERLAY=1 (the vLLM default arms run stock FlashInfer).
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
if [ "${NO_OVERLAY:-0}" != 1 ]; then
  [ -f "$OVERLAY/flashinfer/trtllm_cpp.py" ] || { echo "run bin/make_overlay.sh first" >&2; exit 2; }
  export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi
# a source-built torch does not preload cuBLAS globally; vLLM's _C needs it (harmless if not needed)
for c in "$CUDA_HOME"/lib64/libcublas.so.13 "$CUDA_HOME"/lib64/libcublas.so.12; do
  [ -f "$c" ] && { export LD_PRELOAD="$c${LD_PRELOAD:+:$LD_PRELOAD}"; break; }
done
# the host-trace arm runs with Inductor caches off; the default (compiled) arm needs them (HOSTTRACE_TORCH_DISABLE_CACHES=0)
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
exec python "$@"
