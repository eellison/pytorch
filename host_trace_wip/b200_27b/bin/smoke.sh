#!/usr/bin/env bash
# Quick checks before the 27B runs (one GPU, minutes, small memory):
#  1. the vLLM _C launchers (fp4 / fp8 quant, fp8 KV cache write, silu_and_mul) traced and bitwise over M = 1..2056;
#  2. the traced trtllm-gen launcher tests (Qwen3-8B-class bf16 and the 27B class: fp8 Q/KV, head 256, page 32), bitwise,
#     0 eager steps (downloads the cubins those shapes need unless FLASHINFER_NO_DOWNLOAD=1).
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; source "$B/env.sh"
mkdir -p "$WORK/logs"
bash "$B/bin/python_27b.sh" -u "$B/tools/ports_test.py" 2>&1 | tee "$WORK/logs/smoke_ports.log" | grep "^PORT\|Error"
FI_HT_CPP_ROOT=$OVERLAY bash "$B/bin/python_27b.sh" -u "$FI_HT_CPP_ROOT/tests/test_trtllm_cpp_trace.py" -v 2>&1 | tee "$WORK/logs/smoke_trace.log" | grep -E "^test_|OK|FAIL|Error|sweep|traces"
