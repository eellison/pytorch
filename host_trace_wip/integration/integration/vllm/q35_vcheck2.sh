#!/usr/bin/env bash
# Qwen3.5-9B V-stock --check, second attempt: the C++ launcher lane's armV has no hybrid (GDN) path ('GDNAttentionMetadata' object has
# no attribute 'decode'); vllm/armV_stock_fast_cpp = serve/armV_stock (ARMV_HYBRID path) + once-per-form key + ARMV_TRTLLM_CPP import.
source $(dirname $0)/timing_lib.sh
export UTIL_CAP=0.6 FLASHINFER_NO_DOWNLOAD=1  # the hybrid check runs an eager reference forward + saved state rows: 0.9 OOMed
M=/data/eellison/models/Qwen3.5-9B
VS=(CAND_LINE=trtcpp ARMV_TRTLLM_CPP=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 FLASHINFER_NO_DOWNLOAD=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_cpp)
C=(--model $M --trust-remote-code --language-model-only --max-model-len 16384)
trun q35_vstock_Vcheck "${VS[@]}" -- vllm/bench/drive_q35.py --arm V --check "${C[@]}" --batch-size 1 8 64 --prefill-lens 64 512 --prefill-reqs 4 8 --mixed 4 --mixed-spec 32x512 64x64 --out vllm/out/q35_vstock_Vcheck.json
echo ALLDONE
