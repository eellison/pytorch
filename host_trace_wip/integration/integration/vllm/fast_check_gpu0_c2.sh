#!/usr/bin/env bash
# Bitwise checks for the adapter's once-per-form key (armV_fast) on the attn line and on attn+lc (c8l C++): V-stock drive --check.
source $(dirname $0)/timing_lib.sh
F=$PWD/vllm/armV_fast
VS=(CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 INTEG_ARMV=$F)
trun c2_fast_attn_vstock_Vcheck CAND_LINE=attn "${VS[@]}" -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 64x64 --out vllm/out/c2_fast_attn_vstock_Vcheck.json
trun c2_fast_attnlc_vstock_Vcheck CAND_LINE=attnlc "${VS[@]}" -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 64x64 --out vllm/out/c2_fast_attnlc_vstock_Vcheck.json
echo ALLDONE
