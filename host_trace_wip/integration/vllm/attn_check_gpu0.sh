#!/usr/bin/env bash
# V-stock bitwise counts on CAND_LINE=attn (pinned + core attn A1+A3, fork F1): drive --check, rangesweep --check, greedy tokens.
source $(dirname $0)/timing_lib.sh
VS=(CAND_LINE=attn CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
trun c2_attn_vstock_Vcheck "${VS[@]}" -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 64x64 --out vllm/out/c2_attn_vstock_Vcheck.json
trun c2_attn_vstock_range "${VS[@]}" ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_attn_vstock_range.json
trun e2e_a_vstock "${VS[@]}" -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_a_vstock.json
echo ALLDONE
