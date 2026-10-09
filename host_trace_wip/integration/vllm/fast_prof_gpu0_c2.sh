#!/usr/bin/env bash
# Decode bs 128 at KV 6000-8000, step timing + 3-step profile: V-stock with armV_fast on attn and on attn+lc, and default-nc (same hold).
source $(dirname $0)/timing_lib.sh
export UTIL_CAP=0.9
F=$PWD/vllm/armV_fast
VS=(CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 INTEG_ARMV=$F)
W=(--max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --ix-decode 128:6000-8000 --ix-reps 6 --prof-ix-decode vllm/out/prof_dec128b)
trun pd_fast_attn_vstock CAND_LINE=attn "${VS[@]}" -- vllm/bench/drive_ix3.py --arm V "${W[@]}" --out vllm/out/pd_fast_attn_vstock.json
trun pd_fast_attnlc_vstock CAND_LINE=attnlc "${VS[@]}" -- vllm/bench/drive_ix3.py --arm V "${W[@]}" --out vllm/out/pd_fast_attnlc_vstock.json
trun pd_fullnc2 CAND_LINE=attn -- vllm/bench/drive_ix3.py --arm fullnc "${W[@]}" --out vllm/out/pd_fullnc2.json
echo ALLDONE
