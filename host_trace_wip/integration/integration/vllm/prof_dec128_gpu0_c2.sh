#!/usr/bin/env bash
# V-stock vs default-nc at steady decode bs 128, KV 6000-8000 (attn line): step timing (16 steps) then a torch profiler trace of 3 steps.
source $(dirname $0)/timing_lib.sh
export UTIL_CAP=0.9
VS=(CAND_LINE=attn CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
W=(--max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --ix-decode 128:6000-8000 --ix-reps 6 --prof-ix-decode vllm/out/prof_dec128)
trun pd_vstock "${VS[@]}" -- vllm/bench/drive_ix3.py --arm V "${W[@]}" --out vllm/out/pd_vstock.json
trun pd_fullnc CAND_LINE=attn -- vllm/bench/drive_ix3.py --arm fullnc "${W[@]}" --out vllm/out/pd_fullnc.json
echo ALLDONE
