#!/usr/bin/env bash
# V-stock vs vLLM default decode-step profiles (pinned build), bs 1 and 8: host-time breakdown.
source $(dirname $0)/timing_lib.sh
VS=(CAND_LINE=pinned CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
trun p_vstock "${VS[@]}" -- vllm/bench/prof_decode.py --arm V --bs 1 8 --out-dir vllm/out/prof_vstock
trun p_pdefault CAND_LINE=pinned HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/prof_decode.py --arm default --bs 1 8 --out-dir vllm/out/prof_pdefault
echo ALLDONE
