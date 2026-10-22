#!/usr/bin/env bash
# V-stock without the attention direct-call switch (vLLM's custom-op call path): does it trace?
source $(dirname $0)/timing_lib.sh
VS=(CAND_LINE=pinned CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
trun c2_vstock_nodirect_Vcheck "${VS[@]}" ARMV_DIRECT_CALL=0 -- vllm/bench/drive.py --arm V --check --batch-size 1 8 64 --prefill-lens 64 512 --mixed 2 --out vllm/out/c2_vstock_nodirect_Vcheck.json
echo ALLDONE
