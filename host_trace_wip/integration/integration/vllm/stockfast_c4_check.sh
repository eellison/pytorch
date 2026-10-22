#!/usr/bin/env bash
# Bitwise check of armV_stock_fast on CAND_LINE=c4 (candidate 4: step290 + build_cpp9, has F24k = the F26 fix): drive + rangesweep --check (GPU 3).
# drive --check (as c2_fast_attnlc_vstock_Vcheck) and rangesweep --check. Runs on GPU 3 (queue_gpu3.sh).
source $(dirname $0)/timing_lib.sh
VS=(CAND_LINE=c4 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0 INTEG_ARMV=$PWD/vllm/armV_stock_fast)
trun c2_stockfast_c4_Vcheck "${VS[@]}" -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 64x64 --out vllm/out/c2_stockfast_c4_Vcheck.json
trun c2_stockfast_c4_range "${VS[@]}" ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_stockfast_c4_range.json
echo ALLDONE
