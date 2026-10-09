#!/usr/bin/env bash
# (e) vLLM's stock metadata builder inside the trace (ARMV_MD_STOCK=1): does it replay correctly? --check, pinned line.
source $(dirname $0)/timing_lib.sh
trun diag_md_stock_check CAND_LINE=pinned ARMV_MD_STOCK=1 ARMV_BOUND=0 -- vllm/bench/drive.py --arm V --check --batch-size 1 8 64 --prefill-lens 64 512 --prefill-reqs 4 --mixed 4 --out vllm/out/diag_md_stock_check.json
echo ALLDONE
