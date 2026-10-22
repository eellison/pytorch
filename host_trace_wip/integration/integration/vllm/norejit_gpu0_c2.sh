#!/usr/bin/env bash
# (b) stock Triton specialization of vLLM's block-table gather and the indptr kernel (no do_not_specialize re-jits):
# range --check counts (traces/variants/redispatches), and decode step time vs the re-jitted adapter (interleaved).
source $(dirname $0)/timing_lib.sh
trun c2_pinned_norejit_range CAND_LINE=pinned ARMV_NO_REJIT=1 ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_pinned_norejit_range.json
W=(--batch-size 1 8 64 5 100 --prefill-lens 64 512 --mixed 4 --mixed-spec 64x64)
for r in 1 2; do
  trun t_norejit_r$r CAND_LINE=pinned ARMV_NO_REJIT=1 -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_norejit_r$r.json
  trun t_rejit_r$r CAND_LINE=pinned ARMV_NO_REJIT=0 -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_rejit_r$r.json
done
trun c2_pinned_rejit_range CAND_LINE=pinned ARMV_NO_REJIT=0 ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_pinned_rejit_range.json
trun diag_md_stock CAND_LINE=pinned ARMV_MD_STOCK_DIAG=1 ARMV_BOUND=0 -- vllm/bench/drive.py --arm V --batch-size 8 --prefill-lens 64 --mixed 2 --out vllm/out/diag_md_stock.json
echo ALLDONE
