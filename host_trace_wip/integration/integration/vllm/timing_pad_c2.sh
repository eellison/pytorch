#!/usr/bin/env bash
# Candidate 2 padding A/B (GPU 0, one hold): V cpp with ARMV_PAD=0 (exact sizes) vs 1 (vLLM-default padding), closed and
# open, interleaved, 2 rounds. Same workload as timing_c2.sh (aligned and non-aligned sizes).
source $(dirname $0)/timing_lib.sh
W=(--batch-size 1 8 64 5 100 --prefill-lens 64 512 2048 300 100 --prefill-reqs 4 8 --mixed-spec 32x512 4x2048 64x64 128x16)
for r in ${ROUNDS:-1 2}; do
  trun t_Vcpp_closed_ab_r$r CAND_LINE=cpp ARMV_PAD=0 -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_closed_ab_r$r.json
  trun t_Vcpp_closed_pad_r$r CAND_LINE=cpp ARMV_PAD=1 -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_closed_pad_r$r.json
  trun t_Vcpp_open_ab_r$r CAND_LINE=cpp ARMV_PAD=0 "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_open_ab_r$r.json
  trun t_Vcpp_open_pad_r$r CAND_LINE=cpp ARMV_PAD=1 "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_open_pad_r$r.json
done
echo ALLDONE
