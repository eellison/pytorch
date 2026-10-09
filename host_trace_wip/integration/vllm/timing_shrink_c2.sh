#!/usr/bin/env bash
# Candidate 2 fluctuating decode batch (GPU 0, one hold): N requests, one finishing per step (bs N..1), pass 1 + settled
# pass 2; vLLM default vs V (cpp closed/open, py closed) and eager, 2 interleaved rounds.
source $(dirname $0)/timing_lib.sh
W=(--batch-size 1 --prefill-lens 64 --mixed 0 --shrink 64 128)
for r in ${ROUNDS:-1 2}; do
  trun t_shrink_default_r$r CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${W[@]}" --out vllm/out/t_shrink_default_r$r.json
  trun t_shrink_Vcpp_closed_r$r CAND_LINE=cpp -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_shrink_Vcpp_closed_r$r.json
  trun t_shrink_Vcpp_open_r$r CAND_LINE=cpp "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_shrink_Vcpp_open_r$r.json
  trun t_shrink_Vpy_closed_r$r CAND_LINE=py -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_shrink_Vpy_closed_r$r.json
  trun t_shrink_eager_r$r CAND_LINE=cpp -- vllm/bench/drive.py --arm eager "${W[@]}" --out vllm/out/t_shrink_eager_r$r.json
done
echo ALLDONE
