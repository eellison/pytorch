#!/usr/bin/env bash
# Candidate 2 cold/warm start (GPU 0, one hold). Pre-warm: one throwaway run per launcher config (third-party JIT /
# the bindings a pre-warm run saved (HarvestProvider.save -> load).
source $(dirname $0)/timing_lib.sh
B=$G/vllm/out/bindings_c2
mkdir -p $B
for r in ${ROUNDS:-1 2 3}; do
  trun s_default_r$r CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/startup.py --arm default --out vllm/out/s_default_r$r.json
  trun s_eager_r$r CAND_LINE=cpp -- vllm/bench/startup.py --arm eager --out vllm/out/s_eager_r$r.json
  trun s_Vcpp_closed_r$r CAND_LINE=cpp -- vllm/bench/startup.py --arm V --out vllm/out/s_Vcpp_closed_r$r.json
  trun s_Vcpp_open_r$r CAND_LINE=cpp "${OPEN[@]}" -- vllm/bench/startup.py --arm V --out vllm/out/s_Vcpp_open_r$r.json
  trun s_Vpy_closed_r$r CAND_LINE=py -- vllm/bench/startup.py --arm V --out vllm/out/s_Vpy_closed_r$r.json
  trun s_Vcpp_closed_warm_r$r CAND_LINE=cpp -- vllm/bench/startup.py --arm V --load $B/cpp_closed.pkl --out vllm/out/s_Vcpp_closed_warm_r$r.json
  trun s_Vcpp_open_warm_r$r CAND_LINE=cpp "${OPEN[@]}" -- vllm/bench/startup.py --arm V --load $B/cpp_open.pkl --out vllm/out/s_Vcpp_open_warm_r$r.json
done
echo ALLDONE
