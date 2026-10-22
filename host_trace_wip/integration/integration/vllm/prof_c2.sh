#!/usr/bin/env bash
# Decode-step profiles (GPU 0, one hold): vLLM default and V (cpp closed), bs 8 and 64.
source $(dirname $0)/timing_lib.sh
trun p_default CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/prof_decode.py --arm default --bs 8 64 --out-dir vllm/out/prof_default
trun p_Vcpp_closed CAND_LINE=cpp -- vllm/bench/prof_decode.py --arm V --bs 8 64 --out-dir vllm/out/prof_Vcpp_closed
echo ALLDONE
