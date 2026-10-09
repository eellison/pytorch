#!/usr/bin/env bash
# Candidate 2 arm VP (GPU 0, one hold): default / VAf / V / VP (cpp closed) at decode bs 1/8/64/128.
# Pass A: synced step + async step (--async-decode). Pass B: --launch-stamps (pre-launch host us, LD_PRELOAD launchshim).
# Then decode-step profiles (launches outside graphs) for all four arms.
source $(dirname $0)/timing_lib.sh
SHIM=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
W=(--batch-size 1 8 64 128 --prefill-lens 64 --mixed 0)
# correctness first, inside this hold (GPU 1 is reserved for candidate 3's gates): VP --check, then greedy tokens VP vs V, VAf vs default
trun c2_cpp_VP3_Vcheck CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/drive.py --arm V --check --batch-size 1 8 64 128 --prefill-lens 64 --mixed 2 --out vllm/out/c2_cpp_VP3_Vcheck.json
trun e2e_VP3 CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_VP3.json
trun e2e_V CAND_LINE=cpp -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_V.json
trun e2e_VAf CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/e2e_tokens.py --arm VAf --out vllm/out/e2e_VAf.json
trun e2e_default CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/e2e_tokens.py --arm default --out vllm/out/e2e_default.json
for r in ${ROUNDS:-1 2}; do
  trun t_vp_default_r$r CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_vp_default_r$r.json
  trun t_vp_VAf_r$r CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm VAf "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_vp_VAf_r$r.json
  trun t_vp_V_r$r CAND_LINE=cpp -- vllm/bench/drive.py --arm V "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_vp_V_r$r.json
  trun t_vp_VP_r$r CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/drive.py --arm V "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_vp_VP_r$r.json
done
trun t_vpls_default CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 LD_PRELOAD=$SHIM -- vllm/bench/drive.py --arm default "${W[@]}" --launch-stamps --out vllm/out/t_vpls_default.json
trun t_vpls_VAf CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 LD_PRELOAD=$SHIM -- vllm/bench/drive.py --arm VAf "${W[@]}" --launch-stamps --out vllm/out/t_vpls_VAf.json
trun t_vpls_V CAND_LINE=cpp LD_PRELOAD=$SHIM -- vllm/bench/drive.py --arm V "${W[@]}" --launch-stamps --out vllm/out/t_vpls_V.json
trun t_vpls_VP CAND_LINE=cpp ARMV_VP=1 LD_PRELOAD=$SHIM -- vllm/bench/drive.py --arm V "${W[@]}" --launch-stamps --out vllm/out/t_vpls_VP.json
trun p_VAf CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/prof_decode.py --arm VAf --bs 1 8 64 128 --out-dir vllm/out/prof_VAf
trun p_VP CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/prof_decode.py --arm V --bs 1 8 64 128 --out-dir vllm/out/prof_VP
trun p_V4 CAND_LINE=cpp -- vllm/bench/prof_decode.py --arm V --bs 1 128 --out-dir vllm/out/prof_V4
trun p_default4 CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/prof_decode.py --arm default --bs 1 128 --out-dir vllm/out/prof_default4
echo ALLDONE
