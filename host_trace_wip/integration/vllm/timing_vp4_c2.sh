#!/usr/bin/env bash
# Candidate 2 arm VP (GPU 0, one hold): default / VAf / V / VP (cpp closed) at decode bs 1/8/64/128.
# Pass A: synced step + async step (--async-decode). Pass B: --launch-stamps (pre-launch host us, LD_PRELOAD launchshim).
# Then decode-step profiles (launches outside graphs) for all four arms.
source $(dirname $0)/timing_lib.sh
SHIM=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
W=(--batch-size 1 8 64 128 --prefill-lens 64 --mixed 0)
# VP4 (float32 one-hot sampler tail: no eager step left) check + tokens, then the VP rows again
trun c2_cpp_VP4_Vcheck CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/drive.py --arm V --check --batch-size 1 8 64 128 --prefill-lens 64 --mixed 2 --out vllm/out/c2_cpp_VP4_Vcheck.json
trun e2e_VP4 CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_VP4.json
for r in ${ROUNDS:-1 2}; do
  trun t_vp_VP_r$r CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/drive.py --arm V "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_vp_VP_r$r.json
done
trun t_vpls_VP CAND_LINE=cpp ARMV_VP=1 LD_PRELOAD=$SHIM -- vllm/bench/drive.py --arm V "${W[@]}" --launch-stamps --out vllm/out/t_vpls_VP.json
trun p_VP CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/prof_decode.py --arm V --bs 1 8 64 128 --out-dir vllm/out/prof_VP
echo ALLDONE
