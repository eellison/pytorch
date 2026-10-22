#!/usr/bin/env bash
# Arm VP on the core pinned-copy build: stock pinned copy_(non_blocking) staging/outputs as memcpy nodes (ARMV_VP_UVA=0
# ARMV_VP_PINNED=1) vs the UVA stopgap, same build. Check + tokens, async/synced timing, launch stamps, profiles.
source $(dirname $0)/timing_lib.sh
SHIM=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/armF/prep/launchshim.so
W=(--batch-size 1 8 64 128 --prefill-lens 64 --mixed 0)
PIN=(CAND_LINE=pinned ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_PINNED=1)
UVA=(CAND_LINE=pinned ARMV_VP=1 ARMV_VP_UVA=1)
trun c2_pinned_VPpin_Vcheck "${PIN[@]}" -- vllm/bench/drive.py --arm V --check --batch-size 1 8 64 128 --prefill-lens 64 --mixed 2 --out vllm/out/c2_pinned_VPpin_Vcheck.json
trun e2e_pinned_VPpin "${PIN[@]}" -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_pinned_VPpin.json
trun e2e_pinned_V CAND_LINE=pinned -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_pinned_V.json
for r in 1 2; do
  trun t_pvp_VPpin_r$r "${PIN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_pvp_VPpin_r$r.json
  trun t_pvp_VPuva_r$r "${UVA[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_pvp_VPuva_r$r.json
  trun t_pvp_V_r$r CAND_LINE=pinned -- vllm/bench/drive.py --arm V "${W[@]}" --async-decode 1 8 64 128 --out vllm/out/t_pvp_V_r$r.json
done
trun t_pvpls_VPpin "${PIN[@]}" LD_PRELOAD=$SHIM -- vllm/bench/drive.py --arm V "${W[@]}" --launch-stamps --out vllm/out/t_pvpls_VPpin.json
trun p_VPpin "${PIN[@]}" -- vllm/bench/prof_decode.py --arm V --bs 1 8 64 128 --out-dir vllm/out/prof_VPpin
echo ALLDONE
