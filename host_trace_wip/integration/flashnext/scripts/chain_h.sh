#!/usr/bin/env bash
# chain_h.sh: re-queue (02:1x, gpu3.sh now yields to the MiMo lane's and side-streams' waiting jobs) of the PLE repro and
# stock fn_td processes 2 and 3 (full diagnosis config: Triton GDN + deterministic QSA order), then the 3-way compare.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
M=/data/eellison/models/Qwen3.8-Flash-Next-NVFP4
CMD_TIMEOUT=600 bash gpu3.sh r_ple4 bash python_fn.sh tests/repro_ple_conv.py
for i in 2 3; do
  sleep 60
  INTEG_GPU_UTIL=0.88 CMD_TIMEOUT=1200 bash gpu3.sh fn_td_$i bash python_fn.sh scripts/stock_ref.py --model $M --out out/fn_td_$i.json \
    --llm-kw '{"language_model_only": true, "gdn_prefill_backend": "triton"}' \
    --quality data/ppl_passage_pytorch_README.md data/ppl_passage_lane_status_20261013.md --qsa-det
  echo "$(date +%T) fn_td_$i rc=$?"
done
python3 scripts/cmp_refs.py out/fn_td_1.json out/fn_td_2.json out/fn_td_3.json
