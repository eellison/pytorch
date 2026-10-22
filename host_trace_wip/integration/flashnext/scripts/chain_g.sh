#!/usr/bin/env bash
# chain_g.sh: (after the side-streams lane's waiting jobs, gpu3.sh yields) the PLE repro, then 3 stock processes in the full
# diagnosis config (Triton GDN + deterministic QSA order).
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
CMD_TIMEOUT=600 bash gpu3.sh r_ple2 bash python_fn.sh tests/repro_ple_conv.py
sleep 60
FN_DL1=0 FN_STOCK_ARGS=--qsa-det bash scripts/stock3.sh fn_td '"gdn_prefill_backend": "triton"'
