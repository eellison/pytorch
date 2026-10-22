#!/usr/bin/env bash
# chain_c.sh WAIT_PID: the Triton-GDN diagnosis config (gdn_prefill_backend=triton, as the 27B lane): 3 fresh stock eager processes.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
while kill -0 $1 2>/dev/null; do sleep 15; done
FN_DL1=0 bash scripts/stock3.sh fn_t '"gdn_prefill_backend": "triton"'
