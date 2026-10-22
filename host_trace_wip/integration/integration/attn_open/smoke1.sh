#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
source attn_open/vs_env.sh
bash attn_open/gpu1.sh q8_triton_V_check "${VS[@]}" ATTN_BACKEND=TRITON_ATTN INTEG_KV_BYTES=17179869184 INTEG_GPU_UTIL=0.3 -- attn_open/drive_attn.py --arm V --check "${CHECKW[@]}" --out attn_open/out/q8_triton_V_check.json
