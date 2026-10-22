#!/usr/bin/env bash
# c3v7 + only scaled_fp4_quant as an extra extern (static_scaled_fp8_quant stays an eager step): the fp8 harvest fill gap avoided



export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
ARMV_EXTRA_EXTERN=scaled_fp4_quant.out bash serve/python_vllm_cand3.sh serve/r1_probe.py --out serve/out/c3v8.json --cute-observe --sym-origins --check --gdn-prefill triton
