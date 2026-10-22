#!/usr/bin/env bash
# R1-V probe with cute.compile observed from process start AND a FlashInfer workspace without the AOT-cached CuTe DSL
# GEMMs (serve/fi_ws_ht: hard-link copy of scratch/vllm/cache/.cache/flashinfer minus cached_ops/mm_fp4_sm103a_cute_dsl),
# so FlashInfer compiles mm_fp4's CuTe DSL kernels under observation.
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
bash serve/python_vllm_cand3.sh serve/r1_probe.py --out serve/out/c3v7.json --cute-observe --sym-origins --check --gdn-prefill triton
