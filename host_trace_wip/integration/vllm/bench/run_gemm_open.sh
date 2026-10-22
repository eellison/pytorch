#!/usr/bin/env bash
# OPEN_PLAN.md GEMM scoping: bench/gemm_open.py in the vLLM env (cuBLAS, vLLM batch-invariant Triton, FlashInfer
# cute-dsl, Inductor max-autotune Triton) and the SGLang env (cuBLAS, TGV), GPU 2 under gpu2.lock per run.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
L=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu2.lock
cd $V
echo "=== vllm env $(date)"
CUDA_VISIBLE_DEVICES=2 flock $L timeout 3000 taskset -c 108-143 bash python_vllm_cpp.sh bench/gemm_open.py --out bench/out/gemm_open_v.json --backends cublas triton_bi fi_cute
echo "rc=$? $(date)"
echo "=== sglang env tgv $(date)"
CUDA_VISIBLE_DEVICES=2 flock $L timeout 1800 taskset -c 108-143 bash $S/python_sgltgv.sh bench/gemm_open.py --out bench/out/gemm_open_tgv.json --backends cublas tgv
echo "rc=$? $(date)"
echo "=== vllm env inductor $(date)"
CUDA_VISIBLE_DEVICES=2 TORCHINDUCTOR_CACHE_DIR=$V/cache/inductor_gemm flock $L timeout 4000 taskset -c 108-143 bash python_vllm_cpp.sh bench/gemm_open.py --out bench/out/gemm_open_ind.json --backends cublas inductor --ms 1 8 32 64 128 512 2048 8192
echo "rc=$? $(date)"
echo GODONE
