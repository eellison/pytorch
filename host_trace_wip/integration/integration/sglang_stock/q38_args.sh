# Qwen3.8-Flash-Next NVFP4 at TP1: InferenceX's SGLang SKU args (inferencex/b300-fp4-mtp_agentic.yaml, read at e0315d2a)
# MoE: --moe-runner-backend flashinfer_cutlass (deviation as in the vLLM serve lane: SGLang auto-picks FlashInfer's
# trtllm-gen FP4 MoE, whose batched-GEMM run fails here (q38_probe1: 'Error occurred when running GEMM', bmm_E2m1 cubin);
# Q38_MOE='' restores the auto choice).
# Default MoE is now the recipe default (trtllm-gen, cubins fetched 10-09): Q38_MOE is set to "" below unless given.
# Linear-attn prefill: triton (deviation; FlashInfer GDN prefill is degenerate in stock eager here, QUESTIONS_FOR_USER 4a).
# without MTP (the NEXTN arm comes later; SGL_ENABLE_JIT_DEEPGEMM is spelled SGLANG_ENABLE_JIT_DEEPGEMM at cc012ab), as model args for run_s.sh: MODEL_ARGS="$Q38_ARGS".
Q38=/data/eellison/models/Qwen3.8-Flash-Next-NVFP4
export Q38_ARGS="--model-path $Q38 --trust-remote-code --tensor-parallel-size 1 --linear-attn-prefill-backend ${Q38_LIN:-triton} --linear-attn-decode-backend ${Q38_LIN_DEC:-flashinfer} --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.8 --max-running-requests ${Q38_MAXREQ:-32} ${Q38_MOE-}"
export SGLANG_ENABLE_JIT_DEEPGEMM=false SGLANG_ENABLE_FLASHINFER_GEMM=true
