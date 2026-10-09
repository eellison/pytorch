# Qwen3.8-Flash-Next NVFP4 at TP1: InferenceX's SGLang SKU args (inferencex/b300-fp4-mtp_agentic.yaml, read at e0315d2a)
# without MTP (the NEXTN arm comes later; SGL_ENABLE_JIT_DEEPGEMM is spelled SGLANG_ENABLE_JIT_DEEPGEMM at cc012ab), as model args for run_s.sh: MODEL_ARGS="$Q38_ARGS".
Q38=/data/eellison/models/Qwen3.8-Flash-Next-NVFP4
export Q38_ARGS="--model-path $Q38 --trust-remote-code --tensor-parallel-size 1 --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.8 --max-running-requests ${Q38_MAXREQ:-32}"
export SGLANG_ENABLE_JIT_DEEPGEMM=false SGLANG_ENABLE_FLASHINFER_GEMM=true
