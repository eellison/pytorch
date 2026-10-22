#!/usr/bin/env bash
# GPU 1 chain after smoke1: V-stock --check per backend (Qwen3-8B, R1), then fp8-KV acceptance runs. One hold per run.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
source attn_open/vs_env.sh
until grep -q "end q8_triton_V_check" attn_open/logs/jobs.log; do sleep 30; done
run() { local t=$1; shift; [ -f attn_open/out/$t.json ] || bash attn_open/gpu1.sh $t "$@"; }
K16=(INTEG_KV_BYTES=17179869184 INTEG_GPU_UTIL=0.3)
K24=(INTEG_KV_BYTES=25769803776 INTEG_GPU_UTIL=0.3)
run q8_fa_V_check "${VS[@]}" ATTN_BACKEND=FLASH_ATTN ATTN_CUTE_OBSERVE=1 "${K16[@]}" -- attn_open/drive_attn.py --arm V --check "${CHECKW[@]}" --out attn_open/out/q8_fa_V_check.json
run q8_flex_V_check "${VS[@]}" ATTN_BACKEND=FLEX_ATTENTION "${K16[@]}" -- attn_open/drive_attn.py --arm V --check "${CHECKW[@]}" --out attn_open/out/q8_flex_V_check.json
# R1: NVFP4 GEMM swapped to the CUTLASS kernel (diagnosis: vLLM's default CuTe DSL mm_fp4 sinks every trace, CORE_TASKS h(ii)),
# static fp8 quant left eager (h(i)); Triton GDN prefill (FlashInfer GDN prefill is nondeterministic, h(iii))
R1D=(VLLM_DISABLED_KERNELS=FlashInferCuteDslNvFp4LinearKernel ARMV_EXTRA_EXTERN=scaled_fp4_quant.out)
R1W=(--batch-size 1 8 --prefill-lens 64 512 --prefill-reqs 4 --mixed 3 --decode-tokens 16 --no-host-probe)
run r1_fp8_TRITON_ATTN_V_check "${VS[@]}" "${R1D[@]}" ATTN_BACKEND=TRITON_ATTN "ATTN_LLM_KW=$R1KW_FP8" "${K24[@]}" -- attn_open/drive_attn.py --arm V --check --model $R1 "${R1W[@]}" --out attn_open/out/r1_fp8_TRITON_ATTN_V_check.json
run r1_bf16_TRITON_ATTN_V_check "${VS[@]}" "${R1D[@]}" ATTN_BACKEND=TRITON_ATTN "ATTN_LLM_KW=$R1KW_BF16" "${K24[@]}" -- attn_open/drive_attn.py --arm V --check --model $R1 "${R1W[@]}" --out attn_open/out/r1_bf16_TRITON_ATTN_V_check.json
run r1_bf16_FLASH_ATTN_V_check "${VS[@]}" "${R1D[@]}" ATTN_BACKEND=FLASH_ATTN ATTN_CUTE_OBSERVE=1 "ATTN_LLM_KW=$R1KW_BF16" "${K24[@]}" -- attn_open/drive_attn.py --arm V --check --model $R1 "${R1W[@]}" --out attn_open/out/r1_bf16_FLASH_ATTN_V_check.json
run r1_bf16_FLEX_ATTENTION_V_check "${VS[@]}" "${R1D[@]}" ATTN_BACKEND=FLEX_ATTENTION "ATTN_LLM_KW=$R1KW_BF16" "${K24[@]}" -- attn_open/drive_attn.py --arm V --check --model $R1 "${R1W[@]}" --out attn_open/out/r1_bf16_FLEX_ATTENTION_V_check.json
ACC=(--batch-size 1 8 --prefill-lens 64 512 --mixed 2 --decode-tokens 12 --no-host-probe)
for b in FLASH_ATTN TRITON_ATTN FLEX_ATTENTION; do
  run acc_q8_fp8_$b ATTN_BACKEND=$b 'ATTN_LLM_KW={"kv_cache_dtype": "fp8"}' "${K16[@]}" -- attn_open/drive_attn.py --arm eager "${ACC[@]}" --out attn_open/out/acc_q8_fp8_$b.json
done
echo ALLDONE
