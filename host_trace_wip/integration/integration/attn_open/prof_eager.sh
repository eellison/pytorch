#!/usr/bin/env bash
# Eager attention-kernel profiles on GPU 0, one hold per run (acceptance is recorded by the same runs).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
source attn_open/vs_env.sh
for b in FLASHINFER FLASH_ATTN TRITON_ATTN FLEX_ATTENTION; do
  t=pe_q8_$b
  [ -f attn_open/out/$t.json ] || bash attn_open/gpu0.sh $t ATTN_BACKEND=$b INTEG_KV_BYTES=17179869184 INTEG_GPU_UTIL=0.3 -- attn_open/drive_attn.py --arm eager "${PROFW[@]}" --profile-dir attn_open/prof/$t --out attn_open/out/$t.json
done
for kv in fp8 bf16; do
  kw=$R1KW_FP8; [ $kv = bf16 ] && kw=$R1KW_BF16
  for b in FLASHINFER FLASH_ATTN TRITON_ATTN FLEX_ATTENTION; do
    t=pe_r1_${kv}_$b
    [ -f attn_open/out/$t.json ] || bash attn_open/gpu0.sh $t ATTN_BACKEND=$b "ATTN_LLM_KW=$kw" INTEG_KV_BYTES=25769803776 INTEG_GPU_UTIL=0.3 -- attn_open/drive_attn.py --arm eager --model $R1 "${PROFW[@]}" --profile-dir attn_open/prof/$t --out attn_open/out/$t.json
  done
done
echo ALLDONE
