#!/usr/bin/env bash
# One-time download (user-approved 10-09 "fine to download whatever, just make sure we don't blow up disk space"):
# FlashInfer's trtllm-gen MoE (batched-GEMM) cubins, fetched by one stock-eager Flash-Next run with vLLM's default NVFP4
# MoE backend (FLASHINFER_TRTLLM) and FLASHINFER_NO_DOWNLOAD unset, into the scratch FlashInfer cache serve/fi_ws_ht.
# Cache listing and df before/after -> serve/out/moe_cubins_{before,after,downloaded}.txt, serve/out/moe_df.txt.
S=serve; CUB=$S/fi_ws_ht/.cache/flashinfer/cubins
M=/data/eellison/models/Qwen3.8-Flash-Next-NVFP4
KW='{"max_num_seqs": 32, "max_num_batched_tokens": 8192, "attention_config": {"indexer_kv_dtype": "fp8"}, "enable_flashinfer_autotune": false}'
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
{ date; df -h /data; } > $S/out/moe_df.txt
find $CUB -type f ! -name "*.lock" -printf "%s %p\n" | sort -k2 > $S/out/moe_cubins_before.txt
env -u FLASHINFER_NO_DOWNLOAD bash $S/python_vllm_s288.sh $S/r1_probe.py --out $S/out/dl_moe.json --arm eager --model $M --max-model-len 16384 --kv-gib 30 --kv-dtype auto --cute-observe --llm-kw "$KW"
echo "dl_moe rc=$?"
find $CUB -type f ! -name "*.lock" -printf "%s %p\n" | sort -k2 > $S/out/moe_cubins_after.txt
join -1 2 -2 2 -v 2 <(sort -k2 $S/out/moe_cubins_before.txt) <(sort -k2 $S/out/moe_cubins_after.txt) > $S/out/moe_cubins_downloaded.txt
echo "downloaded: $(wc -l < $S/out/moe_cubins_downloaded.txt) files, $(awk '{s+=$2} END {print s}' $S/out/moe_cubins_downloaded.txt) bytes"
{ date; df -h /data; } >> $S/out/moe_df.txt
cat $S/out/moe_df.txt
