#!/usr/bin/env bash
S=serve; CUB=$S/fi_ws_ht/.cache/flashinfer/cubins
export FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht
{ date; df -h /data; } > $S/out/moe2_df.txt
find $CUB -type f ! -name "*.lock" -printf "%s %p\n" | sort -k2 > $S/out/moe2_before.txt
CUDA_VISIBLE_DEVICES= env -u FLASHINFER_NO_DOWNLOAD bash $S/python_vllm_s288.sh $S/repro/fetch_moe_cubins.py
echo "fetch rc=$?"
find $CUB -type f ! -name "*.lock" -printf "%s %p\n" | sort -k2 > $S/out/moe2_after.txt
join -1 2 -2 2 -v 2 <(sort -k2 $S/out/moe2_before.txt) <(sort -k2 $S/out/moe2_after.txt) > $S/out/moe2_downloaded.txt
echo "downloaded: $(wc -l < $S/out/moe2_downloaded.txt) files, $(awk '{s+=$2} END {print s}' $S/out/moe2_downloaded.txt) bytes"
{ date; df -h /data; } >> $S/out/moe2_df.txt; cat $S/out/moe2_df.txt
