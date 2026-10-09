#!/usr/bin/env bash
# Phase B round 3 on GPU 0 (one gpu0.lock hold; GPU 1 is reserved for candidate 3 gates; correctness runs are OK on GPU 0 while no timing job holds it): R1 with the recipe config (FlashInfer chosen by vLLM). Downloads of
# FlashInfer's trtllm-gen cubins are allowed for this batch only (user-approved 2026-10-07: fp8 KV, head 256, sm103a;
# FlashInfer fetches only the kernels it selects). The cubin cache is listed before and after (out/cubins_{before,after}.txt,
# out/cubins_downloaded.txt). Everything after this batch runs with FLASHINFER_NO_DOWNLOAD=1 again.
# Order: Qwen3-8B default check (a4, no downloads), then R1 default, eager, V. Each arm waits (<= 20 min) for >= 150 GiB free.
SV=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
CUB=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/cache/.cache/flashinfer/cubins
export TAG=${TAG:-b3} GPU=0 M=/data/eellison/models/Qwen3.8-27B-NVFP4 MAXLEN=262144 ATTN= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UTIL=0.5 SERVER_EXTRA="--kv-cache-dtype fp8 --language-model-only --kv-cache-memory-bytes 107374182400"
# GPU 0 must have >= 150 GiB free before this takes the lock (foreign jobs can hold all GPUs for hours; don't hold the lock idle)
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | tr -d , | awk '{print $1 - $2}') -ge 153600 ]; do sleep 60; done
echo "$(date +%T) GPU 0 free; waiting for gpu0.lock"
exec flock -x $LOCK bash -c '
echo "$(date +%T) holding gpu0.lock"
S=$0 CUB=$1
free_wait() {
  local t0=$(date +%s) used total
  while :; do
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 | tr -d ,)
    [ $(( total - used )) -ge 153600 ] && { echo "free ok ($(( (total - used) / 1024 )) GiB) after $(( $(date +%s) - t0 )) s"; return 0; }
    [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo "free CONTENDED: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0 | tr "\n" " ")"; return 1; }
    sleep 20
  done
}
snap() { find $CUB -type f ! -name "*.lock" -printf "%s %p\n" | sort -k2 > $S/out/cubins_$1.txt; echo "cubins_$1: $(wc -l < $S/out/cubins_$1.txt) files"; }
snap before
free_wait && env TAG=a4 M=/data/eellison/models/Qwen3-8B MAXLEN=10240 ATTN=FLASHINFER UTIL=0.15 SERVER_EXTRA="--kv-cache-memory-bytes 34359738368" FLASHINFER_NO_DOWNLOAD=1 timeout 2400 bash $S/session.sh default check1 burst8
unset FLASHINFER_NO_DOWNLOAD
free_wait && timeout 3000 bash $S/session.sh default check1 burst8 sweep:s8k:7373:922:0.1111:1,16 sweep:s1k:922:922:0.1111:64
free_wait && timeout 2700 bash $S/session.sh eager check1 burst8
free_wait && timeout 3600 bash $S/session.sh V check1 ctr:c1 checkon burst8:on ctr:burst checkoff sweep:s1k:922:922:0.1111:8 ctr:end summary:end
snap after
join -1 2 -2 2 -v 2 <(cut -d" " -f1,2 $S/out/cubins_before.txt | sort -k2) <(sort -k2 $S/out/cubins_after.txt) > $S/out/cubins_downloaded.txt
echo "downloaded: $(wc -l < $S/out/cubins_downloaded.txt) files, $(awk "{s+=\$2} END {print s}" $S/out/cubins_downloaded.txt) bytes"
echo PHASEB3_GPU0_DONE' $SV $CUB
