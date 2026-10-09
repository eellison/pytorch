#!/usr/bin/env bash
# Phase B round 1 on GPU 1 (one gpu1.lock hold): R1 = Qwen3.8-27B NVFP4 under `vllm serve` with the recipe's TP1 args
# (--kv-cache-dtype fp8, max_model_len 262144; attention backend left to vLLM) plus --language-model-only (the recipe's
# text_only option; without it vLLM computes inputs_embeds outside the model forward). No downloads: HF offline,
# FLASHINFER_NO_DOWNLOAD=1 (a missing cubin fails loudly instead of being fetched).
# Fixed 100 GiB KV pool (48 GiB gave 1003 Mamba blocks < max_num_seqs 1024, which vLLM rejects for graph capture) and
# util 0.5 for the startup check. Foreign processes on GPU 1 (no lock) swing by ~180-230 GB: before each arm wait
# (<= 20 min) for >= 150 GiB free.
SV=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock
export TAG=${TAG:-b1} GPU=1 M=/data/eellison/models/Qwen3.8-27B-NVFP4 MAXLEN=262144 ATTN= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FLASHINFER_NO_DOWNLOAD=1
export UTIL=0.5 SERVER_EXTRA="--kv-cache-dtype fp8 --language-model-only --kv-cache-memory-bytes 107374182400"
echo "$(date +%T) waiting for gpu1.lock"
exec flock -x $LOCK bash -c '
echo "$(date +%T) holding gpu1.lock"
S=$0
free_wait() {
  local t0=$(date +%s) used total
  while :; do
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | tr -d ,)
    [ $(( total - used )) -ge 153600 ] && { echo "free ok ($(( (total - used) / 1024 )) GiB) after $(( $(date +%s) - t0 )) s"; return 0; }
    [ $(( $(date +%s) - t0 )) -gt 1200 ] && { echo "free CONTENDED: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 1 | tr "\n" " ")"; return 1; }
    sleep 20
  done
}
for arm in ${ARMS:-default eager}; do free_wait && timeout 2700 bash $S/session.sh $arm check1; done
[ -n "${WITH_V:-}" ] && free_wait && timeout 2700 bash $S/session.sh V check1 ctr:c1 checkon burst8:on ctr:burst summary:end
echo PHASEB1_GPU1_DONE' $SV
