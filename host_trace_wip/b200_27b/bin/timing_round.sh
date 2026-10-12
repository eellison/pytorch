#!/usr/bin/env bash
# timing_round.sh R: one round of the 27B DECODE-ONLY timing set, arms rotated by R: default (vLLM's recipe: torch.compile +
# FULL_AND_PIECEWISE graphs), default-nc (compile off + FULL_DECODE_ONLY graphs), vstock (enforce_eager + host-trace replay,
# SOUND, prefill / mixed steps run vLLM's eager forward by configuration: ARMV_EAGER_KINDS=prefill,mixed, as full_decode_only
# does). Each run waits for the GPU to have no other compute process; a busy GPU skips the run (tagged SKIPPED_BUSY).
# Run on one GPU: CUDA_VISIBLE_DEVICES=N bash bin/timing_round.sh 1
set -uo pipefail
B="$(cd "$(dirname "$0")/.." && pwd)"; source "$B/env.sh"
R=$1
OUT=$WORK/timing; mkdir -p "$OUT"
GPU=${CUDA_VISIBLE_DEVICES:-0}
KVB=$(( ${KV_GIB_TIMING:-60} << 30 ))
W=(--language-model-only --max-model-len 16384 --batch-size 1 8 64 --prefill-lens 64 --mixed 0
   --ix-decode 8:6000-8000 64:6000-8000 --ix-reps 6 --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128}')
idle() {  # no compute process on the GPU and <= 5% util over 10 samples (waits up to IDLE_MAX s)
  local t0=$(date +%s)
  while :; do
    local apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | wc -l) busy=0
    for i in $(seq 10); do [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU")" -gt 5 ] && busy=1; sleep 1; done
    [ "$apps" = 0 ] && [ $busy = 0 ] && return 0
    [ $(( $(date +%s) - t0 )) -gt ${IDLE_MAX:-600} ] && return 1
    sleep 20
  done
}
run() {  # tag arm env... ; skips a tag whose json exists
  local tag=$1 arm=$2; shift 2
  [ -f "$OUT/$tag.json" ] && { echo "have $tag"; return; }
  if ! idle; then echo "$(date +%T) SKIPPED_BUSY $tag" | tee -a "$OUT/timing.log"; return; fi
  echo "$(date +%T) start $tag load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/timing.log"
  env "$@" INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.8} INTEG_KV_BYTES=$KVB FLASHINFER_NO_DOWNLOAD=1 timeout 3600 \
    bash "$B/bin/python_27b.sh" "$B/bench/drive_fn.py" --arm "$arm" "${W[@]}" --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  echo "$(date +%T) end $tag rc=$?" | tee -a "$OUT/timing.log"
}
# the stock arms run stock FlashInfer (no overlay) in their own JIT workspace, sharing the downloaded cubins
STOCK=(NO_OVERLAY=1 FLASHINFER_WORKSPACE_BASE=$WORK/fi_ws_stock FLASHINFER_CUBIN_DIR=$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/cubins)
VS=($(sed -n 's/^export //p' "$B/bin/vstock_env.sh" | tr ' ' '\n' | grep =) ARMV_EAGER_KINDS=prefill,mixed)
arms=(default fullnc vstock); k=$(( (R - 1) % 3 )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  case $a in
    default) run t_default_r$R default "${STOCK[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ;;
    fullnc) run t_fullnc_r$R fullnc "${STOCK[@]}" ;;
    vstock) run t_vstock_r$R V "${VS[@]}" ;;
  esac
done
echo ALLDONE
