#!/usr/bin/env bash
# r27 timing round (DRAFT for the coordinator; GPU 0 / B200, not run by the r27 lane): Qwen3.8-27B NVFP4, one round R of set SET,
# arms default (recipe: compile + FULL_AND_PIECEWISE graphs), default-nc (compile off + FULL_DECODE_ONLY) and V-stock (SOUND:
# ARMV_SOUND=1, only cuBLAS harvested; trtllm-gen through FlashInfer's traced C++ launcher; r27/armV_r27 with pinned host args and the
# AutoTuner fix), arms rotated per round, all on candidate 5 (int6 snap/step310 + build_cpp10) through r27/python_r27.sh.
# The default arms run stock FlashInfer (R27_OVERLAY = the site itself, no trtllm_cpp hook in their import path).
# DIAGNOSIS CONFIG: gdn_prefill_backend=triton in every arm (FlashInfer's CuTe GDN prefill NaNs in some processes, README_repro_fi_gdn_nan.md).
# Shapes: decode bs 1/8/64 at short KV and bs 8/64 at KV 6000-8000; mixed 1k1k-shaped (1024/2048-token chunk + 64 decoders at KV
# 1024-2048) and 8k1k prompt-tail (the same at KV 6000-8000). Recipe flags: --language-model-only, kv fp8, no prefix caching.
#   SET=n27 R=1 bash round_27s_line.sh        (uses scratch/integration/vllm/timing_lib.sh: idle_check, util_now)
# Env knobs: R27_KV_GIB (default 40), HT_GPU (timing_lib's GPU index, default 0), R27_EAGER_KINDS (default prefill,mixed: V-stock
# replays decode only and runs vLLM's eager forward for prefill / mixed by configuration, as full_decode_only does, until the
# prefill replay mismatch (r27 STATUS) is fixed; R27_EAGER_KINDS= (empty) replays every kind). ARMV_PORTS=1: the vLLM _C launchers
# (integration/vllm/vllm_ports), so a decode variant is one segment with no eager step.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
source /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/timing_lib.sh
export UTIL_CAP=0.9
OUT=$R27/out/timing; LOGD=$R27/logs/timing; mkdir -p $OUT $LOGD
KVB=$(( ${R27_KV_GIB:-40} * 1073741824 ))
VS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=${R27_EAGER_KINDS-prefill,mixed})
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 8 64 --prefill-lens 64 --mixed 0
   --ix-decode 8:6000-8000 64:6000-8000
   --ix-mixed 1024x64@1024-2048 2048x64@1024-2048 1024x64@6000-8000 2048x64@6000-8000 --ix-reps 6 ${R27_W_EXTRA:-}
   --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}')
# R27_W (bash words, --llm-kw last): replaces W, e.g. the mixed-step set (timing_gpu0_m.sh)
[ -n "${R27_W:-}" ] && eval "W=($R27_W)"
trun27() {  # tag ENV=.. -- script args   (timing_lib's trun on r27's launcher)
  local tag=$1; shift; local envs=()
  if [ -f $OUT/$tag.json ] && [ -z "${RERUN:-}" ]; then echo "$(date +%T) have $tag"; return; fi
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local verdict=$(idle_check | tail -1)
  echo "$(date +%T) start $tag $verdict load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOGD/timing_idle.log
  case "$verdict" in "idle_check ok"*) ;; *) echo "$(date +%T) SKIPPED_CONTENDED $tag" | tee -a $LOGD/timing_idle.log; return ;; esac
  local u=$(util_now)
  env "${envs[@]}" INTEG_GPU_UTIL=$u INTEG_KV_BYTES=$KVB CUDA_VISIBLE_DEVICES=${HT_GPU:-0} timeout 3000 bash $R27/python_r27.sh "$@" > $LOGD/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$? util=$u"
}
arms=(${R27_ARMS:-default fullnc vstock}); na=${#arms[@]}; k=$(( (R - 1) % na )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
VC=("${VS[@]}" ARMV_INDUCTOR_SEAMS=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 ${R27_VC_EXTRA:-})
WC=("${W[@]:0:${#W[@]}-2}" --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}')
for a in "${arms[@]}"; do
  case $a in
    default) trun27 t_${SET}_default_r$R R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_fn.py --arm default "${W[@]}" --out $OUT/t_${SET}_default_r$R.json ;;
    fullnc) trun27 t_${SET}_fullnc_r$R R27_OVERLAY=$S -- vllm/bench/drive_fn.py --arm fullnc "${W[@]}" --out $OUT/t_${SET}_fullnc_r$R.json ;;
    vstock) trun27 t_${SET}_vstock_r$R "${VS[@]}" -- vllm/bench/drive_fn.py --arm V "${W[@]}" --out $OUT/t_${SET}_vstock_r$R.json ;;
    vc) trun27 t_${SET}_vc_r$R "${VC[@]}" -- vllm/bench/drive_fn.py --arm Vc "${WC[@]}" --out $OUT/t_${SET}_vc_r$R.json ;;
    vcvp) trun27 t_${SET}_vcvp_r$R "${VC[@]}" ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 -- vllm/bench/drive_fn.py --arm Vc "${WC[@]}" --out $OUT/t_${SET}_vcvp_r$R.json ;;
  esac
done
echo ALLDONE
