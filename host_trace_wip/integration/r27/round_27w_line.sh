#!/usr/bin/env bash
# r27 slide sweeps (10-13, matching the Qwen3-8B set in integration/vllm/round_sweep_line.sh): Qwen3.8-27B NVFP4 on c5h (int6 snap/step324 +
# build_cpp12, build_c12), full-step metric (DRIVE_STEP_WITH_SAMPLE=1: execute_model + sample_tokens, every arm), one engine per arm-run.
# Rows: decode every bs 1-128 (20 tokens per point, first 4 steps dropped; vLLM's capture sizes in the default arm's cg_config), prefill
# 64-16000, mixed chunk 256-4096 x {16, 64} decodes at KV 1024-2048 (first two mixed steps per shape dropped), alternating decode
# 37<->50, 63<->65, 97<->100. Arms: default (compile + FULL_AND_PIECEWISE), fullnc (compile off + FULL_DECODE_ONLY), vcvp (Vc+VP: our
# step around vLLM's compiled forward with vLLM's own sampler in the graph; SOUND, every hybrid kind replayed).
# DIAGNOSIS CONFIG: gdn_prefill_backend=triton in every arm (FlashInfer's CuTe GDN prefill NaNs in some processes); in the n27s set
# mixed steps took ~250 ms in every arm, so Triton GDN prefill may dominate the mixed and prefill rows.
#   SET=n27w R=1 R27_ARMS=default bash round_27w_line.sh      (timing_gpu0_sweep.sh runs it per arm under gpu0.lock)
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
source /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/timing_lib.sh
export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
# no first-time trace or cuBLAS learn inside a measured window of ours (user rule 10-13): drive_sweep27 exits 3 otherwise
export DRIVE_REQUIRE_WARM=1
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
OUT=$R27/out/timing; LOGD=$R27/logs/timing; mkdir -p $OUT $LOGD
KVB=$(( ${R27_KV_GIB:-40} * 1073741824 ))
D=$R27/bench/drive_sweep27.py
VC=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1)
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64; do MIX+=(${p}x${nd}@1024-2048); done; done
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
KWC='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size $(seq 1 128) --decode-tokens 20
   --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0 --ix-mixed "${MIX[@]}"
   --alt-decode 37:50@1024-2048 63:65@1024-2048 97:100@1024-2048 --ix-reps 6)
trun27() {  # tag ENV=.. -- script args   (timing_lib's idle_check / util_now on r27's launcher)
  local tag=$1; shift; local envs=()
  if [ -f $OUT/$tag.json ] && [ -z "${RERUN:-}" ]; then echo "$(date +%T) have $tag"; return; fi
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  # host load gate (r27, 10-13): another user's build at 37 cores loaded every CPU set; wait (at most 30 min) for the 1-min load
  # average to drop under ${LOAD_MAX:-60}, then run anyway; the start line records the load either way
  local w=0; while [ "$(cut -d. -f1 /proc/loadavg)" -ge ${LOAD_MAX:-60} ] && [ $w -lt 1800 ]; do sleep 30; w=$((w + 30)); done
  local verdict=$(idle_check | tail -1)
  echo "$(date +%T) start $tag $verdict load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOGD/timing_idle.log
  case "$verdict" in "idle_check ok"*) ;; *) echo "$(date +%T) SKIPPED_CONTENDED $tag" | tee -a $LOGD/timing_idle.log; return ;; esac
  local u=$(util_now)
  env "${envs[@]}" INTEG_GPU_UTIL=$u INTEG_KV_BYTES=$KVB CUDA_VISIBLE_DEVICES=${HT_GPU:-0} timeout 3000 bash $R27/python_r27.sh "$@" > $LOGD/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$? util=$u"
}
# cuBLAS bindings learned once offline (arm learn: the same workload, untimed, ARMV_BINDINGS_SAVE) and preloaded by every timed vcvp
# run (ARMV_BINDINGS_LOAD), so no cuBLAS learn lands in a timed step (user rule 10-13: GEMMs warmed, the deployed configuration)
BIND=$R27/bindings/$SET.pkl
for a in ${R27_ARMS:-default fullnc vcvp}; do
  t=t_${SET}_${a}_r$R
  case $a in
    default) trun27 $t R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0 -- $D --arm default "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    fullnc) trun27 $t R27_OVERLAY=$S -- $D --arm fullnc "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    # vLLM's breakable CUDA graphs (0.31's default for new architectures): no compile, FULL_AND_PIECEWISE, vLLM's default capture sizes
    breakable) trun27 $t R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- $D --arm fullnc_full "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    learn) trun27 t_${SET}_learn "${VC[@]}" ARMV_BINDINGS_SAVE=$BIND DRIVE_PHASES=1 -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/t_${SET}_learn.json ;;
    vcvp) [ -f $BIND ] || { echo "$(date +%T) no $BIND: refusing a cold-learn vcvp timing run"; continue; }
      trun27 $t "${VC[@]}" ARMV_BINDINGS_LOAD=$BIND -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/$t.json ;;
  esac
done
