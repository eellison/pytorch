#!/usr/bin/env bash
# r27 decode curves (10-13, InferenceX / vllm bench latency convention): Qwen3.8-27B NVFP4 on c5h, full-step metric (execute_model +
# sample_tokens, every arm). Per point: bs decoders filled at a ~1K or ~8K-token context, 16 warm-up decode steps, then 128 measured
# decode steps; per-token decode time = the mean step over the span. Every bs 1-128 at both contexts; the alternating decode rows
# (37<->50, 63<->65, 97<->100) again here so they are measured with the arms interleaved per round. Arms default,
# fullnc, vcvp (preloaded cuBLAS bindings; 0 traces / learns in the measured steps or the run fails). KV pool 72 GiB (a request holds ceil(8.2K / block) attention pages plus a GDN state page per group: 48 GiB stalled the 8K fill at bs 112).
#   SET=n27d R=1 R27_ARMS=default bash round_27d_line.sh      (timing_gpuN.sh runs it per arm under the GPU's lock)
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
# ours-uncompiled (user rule 10-13: no compile vs compile mixing): the eager forward with the registered _C launchers and VP_STOCK, the
# same kernels as default-nc / breakable
VS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1)
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
KWC='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
BS="$(seq -s ' ' 1 128)"
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
   --dcurve-ctx 1024 8192 --dcurve-bs $BS --dcurve-steps 128 --dcurve-warm 16
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
  env "${envs[@]}" INTEG_GPU_UTIL=$u INTEG_KV_BYTES=$KVB CUDA_VISIBLE_DEVICES=${HT_GPU:-0} timeout ${R27_TIMEOUT:-3000} bash $R27/python_r27.sh "$@" > $LOGD/$tag.log 2>&1
  echo "$(date +%T) end $tag rc=$? util=$u"
}
# cuBLAS bindings learned once offline (arm learn: the same workload, untimed, ARMV_BINDINGS_SAVE) and preloaded by every timed vcvp
# run (ARMV_BINDINGS_LOAD), so no cuBLAS learn lands in a timed step (user rule 10-13: GEMMs warmed, the deployed configuration)
BIND=$R27/bindings/$SET.pkl
BINDV=$R27/bindings/${SET}_v.pkl
# our decode-curve arms run the (untimed) fills' prefill / mixed steps on vLLM's eager forward by configuration: every fill step is a
# new prefill-request count (CORE_TASKS xi), so tracing them only made the fills slow; the measured steps are pure decode replays
WL=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size $(seq -s ' ' 1 128) --decode-tokens 8 --prefill-lens 64 --mixed 0)
for a in ${R27_ARMS:-default fullnc vcvp}; do
  t=t_${SET}_${a}_r$R
  case $a in
    default) trun27 $t R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0 -- $D --arm default "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    fullnc) trun27 $t R27_OVERLAY=$S -- $D --arm fullnc "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    # vLLM's breakable CUDA graphs (0.31's default for new architectures): no compile, FULL_AND_PIECEWISE, vLLM's default capture sizes
    breakable) trun27 $t R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- $D --arm fullnc_full "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    learn) trun27 t_${SET}_learn "${VC[@]}" ARMV_BINDINGS_SAVE=$BIND DRIVE_PHASES=1 -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/t_${SET}_learn.json ;;
    # decode curves bind only decode cuBLAS keys (M = bs): learn them on a short-context decode sweep of every bs
    learnv) trun27 t_${SET}_learnv "${VS[@]}" ARMV_BINDINGS_SAVE=$BINDV DRIVE_PHASES=1 -- $D --arm V "${WL[@]}" --llm-kw "$KW" --out $OUT/t_${SET}_learnv.json ;;
    vsvp) [ -f $BINDV ] || { echo "$(date +%T) no $BINDV: refusing a cold-learn vsvp timing run"; continue; }
      trun27 $t "${VS[@]}" ARMV_EAGER_KINDS=prefill,mixed ARMV_BINDINGS_LOAD=$BINDV DRIVE_PHASES=1 -- $D --arm V "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    vcvp) [ -f $BIND ] || { echo "$(date +%T) no $BIND: refusing a cold-learn vcvp timing run"; continue; }
      trun27 $t "${VC[@]}" ARMV_EAGER_KINDS=prefill,mixed ARMV_BINDINGS_LOAD=$BIND -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/$t.json ;;
  esac
done
