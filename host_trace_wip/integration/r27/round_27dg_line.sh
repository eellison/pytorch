#!/usr/bin/env bash
# n27dg (coordinator 10-15, the vLLM slides' only model): round_27dv2_line.sh (n27d) plus
#   - ours re-freezes the GC after its warm-up traces: ARMV_GC_FREEZE=1 (freeze after warm-up) and ARMV_REFREEZE=1 (gc.collect();
#     gc.freeze() after any later forward that traced or learned);
#   - every arm: DRIVE_GC_LOG=1 (gc.callbacks: generation, ms, step index) and per-point step stats (max, p99, steps > 10x median,
#     GC passes per slow step, PSI / outlier flag) in R["dcurve"][..]["stalls"];
#   - the own-core (0.30) and PSI (< 2% avg60) start gates run in timing_gpuN4.sh before the GPU lock.
# R27_DG_SMOKE=1: a ~2 min smoke workload (1K context, bs 1 / 2 / 64, 16 measured steps, no alternating rows).
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
export DRIVE_REQUIRE_WARM=1 DRIVE_GC_LOG=1
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
OUT=$R27/out/timing; LOGD=$R27/logs/timing; mkdir -p $OUT $LOGD
KVB=$(( ${R27_KV_GIB:-40} * 1073741824 ))
D=$R27/bench/drive_sweep27.py
VC=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_REFREEZE=1)
# ours-uncompiled (user rule 10-13: no compile vs compile mixing): the eager forward with the registered _C launchers and VP_STOCK, the
# same kernels as default-nc / breakable
VS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_REFREEZE=1)
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
KWC='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
BS="$(seq -s ' ' 1 128)"
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
   --dcurve-ctx 1024 8192 --dcurve-bs $BS --dcurve-steps 128 --dcurve-warm 16
   --alt-decode 37:50@1024-2048 63:65@1024-2048 97:100@1024-2048 --ix-reps 6)
# R27_DG_CTX / R27_DG_BS (flagged-point reruns, coordinator 10-15): only those points, no alternating rows
if [ -n "${R27_DG_CTX:-}" ]; then
  W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
     --dcurve-ctx $R27_DG_CTX --dcurve-bs $R27_DG_BS --dcurve-steps 128 --dcurve-warm 16)
fi
# R27_DG_CTX_V: the V+VP arms' contexts (the A/B runs V+VP at 1K only)
if [ -n "${R27_DG_CTX_V:-}" ] && [[ " ${R27_ARMS} " == *" vsvp"* ]]; then
  W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
     --dcurve-ctx $R27_DG_CTX_V --dcurve-bs $R27_DG_BS --dcurve-steps 128 --dcurve-warm 16)
fi
if [ -n "${R27_DG_SMOKE:-}" ]; then
  W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
     --dcurve-ctx 1024 --dcurve-bs 1 2 64 --dcurve-steps 16 --dcurve-warm 4)
fi
trun27() {  # tag ENV=.. -- script args   (timing_lib's idle_check / util_now on r27's launcher)
  local tag=$1; shift; local envs=()
  if [ -f $OUT/$tag.json ] && [ -z "${RERUN:-}" ]; then echo "$(date +%T) have $tag"; return; fi
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  # n27dg (10-16): the own-core gate again right before the run, inside the hold (the box's load went to ~600 with other users' jobs
  # on our pinned cores after the pre-lock gate passed): own cores 36-71 mean busy < ${OWN_BUSY_FRAC:-0.30} over 10 s, waiting up to
  # 4 h; past that the run starts flagged OWN_CORES_GATE_TIMEOUT. The start line records own_busy and the load either way.
  local w=0 own=1
  while own=$(python3 /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/cpubusy.py 10 36 71 | cut -d' ' -f1); \
        python3 -c "import sys; sys.exit(not ($own / 36 >= ${OWN_BUSY_FRAC:-0.30}))" && [ $w -lt 14400 ]; do sleep 50; w=$((w + 60)); done
  [ $w -ge 14400 ] && echo "$(date +%T) OWN_CORES_GATE_TIMEOUT $tag (flag this run)" | tee -a $LOGD/timing_idle.log
  echo "{\"own_busy_at_start\": $(python3 -c "print(round($own / 36, 3))"), \"own_wait_s\": $w}" > $OUT/$tag.startgate
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
    # breakable+gdnbreak (coordinator 10-15: the breakable decode arm; plain breakable's FULL decode is wrong at unpadded capture sizes,
    # out/dp27): Qwen GDN as an eager break (patches/vllm_gdnbreak)
    breakablegb) trun27 $t R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- $D --arm fullnc_full "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    learn) trun27 t_${SET}_learn "${VC[@]}" ARMV_BINDINGS_SAVE=$BIND DRIVE_PHASES=1 -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/t_${SET}_learn.json ;;
    # decode curves bind only decode cuBLAS keys (M = bs): learn them on a short-context decode sweep of every bs
    learnv) trun27 t_${SET}_learnv "${VS[@]}" ARMV_BINDINGS_SAVE=$BINDV DRIVE_PHASES=1 -- $D --arm V "${WL[@]}" --llm-kw "$KW" --out $OUT/t_${SET}_learnv.json ;;
    vsvp) [ -f $BINDV ] || { echo "$(date +%T) no $BINDV: refusing a cold-learn vsvp timing run"; continue; }
      trun27 $t "${VS[@]}" ARMV_EAGER_KINDS=prefill,mixed ARMV_BINDINGS_LOAD=$BINDV DRIVE_PHASES=1 -- $D --arm V "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    # re-freeze A/B (coordinator 10-16): ours without ARMV_REFREEZE (the later assignment wins in env)
    vcvpnr) [ -f $BIND ] || { echo "$(date +%T) no $BIND: refusing a cold-learn vcvp timing run"; continue; }
      trun27 $t "${VC[@]}" ARMV_REFREEZE=0 ARMV_EAGER_KINDS=prefill,mixed ARMV_BINDINGS_LOAD=$BIND -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/$t.json ;;
    vsvpnr) [ -f $BINDV ] || { echo "$(date +%T) no $BINDV: refusing a cold-learn vsvp timing run"; continue; }
      trun27 $t "${VS[@]}" ARMV_REFREEZE=0 ARMV_EAGER_KINDS=prefill,mixed ARMV_BINDINGS_LOAD=$BINDV DRIVE_PHASES=1 -- $D --arm V "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    vcvp) [ -f $BIND ] || { echo "$(date +%T) no $BIND: refusing a cold-learn vcvp timing run"; continue; }
      trun27 $t "${VC[@]}" ARMV_EAGER_KINDS=prefill,mixed ARMV_BINDINGS_LOAD=$BIND -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/$t.json ;;
  esac
done
