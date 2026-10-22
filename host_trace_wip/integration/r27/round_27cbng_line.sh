#!/usr/bin/env bash
# n27cbng (10-14): n27cbfg rerun after the unsynced-window fix
# round_27cbf_line.sh with the breakable+gdnbreak arm (10-14): vLLM breakable with Qwen's GDN op as an eager break (correctness-forced deviation); the whole churn workload (its windows mix sizes)
# r27 continuous-batching churn on vLLM's defaults (10-13): FlashInfer GDN prefill, max_num_seqs default, unsynced wall per output token
# (DRIVE_CB_NOSYNC). C = 16 / 64, outputs 256-2048, prompts 64-512, the seeded sequence once untimed then measured. Five arms by forward.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
source /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/timing_lib.sh
export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
# no first-time trace or cuBLAS learn inside a measured window of ours (user rule 10-13): drive_sweep27 exits 3 otherwise
export DRIVE_REQUIRE_WARM=1
# FlashInfer GDN: whether this process's GDN prefill NaNs, probed after the measurements (nan_probe in the JSON)
export DRIVE_NAN_PROBE=1
# churn wall clock without per-step syncs (NONDEFAULTS.md)
export DRIVE_CB_NOSYNC=1
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
OUT=$R27/out/timing; LOGD=$R27/logs/timing; mkdir -p $OUT $LOGD
KVB=$(( ${R27_KV_GIB:-40} * 1073741824 ))
D=$R27/bench/drive_sweep27.py
VC=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
# ours-uncompiled (user rule 10-13: no compile vs compile mixing): the eager forward with the registered _C launchers and VP_STOCK, the
# same kernels as default-nc / breakable
VS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
KW='{"kv_cache_dtype": "fp8"}'
KWC='{"kv_cache_dtype": "fp8", "compilation_config": {"cudagraph_mode": "NONE"}}'
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0
   --cb ${R27_CB:-16 64} --cb-steps ${R27_CB_STEPS:-3000} --cb-warm 400 --cb-out 256 2048 --cb-prompt 64 512 --cb-prewarm)
# 10-19 InferenceX-shaped fixed-seq (R27_IFX="L_in L_out", drive_sweep27 --ifx-*): the same engine and arms, one fixed-seq window per C
# (R27_CB) instead of the churn; untimed first: R27_IFX_WAVE (default 4) reps of the first C prompts (outputs capped at 64), then R27_IFX_PRE (default all) of the same request list. Bitwise check (vsvpchk_*):
# C=16, R27_IFX_CHK_MULT x 16 prompts, no untimed pass. learnifx_<line>: one untimed run of ours in the timed configuration (a step size first seen
# on a still-eager call of the untimed pass is learned in the window), bindings saved to bindings/$SET_v.pkl for every timed run of ours
CHKW=(--cb 16 64 --cb-steps 200 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512 --cb-prewarm)
if [ -n "${R27_IFX:-}" ]; then
  W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0
     --ifx-c ${R27_CB:-16 64} --ifx-shape $R27_IFX --ifx-mult ${R27_IFX_MULT:-10} --ifx-prewarm-frac ${R27_IFX_PRE:-1} --ifx-wave-reps ${R27_IFX_WAVE:-4})
  CHKW=(--ifx-c 16 --ifx-shape $R27_IFX --ifx-mult ${R27_IFX_CHK_MULT:-1} --ifx-prewarm-frac 0)
fi
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
for a in ${R27_ARMS:-default fullnc vcvp}; do
  t=t_${SET}_${a}_r$R
  case $a in
    default) trun27 $t R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0 -- $D --arm default "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    fullnc) trun27 $t R27_OVERLAY=$S -- $D --arm fullnc "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    # vLLM's breakable CUDA graphs (0.31's default for new architectures): no compile, FULL_AND_PIECEWISE, vLLM's default capture sizes
    breakable) trun27 $t R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- $D --arm fullnc_full "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    # breakable+gdnbreak (correctness-forced, 10-14): Qwen GDN as an eager break (patches/vllm_gdnbreak; out/BREAKABLE_GDN.md)
    breakablegb) trun27 $t R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- $D --arm fullnc_full "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    # the churn step sizes are timing-dependent (the engine batch queue), so the learn run also learns every size 1-1600 once (about 1 s per size; churn steps here are at most C + 2 x 512)
    learn) trun27 t_${SET}_learn "${VC[@]}" ARMV_BINDINGS_SAVE=$BIND DRIVE_PHASES=1 -- $D --arm Vc "${W[@]}" --learn-range 1 1600 --llm-kw "$KWC" --out $OUT/t_${SET}_learn.json ;;
    learnv) trun27 t_${SET}_learnv "${VS[@]}" ARMV_BINDINGS_SAVE=$BINDV DRIVE_PHASES=1 -- $D --arm V "${W[@]}" --learn-range 1 1600 --llm-kw "$KW" --out $OUT/t_${SET}_learnv.json ;;
    vsvp) [ -f $BINDV ] || { echo "$(date +%T) no $BINDV: refusing a cold-learn vsvp timing run"; continue; }
      trun27 $t "${VS[@]}" ARMV_BINDINGS_LOAD=$BINDV DRIVE_PHASES=1 -- $D --arm V "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json ;;
    vcvp) [ -f $BIND ] || { echo "$(date +%T) no $BIND: refusing a cold-learn vcvp timing run"; continue; }
      trun27 $t "${VC[@]}" ARMV_BINDINGS_LOAD=$BIND DRIVE_PHASES=1 -- $D --arm Vc "${W[@]}" --llm-kw "$KWC" --out $OUT/$t.json ;;
    # 10-17 remeasure: vsvp on the crash-fixed head (snap/step_ms17 + build_cpp_ms11; traced FlashInfer extension memset/tc/fi_build_ms8 as
    # learnfast's head runs; own extensions dir); bindings R27_V17_BIND (default n27cbn_v.pkl; DRIVE_REQUIRE_WARM exits 3 on a missed key)
    vsvp17|vsvp17chk|vsvp20|vsvp20chk|vsvp27|vsvp27chk) case $a in *27*) H17=(R27_WT=$I/snap/step_ms27 R27_INSTALL=$I/build_cpp_ms12c/install FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=$R27/torch_ext_ms27);; *20*) H17=(R27_WT=$I/snap/step_ms20 R27_INSTALL=$I/build_cpp_ms12b/install FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=$R27/torch_ext_ms20);; *) H17=(R27_WT=$I/snap/step_ms17 R27_INSTALL=$I/build_cpp_ms11/install FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=$R27/torch_ext_ms17);; esac
      B17=${R27_V17_BIND:-$R27/bindings/n27cbn_v.pkl}
      if [ $a = vsvp17 ] || [ $a = vsvp20 ] || [ $a = vsvp27 ]; then trun27 $t "${VS[@]}" "${H17[@]}" ARMV_BINDINGS_LOAD=$B17 DRIVE_PHASES=1 -- $D --arm V "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json
      else  # bitwise --check of the churn (C=16/64, 200 steps after 100 warm, Triton GDN prefill: the checkable configuration), untimed
        RERUN=1 DRIVE_REQUIRE_WARM=0 trun27 t_${SET}_${a} "${VS[@]}" "${H17[@]}" ARMV_BINDINGS_LOAD=$B17 ARMV_CHECK_TWICE=1 -- $D --arm V --check \
          --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 \
          --cb 16 64 --cb-steps 200 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512 --cb-prewarm \
          --llm-kw '{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton"}' --out $OUT/t_${SET}_${a}.json; fi ;;
    # 10-18: vsvp_<CAND_LINE> / vsvpchk_<CAND_LINE>: ours on any line of scratch/integration/python_vllm_cand2.sh (its snap + install), traced
    # FlashInfer extension memset/tc/fi_build_ms8 as learnfast's head runs, own extensions dir
    vsvp_*|vsvpchk_*|learnifx_*) ln=${a#*_}; spec=$(grep -E "^  $ln\)" /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/python_vllm_cand2.sh)
      WTL=$(echo "$spec" | sed -E 's/.*HOSTTRACE_WT=\$\{[A-Z0-9_]+:-([^}]*)\}.*/\1/'); WTL=${WTL//\$I/$I}
      INL=$(echo "$spec" | sed -E 's/.*HOSTTRACE_INSTALL=([^ ]*) .*/\1/'); INL=${INL//\$I/$I}
      [ -d "$WTL" ] && [ -d "$INL" ] || { echo "$(date +%T) $a: bad line spec ($WTL / $INL)"; continue; }
      HL=(R27_WT=$WTL R27_INSTALL=$INL FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=$R27/torch_ext_$ln)
      B17=${R27_V17_BIND:-$R27/bindings/n27cbn_v.pkl}
      if [ ${a%%_*} = vsvp ]; then trun27 $t "${VS[@]}" "${HL[@]}" ARMV_BINDINGS_LOAD=$B17 DRIVE_PHASES=1 -- $D --arm V "${W[@]}" --llm-kw "$KW" --out $OUT/$t.json
      elif [ ${a%%_*} = learnifx ]; then DRIVE_REQUIRE_WARM=0 trun27 t_${SET}_${a} "${VS[@]}" "${HL[@]}" ARMV_BINDINGS_LOAD=$B17 ARMV_BINDINGS_SAVE=$R27/bindings/${SET}_v.pkl DRIVE_PHASES=1 -- \
          $D --arm V "${W[@]}" --llm-kw "$KW" --out $OUT/t_${SET}_${a}.json
      else RERUN=1 DRIVE_REQUIRE_WARM=0 trun27 t_${SET}_${a} "${VS[@]}" "${HL[@]}" ARMV_BINDINGS_LOAD=$B17 ARMV_CHECK_TWICE=1 -- $D --arm V --check \
          --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 \
          "${CHKW[@]}" \
          --llm-kw '{"kv_cache_dtype": "fp8", "gdn_prefill_backend": "triton"}' --out $OUT/t_${SET}_${a}.json; fi ;;
  esac
done
