#!/usr/bin/env bash
# startup27.sh STEP ARM [R]: fair startup for Qwen3.8-27B-NVFP4 (the 8B lane's serve/repro/run_startup.sh, ported to r27's driver as
# learnfast/drive_startup27.py; r27's vLLM-default line: FlashInfer GDN, fp8 KV, KV 40 GiB). GPU $LF_GPU (caller holds the lock).
#  seed  ARM=vc|vcl: minimal warm-up + 20 stream steps, harvested cuBLAS keys saved -> out/startup27_seed_<arm>.pkl
#  learn ARM=vc|vcl: learnfast learn_range.sh from the seed over token counts 1-16384 (sampled rows 1-128) -> out/bind_startup27_<arm>.pkl
#  run   ARM=default|fullnc|vc|vcl R=n: engine ready + the request stream -> out/startup27_<arm>_r<R>.json
#        vc = ours (Vc + VP, adapters as shipped); vcl = ours + Delta 1 (lifted per-step state, provenance criterion) + the fla overlay
#        brk = vLLM breakable+gdnbreak (no compile, FULL_AND_PIECEWISE, Qwen GDN as an eager break; r27's breakablegb)
#        v = ours on the eager forward (r27's VS + VP); vt = v + Delta 1 + fla overlay + templates (the talk's primary ours arm)
#  RANGE=1: ours' warm-up is the range warm-up (DRIVE_RANGE_WARM) instead of the minimal one. TRITON=1 (default): Triton GDN prefill,
#  max_num_seqs 128 (r27's timing line) for every arm; TRITON=0: the r1 line (vLLM default GDN, default max_num_seqs).
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
LF=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
STEP=$1 ARM=$2 R=${3:-1}
SFX=; [ "${HEAD:-0}" = 1 ] && SFX=_h
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export UTIL_CAP=${UTIL:-0.9} DRIVE_STEP_WITH_SAMPLE=1 DRIVE_REQUIRE_WARM=0 INTEG_GPU_UTIL=${UTIL:-0.9} INTEG_KV_BYTES=$(( ${KV_GIB:-64} << 30 ))
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
# HEAD=1: the head (core agent 10-15: snap/step_ms10 = step_ms9 + the template core patch, on build_cpp_ms8, the traced FlashInfer
# extension rebuilt against it); HEAD_SNAP=step_ms8 for the earlier head
[ "${HEAD:-0}" = 1 ] && export R27_WT=${HEAD_WT:-$I/snap/${HEAD_SNAP:-step_ms11}} R27_INSTALL=$I/build_cpp_ms8/install \
  FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=$LF/torch_ext_ms8
# HEAD=2: step_ms12 on build_cpp_ms9 (core agent 10-15: the nccl2 hook fix, so cuBLAS harvests work under a process group); the
# traced FlashInfer extension as for HEAD=1; fresh torch extensions dir
[ "${HEAD:-0}" = 2 ] && export R27_WT=${HEAD_WT:-$I/snap/${HEAD_SNAP:-step_ms12}} R27_INSTALL=$I/build_cpp_ms9/install \
  FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=$LF/torch_ext_ms9 && SFX=_h2
# HEAD=3: step_ms15 on build_cpp_ms10 (run-buffer section temporaries placed over their op's launches); fresh extensions dir
[ "${HEAD:-0}" = 3 ] && export R27_WT=${HEAD_WT:-$I/snap/${HEAD_SNAP:-step_ms15}} R27_INSTALL=${HEAD3_INSTALL:-$I/build_cpp_ms10/install} \
  FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=${HEAD3_EXT:-$LF/torch_ext_ms10} && SFX=_h3
export CUDA_VISIBLE_DEVICES=${LF_GPU:-0}
VC=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
# SECTIONS=1: sections on (no_sections lifted; the core's check_section, Delta 3 on a snap that has it); MEMO=1: the refusal memo
# sections on -> eager-order memory (notes/MEMORY_DECODE_THEN_PREFILL.md: run_buffer holds every section's temporaries for the run)
EXTRA=(); [ "${SECTIONS:-0}" = 1 ] && EXTRA+=(DIAG27_SECTIONS_ON=1 ARMV_MEMORY=${ARMV_MEMORY:-$([ "${HEAD:-0}" = 3 ] && echo run_buffer || echo eager)}) && OSFX=_sec; [ "${MEMO:-0}" = 1 ] && EXTRA+=(DIAG27_REFUSAL_MEMO=1) && OSFX=${OSFX:-}_m
[ "${KEYOUT:-1}" = 1 ] && EXTRA+=(TPL_LAYER_KEYOUT=1)
LIFT=(DIAG27_QUIET=${DIAG27_QUIET:-1} DIAG27_PROV=1 PROV_LIFT=1 PROV_RELAX_GUARDS=1 DIAG27_CLOSURE=1 DIAG27_CLOSURE_STORAGE=1 DIAG27_FNRESOLVE=1 DIAG27_CUTE_RESOLVE=1
      DIAG27_SET=redo_resolves_arguments=1 DIAG27_REDOCTX=1 DIAG27_REDOCTX_LAZY=1 DIAG27_LIFT=1 DIAG27_FLA_SYMLEN=1)
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0)
KW='{"kv_cache_dtype": "fp8"}'
KWC='{"kv_cache_dtype": "fp8", "compilation_config": {"cudagraph_mode": "NONE"}}'
if [ "${TRITON:-1}" = 1 ]; then
  KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
  KWC='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
  SFX=${SFX}t
fi
VS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
RW=(); [ "${RANGE:-0}" = 1 ] && RW=(DRIVE_RANGE_WARM=1) && RSFX=_rw
# EAGER=1: ours traces decode only; prefill / mixed steps run vLLM's eager forward (r27's vsvp convention); the range warm-up is
# then its decode phase alone
[ "${EAGER:-0}" = 1 ] && RW+=(ARMV_EAGER_KINDS=prefill,mixed DRIVE_RANGE_PHASES=decode) && RSFX=${RSFX:-}_ek
case $ARM in vc) OURS=("${VC[@]}");; vcl) OURS=("${VC[@]}" "${LIFT[@]}");; vclt) OURS=("${VC[@]}" "${LIFT[@]}" DIAG27_TPL=1);;
  v) OURS=("${VS[@]}");; vt) OURS=("${VS[@]}" "${LIFT[@]}" DIAG27_TPL=1 "${EXTRA[@]}");; esac
# the forward ours runs: Vc = the compiled forward (inductor seams), V = vLLM's eager forward (KW: r27's V arm builds its own)
case $ARM in v|vt) OARM=V OKW=$KW BARM=v;; *) OARM=Vc OKW=$KWC BARM=vc;; esac
D=$LF/drive_startup27.py
run_py() { timeout ${RUN_TIMEOUT:-1150} taskset -c ${LF_CPUS:-0-35} env "$@"; }
case $STEP in
  seed) t=$(date +%s.%N)
        run_py "${OURS[@]}" DRIVE_STARTUP=1 DRIVE_STARTUP_STEPS=20 DIAG27_FILE=$LF/out/startup27${SFX}_seed_$ARM.diag.jsonl ARMV_BINDINGS_SAVE=$LF/out/startup27${SFX}_seed_$ARM.pkl \
          bash $R27/python_r27.sh $LF/diag27.py $D --arm $OARM "${W[@]}" --llm-kw "$OKW" --out $LF/out/startup27${SFX}_seed_$ARM.json
        echo "seed $ARM rc=$? wall $(python3 -c "print(round($(date +%s.%N) - $t, 1))") s" ;;
  learn) t=$(date +%s.%N)
        (cd $LF && { [ "${HEAD:-0}" = 1 ] && export LF_WT=$I/snap/${HEAD_SNAP:-step_ms11} LF_INSTALL=$I/build_cpp_ms8/install; [ "${HEAD:-0}" = 2 ] && export LF_WT=$I/snap/${HEAD_SNAP:-step_ms12} LF_INSTALL=$I/build_cpp_ms9/install; [ "${HEAD:-0}" = 3 ] && export LF_WT=$I/snap/${HEAD_SNAP:-step_ms15} LF_INSTALL=$I/build_cpp_ms10/install; true; } && PY=./py.sh LF_GPU=${LF_GPU:-0} CPU0=${LF_CPU0:-0} N=8 bash learn_range.sh $LF/out/startup27${SFX}_seed_$ARM.pkl $LF/out/bind_startup27${SFX}_$ARM.pkl 16384 128 248320)
        echo "learn $ARM rc=$? wall $(python3 -c "print(round($(date +%s.%N) - $t, 1))") s" ;;
  run) case $ARM in
         default) run_py R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0 DRIVE_STARTUP=1 bash $R27/python_r27.sh $D --arm default "${W[@]}" --llm-kw "$KW" --out $LF/out/startup27${SFX}_default_r$R.json ;;
         eager) run_py R27_OVERLAY=$S DRIVE_STARTUP=1 DRIVE_RANGE_ANY=1 bash $R27/python_r27.sh $D --arm eager "${W[@]}" --llm-kw "$KW" --out $LF/out/startup27${SFX}_eager${RSFX:-}_r$R.json ;;
         fullnc) run_py R27_OVERLAY=$S DRIVE_STARTUP=1 bash $R27/python_r27.sh $D --arm fullnc "${W[@]}" --llm-kw "$KW" --out $LF/out/startup27${SFX}_fullnc_r$R.json ;;
         brk) run_py R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' DRIVE_STARTUP=1 \
                bash $R27/python_r27.sh $D --arm fullnc_full "${W[@]}" --llm-kw "$KW" --out $LF/out/startup27${SFX}_brk_r$R.json ;;
         vc|vcl|vclt|v|vt) run_py "${OURS[@]}" "${RW[@]}" DRIVE_STARTUP=1 ARMV_BINDINGS_LOAD=$LF/out/bind_startup27${SFX}_$BARM.pkl DIAG27_FILE=$LF/out/startup27${SFX}_${ARM}${RSFX:-}${OSFX:-}_r$R.diag.jsonl \
                   ${PY_PREFIX:-} bash $R27/python_r27.sh $LF/diag27.py $D --arm $OARM "${W[@]}" --llm-kw "$OKW" --out $LF/out/startup27${SFX}_${ARM}${RSFX:-}${OSFX:-}_r$R.json ;;
       esac
       echo "run $ARM r$R rc=$?" ;;
esac
