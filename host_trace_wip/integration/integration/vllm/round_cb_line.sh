#!/usr/bin/env bash
# cbh (serve lane, 10-13): continuous-batching churn, in-process, c5h, Qwen3-8B, full-step metric. C = 16 / 64 / 128 requests in
# flight; output lengths seeded uniform 256-2048 (ignore_eos), prompts 64-512; a finished request is replaced at once (its prefill
# lands as a mixed step). Untimed warm-up (every arm): one prefill per token count 1-3072 (1-1300 for cbh3 r1-r3 vcsl; every step size up to ~5 simultaneous admissions at C = 128), a short decode at every bs 1-160, three mixed steps; then 400 warm-up steps, 3000 measured steps per C. Arms default, fullnc, vcs (Vc + sampling, DRIVE_PHASES;
# cuBLAS keys learned cold as they appear), vcsl (vcs with the cuBLAS bindings preloaded from vllm/out/bindings_<SET>.pkl, which
# the "learn" step writes: one pass over token counts 1-1300 (single prefills) and decode batch sizes 1-160, same build and config).
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5h}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
[ "${CB_NOSYNC:-0}" = 1 ] && export DRIVE_CB_NOSYNC=1  # cbh4+: no syncs in the measured window (async scheduling as shipped); wall ms/token
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
# cbh4+ (main 10-14): ours re-freezes the GC heap after post-warm-up traces / learns (ARMV_REFREEZE, labelled in the tables)
RF=(); [ "$SET" != cbh ] && [ "$SET" != cbh2 ] && [ "$SET" != cbh3 ] && RF=(ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1)  # + ours' warm-up covers short-chunk mixed steps
VC=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 "${RF[@]}")
VR=("${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 "${RF[@]}")  # ours on vLLM's eager (uncompiled) forward: V-registered + sampling in the step
W=(--max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 64 128 --cb-steps 3000 --cb-warm 400 --cb-out 256 2048 --cb-prompt 64 512 --warm-range 1 3072 160)
# 10-19 InferenceX-shaped fixed-seq (IFX="L_in L_out"; r27 round_27cbng_line.sh's R27_IFX): one fixed-seq window per C in IFX_C (default 16 64)
# instead of the churn, after the same range warm-up; untimed first: IFX_WAVE (default 4) reps of the first C prompts (outputs capped at 64), then
# the whole seeded request list. Ours' cuBLAS bindings: VRS_EXTRA="ARMV_BINDINGS_LOAD=..." (learn run: VRS_EXTRA="DRIVE_STRICT_WINDOW=0 ARMV_BINDINGS_SAVE=...")
[ -n "${IFX:-}" ] && W=(--max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --ifx-c ${IFX_C:-16 64} --ifx-shape $IFX --ifx-mult ${IFX_MULT:-10}
                        --ifx-prewarm-frac ${IFX_PRE:-1} --ifx-wave-reps ${IFX_WAVE:-4} --warm-range 1 3072 160)
arms=(${ARMS:-default fullnc vcs}); k=$(( (R - 1) % ${#arms[@]} )); arms=("${arms[@]:$k}" "${arms[@]:0:$k}")
for a in "${arms[@]}"; do
  t=t_${SET}_${a}_r$R
  case $a in
    default) trun $t CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_sweep.py --arm default "${W[@]}" --out vllm/out/$t.json ;;
    fullnc) trun $t CAND_LINE=$L -- vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/$t.json ;;
    vrs) trun $t "${VR[@]}" ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;  # GEMMs learned in untimed rows / warm-up; strict windows
    bnc) trun $t CAND_LINE=$L VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- vllm/bench/drive_sweep.py --arm fullnc_full "${W[@]}" --out vllm/out/$t.json ;;  # vLLM breakable CUDA graphs (no compile, default capture sizes; vLLM 0.31's default for new architectures)
    vcs) trun $t "${VC[@]}" ARMV_VPS=1 -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;
    vcsl) trun $t "${VC[@]}" ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_${SET}.pkl -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;
    vcsl_h) trun $t "${VC[@]}" CAND_LINE=ms3 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;  # 10-14: vcsl on the new candidate head (snap/step_ms2b + build_cpp_ms3); cuBLAS keys learned in the untimed warm-up (bindings files are per build)
    vrs_h) trun $t "${VR[@]}" CAND_LINE=ms3 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;  # 10-14: vrs on the new candidate head
    vcsl_fh) trun $t "${VC[@]}" CAND_LINE=ms3 DRIVE_FORM_HELD=1 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;  # cbh4c: head + form_held on
    vrs_fh) trun $t "${VR[@]}" CAND_LINE=ms3 DRIVE_FORM_HELD=1 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;  # cbh4c: head + form_held on
    vcsl_h7) trun $t "${VC[@]}" CAND_LINE=ms7 DRIVE_SPLIT=-1 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;  # 10-15: head ms7, split off
    vcsl_h7s) trun $t "${VC[@]}" CAND_LINE=ms7 DRIVE_SPLIT=64 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;  # 10-15: head ms7, executor-level split threshold 64
    vrs17) trun $t "${VR[@]}" CAND_LINE=ms17 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;  # 10-17 remeasure: vrs on the crash-fixed head (snap/step_ms17 + build_cpp_ms11)
    vrs17chk) RERUN=1 trun t_${SET}_vrs17chk "${VR[@]}" CAND_LINE=ms17 ARMV_VPS=1 DRIVE_GC_FREEZE=1 -- vllm/bench/drive_sweep.py --arm V --check --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 64 128 --cb-steps 200 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512 --warm-range 1 3072 160 --out vllm/out/t_${SET}_vrs17chk.json ;;  # 10-17: bitwise --check of the vrs17 churn (all C, mixed steps included; untimed)
    vrs_*) trun $t "${VR[@]}" CAND_LINE=${a#vrs_} ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 ${VRS_EXTRA:-} -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;  # 10-17 remeasure: vrs on CAND_LINE=<suffix> (core landings); VRS_EXTRA: more env (e.g. the split switch)
    ifxchk_*) RERUN=1 trun t_${SET}_$a "${VR[@]}" CAND_LINE=${a#ifxchk_} ARMV_VPS=1 DRIVE_GC_FREEZE=1 ${VRS_EXTRA:-} -- vllm/bench/drive_sweep.py --arm V --check --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --ifx-c 16 --ifx-shape ${IFX:-1024 1024} --ifx-mult 1 --ifx-prewarm-frac 0 --warm-range 1 3072 160 --out vllm/out/t_${SET}_$a.json ;;  # 10-19: bitwise --check of ours on the fixed-seq (C=16, 16 prompts; as r27's vsvpchk_* with R27_IFX)
    chk_*) RERUN=1 trun t_${SET}_$a "${VR[@]}" CAND_LINE=${a#chk_} ARMV_VPS=1 DRIVE_GC_FREEZE=1 ${VRS_EXTRA:-} -- vllm/bench/drive_sweep.py --arm V --check --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 64 128 --cb-steps 200 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512 --warm-range 1 3072 160 --out vllm/out/t_${SET}_$a.json ;;  # 10-17: bitwise --check of vrs_<line>
    splitab_*) SQ="0,1,0,1,0,1,0,1"; [ $((R % 2)) = 0 ] && SQ="1,0,1,0,1,0,1,0"  # 10-18: split off/on alternating per window in ONE process (C=64 x4 then C=128 x4; the same seeded request sequence per window); odd R starts off, even R starts on
      trun $t "${VR[@]}" CAND_LINE=${a#splitab_} ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 DRIVE_HOSTFLAG_FAIL=0 DRIVE_CB_SPLIT_SEQ=$SQ -- vllm/bench/drive_sweep.py --arm V --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 64 64 64 64 128 128 128 128 --cb-steps 3000 --cb-warm 400 --cb-out 256 2048 --cb-prompt 64 512 --warm-range 1 3072 160 --out vllm/out/$t.json ;;
    chksplit_*) RERUN=1 trun t_${SET}_$a "${VR[@]}" CAND_LINE=${a#chksplit_} ARMV_VPS=1 DRIVE_GC_FREEZE=1 DRIVE_HOSTFLAG_FAIL=0 DRIVE_CB_SPLIT_SEQ=1,0,1 -- vllm/bench/drive_sweep.py --arm V --check --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 64 64 128 --cb-steps 200 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512 --warm-range 1 3072 160 --out vllm/out/t_${SET}_$a.json ;;  # 10-18: bitwise with the split toggled per window
    smoke_*) RERUN=1 trun t_${SET}_$a "${VR[@]}" CAND_LINE=${a#smoke_} ARMV_VPS=1 DRIVE_GC_FREEZE=1 DRIVE_HOSTFLAG_FAIL=0 -- vllm/bench/drive_sweep.py --arm V --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 64 --cb-steps 300 --cb-warm 50 --cb-out 256 2048 --cb-prompt 64 512 --out vllm/out/t_${SET}_$a.json ;;  # 10-18: harness smoke (ctx probe), untimed
    smokevl_*) RERUN=1 trun t_${SET}_$a CAND_LINE=c5h DRIVE_HOSTFLAG_FAIL=0 -- vllm/bench/drive_sweep.py --arm ${a#smokevl_} --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 64 --cb-steps 300 --cb-warm 50 --cb-out 256 2048 --cb-prompt 64 512 --out vllm/out/t_${SET}_$a.json ;;  # 10-18: smoke of a vLLM arm (fullnc)
    learn) RERUN=1 trun t_${SET}_learn "${VC[@]}" ARMV_VPS=1 ARMV_BINDINGS_SAVE=$PWD/vllm/out/bindings_${SET}.pkl -- vllm/bench/drive_sweep.py --arm Vc --max-model-len 16384 \
             --batch-size $(seq 1 160) --decode-tokens 3 --prefill-lens $(seq 1 1300) --prefill-reps 1 --mixed 0 --out vllm/out/t_${SET}_learn.json ;;
  esac
done
echo ALLDONE
