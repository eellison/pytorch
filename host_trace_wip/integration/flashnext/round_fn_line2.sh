#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 eager-forward timing line (the 8B / 27B / MiMo spec; r27's drive_sweep27.py, read only), one engine
# per arm-run:
#   SET=<tag> KIND=id|d|d20|w|cb R=<1..3> ARMS="fullnc breakable v" HT_GPU=<n> HT_CPUS=<lo-hi> bash round_fn_line.sh
# Pairs (forward held fixed, DECISIONS 10-13): vLLM fullnc (no compile, FULL_DECODE_ONLY) and breakable (no compile,
# VLLM_USE_BREAKABLE_CUDAGRAPH=1, FULL_AND_PIECEWISE) vs ours v (the eager forward + registered launchers, VP_STOCK).
# Every arm: vLLM 0.29, TP1, language model only, max_model_len 16384, bf16 KV, Triton GDN prefill (coordinator recommendation,
# pending the user), fixed 40 GiB KV pool (>= 1024 Mamba blocks for vLLM's default max_num_seqs), vLLM's default kernels, ONE
# FlashInfer autotune file (scripts/with_fi_autotune.py: tuned once by fullnc, loaded by every arm; check_autotune_hit.py),
# the same cache dirs. NEVER the deterministic-QSA diagnosis config (FN_QSA_DET) in a timed arm.
# Torch line: int6 snap/step334_ss (side streams) on build_cpp15/install + the side-streams lane's port overlay
# (SGL_OPS_TRACE_DIR), every arm. Ours: ARMV_SOUND=1, learnfast's no_sections + no_lib_redo (adapter top), gc.freeze after
# warm-up, held tensors lifted, cuBLAS bindings learned offline first (learnv: the same workload, untimed, ARMV_BINDINGS_SAVE)
# and preloaded by every timed run (ARMV_BINDINGS_LOAD); DRIVE_REQUIRE_WARM=1: no first-time trace or learn inside a measured
# window. Ours' remaining 1-3 PLE ATen eager steps per variant are labelled (atengaps lane), not waited for.
# KIND idall: kernel identity at EVERY timed size before timing (user 10-14): the full decode ladder bs 1-128 at both
#          contexts (few steps per point), every prefill / mixed shape and the alternating pairs, per arm, untimed; the
#          driver's R["kid"] per point; compare: scripts/kid_compare.py out/timing/t_<SET>_v_r1.json ..._fullnc_r1.json ..._breakable_r1.json
# KIND id: kernel identity: Kineto profiles of single decode steps (bs 8 / 13 / 31 / 33 / 64; 13, 31, 33 = padded sizes for the graph arms; 31 / 33 straddle the QSA top-k switch at 32 rows)
#          per arm (out/timing/prof_<SET>_<arm>/), compared by scripts/kernel_identity.py;
#      d:  decode TPOT curves, every bs 1-128 at ~1K and ~8K context (128 measured steps after 16) + alternating decode (r1);
#      d20: the same on a 20-point grid incl. padded sizes (r2 / r3);
#      cb: continuous batching at C = 16 / 64 / 128 (outputs 256-2048, prompts 64-512; seeded sequence once untimed, then
#          measured; unsynced wall per output token, async scheduling as shipped: DRIVE_CB_NOSYNC=1);
#      w:  prefill 64-16000, mixed chunk 256-4096 x {16, 64} decodes at KV 1024-2048, alternating decode.
set -u
ROOT=${FN_ROOT:-/data/eellison/src/pytorch/agent_space/paramgraph}
L=$ROOT/land
D=$L/core/flashnext
I=$L/int6
R27=$L/core/r27
source $L/scratch/integration/vllm/timing_lib.sh   # idle_check, util_now (on HT_GPU)
# dev-box own-core gate 0.20 (coordinator 10-14: the shared host's non-ours floor on CPUs 36-71 is ~0.15); elsewhere 0.10
[ "$(hostname -s)" = dev-box ] && export OWN_BUSY_FRAC=${OWN_BUSY_FRAC:-0.20}
# gb300-box (round_fn_line2 chains, coordinator 10-14): other users' floor on CPUs 0-35 / 36-71 is 0.19-0.26
[ "$(hostname -s)" = gb300-box ] && export OWN_BUSY_FRAC=${OWN_BUSY_FRAC:-0.30}
# kernel identity (DRIVE_KID) only in untimed processes: a profiled step leaves CUPTI attached and slows every later step of the
# process by 3-10%, unevenly across arms (27B lane, 10-14)
case ${KIND:-} in id|idall|idmem|idchk) KID_DEF=1 ;; mset|rcx|cbsmoke) KID_DEF=0 ;; *) KID_DEF=0; [ "${DRIVE_KID:-0}" = 1 ] && { echo "DRIVE_KID=1 in a timed KIND=$KIND refused"; exit 2; } ;; esac
export UTIL_CAP=${UTIL_CAP:-0.9} DRIVE_STEP_WITH_SAMPLE=1 DRIVE_REQUIRE_WARM=${DRIVE_REQUIRE_WARM:-1} DRIVE_KID=${DRIVE_KID:-$KID_DEF}
OUT=$D/out/timing; LOGD=$D/logs/timing; BINDD=$D/bindings; mkdir -p $OUT $LOGD $BINDD
KVB=$(( ${FN_KV_GIB:-40} * 1073741824 ))
DRV=${FN_DRV:-$D/bench/drive_sweep27.py}  # snapshot of r27/bench/drive_sweep27.py (bench/SNAPSHOT.sha256; DRIVE_KID kernel identity included)
COMMON=(FN_WT=$I/snap/${FN_STEP:-step334_ss} FN_INSTALL=$I/${FN_BUILD:-build_cpp15}/install TORCH_EXTENSIONS_DIR=$D/cache/torch_ext_c15
        SGL_OPS_TRACE_DIR=${FN_PORT_OVERLAY:-$L/core/sidestreams/port_overlay} FN_FI_AUTOTUNE_FILE=$D/cache/fi_autotune/${FN_FI_TAG:-${SET:?SET}}_autotune.json
        HOSTTRACE_TORCH_DISABLE_CACHES=0)
# stock arms: stock FlashInfer (no MoE-hook overlay: no custom-op hop in their calls), the same cubins / workspace / caches
STOCK=(FN_MOE_OVERLAY=$L/scratch/sglang/site)
OURS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
      ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
      ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1)
M=(--model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --trust-remote-code --language-model-only --max-model-len 16384
   --llm-kw '{"gdn_prefill_backend": "triton"}')
ALT=(--alt-decode 37:50@1024-2048 63:65@1024-2048 97:100@1024-2048 --ix-reps 6)
GRID20="1 2 3 4 5 7 8 12 16 17 24 31 32 48 63 64 65 96 127 128"
case ${KIND:?KIND id, idall, idmem, d, d20, w or cb} in
  id) W=("${M[@]}" --batch-size 1 8 13 31 33 64 --decode-tokens 20 --prefill-lens 64 --mixed 0 --profile-decode 8 13 31 33 64 --profile-dir $OUT/prof_${SET}_ARM);;
  idall) MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64; do MIX+=(${p}x${nd}@1024-2048); done; done
     # every timed size (user, 10-14): the full decode ladder at both contexts (few steps per point: one profiled step per
     # class, scripts/with_kprof.py), every prefill length and mixed shape of w, the alternating pairs
     W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0
        --ix-mixed "${MIX[@]}" --dcurve-ctx ${FN_IDALL_CTX:-1024} --dcurve-bs $(seq -s ' ' 1 128) --dcurve-steps 4 --dcurve-warm 4 "${ALT[@]}");;
     # (the 8K ladder's fills alone take ~1.5 h per arm; its identity comes from idmem)
  rcx) # harness test (untimed): a fill that cannot fit (FN_KV_GIB small) must fail the run with a nonzero rc
     W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --dcurve-ctx 8192 --dcurve-bs ${FN_RCX_BS:-99} --dcurve-steps 2 --dcurve-warm 2);;
  mset) # untimed: held memset widths of ours' replays (FN_MEMSET_CHECK=1, land/core/memset/check_held_memsets.py), ARMS=v
     W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --dcurve-ctx 1024 --dcurve-bs ${FN_MSET_BS:-5 13 16 24 64}
        --dcurve-steps 3 --dcurve-warm 2);;
  # the 9 sizes where plain breakable replays the GDN decode branch (INVALID there): prefill 64-1024, mixed 256 / 512 x 16 / 64
  w9) export DRIVE_FWD_SYNC=0
      W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 128 256 512 1024 --mixed 0
         --ix-mixed 256x16@1024-2048 256x64@1024-2048 512x16@1024-2048 512x64@1024-2048 --ix-reps 6);;
  # untimed kernel check: the 9 sizes + decode points where breakable's step time differs from fullnc / ours at unpadded capture sizes
  idchk) W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 128 256 512 1024 --mixed 0
         --ix-mixed 256x16@1024-2048 256x64@1024-2048 512x16@1024-2048 512x64@1024-2048 --ix-reps 3
         --dcurve-ctx 1024 8192 --dcurve-bs 8 9 31 32 33 64 128 --dcurve-cap 8192:9 --dcurve-steps 4 --dcurve-warm 4);;
  idmem) # untimed: memset / memcpy bytes and launch shapes vs bs at both contexts (R["kid"] memops), per arm
     W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --dcurve-ctx 1024 8192
        --dcurve-bs ${FN_IDMEM_BS:-1 2 3 4 5 7 8 12 16 17 24 31 32 48 63 64 65 96} --dcurve-steps 4 --dcurve-warm 4 "${ALT[@]}");;
  # the 8K curve stops at bs 96 in every arm: the fixed 40 GiB pool (1,105,920 tokens) holds at most 98 decoders at 8K (coordinator 10-14)
  d) W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --dcurve-ctx 1024 8192 --dcurve-bs $(seq -s ' ' 1 128)
        --dcurve-cap 8192:96 --dcurve-steps 128 --dcurve-warm 16 "${ALT[@]}");;
  d20) W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --dcurve-ctx 1024 8192 --dcurve-bs $GRID20 --dcurve-cap 8192:96
          --dcurve-steps 128 --dcurve-warm 16 "${ALT[@]}");;
  # w: step-level syncs only in every arm (DRIVE_FWD_SYNC=0, coordinator 10-14): the forward-level pair (run_pw_graph / V.forward)
  # hit breakable's PIECEWISE and ours' prefill / mixed steps once per step but not fullnc's eager prefill
  w) export DRIVE_FWD_SYNC=0
     MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64; do MIX+=(${p}x${nd}@1024-2048); done; done
     W=("${M[@]}" --batch-size 1 --decode-tokens 8 --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0 --ix-mixed "${MIX[@]}" "${ALT[@]}");;
  cbsmoke) # untimed: Python-level synchronizes per step (STEP_SYNCS) for decode / prefill 512 / 2048 / mixed 512x16 in the timed
      # d / w configuration (DRIVE_FWD_SYNC=${FN_SMOKE_FWD_SYNC:-1}), and the CUDA sync calls inside the unsynced churn window
      export DRIVE_CB_NOSYNC=1 DRIVE_CB_SYNCPROF=${FN_SYNCPROF_STEPS:-40} DRIVE_SYNC_CENSUS=1 DRIVE_FWD_SYNC=${FN_SMOKE_FWD_SYNC:-1}
      W=("${M[@]}" --batch-size 1 8 --decode-tokens 8 --prefill-lens 512 2048 --mixed 0 --ix-mixed 512x16@1024-2048 --ix-reps 4
         --cb 16 64 --cb-steps 80 --cb-warm 40 --cb-out 256 2048 --cb-prompt 64 512 --cb-prewarm);;
  cb) export DRIVE_CB_NOSYNC=1
      W=("${M[@]}" --batch-size 1 --prefill-lens 64 --mixed 0 --cb 16 64 128 --cb-steps ${FN_CB_STEPS:-3000} --cb-warm 400 --cb-out 256 2048
         --cb-prompt 64 512 --cb-prewarm);;
esac
trun() {  # tag ENV=.. -- script args: idle GPU (no foreign process), own cores quiet, then the run (detached callers: setsid nohup)
  local tag=$1; shift; local envs=()
  if [ -f $OUT/$tag.json ] && [ -z "${RERUN:-}" ]; then echo "$(date +%T) have $tag"; return; fi
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local own=${HT_CPUS:?HT_CPUS lo-hi} lw=0 lo hi n
  lo=${own%-*}; hi=${own#*-}; n=$((hi - lo + 1))
  # own-core gate BEFORE the lock (round_fn_line2, 10-14): a gate wait no longer holds the GPU from other lanes
  while python3 -c "import sys; sys.exit(not ($(python3 $L/scratch/integration/vllm/cpubusy.py 10 $lo $hi | cut -d' ' -f1) / $n >= ${OWN_BUSY_FRAC:-0.10}))" && [ $lw -lt 1800 ]; do sleep 20; lw=$((lw + 30)); done
  [ $lw -ge 1800 ] && echo "$(date +%T) OWN_CORES_GATE_TIMEOUT $tag (flag this run)" | tee -a $LOGD/timing_idle.log
  # host memory pressure gate (main 10-14, all lanes; timing_lib's trun): /proc/pressure/memory full avg60 < 2% before the run, <= 60 min
  local pw=0 p60; p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory)
  while python3 -c "import sys; sys.exit(not ($p60 >= ${PSI_GATE:-2}))" && [ $pw -lt 3600 ]; do sleep 30; pw=$((pw + 30)); p60=$(awk '/^full/{split($3,a,"="); print a[2]}' /proc/pressure/memory); done
  echo "$(date +%T) PSI_GATE $tag waited ${pw}s full_avg60 $p60$( [ $pw -ge 3600 ] && echo ' TIMEOUT (flag this run)')" | tee -a $LOGD/timing_idle.log
  # FN_LOCK (gb300-box): hold the GPU's lock for this run only (GPU_LANES.md: per command), from the idle check on
  local lk; if [ -n "${FN_LOCK:-}" ]; then exec {lk}>$FN_LOCK; echo "$(date +%T) $tag waiting for $FN_LOCK"; flock -x $lk; echo "$(date +%T) $tag holding $FN_LOCK"; fi
  local verdict
  if [ "${FN_SKIP_IDLE:-0}" = 1 ]; then verdict="idle_check ok (skipped: untimed functional test)"; else verdict=$(idle_check | tail -1); fi
  echo "$(date +%T) start $tag $verdict load $(cut -d' ' -f1-3 /proc/loadavg) own_busy $(python3 $L/scratch/integration/vllm/cpubusy.py 5 $lo $hi | cut -d' ' -f1)/$n" | tee -a $LOGD/timing_idle.log
  case "$verdict" in "idle_check ok"*) ;; *) echo "$(date +%T) SKIPPED_CONTENDED $tag" | tee -a $LOGD/timing_idle.log; [ -n "${FN_LOCK:-}" ] && exec {lk}>&-; return ;; esac
  local u=$(util_now) pre=()
  [ -n "${FN_STRACE:-}" ] && pre=(strace -f -qq -e trace=exit_group,kill,tgkill,wait4 -o $LOGD/$tag.strace)  # harness diagnosis only
  "${pre[@]}" env "${COMMON[@]}" "${envs[@]}" INTEG_GPU_UTIL=$u INTEG_KV_BYTES=$KVB CUDA_VISIBLE_DEVICES=${HT_GPU:?HT_GPU} \
    timeout --foreground ${FN_TIMEOUT:-7200} taskset -c $own bash $D/python_fn.sh $D/scripts/with_fi_autotune.py "$@" > $LOGD/$tag.log 2>&1
  local rc=$?
  [ -f $OUT/$tag.json ] || { echo "$(date +%T) NO_JSON $tag (process rc $rc)" | tee -a $LOGD/timing_idle.log; [ $rc = 0 ] && rc=97; }
  echo "$(date +%T) end $tag rc=$rc util=$u" | tee -a $LOGD/timing_idle.log
  python3 $D/scripts/check_autotune_hit.py $LOGD/$tag.log | tee -a $LOGD/timing_idle.log
  [ $rc = 0 ] || FAILED+=("$tag")
  [ -n "${FN_LOCK:-}" ] && exec {lk}>&-
  return $rc
}
FAILED=()
# kernel identity: drive_sweep27's DRIVE_KID profiles one step after each point's window into R["kid"], in the untimed KINDs only
# (id / idall / idmem, separate processes); scripts/kid_compare.py voids points that differ beyond labelled padding.
# (scripts/with_kprof.py, this lane's earlier per-class profiler, is kept but off: FN_KPROF=1 stacks it.)
KPROF=(); [ "${FN_KPROF:-0}" = 1 ] && KPROF=($D/scripts/with_kprof.py)
w_for() { local a=$1; printf '%s\n' "${W[@]}" | sed "s#prof_${SET}_ARM#prof_${SET}_$a#"; }
BINDV=$BINDD/${SET}_v.pkl
# learnfast's fast learner (coordinator 05:0x, replaces learnv): a short traced seed run saves its harvested cuBLAS keys
# (decode, prefill, mixed traced once), enum_keys.py enumerates M over [1, 16384] ([1, 1024] for the logits GEMM, N = vocab
# 248320), batchlearn.py learns them in 8 processes; the merged file is preloaded. A key it misses fails the strict window
# loudly (DRIVE_REQUIRE_WARM exit 3): the run is then redone with learnv's bindings.
SEEDV=$BINDD/fn_v.seed.pkl FASTV=${FN_FASTBIND_V:-$BINDD/fn_v.fast.pkl}
run_arm() {
  local a=$1 t=t_${SET}_${1}_r${R:-1} w
  mapfile -t w < <(w_for $a)
  case $a in
    fullnc) trun $t "${STOCK[@]}" FN_KPROF_OUT=$OUT/kprof_${SET}_fullnc.json -- "${KPROF[@]}" $DRV --arm fullnc "${w[@]}" --out $OUT/$t.json ;;
    breakable) trun $t "${STOCK[@]}" VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' FN_KPROF_OUT=$OUT/kprof_${SET}_breakable.json -- \
                 "${KPROF[@]}" $DRV --arm fullnc_full "${w[@]}" --out $OUT/$t.json ;;
    # breakable+gdnbreak (correctness-forced deviation arm, r27/out/BREAKABLE_GDN.md): r27's patch makes Qwen's GDN core an eager break
    gdnbreak) trun $t "${STOCK[@]}" VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- \
                 $DRV --arm fullnc_full "${w[@]}" --out $OUT/$t.json ;;
    seedv) FN_SKIP_IDLE=1 trun t_fn_seedv "${OURS[@]}" ARMV_BINDINGS_SAVE=$SEEDV DRIVE_REQUIRE_WARM=0 -- $DRV --arm V "${M[@]}" --batch-size 1 8 \
             --decode-tokens 8 --prefill-lens 64 512 --mixed 2 --out $OUT/t_fn_seedv.json ;;
    fastv) [ -f $FASTV ] && { echo "$(date +%T) have $FASTV"; return; }
       [ -f $SEEDV ] || run_arm seedv
       [ -f $SEEDV ] || { echo "$(date +%T) no seed $SEEDV"; return; }
       echo "$(date +%T) start fast learn $FASTV" | tee -a $LOGD/timing_idle.log
       env "${COMMON[@]}" PY=$D/learnfast_tools/py_fn.sh N=8 CPU0=${HT_CPUS%-*} LF_GPU=$HT_GPU \
         ${FN_LOCK:+flock -x $FN_LOCK} bash $D/learnfast_tools/learn_range.sh $SEEDV $FASTV 16384 1024 248320 > $LOGD/fast_fn_v.log 2>&1
       echo "$(date +%T) end fast learn $FASTV rc=$?" | tee -a $LOGD/timing_idle.log ;;
    learnv) trun t_${SET}_learnv "${OURS[@]}" ARMV_BINDINGS_SAVE=$BINDV DRIVE_PHASES=1 -- $DRV --arm V "${w[@]}" --out $OUT/t_${SET}_learnv.json ;;
    v) if [ "$KIND" = id ] || [ "$KIND" = idall ] || [ "$KIND" = idmem ] || [ "$KIND" = mset ] || [ "$KIND" = cbsmoke ] || [ "$KIND" = idchk ]; then  # not timed: no preloaded bindings, learns allowed
         DRIVE_REQUIRE_WARM=0 trun $t "${OURS[@]}" FN_KPROF_OUT=$OUT/kprof_${SET}_v.json -- "${KPROF[@]}" $DRV --arm V "${w[@]}" --out $OUT/$t.json; return
       fi
       [ -f $FASTV ] || run_arm fastv
       if [ -f $FASTV ]; then
         trun $t "${OURS[@]}" ARMV_BINDINGS_LOAD=$FASTV -- $DRV --arm V "${w[@]}" --out $OUT/$t.json
         [ -f $OUT/$t.json ] && return
         echo "$(date +%T) $t: no result with the fast bindings ($FASTV; a missing key exits 3): the learnv bindings" | tee -a $LOGD/timing_idle.log
       fi
       [ -f $BINDV ] || run_arm learnv
       [ -f $BINDV ] || { echo "$(date +%T) no $BINDV: refusing a cold-learn v timing run"; return; }
       trun $t "${OURS[@]}" ARMV_BINDINGS_LOAD=$BINDV -- $DRV --arm V "${w[@]}" --out $OUT/$t.json ;;
  esac
}
# dev-box GPU 1 (coordinator 10-14): at every arm boundary yield to a queued 8B cbh4 job (its dev3_run.sh waits on
# gpu1.lock with a tag containing cbh4): exit 75 so the chain stops and releases the lock; resumable (outputs are skipped)
yield_cbh4() { [ -n "${FN_YIELD_PATTERN:-}" ] && pgrep -u "$(id -un)" -f "$FN_YIELD_PATTERN" > /dev/null 2>&1; }
for a in ${ARMS:-fullnc breakable v}; do
  if yield_cbh4; then echo "$(date +%T) YIELD: a queued job matches '$FN_YIELD_PATTERN'; stopping before arm $a" | tee -a $LOGD/timing_idle.log; exit 75; fi
  run_arm $a
done
missing=(); for t in "${FAILED[@]}"; do [ -f $OUT/$t.json ] || missing+=("$t"); done
[ ${#missing[@]} = 0 ] || { echo "$(date +%T) ROUND_FAILED ${missing[*]}" | tee -a $LOGD/timing_idle.log; exit 1; }
