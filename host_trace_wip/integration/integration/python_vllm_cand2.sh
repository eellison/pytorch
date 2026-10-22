#!/usr/bin/env bash
# arm V on int6 candidate 2. CAND_LINE=cpp (default): candidate2/src_cpp on int6/build_cpp7/install (C++ entry);
# CAND_LINE=py: candidate2/src_py on install_land_int5b_frozen. CAND_LINE=pinned: the core pinned-copy lane's build (land/core/pinned). CAND_LINE=attn: pinned + the core attn lane's A1+A3 (attnline/py), fork attnline/fork (F1). CAND_LINE=lc: the core learncost lane's build (land/core/lc). CAND_FORK=1 puts the FlashInfer trtllm fork ($S/fork) ahead
# of $S/site with its own caches, as scratch/vllm/python_vllm_cpp_fork.sh (the open variant: ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1).
# INTEG_ARMV=vllm/armV_cand2 (the scratch adapter copy) for the drivers in vllm/bench.
#   CUDA_VISIBLE_DEVICES=1 CAND_LINE=cpp bash python_vllm_cand2.sh <script> [args]
set -euo pipefail
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
P=/data/eellison/src/pytorch/agent_space/paramgraph
I=$P/land/int6
G=$P/land/scratch/integration
source /data/eellison/src/pytorch/agent_space/env.sh
unset TORCH_WT WT_PRELUDE
LINE=${CAND_LINE:-cpp} FORK=${CAND_FORK:-0}
if [ "$LINE" = trtcpp ]; then  # the TRT-LLM C++ launcher lane's run (attn snap s5 Python on the pinned install, FlashInfer overlay, its armV adapter): ARMV_TRTLLM_CPP=1
  export INTEG_ARMV=${INTEG_ARMV:-$P/land/core/trtllm_cpp/armV} VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  exec bash $P/land/core/trtllm_cpp/python_trtllm_cpp.sh "$@"
fi
case $LINE in
  cpp) export HOSTTRACE_WT=$I/candidate2/src_cpp HOSTTRACE_INSTALL=$I/build_cpp7/install HOSTTRACE_PREFIX=$I/build_cpp7/install ;;
  c3) export HOSTTRACE_WT=${C3_WT:-$I/snap/step246} HOSTTRACE_INSTALL=$I/build_cpp8/install HOSTTRACE_PREFIX=$I/build_cpp8/install ;;  # candidate 3 even top (int6 snap/step246 on build_cpp8)
  pinned) export HOSTTRACE_WT=$P/land/core/pinned/py HOSTTRACE_INSTALL=$P/land/core/pinned/c8x/install HOSTTRACE_PREFIX=$P/land/core/pinned/c8x/install ;;  # core pinned lane: step242 + build_cpp8 + pinned memcpy nodes
  lc) export HOSTTRACE_WT=$P/land/core/lc/py HOSTTRACE_INSTALL=$P/land/core/lc/c8l_bld/install HOSTTRACE_PREFIX=$P/land/core/lc/c8l_bld/install ;;  # core learncost lane: step242 + build_cpp8 + L1/L1b/L1c/4/8
  lcbase) export HOSTTRACE_WT=$P/land/int6/snap/step242 HOSTTRACE_INSTALL=$I/build_cpp8/install HOSTTRACE_PREFIX=$I/build_cpp8/install ;;  # learncost lane's baseline: step242 on build_cpp8
  attn) export HOSTTRACE_WT=$G/attnline/py HOSTTRACE_INSTALL=$P/land/core/pinned/c8x/install HOSTTRACE_PREFIX=$P/land/core/pinned/c8x/install ;;  # A1 + F1 (choice() present but unused): pinned + core attn A1 + A3 (A3 = choice(), not adopted; only F1's max_q > 1 decode path calls it); fork = attnline/fork (F1)
  attnlc) export HOSTTRACE_WT=$G/attnlcline/py HOSTTRACE_INSTALL=$P/land/core/lc/c8l_bld/install HOSTTRACE_PREFIX=$P/land/core/lc/c8l_bld/install ;;  # attn line Python + LC_core Python (L1c) on the learn-cost lane's c8l install (LC C++, no pinned memcpy C++); fork attnline/fork
  entry) export HOSTTRACE_WT=${ENTRY_WT:-$P/land/core/entry/py} HOSTTRACE_INSTALL=$P/land/core/entry/c8e0/install HOSTTRACE_PREFIX=$P/land/core/entry/c8e0/install ;;  # core entry lane (land/core/entry): pinned copy + entry/bind/evaluate changes
  attnentry) export HOSTTRACE_WT=$G/attnline/py HOSTTRACE_INSTALL=$P/land/core/entry/c8e0/install HOSTTRACE_PREFIX=$P/land/core/entry/c8e0/install ;;  # the attn line (Python + fork) on the core entry lane's C++ (land/core/entry/c8e0)
  c4) export HOSTTRACE_WT=$I/snap/step290 HOSTTRACE_INSTALL=$I/build_cpp9/install HOSTTRACE_PREFIX=$I/build_cpp9/install ;;  # candidate 4 even top (snap/step290 = cand4 src_cpp; build_cpp9: LC_core + F24k + pinned); fork attnline/fork
  c5) export HOSTTRACE_WT=$I/snap/step310 HOSTTRACE_INSTALL=$I/build_cpp10/install HOSTTRACE_PREFIX=$I/build_cpp10/install ;;  # candidate 5 even top, PRE-GATE (snap/step310: mm_fp4 stack 97b_c3 + R1_ii + 99; build_cpp10 with E1 + RD2-RD5); fork attnline/fork
  c5s) export HOSTTRACE_WT=$P/land/core/symti/wt314 HOSTTRACE_INSTALL=$P/land/core/symti/install15 HOSTTRACE_PREFIX=$P/land/core/symti/install15 ;;  # c5s (10-12): the symti lane's install15 (build_cpp10/src + symti_cpp) + wt314 (step314 + symti_py), as the 27B lane's; fork attnline/fork
  c5g) export HOSTTRACE_WT=${C5G_WT:-$I/snap/step318} HOSTTRACE_INSTALL=$I/build_cpp11/install HOSTTRACE_PREFIX=$I/build_cpp11/install ;;  # candidate 5 GATED (10-12): snap/step318 + build_cpp11 (build_cpp10 + symti C++), conformance 0/0; C5G_WT=$I/snap/step320 once it gates; fork attnline/fork
  c5h) export HOSTTRACE_WT=${C5H_WT:-$I/snap/step324} HOSTTRACE_INSTALL=$I/build_cpp12/install HOSTTRACE_PREFIX=$I/build_cpp12/install ;;  # candidate 5 green (10-13): snap/step324 (symti, LC_compact, S1R, cute_arith) + build_cpp12 (build_cpp11 + GIL-exit fix); fork attnline/fork_c5
  ms3) export HOSTTRACE_WT=${MS3_WT:-$I/snap/step_ms2b} HOSTTRACE_INSTALL=$I/build_cpp_ms3/install HOSTTRACE_PREFIX=$I/build_cpp_ms3/install ;;  # new candidate head (10-14): snap/step_ms2b + build_cpp_ms3 (build_cpp19/src + memset lane row-switch skip); fork attnline/fork_c5
  ms7) export HOSTTRACE_WT=${MS7_WT:-$I/snap/step_ms6} HOSTTRACE_INSTALL=$I/build_cpp_ms7/install HOSTTRACE_PREFIX=$I/build_cpp_ms7/install ;;  # head 10-15: snap/step_ms6 + build_cpp_ms7 (executor-level split, opt-in _host_trace_split); fork attnline/fork_c5
  ms8) export HOSTTRACE_WT=${MS8_WT:-$I/snap/step_ms10} HOSTTRACE_INSTALL=$I/build_cpp_ms8/install HOSTTRACE_PREFIX=$I/build_cpp_ms8/install ;;  # core head 10-15: snap/step_ms10 + build_cpp_ms8; fork attnline/fork_c5
  ms9) export HOSTTRACE_WT=${MS9_WT:-$I/snap/step_ms14} HOSTTRACE_INSTALL=$I/build_cpp_ms9/install HOSTTRACE_PREFIX=$I/build_cpp_ms9/install ;;  # core head 10-15 (memset): build_cpp_ms9 = build_cpp_ms8 + gc untrack helper + nccl2 hook skips unmapped segments; MS9_WT overrides; fork attnline/fork_c5
  ms10) export HOSTTRACE_WT=${MS10_WT:-$I/snap/step_ms15} HOSTTRACE_INSTALL=$I/build_cpp_ms10/install HOSTTRACE_PREFIX=$I/build_cpp_ms10/install ;;  # core head candidate (memset): build_cpp_ms10 = build_cpp_ms9 + run-buffer section placement; MS10_WT overrides; fork attnline/fork_c5
  ms11) export HOSTTRACE_WT=${MS11_WT:-$I/snap/step_ms16} HOSTTRACE_INSTALL=$I/build_cpp_ms11/install HOSTTRACE_PREFIX=$I/build_cpp_ms11/install ;;  # core head candidate (memset): build_cpp_ms11 = ms10 + pipelined split (opt-in) + trusted statics; MS11_WT overrides; fork attnline/fork_c5
  ms17) export HOSTTRACE_WT=${MS17_WT:-$I/snap/step_ms17} HOSTTRACE_INSTALL=$I/build_cpp_ms11/install HOSTTRACE_PREFIX=$I/build_cpp_ms11/install ;;  # core head step_ms17 (= step_ms16 + _canonical scoped to cuDNN / tensor-map shape guard) on build_cpp_ms11; MS17_WT overrides
  ms18) export HOSTTRACE_WT=${MS18_WT:-$I/snap/step_ms18} HOSTTRACE_INSTALL=$I/build_cpp_ms11/install HOSTTRACE_PREFIX=$I/build_cpp_ms11/install ;;  # core head step_ms18 (stale-constant refusal, no rewrite) on build_cpp_ms11
  ms19) export HOSTTRACE_WT=${MS19_WT:-$I/snap/step_ms19} HOSTTRACE_INSTALL=$I/build_cpp_ms12a/install HOSTTRACE_PREFIX=$I/build_cpp_ms12a/install ;;  # core head step_ms19 (incremental gc_untrack + incremental users index) on build_cpp_ms12a
  ms20) export HOSTTRACE_WT=${MS20_WT:-$I/snap/step_ms20} HOSTTRACE_INSTALL=$I/build_cpp_ms12b/install HOSTTRACE_PREFIX=$I/build_cpp_ms12b/install ;;  # core head step_ms20 (native binding table fill + split dirty_min) on build_cpp_ms12b
  ms21) export HOSTTRACE_WT=${MS21_WT:-$I/snap/step_ms21} HOSTTRACE_INSTALL=$I/build_cpp_ms12b/install HOSTTRACE_PREFIX=$I/build_cpp_ms12b/install ;;  # core head step_ms21 (dispatch cache keyed by extent length/overlap: cross-variant redo reuse) on build_cpp_ms12b
  ms22) export HOSTTRACE_WT=${MS22_WT:-$I/snap/step_ms22} HOSTTRACE_INSTALL=$I/build_cpp_ms12b/install HOSTTRACE_PREFIX=$I/build_cpp_ms12b/install ;;  # core head step_ms22 (+ redo templates over declared symbols in mixed formulas) on build_cpp_ms12b
  ms23) export HOSTTRACE_WT=${MS23_WT:-$I/snap/step_ms23} HOSTTRACE_INSTALL=$I/build_cpp_ms12b/install HOSTTRACE_PREFIX=$I/build_cpp_ms12b/install ;;  # core head step_ms23 (gc full walk triggered by untracked count) on build_cpp_ms12b
  ms24) export HOSTTRACE_WT=${MS24_WT:-$I/snap/step_ms24} HOSTTRACE_INSTALL=$I/build_cpp_ms12c/install HOSTTRACE_PREFIX=$I/build_cpp_ms12c/install ;;  # core head step_ms24 (gc walk: sealed referents take no node) on build_cpp_ms12c
  ms25) export HOSTTRACE_WT=${MS25_WT:-$I/snap/step_ms25} HOSTTRACE_INSTALL=$I/build_cpp_ms12c/install HOSTTRACE_PREFIX=$I/build_cpp_ms12c/install ;;  # core head step_ms25 (redo instance guard trim + counters) on build_cpp_ms12c
  ms26) export HOSTTRACE_WT=${MS26_WT:-$I/snap/step_ms26} HOSTTRACE_INSTALL=$I/build_cpp_ms12c/install HOSTTRACE_PREFIX=$I/build_cpp_ms12c/install ;;  # core head step_ms26 (check_section caches) on build_cpp_ms12c
  ms27) export HOSTTRACE_WT=${MS27_WT:-$I/snap/step_ms27} HOSTTRACE_INSTALL=$I/build_cpp_ms12c/install HOSTTRACE_PREFIX=$I/build_cpp_ms12c/install ;;  # core head step_ms27 (redo instances: per-root pin rows) on build_cpp_ms12c
  ms28) export HOSTTRACE_WT=${MS28_WT:-$I/snap/step_ms28} HOSTTRACE_INSTALL=$I/build_cpp_ms12d/install HOSTTRACE_PREFIX=$I/build_cpp_ms12d/install ;;  # core head step_ms28 (split default on, dirty_min 256) on build_cpp_ms12d
  ms29) export HOSTTRACE_WT=${MS29_WT:-$I/snap/step_ms29} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms29 (lazy exec upload) on build_cpp_ms12e
  ms30) export HOSTTRACE_WT=${MS30_WT:-$I/snap/step_ms30} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms30 (guard_sloc memo) on build_cpp_ms12e
  ms31) export HOSTTRACE_WT=${MS31_WT:-$I/snap/step_ms31} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms31 (deterministic-fill IMA fix) on build_cpp_ms12e
  ms37) export HOSTTRACE_WT=${MS37_WT:-$I/snap/step_ms37} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms37 (section confirmation: guard membership on renamed guards) on build_cpp_ms12e
  ms37w) export HOSTTRACE_WT=${MS37W_WT:-$I/snap/wip48} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # wip: step_ms36 + section-path guard membership on renamed guards (ms37 candidate), not a landed line
  ms38w) export HOSTTRACE_WT=${MS38W_WT:-$I/snap/wip49} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # wip: step_ms37 + Triton launch closures as objects, triton types clean for the GC walk (ms38 candidate, 10-22)
  ms38) export HOSTTRACE_WT=${MS38_WT:-$I/snap/step_ms38} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms38 (Triton launch closures as objects, triton types clean for the GC walk; 10-22)
  ms39w) export HOSTTRACE_WT=${MS39W_WT:-$I/snap/wip50} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # wip: step_ms38 + X1, the unread unpinned CPU argument kind (ms39 candidate, 10-22)
  ms39) export HOSTTRACE_WT=${MS39_WT:-$I/snap/step_ms39} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms39 (X1: the unread unpinned CPU argument kind; 10-22)
  ms36) export HOSTTRACE_WT=${MS36_WT:-$I/snap/step_ms36} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms36 (library-body confirmation of a section redo + the drops check) on build_cpp_ms12e
  ms36w) export HOSTTRACE_WT=${MS36W_WT:-$I/snap/wip47} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # wip: step_ms35 + section confirmation (ms36 candidate), not a landed line
  ms35) export HOSTTRACE_WT=${MS35_WT:-$I/snap/step_ms35} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms35 (harvest preload + restore index) on build_cpp_ms12e
  ms34) export HOSTTRACE_WT=${MS34_WT:-$I/snap/step_ms34} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms34 (cold-start redo API) on build_cpp_ms12e
  ms34w) export HOSTTRACE_WT=${MS34W_WT:-$I/snap/wip43} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # wip: step_ms33 + the ms34 cold-start redo API (sheet), not a landed line
  ms33) export HOSTTRACE_WT=${MS33_WT:-$I/snap/step_ms33} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms33 (library-body redo inputs only) on build_cpp_ms12e
  ms32) export HOSTTRACE_WT=${MS32_WT:-$I/snap/step_ms32} HOSTTRACE_INSTALL=$I/build_cpp_ms12e/install HOSTTRACE_PREFIX=$I/build_cpp_ms12e/install ;;  # core head step_ms32 (deterministic fill recorded as nodes) on build_cpp_ms12e
  ms12f35) export HOSTTRACE_WT=${MS12F35_WT:-$I/snap/wip35} HOSTTRACE_INSTALL=$I/build_cpp_ms12f/install HOSTTRACE_PREFIX=$I/build_cpp_ms12f/install ;;  # wip: step_ms30 + delta row bits (core, not a landed line)
  ms9hc) export HOSTTRACE_WT=${MS9HC_WT:-$I/snap/step_ms14} HOSTTRACE_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/mem_eager/build_cpp_ms9hc/install HOSTTRACE_PREFIX=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/mem_eager/build_cpp_ms9hc/install ;;  # mem_eager 10-15: build_cpp_ms9 + hostcall written-bytes compare (patch phase); fork attnline/fork_c5
  ms9ts) export HOSTTRACE_WT=${MS9TS_WT:-/data/eellison/src/pytorch/agent_space/paramgraph/land/core/mem_eager/snap_ts} HOSTTRACE_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/mem_eager/build_cpp_ms9ts/install HOSTTRACE_PREFIX=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/mem_eager/build_cpp_ms9ts/install ;;  # mem_eager 10-16: build_cpp_ms9 + trusted statics for unbound calls; snap_ts = step_ms14 + its test/doc; fork attnline/fork_c5
  c4entry) export HOSTTRACE_WT=${ENTRY_WT:-$P/land/core/entry/py9} HOSTTRACE_INSTALL=$P/land/core/entry/c9e0/install HOSTTRACE_PREFIX=$P/land/core/entry/c9e0/install ;;  # candidate 4 even top (snap/step290 + build_cpp9 incl. LC + F24k) + the core entry lane's E1 (land/core/entry/c9e0); fork attnline/fork
  py) export HOSTTRACE_WT=$I/candidate2/src_py HOSTTRACE_INSTALL=$P/hosttrace_build/install_land_int5b_frozen HOSTTRACE_PREFIX=$P/hosttrace_build/install_land_int5b_frozen ;;
  *) echo "CAND_LINE=$LINE" >&2; exit 2 ;;
esac
SITE="$V/site:$S/site"
# CAND_FORK=1: each line's FlashInfer trtllm fork; a line with no mapping exits (it would otherwise measure the old $S/fork silently)
if [ "$FORK" = 1 ]; then
  case $LINE in cpp|c3|pinned|lc|lcbase|entry|py) SITE="$V/site:$S/fork:$S/site" ;; attn|attnlc|attnentry|c4entry|c4|c5) SITE="$V/site:$G/attnline/fork:$S/site" ;; c5s|c5g|c5h|ms3|ms7|ms8|ms9|ms9hc|ms10|ms9ts|ms11|ms17|ms18|ms19|ms20|ms21|ms22|ms23|ms24|ms25|ms26|ms27|ms28|ms29|ms30|ms31|ms32|ms33|ms34|ms34w|ms35|ms36|ms36w|ms37|ms37w|ms38|ms38w|ms39|ms39w|ms12f35) SITE="$V/site:$G/attnline/fork_c5:$S/site" ;;
    *) echo "python_vllm_cand2.sh: CAND_LINE=$LINE has no FlashInfer fork mapping (CAND_FORK=1): add it to the fork case" >&2; exit 2 ;;
  esac  # fork_c5: the fork's trace-time check from the call's values (candidate 5 API)
fi
export PYTHONPATH="$P/hosttrace_build/site_land:$SITE${VLLM_EXTRA_PYTHONPATH:+:$VLLM_EXTRA_PYTHONPATH}"
export TORCH_CUSTOM_PYTHONPATH="$PYTHONPATH"
export LD_LIBRARY_PATH="$HOSTTRACE_INSTALL/torch/lib:$LD_LIBRARY_PATH"
# vllm _C references cublas without a DT_NEEDED entry; the pip torch preloads it globally, the source build does not
export LD_PRELOAD="/usr/local/cuda-13.0/lib64/libcublas.so.13${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="$V/site/bin:$PATH"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$G/vllm/ext_cand2_$LINE}
if [ "$FORK" = 1 ]; then
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$V/cache_fork} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache_fork/tvm-ffi}
  export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR:-$V/cache/.cache/flashinfer/cubins} FLASHINFER_NO_DOWNLOAD=1 MAX_JOBS=${MAX_JOBS:-16}
else
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$V/cache} TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-$V/cache/tvm-ffi}
fi
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$V/cache/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$V/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$V/cache/inductor}
export CUDA_HOME=/usr/local/cuda-13.0
export TORCHINDUCTOR_FORCE_DISABLE_CACHES="${HOSTTRACE_TORCH_DISABLE_CACHES:-1}"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export INTEG_ARMV=${INTEG_ARMV:-$G/vllm/armV_cand2}
exec python "$@"
