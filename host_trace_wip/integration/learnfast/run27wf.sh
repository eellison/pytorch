#!/usr/bin/env bash
# run27wf.sh TAG [chk]: r27's n27wf workload (round_27wf_line.sh vcvp arm: Vc+VP on vLLM defaults, prefill 64-16000, mixed
# 256-4096 x {16, 64}, decode bs 1; c5h install; r27's n27wf cuBLAS bindings preloaded, read only) under diag27.py, untimed
# (DRIVE_REQUIRE_WARM=0). chk: --check (every forward vs eager, twice). GPU $LF_GPU (caller holds the lock), CPUs $LF_CPUS.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
LF=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
TAG=$1 MODE=${2:-run}
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_REQUIRE_WARM=0
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
# HEAD=1: the candidate head (memset lane 10-15: snap/step_ms8 on build_cpp_ms8, the traced FlashInfer extension rebuilt against it)
[ "${HEAD:-0}" = 1 ] && export R27_WT=${HEAD_WT:-$I/snap/step_ms11} R27_INSTALL=${HEAD_INSTALL:-$I/build_cpp_ms8/install} \
  FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/memset/tc/fi_build_ms8 TORCH_EXTENSIONS_DIR=${HEAD_EXT:-$LF/torch_ext_ms8}
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1
export ARMV_BINDINGS_LOAD=$R27/bindings/n27wf.pkl INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.9} INTEG_KV_BYTES=$(( ${KV_GIB:-64} << 30 ))
export DIAG27_FILE=$LF/out/$TAG.diag.jsonl CUDA_VISIBLE_DEVICES=${LF_GPU:-0} ARMV_DUMP_VARIANTS=$LF/out/${TAG}_variants.json
[ "$MODE" = chk ] && { export ARMV_CHECK_TWICE=1; X=--check; } || X=
# ARM=V: ours on vLLM's eager forward (no inductor seams; the startup arm), our 1-16384 bindings
ARM=${ARM:-Vc}; CGN=', "compilation_config": {"cudagraph_mode": "NONE"}'
[ "$ARM" = V ] && { export ARMV_INDUCTOR_SEAMS=0 ARMV_BINDINGS_LOAD=$LF/out/bind_startup27_ht_v.pkl; CGN=; }
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64; do MIX+=(${p}x${nd}@1024-2048); done; done
PL=(64 128 256 512 1024 2048 4096 8192 16000)
# PART=a: decode + the prefill lengths; PART=b: decode + the mixed shapes (a --check run of the whole set exceeds a 20 min hold)
case ${PART:-all} in a) MIXA=();; b) PL=(64); MIXA=(--ix-mixed "${MIX[@]}" --ix-reps 6);;
  # ab: the split A/B shapes (decode bs 1 / 8 / 32 at 1K, alternating 9<->16, mixed 512 x 8 decoders)
  ab) PL=(64); MIXA=(--ix-mixed 512x8@1024-2048 --ix-reps 6 --alt-decode 9:16@1024-2048 --dcurve-ctx 1024 --dcurve-bs 1 8 32 --dcurve-steps 16 --dcurve-warm 4);;
  *) MIXA=(--ix-mixed "${MIX[@]}" --ix-reps 6);; esac
# GDN=triton: Triton GDN prefill (the lanes' correctness configuration; FlashInfer's GDN prefill NaNs / is nondeterministic in
# some processes, stock bug)
KWG=; [ "${GDN:-}" = triton ] && KWG=', "gdn_prefill_backend": "triton"'
timeout ${RUN_TIMEOUT:-1150} taskset -c ${LF_CPUS:-0-35} bash $R27/python_r27.sh $LF/diag27.py $R27/bench/drive_sweep27.py --arm $ARM \
  --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 \
  --prefill-lens "${PL[@]}" --mixed 0 "${MIXA[@]}" $X \
  --llm-kw "{\"kv_cache_dtype\": \"fp8\"$KWG$CGN}" --out $LF/out/$TAG.json
echo "rc=$?"
