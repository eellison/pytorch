#!/usr/bin/env bash
# fnrc.sh TAG MODE: Qwen3.8-Flash-Next NVFP4, ours (arm V, eager forward + registered launchers), the Flash-Next lane's untimed
# config (STATUS 10:26 KIND=mset launcher: step334_ss + build_cpp15, sidestreams port overlay, OURS flags, no preloaded
# bindings, DRIVE_REQUIRE_WARM=0), under diag27.py; 1K decode ladder bs $BS (default 1..128). MODE chk: --check (every
# forward vs eager, twice), dc: no check. Writable caches are copies in learnfast/fncache; the lane's files are read only.
# GPU $LF_GPU (caller holds the lock), CPUs $LF_CPUS.
set -u
TAG=$1 MODE=$2
L=/data/eellison/src/pytorch/agent_space/paramgraph/land
F=$L/core/flashnext I=$L/int6 LF=$L/core/learnfast C=$L/core/learnfast/fncache
export FN_WT=$I/snap/step334_ss FN_INSTALL=$I/build_cpp15/install TORCH_EXTENSIONS_DIR=$C/torch_ext_c15
export SGL_OPS_TRACE_DIR=$L/core/sidestreams/port_overlay FN_FI_AUTOTUNE_FILE=$C/fn_autotune.json HOSTTRACE_TORCH_DISABLE_CACHES=0
export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor FLASHINFER_WORKSPACE_BASE=$C/fi_ws
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
export ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1
export DRIVE_STEP_WITH_SAMPLE=1 DRIVE_REQUIRE_WARM=0 DRIVE_KID=0
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.9} INTEG_KV_BYTES=$(( ${FN_KV_GIB:-40} * 1073741824 ))
export ARMV_DUMP_VARIANTS=$LF/out/${TAG}_variants.json DIAG27_FILE=$LF/out/$TAG.diag.jsonl CUDA_VISIBLE_DEVICES=${LF_GPU:-3}
BS=${BS:-$(seq -s ' ' 1 128)}
# check mode (Flash-Next lane 10-14): QSA's deterministic top-k tie-break (FN_QSA_DET, check-only diagnosis config) and GDN
# state copies on the CPU (device copies OOM at 64 requests)
if [ "$MODE" = chk ]; then export ARMV_CHECK_TWICE=1 FN_QSA_DET=1 ARMV_CHECK_STATE_CPU=1; X="--check"; else X=""; fi
timeout ${RUN_TIMEOUT:-1150} taskset -c ${LF_CPUS:-108-143} bash $F/python_fn.sh $LF/diag27.py $LF/fnwrap.py $F/bench/drive_sweep27.py --arm V \
  --model /data/eellison/models/Qwen3.8-Flash-Next-NVFP4 --trust-remote-code --language-model-only --max-model-len 16384 \
  --llm-kw '{"gdn_prefill_backend": "triton"}' --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 \
  --dcurve-ctx 1024 --dcurve-bs $BS --dcurve-steps ${STEPS:-2} --dcurve-warm ${WARM:-1} $X --out $LF/out/$TAG.json
echo "rc=$?"
