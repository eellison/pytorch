#!/usr/bin/env bash
# cbrc.sh TAG C: the MiMo lane's validate_fix cb block (a) config (build_fix, eager tactics file, l13mimo_v fast bindings; MiMo
# lane's files read only) under diag27.py, ours V, continuous batching at concurrency C with --cb-prewarm, untimed.
# GPU $LF_GPU (caller holds the lock), CPUs $LF_CPUS.
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/trtllm_moe_cpp
LF=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
TAG=$1 C=$2
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export CUDA_VISIBLE_DEVICES=${LF_GPU:-0} GPU_IDX=${LF_GPU:-0} DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1
export MIMO_FI_AUTOTUNE_FILE=$D/fi_tactics/local_eager.json MIMO_MOE_BUILD=build_fix
export VLLM_USE_DEEP_GEMM=1 MIMO_BACKPORT=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((40<<30))
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_VP=1 ARMV_VP_UVA=0
export ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GC_FREEZE_AFTER_WARM=1 ARMV_BINDINGS_LOAD=$D/bindings/l13mimo_v.fast.pkl
export ARMV_REFUSE_SECTIONS=1 DIAG27_FILE=$LF/out/$TAG.diag.jsonl ARMV_DUMP_VARIANTS=$LF/out/${TAG}_variants.json
timeout ${RUN_TIMEOUT:-1150} taskset -c ${LF_CPUS:-0-35} bash $D/python_mimo.sh $LF/diag27.py $D/bench/drive_sweep_mimo.py --arm V \
  --model /data/eellison/models/MiMo-V2.6-Flash-MOPD --trust-remote-code --language-model-only --max-model-len 16384 \
  --batch-size 1 --prefill-lens 64 --mixed 0 --cb $C --cb-steps 3000 --cb-warm 400 --cb-out 256 2048 --cb-prompt 64 512 \
  --cb-prewarm --out $LF/out/$TAG.json
echo "rc=$?"
