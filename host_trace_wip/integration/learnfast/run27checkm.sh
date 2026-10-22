#!/usr/bin/env bash
# run27check.sh TAG "PREFILL LENS": the learn arm's config with --check (every traced/replayed forward vs eager) under diag27.py
# (fix flags from the caller's DIAG27_* env); prefill lengths chosen to land on redispatched classes. GPU $LF_GPU, CPUs $LF_CPUS.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
LF=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
TAG=$1 PL=$2
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export DRIVE_STEP_WITH_SAMPLE=1
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1 ARMV_CHECK_TWICE=1
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 DRIVE_PHASES=1
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.25} INTEG_KV_BYTES=$(( ${KV_GIB:-8} << 30 ))
export DIAG27_FILE=$LF/out/$TAG.diag.jsonl CUDA_VISIBLE_DEVICES=${LF_GPU:-3}
KWC='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
timeout ${CHECK_TIMEOUT:-1100} taskset -c ${LF_CPUS:-108-143} bash $R27/python_r27.sh $LF/diag27.py $R27/bench/drive_sweep27.py --arm Vc --check --model ${MODEL:-/data/eellison/models/Qwen3.8-27B-NVFP4} --language-model-only ${MODEL_EXTRA:-} \
  --max-model-len 16384 --batch-size 1 --prefill-lens $PL --mixed 0 --llm-kw "$KWC" --out $LF/out/$TAG.json
echo "rc=$?"
python3 $R27/b200_bundle/tools/verdict.py $LF/out/$TAG.json
