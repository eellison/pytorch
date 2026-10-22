#!/usr/bin/env bash
# run27.sh TAG LO HI: r27's churn learn arm (round_27cbv_line.sh "learn": c5h = snap/step324 + build_cpp12 + r27/build_c12, Vc+VP,
# SOUND, gdn_prefill_backend=triton) with --learn-range LO HI, under diag27.py. GPU 3 (caller holds gpu3.lock), CPUs 108-143.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
LF=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
TAG=$1 LO=$2 HI=$3
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export DRIVE_STEP_WITH_SAMPLE=1
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS= ARMV_INDUCTOR_SEAMS=1
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 DRIVE_PHASES=1
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.25} INTEG_KV_BYTES=$(( ${KV_GIB:-8} << 30 ))
export DIAG27_FILE=$LF/out/$TAG.diag.jsonl ${ARMV_BINDINGS_SAVE:+ARMV_BINDINGS_SAVE=$ARMV_BINDINGS_SAVE}
export CUDA_VISIBLE_DEVICES=${LF_GPU:-2}
KWC='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
timeout ${LEARN_TIMEOUT:-1500} taskset -c ${LF_CPUS:-36-71} ${SANITIZE:-} bash $R27/python_r27.sh $LF/diag27.py $R27/bench/drive_sweep27.py --arm Vc --model ${MODEL:-/data/eellison/models/Qwen3.8-27B-NVFP4} --language-model-only ${MODEL_EXTRA:-} \
  --max-model-len 16384 --batch-size 1 --prefill-lens ${PL:-64} --mixed 0 --learn-range $LO $HI --llm-kw "$KWC" --out $LF/out/$TAG.json
echo "rc=$?"
