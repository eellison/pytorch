#!/usr/bin/env bash
# run_q35.sh TAG [extra drive_fn args]: Qwen3.5-9B V-stock --check on c5 + r27 (SOUND, ARMV_TRTLLM_CPP, ports, pinned host args,
# eager-twice control) over q35s's failing shapes (16384-token prefill / mixed setup steps of 6-8K prompts); ARMV_MEMORY from the env.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
TAG=$1; shift
cd $G
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=${ARMV_HOST_ARGS:-pinned} ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_PORTS=${ARMV_PORTS:-1} ARMV_CHECK_TWICE=1 ARMV_CHECK_STATE_CPU=1
export FLASHINFER_WORKSPACE_BASE=$V/cache FLASHINFER_CUBIN_DIR=$V/cache/.cache/flashinfer/cubins
export INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.04} INTEG_KV_BYTES=$(( ${R27_KV_GIB:-16} * 1073741824 ))
W=(--model /data/eellison/models/Qwen3.5-9B --trust-remote-code --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8
   --prefill-lens 64 512 --mixed 0 --ix-mixed 2048x8@6000-8000 --ix-decode 8:6000-8000 --ix-reps 3)
bash $R/python_r27.sh vllm/bench/drive_fn.py --arm V --check "${W[@]}" "$@" --out $R/out/$TAG.json
echo "$TAG rc=$?"
python3 $R/summ.py $R/out/$TAG.json
