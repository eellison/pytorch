#!/usr/bin/env bash
# sync check (coordinator 10-14 13:5x): CUDA synchronize API calls per step inside the unsynced churn window (DRIVE_CB_NOSYNC=1), n27cbf
# config, C=64, 60 measured steps after 50 warm (prewarm pass first), torch.profiler over the window (DRIVE_SYNC_PROF=1). Untimed.
# GPU 2, one hold per arm.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=2
export DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1 DRIVE_SYNC_PROF=1 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
COMMON=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
KW='{"kv_cache_dtype": "fp8"}'; KWC='{"kv_cache_dtype": "fp8", "compilation_config": {"cudagraph_mode": "NONE"}}'
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0
   --cb 64 --cb-steps 60 --cb-warm 50 --cb-out 256 2048 --cb-prompt 64 512 --cb-prewarm)
mkdir -p $R27/out/synccheck
for ARM in ${ARMS:-default vcvp fullnc vsvp breakablegb}; do
  case $ARM in
    default) envs=(R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0); args=(--arm default --llm-kw "$KW") ;;
    fullnc) envs=(R27_OVERLAY=$S); args=(--arm fullnc --llm-kw "$KW") ;;
    breakablegb) envs=(R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'); args=(--arm fullnc_full --llm-kw "$KW") ;;
    vcvp) envs=("${COMMON[@]}" ARMV_INDUCTOR_SEAMS=1 ARMV_BINDINGS_LOAD=$R27/bindings/n27cbf.pkl); args=(--arm Vc --llm-kw "$KWC") ;;
    vsvp) envs=("${COMMON[@]}" ARMV_BINDINGS_LOAD=$R27/bindings/n27cbf_v.pkl); args=(--arm V --llm-kw "$KW") ;;
  esac
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start $ARM"
  env "${envs[@]}" CUDA_VISIBLE_DEVICES=$G INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((64 << 30)) timeout 3000 taskset -c 36-71 bash $R27/python_r27.sh $R27/bench/drive_sweep27.py \
    "${W[@]}" "${args[@]}" --out $R27/out/synccheck_sites/$ARM.json > $R27/logs/synccheck_sites_$ARM.log 2>&1
  echo "$(date +%T) end $ARM rc=$?"
  exec 9>&-
  python3 -c "import json; d=json.load(open('$R27/out/synccheck_sites/$ARM.json'))['cb']['64']; print('$ARM', 'steps', d['steps'], 'nosync', d['nosync'], 'per step', d.get('sync_per_step'))"
done
