#!/usr/bin/env bash
# per-shape sync check (coordinator 10-14 14:0x): one KID-profiled step of prefill 512 and of mixed 512x16 (plus prefill 64 / 4096) after
# their windows, n27wf timed configuration, each arm; R["kid"][...]["syncs"] = cuda*Synchronize calls in the full step (harness step-level
# syncs included). Untimed; GPU 2, one hold per arm.
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=2
export DRIVE_STEP_WITH_SAMPLE=1 DRIVE_KID=1 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
COMMON=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1)
KW='{"kv_cache_dtype": "fp8"}'; KWC='{"kv_cache_dtype": "fp8", "compilation_config": {"cudagraph_mode": "NONE"}}'
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 8 --decode-tokens 12
   --prefill-lens 64 512 4096 --mixed 0 --ix-mixed 512x16@1024-2048 --ix-reps 6)
mkdir -p $R27/out/syncstep
for ARM in ${ARMS:-default vcvp fullnc vsvp breakablegb}; do
  case $ARM in
    default) envs=(R27_OVERLAY=$S HOSTTRACE_TORCH_DISABLE_CACHES=0); args=(--arm default --llm-kw "$KW") ;;
    fullnc) envs=(R27_OVERLAY=$S); args=(--arm fullnc --llm-kw "$KW") ;;
    breakablegb) envs=(R27_OVERLAY=$S VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_GDN_EAGER_BREAK=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}'); args=(--arm fullnc_full --llm-kw "$KW") ;;
    vcvp) envs=("${COMMON[@]}" ARMV_INDUCTOR_SEAMS=1 ARMV_BINDINGS_LOAD=$R27/bindings/n27wf.pkl); args=(--arm Vc --llm-kw "$KWC") ;;
    vsvp) envs=("${COMMON[@]}" ARMV_BINDINGS_LOAD=$R27/bindings/n27wf_v.pkl); args=(--arm V --llm-kw "$KW") ;;
  esac
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start $ARM"
  env "${envs[@]}" CUDA_VISIBLE_DEVICES=$G INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((64 << 30)) timeout 3000 taskset -c 36-71 bash $R27/python_r27.sh $R27/bench/drive_sweep27.py \
    "${W[@]}" "${args[@]}" --out $R27/out/syncstep/$ARM.json > $R27/logs/syncstep_$ARM.log 2>&1
  echo "$(date +%T) end $ARM rc=$?"
  exec 9>&-
  python3 -c "
import json; k=json.load(open('$R27/out/syncstep/$ARM.json'))['kid']
for key,v in k.items(): print('$ARM', key, v.get('cg_mode'), v.get('path'), v.get('syncs'))"
done
