#!/usr/bin/env bash
# smoke test of drive_sweep27's kernel-identity fold-in (DRIVE_KID default on): tiny workload, fullnc then vsvp, GPU 0 (short
# correctness jobs allowed between timing rounds, LANES.md), one hold per run
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=1
export DRIVE_STEP_WITH_SAMPLE=1 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12 DRIVE_TACTIC_LOG=1 DRIVE_KID=1
VS=(ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
    ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
    HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1 ARMV_BINDINGS_LOAD=$R27/bindings/n27wf_v.pkl)
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 64}'
W=(--model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8
   --prefill-lens 64 --mixed 0 --ix-mixed 256x16@1024-2048 --ix-reps 3 --dcurve-ctx 1024 --dcurve-bs 2 3 --dcurve-warm 2 --dcurve-steps 2
   --alt-decode 2:4 --cb 4 --cb-steps 40 --cb-warm 10 --cb-out 16 32 --cb-prompt 64 128 --cb-prewarm --llm-kw "$KW")
mkdir -p $R27/out/kidsmoke
for arm in fullnc vsvp; do
  exec 9>$E/gpu$G.lock
  if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
  echo "$(date +%T) start $arm"
  case $arm in
    fullnc) envs=(R27_OVERLAY=$S); a=fullnc ;;
    vsvp) envs=("${VS[@]}"); a=V ;;
  esac
  env "${envs[@]}" INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$((24 << 30)) CUDA_VISIBLE_DEVICES=$G timeout 600 taskset -c 72-107 \
    bash $R27/python_r27.sh $R27/bench/drive_sweep27.py "${W[@]}" --arm $a --out $R27/out/kidsmoke/$arm.json > $R27/logs/kidsmoke_$arm.log 2>&1
  echo "$(date +%T) end $arm rc=$?"
  exec 9>&-
done
