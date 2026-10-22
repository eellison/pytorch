#!/usr/bin/env bash
# held memset widths (land/core/memset/check_held_memsets.py) on r27's V+VP decode config (n27d), untimed: decode at 1K context, bs 10,
# 5, 13, 16, 48, 64 (bs 5 / 14-16 / 64 replay variants traced at another bs). ARMV_HELD_MEMSETS logs each replay's rows into the
# summary (V.held_memsets); GPU 1, one hold.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=1
exec 9>$E/gpu$G.lock
if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
echo "$(date +%T) start held"
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=prefill,mixed
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_HELD_MEMSETS=1 ARMV_BINDINGS_LOAD=$R/bindings/n27d_v.pkl
CUDA_VISIBLE_DEVICES=$G INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 72 << 30 )) timeout 1800 taskset -c 72-107 bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm V \
  --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 12 --prefill-lens 64 --mixed 0 \
  --dcurve-ctx 1024 --dcurve-bs 10 5 13 16 48 64 --dcurve-warm 2 --dcurve-steps 3 \
  --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}' --out $R/out/held_memsets_V.json > $R/logs/held_memsets_V.log 2>&1
echo "$(date +%T) end held rc=$?"
