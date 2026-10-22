#!/usr/bin/env bash
# run_ctr_check.sh ARM (V | Vc | Vk | eager); Vk = V without --check (its KID memsets are the replay's alone): trtllm-gen multi-CTA KV counter check (10-14, after the 8B lane's baked-memset finding).
# Decode at 8K context (multi-CTA region at small bs) for bs 1-24, 32, 48, 64: ours with --check (every decode step bitwise vs
# vLLM's eager reference) and DRIVE_KID (memset byte sizes per bs); "eager" = vLLM enforce_eager, the stock memset sizes.
# Expected stock counter bytes: round_up(max(bs * 24, sm_count), 8) * 4. n27d config. Run inside a GPU hold.
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
ARM=$1
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_KID=1
KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
CHK=()
if [ $ARM != eager ]; then
  export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
  export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=prefill,mixed
  export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_CHECK_KINDS=decode ARMV_CHECK_STATE_CPU=0 ARMV_GC_FREEZE=1
  [ $ARM = Vk ] || CHK=(--check)
  if [ $ARM = Vc ]; then export ARMV_INDUCTOR_SEAMS=1 ARMV_BINDINGS_LOAD=$R/bindings/n27d.pkl
    KW='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "NONE"}}'
  else export ARMV_BINDINGS_LOAD=$R/bindings/n27d_v.pkl; fi
else export R27_OVERLAY=$S; fi
export INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 72 << 30 ))
O=$R/out/ctr_check_$ARM.json
A=$ARM; [ $A = Vk ] && A=V
bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm $A "${CHK[@]}" --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
  --max-model-len 16384 --batch-size 1 --decode-tokens 12 --prefill-lens 64 --mixed 0 \
  --dcurve-ctx 8192 1024 --dcurve-bs $(seq -s ' ' 1 24) 32 48 64 --dcurve-warm 2 --dcurve-steps 4 --llm-kw "$KW" --out $O > $R/logs/ctr_check_$ARM.log 2>&1
echo "ctr_check $ARM rc=$?"
