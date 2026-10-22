#!/usr/bin/env bash
# TRACE_COST_27B (1), low priority, untimed: cProfile of every traced miss of a V+VP process (n27wf config, warm caches, n27wf_v bindings)
# running prefill 4096 and 16000 (plus the warm-up's small prefill / decode traces); dumps in out/traceprof/, PATH lines map them to
# token counts. gb300-box GPU 3, one gpu3.lock hold (LANES.md: flock status checked, used memory < 20 GiB inside the hold).
R27=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
G=3
exec 9>$E/gpu$G.lock
if ! flock -x 9; then echo "$(date +%T) flock gpu$G.lock failed; aborting"; exit 1; fi
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -ge 20480 ]; do sleep 30; done
echo "$(date +%T) start traceprof"
export DRIVE_STEP_WITH_SAMPLE=1 R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R27/build_c12 TORCH_EXTENSIONS_DIR=$R27/torch_ext_c12
export ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_NO_REJIT=1 ARMV_BOUND=0
export ARMV_HOST_ARGS=pinned ARMV_SOUND=1 ARMV_TRTLLM_CPP=1 ARMV_CUTE_OBSERVE=1 ARMV_PORTS=1 ARMV_EAGER_KINDS=
export HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_VP=1 ARMV_VP_UVA=0 ARMV_VP_STOCK=1 ARMV_GC_FREEZE=1 ARMV_GDN_WS=1 ARMV_BINDINGS_LOAD=$R27/bindings/n27wf_v.pkl
export ARMV_TRACE_CPROFILE=$R27/out/traceprof ARMV_DIAG_PATHS=1
rm -rf $R27/out/traceprof; mkdir -p $R27/out/traceprof
CUDA_VISIBLE_DEVICES=$G INTEG_GPU_UTIL=0.9 INTEG_KV_BYTES=$((64 << 30)) timeout 3600 taskset -c 108-143 bash $R27/python_r27.sh $R27/bench/drive_sweep27.py --arm V \
  --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 4096 16000 --mixed 0 \
  --llm-kw '{"kv_cache_dtype": "fp8"}' --out $R27/out/traceprof/run.json > $R27/logs/traceprof.log 2>&1
echo "$(date +%T) end traceprof rc=$?"
