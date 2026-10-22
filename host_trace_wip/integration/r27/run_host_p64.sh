#!/usr/bin/env bash
# CPU attribution of vLLM's prefill-64 step (default arm: compile + FULL_AND_PIECEWISE, dispatches PIECEWISE): py-spy (native off,
# 250 Hz, GIL and idle samples kept) over ~400 back-to-back one-request prefill-64 steps, for gdn_prefill_backend=triton and the default
# (FlashInfer) GDN backend, and vLLM breakable graphs (no compile; triton GDN) for the per-segment comparison. Raw folded stacks:
# out/prof/p64/pyspy_<cfg>.txt
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1
export R27_OVERLAY=$S INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 40 << 30 )) HOSTTRACE_TORCH_DISABLE_CACHES=0
for cfg in triton flashinfer breakable; do
  kw='{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}'
  [ $cfg = flashinfer ] && kw='{"kv_cache_dtype": "fp8", "max_num_seqs": 128}'
  log=$R/logs/host_p64_$cfg.log
  arm=default; envs=()
  # breakable also writes one torch.profiler step (prefill 64, its 6th occurrence) to check its GDN kernels sit inside graph segments
  [ $cfg = breakable ] && { arm=fullnc_full; envs=(VLLM_USE_BREAKABLE_CUDAGRAPH=1 'DRIVE_CC_JSON={"cudagraph_mode": "FULL_AND_PIECEWISE"}' DRIVE_PROFILE_DIR=$R/out/prof/p64 DRIVE_PROFILE_CLASSES=1:64 DRIVE_PROFILE_AT=6 DRIVE_PROFILE_TAG=breakable); }
  env "${envs[@]}" bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm $arm --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
    --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --host-loop 64 400 --llm-kw "$kw" --out $R/out/prof/p64/host_$cfg.json > $log 2>&1 &
  until grep -q HOSTLOOP_START $log || ! kill -0 $! 2>/dev/null; do sleep 2; done
  pid=$(grep -m1 HOSTLOOP_START $log | awk '{print $2}')
  [ -n "$pid" ] && py-spy record --pid $pid --rate 250 --duration 25 --format raw --idle -o $R/out/prof/p64/pyspy_$cfg.txt > $R/logs/pyspy_$cfg.log 2>&1
  wait
  grep -E "HOSTLOOP_END|^prefill" $log | cut -c1-200
done
