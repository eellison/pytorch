#!/usr/bin/env bash
# churnprof3 (main 10-14, untimed): vcsl only, every cuGraphExecKernelNodeSetParams over churn steps 100-400 (no profiler) with its wall, to
# split config-change vs args-only calls by whether the stream was busy (size-changed and mixed steps included).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_CB_NOSYNC=1 DRIVE_CBPROF=100:400 DRIVE_CBSETLOG=100:250 DRIVE_CBPERF=300:350 DRIVE_IXPROF_DIR=
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 64 --cb-steps 400 --cb-warm 100 --cb-out 256 2048 --cb-prompt 64 512)
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1 DRIVE_SETLOG=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_cbh4.pkl DRIVE_PROFILE_TAG=vcsl \
    bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc "${W[@]}" --warm-range 1 3072 160 --out vllm/out/t_churnprof3_vcsl_r1.json > serve/logs/churnprof3_vcsl_run.log 2>&1
echo "churnprof3 vcsl rc=$?"
echo "churnprof3 done"
# (b) main 10-14: the C = 128 stall A/B in the cbh4b harness (drive_sweep, truly unsynced, allocator events + stall stack dumps per window):
# vcsl as is vs torch.cuda.empty_cache() once after the warm-up (DRIVE_EMPTY_CACHE)
CW=(--max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --cb 128 --cb-steps 3000 --cb-warm 400 --cb-out 256 2048 --cb-prompt 64 512 --warm-range 1 3072 160)
for v in asis emptycache; do
  EC=; [ $v = emptycache ] && EC=DRIVE_EMPTY_CACHE=1
  env DRIVE_CBPROF= DRIVE_CBSETLOG= DRIVE_CBPERF= CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual \
      ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 \
      DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_cbh4.pkl $EC \
      bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm Vc --model /data/eellison/models/Qwen3-8B "${CW[@]}" --out vllm/out/t_stallab_${v}_vcsl.json > serve/logs/stallab_${v}_vcsl.log 2>&1
  echo "stallab $v rc=$?"
done
# (c) main 10-14: vcsl on the new head (CAND_LINE=ms3), unsynced C = 64, the set logger over steps 100-250 (exec per set = the form the step
# lands on) with per-step kernel row switches / packs / TMA counters; (d) the same with form_held_enabled (DRIVE_FORM_HELD): does keeping
# each form's held state remove the size-changed steps' identical-byte sets?
for v in head head_formheld; do
  FH=; [ $v = head_formheld ] && FH=DRIVE_FORM_HELD=1
  env CAND_LINE=ms3 CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
      ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
      ARMV_REFREEZE=1 DRIVE_MIXED_WARM=1 DRIVE_SETLOG=1 DRIVE_CBPROF=100:400 DRIVE_CBSETLOG=100:250 DRIVE_CBPERF= DRIVE_IXPROF_DIR= DRIVE_PROFILE_TAG=vcsl_$v $FH \
      bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc "${W[@]}" --warm-range 1 3072 160 --out vllm/out/t_churnprof3_vcsl_$v.json > serve/logs/churnprof3_vcsl_$v.log 2>&1
  echo "churnprof3 vcsl $v rc=$?"
done
