#!/usr/bin/env bash
# setlog (10-14, untimed): (1) ours (vcsl config): every cuGraphExecKernelNodeSetParams over fixed bs 63 / 65 and alternating
# 63<->65 and 37<->50 at the swph1 alternating KV (DRIVE_SETLOG), for the split of the alternating node sets into what a per-size
# ready exec would skip vs still patch; (2) ours and default: profiled decode-curve steps at ~1.1K context bs 63 (DRIVE_DECPROF)
# next to the alternating-KV fixed / alternating steps (DRIVE_IXPROF_DIR), for the curve-vs-alternating-KV comparison.
# serve/ix/diag3_analyze.py reads both.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_IXPROF_DIR=$PWD/serve/out/prof_setlog DRIVE_DECPROF=1
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0
   --dec-ctx 1100 --dec-ctx-sizes 63 --dec-ctx-steps 32 --ix-decode 63:1024-2048 65:1024-2048 --alt-decode 63:65@1024-2048 37:50@1024-2048 --ix-reps 6)
# (4) memprobe (main 10-14, high priority): where vrs's reserved growth goes vs fullnc: init, after warm-range, after a decode /
# prefill-16000 / mixed sweep (DRIVE_MEMPROBE: allocator stats, driver-minus-reserved, pools, active bytes by allocation site)
MW=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 32 64 128 --decode-tokens 8 --prefill-lens 4096 16000 --mixed 0
    --ix-mixed 2048x64@1024-2048 4096x128@1024-2048 --ix-reps 2 --warm-range 1 2048 128)
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 DRIVE_MEMPROBE=1 DRIVE_IXPROF_DIR= DRIVE_DECPROF= \
    bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm V "${MW[@]}" --out vllm/out/t_memprobe_vrs_r1.json > serve/logs/memprobe_vrs_run.log 2>&1
echo "memprobe vrs rc=$?"
env CAND_LINE=c5h DRIVE_MEMPROBE=1 DRIVE_IXPROF_DIR= DRIVE_DECPROF= bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm fullnc "${MW[@]}" \
    --out vllm/out/t_memprobe_fullnc_r1.json > serve/logs/memprobe_fullnc_run.log 2>&1
echo "memprobe fullnc rc=$?"
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    DRIVE_SETLOG=1 DRIVE_PROFILE_TAG=vcsl ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc "${W[@]}" \
    --warm-range 1 2048 128 --out vllm/out/t_setlog_vcsl_r1.json > serve/logs/setlog_vcsl_run.log 2>&1
echo "setlog vcsl rc=$?"
env CAND_LINE=c5h HOSTTRACE_TORCH_DISABLE_CACHES=0 DRIVE_PROFILE_TAG=default bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm default "${W[@]}" \
    --out vllm/out/t_setlog_default_r1.json > serve/logs/setlog_default_run.log 2>&1
echo "setlog default rc=$?"
# (3) marks (main 10-14): ours' decode exec at bs 63, warm vs fresh (DRIVE_MARKS), at the ~1.1K curve point and fixed bs 63 at the
# alternating KV; separate process so its dropped decode entry does not touch (1)'s alternating split
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    DRIVE_MARKS=1 DRIVE_IXPROF_DIR= DRIVE_DECPROF= ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm Vc \
    --model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 \
    --dec-ctx 1100 --dec-ctx-sizes 63 --dec-ctx-steps 32 --ix-decode 63:1024-2048 --ix-reps 6 --warm-range 1 2048 128 \
    --out vllm/out/t_marks_vcsl_r1.json > serve/logs/marks_vcsl_run.log 2>&1
echo "marks vcsl rc=$?"
