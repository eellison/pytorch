#!/usr/bin/env bash
# swpe smoke (10-15, untimed, GPU 3 shared): the three eager arms through the swpe configuration on a tiny grid (ctx 1100, bs 1/2/20,
# 16 measured steps), small KV, to check the new tail / GC / PSI / empty_cache fields and ARMV_REFREEZE.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.40 INTEG_KV_BYTES=21474836480 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_GC_LOG=1 DRIVE_EMPTY_CACHE=1
BASE=(CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
W=(--max-model-len 16384 --batch-size 1 --decode-tokens 8 --dec-ctx 1100 --dec-ctx-sizes 1 2 20 --dec-ctx-steps 16 --prefill-lens 64 --mixed 0)
env CAND_LINE=c5h timeout 900 bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/t_smoke_swpe_fullnc.json > serve/logs/smoke_swpe_fullnc.log 2>&1; echo "fullnc rc=$?"
env CAND_LINE=c5h VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' timeout 900 bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm fullnc_full "${W[@]}" --out vllm/out/t_smoke_swpe_bnc.json > serve/logs/smoke_swpe_bnc.log 2>&1; echo "bnc rc=$?"
env "${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_REFREEZE=1 ARMV_VPS=1 DRIVE_STRICT_WINDOW=1 DRIVE_GC_FREEZE=1 timeout 900 bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm V "${W[@]}" --warm-range 1 256 24 --out vllm/out/t_smoke_swpe_vrs.json > serve/logs/smoke_swpe_vrs.log 2>&1; echo "vrs rc=$?"
