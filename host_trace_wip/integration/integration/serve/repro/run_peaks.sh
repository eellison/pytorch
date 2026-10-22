#!/usr/bin/env bash
# peaks (main 10-14, untimed): vrs vs fullnc memory over swph1's heaviest steps: warm-range, then the ~8.3K decode curve at bs 32 / 128
# (its staggered 16384-token prefill / mixed chunks), prefill 16000; per step the peak allocated above the step's start and the
# reserved growth (DRIVE_PEAKS), plus DRIVE_MEMPROBE's init / after-warm-up / end attribution.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_MEMPROBE=1 DRIVE_PEAKS=1
PW=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 16000 --mixed 0
    --dec-ctx 8300 --dec-ctx-sizes 32 128 --dec-ctx-steps 16 --warm-range 1 2048 128)
env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
    ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
    bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm V "${PW[@]}" --out vllm/out/t_peaks_vrs_r1.json > serve/logs/peaks_vrs_run.log 2>&1
echo "peaks vrs rc=$?"
env CAND_LINE=c5h bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm fullnc "${PW[@]}" --out vllm/out/t_peaks_fullnc_r1.json > serve/logs/peaks_fullnc_run.log 2>&1
echo "peaks fullnc rc=$?"
