#!/usr/bin/env bash
# memrep (main 10-14, untimed, gb300-box GPU 3): the swph1 sweep's memory, by section, vrs vs fullnc in their swph1 configs (vrs:
# warm-range + freeze; fullnc: none), split in three processes to keep each gpu3.lock hold under 20 min (decode curves at 32 measured
# steps per point instead of 128: the same shapes). DRIVE_MEMPROBE (no alloc history) at init / after warm-up / each section start /
# end; DRIVE_PEAKS per step with our trace / harvest / binding deltas (the reserved delta per post-warm-up trace).
#   PART=1|2|3 ARM=vrs|fullnc bash serve/repro/run_memrep.sh
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
: ${PART:?} ${ARM:?}
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=0.9 DRIVE_STEP_WITH_SAMPLE=1 DRIVE_MEMPROBE=1 DRIVE_MEMPROBE_NOHIST=1 DRIVE_MEMPROBE_SECTIONS=1 DRIVE_PEAKS=1
SZ=($(seq 1 128))
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64 128; do MIX+=(${p}x${nd}@1024-2048); done; done
case $PART in
  1) W=(--batch-size "${SZ[@]}" --decode-tokens 36 --dec-ctx 1100 --dec-ctx-sizes "${SZ[@]}" --dec-ctx-steps 32 --prefill-lens 64 --mixed 0) ;;
  2) W=(--batch-size 1 --decode-tokens 8 --dec-ctx 8300 --dec-ctx-sizes "${SZ[@]}" --dec-ctx-steps 32 --prefill-lens 64 --mixed 0) ;;
  3) W=(--batch-size 1 --decode-tokens 8 --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0 --ix-mixed "${MIX[@]}"
        --alt-decode 37:50@1024-2048 63:65@1024-2048 97:100@1024-2048 --ix-decode 37:1024-2048 50:1024-2048 63:1024-2048 65:1024-2048 97:1024-2048 100:1024-2048 --ix-reps 6) ;;
esac
W=(--model /data/eellison/models/Qwen3-8B --max-model-len 16384 "${W[@]}")
case $ARM in
  vrs) env CAND_LINE=c5h CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
         ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 DRIVE_PHASES=1 ARMV_VPS=1 DRIVE_GC_FREEZE=1 \
         bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm V "${W[@]}" --warm-range 1 2048 128 --out vllm/out/t_memrep_vrs_p$PART.json > serve/logs/memrep_vrs_p${PART}_run.log 2>&1 ;;
  fullnc) env CAND_LINE=c5h bash python_vllm_cand2.sh vllm/bench/drive_diag.py --arm fullnc "${W[@]}" --out vllm/out/t_memrep_fullnc_p$PART.json > serve/logs/memrep_fullnc_p${PART}_run.log 2>&1 ;;
esac
echo "memrep $ARM p$PART rc=$?"
