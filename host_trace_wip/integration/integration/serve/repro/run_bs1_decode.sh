#!/usr/bin/env bash
# bs-N decode loop (10-15, untimed, for the core / mem_eager lanes): ARM=vcsl|default [LINE=ms9 (core head) | c5h] [STEPS=200] [BS=1] [KV_GIB=20] [MAXLEN=16384]. Synced harness,
# per-step full-step rows; ours with DRIVE_PHASES=1 (the C++ entry's phase clocks per step). Small KV (20 GiB) for a shared GPU.
#   CUDA_VISIBLE_DEVICES=3 ARM=vcsl bash serve/repro/run_bs1_decode.sh     -> vllm/out/t_bs1_<arm>_<line>.json, phase medians in ["decode"]["1"]["ph_us"]
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
: ${ARM:?}; LINE=${LINE:-ms9}; STEPS=${STEPS:-200}; BS=${BS:-1}; KV_GIB=${KV_GIB:-20}
export FLASHINFER_NO_DOWNLOAD=1 INTEG_GPU_UTIL=${INTEG_GPU_UTIL:-0.40} INTEG_KV_BYTES=$((KV_GIB * 1073741824)) DRIVE_STEP_WITH_SAMPLE=1
W=(--model /data/eellison/models/Qwen3-8B --max-model-len ${MAXLEN:-16384} --batch-size $BS --decode-tokens $STEPS --prefill-lens 64 --mixed 0)
case $ARM in
  vcsl) env CAND_LINE=$LINE CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 \
          ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 DRIVE_PHASES=1 ARMV_VPS=1 \
          bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --warm-range 1 128 8 --out vllm/out/t_bs${BS}_vcsl_$LINE.json ;;
  default) env CAND_LINE=$LINE HOSTTRACE_TORCH_DISABLE_CACHES=0 bash python_vllm_cand2.sh vllm/bench/drive_sweep.py --arm default "${W[@]}" --out vllm/out/t_bs${BS}_default_$LINE.json ;;
esac
