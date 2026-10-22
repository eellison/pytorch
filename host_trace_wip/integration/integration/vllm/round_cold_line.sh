#!/usr/bin/env bash
# cold (serve lane, 10-13): one cold-cache engine init per arm, untimed for step metrics: vLLM's cache root (torch.compile AOT cache,
# FlashInfer autotune cache), the Inductor cache and the Triton cache pointed at fresh empty dirs (serve/cold/<run>/; the warm caches
# are not touched); ours without preloaded bindings (cuBLAS learned in-process). Workload: decode bs 1, 8 tokens; prefill 64.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5h}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1)
W=(--max-model-len ${MML:-16384} --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0)
for a in ${ARMS:-default fullnc bnc vcsl vrs}; do
  t=t_${SET}_${a}_r$R; D=$PWD/serve/cold/${t}_$(date +%H%M%S); mkdir -p $D/vllm $D/inductor $D/triton
  C=(VLLM_CACHE_ROOT=$D/vllm TORCHINDUCTOR_CACHE_DIR=$D/inductor TRITON_CACHE_DIR=$D/triton)
  case $a in
    default) trun $t CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 "${C[@]}" -- vllm/bench/drive_sweep.py --arm default "${W[@]}" --out vllm/out/$t.json ;;
    fullnc) trun $t CAND_LINE=$L "${C[@]}" -- vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/$t.json ;;
    bnc) trun $t CAND_LINE=$L VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' "${C[@]}" -- vllm/bench/drive_sweep.py --arm fullnc_full "${W[@]}" --out vllm/out/$t.json ;;
    vcsl) trun $t "${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 "${C[@]}" -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;
    vrs) trun $t "${BASE[@]}" INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 ARMV_VPS=1 "${C[@]}" -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;
  esac
done
echo ALLDONE
