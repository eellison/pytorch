#!/usr/bin/env bash
# kidall (serve lane, 10-14, untimed): kernel identity at every size. Each arm records, after every point's window, one profiled step's
# kernel multiset of (name, grid, block) (DRIVE_KREC=1): decode bs 1-128 at ~1K and ~8K context, every swph1 prefill length and mixed
# shape. Arms: default / vcsl / vcslm (vcsl with vLLM's FULL-capture decode max_seq_len = max_model_len, ARMV_DECODE_MAX_SEQ=model) on
# the compiled forward; fullnc / bnc / vrs / vrsm on the eager forward. Compared by serve/ix/kid_every_size.py.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5h}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1 OWN_BUSY_FRAC=1 DRIVE_KREC=1
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4 ARMV_VPS=1 DRIVE_GC_FREEZE=1)
VC=("${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl)
MIX=(); for p in 256 512 1024 2048 4096; do for nd in 16 64 128; do MIX+=(${p}x${nd}@1024-2048); done; done
W=(--max-model-len 16384 --batch-size 1 --decode-tokens 8 --dec-ctx 1100 8300 --dec-ctx-sizes $(seq 1 128) --dec-ctx-steps 4
   --prefill-lens 64 128 256 512 1024 2048 4096 8192 16000 --mixed 0 --ix-mixed "${MIX[@]}" --ix-reps 4)
for a in ${ARMS:-default vcsl vcslm fullnc bnc vrs vrsm}; do
  t=t_kidall_${a}_r1
  case $a in
    default) trun $t CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_sweep.py --arm default "${W[@]}" --out vllm/out/$t.json ;;
    fullnc) trun $t CAND_LINE=$L -- vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/$t.json ;;
    bnc) trun $t CAND_LINE=$L VLLM_USE_BREAKABLE_CUDAGRAPH=1 DRIVE_CC_JSON='{"cudagraph_mode": "FULL_AND_PIECEWISE"}' -- vllm/bench/drive_sweep.py --arm fullnc_full "${W[@]}" --out vllm/out/$t.json ;;
    vcsl) trun $t "${VC[@]}" ARMV_DECODE_MAX_SEQ=actual -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;
    vcslm) trun $t "${VC[@]}" ARMV_DECODE_MAX_SEQ=model -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/$t.json ;;
    vrs) trun $t "${BASE[@]}" ARMV_DECODE_MAX_SEQ=actual -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;
    vrsm) trun $t "${BASE[@]}" ARMV_DECODE_MAX_SEQ=model -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/$t.json ;;
  esac
done
echo ALLDONE
