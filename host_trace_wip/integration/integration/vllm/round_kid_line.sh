#!/usr/bin/env bash
# kid (serve lane, 10-14, untimed): kernel-identity check. One profiled full step (torch.profiler, CUPTI sees kernels inside graph
# launches) per class decode 64:64, prefill 1:512, mixed 65:1088, for vcsl / default (compiled forward) and vrs / fullnc (eager forward);
# serve/ix/kid_compare.py diffs the kernel-name multisets within each forward.
source $(dirname $0)/timing_lib.sh
L=${STOCK_LINE:-c5h}; export UTIL_CAP=0.9 DRIVE_STEP_WITH_SAMPLE=1 OWN_BUSY_FRAC=1
BASE=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_SOUND=1 ARMV_PORTS=1 INTEG_ARMV=$PWD/vllm/armV_stock_fast_m4)
W=(--max-model-len 16384 --batch-size 64 --decode-tokens 12 --prefill-lens 512 --mixed 0 --ix-mixed 1024x64@1024-2048 --ix-reps 6)
P=(DRIVE_PROFILE_CLASSES=64:64,1:512,65:1088 DRIVE_PROFILE_AT=5)
for a in ${ARMS:-vcsl default vrs fullnc}; do
  D=$PWD/serve/out/kid/$a; mkdir -p $D
  case $a in
    vcsl) trun t_kid_vcsl_r1 "${BASE[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 ARMV_INDUCTOR_HOSTS=1 ARMV_VPS=1 ARMV_BINDINGS_LOAD=$PWD/vllm/out/bindings_swph1.pkl "${P[@]}" DRIVE_PROFILE_DIR=$D DRIVE_PROFILE_TAG=vcsl -- vllm/bench/drive_sweep.py --arm Vc "${W[@]}" --out vllm/out/t_kid_vcsl_r1.json ;;
    default) trun t_kid_default_r1 CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 "${P[@]}" DRIVE_PROFILE_DIR=$D DRIVE_PROFILE_TAG=default -- vllm/bench/drive_sweep.py --arm default "${W[@]}" --out vllm/out/t_kid_default_r1.json ;;
    vrs) trun t_kid_vrs_r1 "${BASE[@]}" ARMV_VPS=1 "${P[@]}" DRIVE_PROFILE_DIR=$D DRIVE_PROFILE_TAG=vrs -- vllm/bench/drive_sweep.py --arm V "${W[@]}" --out vllm/out/t_kid_vrs_r1.json ;;
    fullnc) trun t_kid_fullnc_r1 CAND_LINE=$L "${P[@]}" DRIVE_PROFILE_DIR=$D DRIVE_PROFILE_TAG=fullnc -- vllm/bench/drive_sweep.py --arm fullnc "${W[@]}" --out vllm/out/t_kid_fullnc_r1.json ;;
  esac
done
echo ALLDONE
