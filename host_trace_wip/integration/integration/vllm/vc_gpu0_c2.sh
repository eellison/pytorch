#!/usr/bin/env bash
# Vc revival, step 1: the Python caller of the escaping "Cannot call numel() on tensor with symbolic sizes/strides" (drive_vc.py's
# INTEG_NUMEL_PROFILE hook prints the stack of every raising numel/size/stride builtin). Vc = compile ON, cudagraph_mode NONE, V-stock flags.
source $(dirname $0)/timing_lib.sh
VS=(CAND_LINE=attn CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
trun vc_diag "${VS[@]}" HOSTTRACE_TORCH_DISABLE_CACHES=0 INTEG_NUMEL_PROFILE=1 -- vllm/bench/drive_vc.py --arm Vc --check --batch-size 1 8 --prefill-lens 64 --mixed 0 --out vllm/out/vc_diag.json
echo ALLDONE
