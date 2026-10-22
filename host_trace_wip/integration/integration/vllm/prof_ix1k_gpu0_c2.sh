#!/usr/bin/env bash
# The g4 1024-token chunk + 64 decodes row (KV 1024-2048), one hold: per-step cudagraph dispatch (cg_mode, padded tokens) and a
# profile of the last 2 reps for default; dispatch log only for default-nc and V-stock. Attn line, as g4.
source $(dirname $0)/timing_lib.sh
export UTIL_CAP=0.9
VS=(CAND_LINE=attn CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
W=(--max-model-len 16384 --batch-size 1 --prefill-lens 64 --mixed 0 --ix-mixed 1024x64@1024-2048 --ix-reps 6)
trun px_default CAND_LINE=attn HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_ix4.py --arm default "${W[@]}" --prof-ix-mixed vllm/out/prof_ix1k --out vllm/out/px_default.json
trun px_fullnc CAND_LINE=attn -- vllm/bench/drive_ix4.py --arm fullnc "${W[@]}" --out vllm/out/px_fullnc.json
trun px_vstock "${VS[@]}" -- vllm/bench/drive_ix4.py --arm V "${W[@]}" --out vllm/out/px_vstock.json
echo ALLDONE
