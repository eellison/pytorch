#!/usr/bin/env bash
# default's mixed step explained: drive_prof.py (drive.py + cudagraph dispatch / request mix / Dynamo counter log per step, torch.profiler
# over the measured mixed steps) for default, default-nc and V-stock on CAND_LINE=attn; the same decode/prefill sections run first, as in g2.
source $(dirname $0)/timing_lib.sh
L=attn
VS=(CAND_LINE=$L CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
W=(--batch-size 1 8 64 128 --prefill-lens 64 512 --mixed 8 --prof-mixed vllm/out/prof_mixed)
trun pm_default CAND_LINE=$L HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_prof.py --arm default "${W[@]}" --out vllm/out/pm_default.json
trun pm_vstock "${VS[@]}" -- vllm/bench/drive_prof.py --arm V "${W[@]}" --out vllm/out/pm_vstock.json
trun pm_fullnc CAND_LINE=$L -- vllm/bench/drive_prof.py --arm fullnc "${W[@]}" --out vllm/out/pm_fullnc.json
echo ALLDONE
