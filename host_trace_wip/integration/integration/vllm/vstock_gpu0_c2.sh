#!/usr/bin/env bash
# Arm V-stock on the pinned build: vLLM's stock metadata build and input prep eagerly, the region from model.forward,
# metadata tensors + lifted symbolic ints as arguments; no template, no hostcuts, no _C ports (harvested), no re-jits,
# stock max_seq_len; attention through the trtllm fork (the closed op keyed on max_seq_len is core gap d).
# --check (drive + range), then decode timing vs default and V on the same build.
source $(dirname $0)/timing_lib.sh
VS=(CAND_LINE=pinned CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
trun c2_vstock_Vcheck "${VS[@]}" -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 64x64 --out vllm/out/c2_vstock_Vcheck.json
trun c2_vstock_range "${VS[@]}" ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_vstock_range.json
W=(--batch-size 1 8 64 128 --prefill-lens 64 512 --mixed 4 --async-decode 1 8 64 128)
for r in 1 2; do
  trun t_vstock_r$r "${VS[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_vstock_r$r.json
  trun t_pdefault_r$r CAND_LINE=pinned HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${W[@]}" --out vllm/out/t_pdefault_r$r.json
  trun t_pV_r$r CAND_LINE=pinned -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_pV_r$r.json
done
echo ALLDONE
