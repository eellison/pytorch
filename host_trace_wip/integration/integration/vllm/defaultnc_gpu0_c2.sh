#!/usr/bin/env bash
# default-nc (vLLM compile off + its own full CUDA graphs: FULL_DECODE_ONLY, and FULL) next to default and V-stock, pinned
# build. Correctness: greedy tokens vs eager and vs V-stock (e2e_tokens). Timing: 2 interleaved rounds.
source $(dirname $0)/timing_lib.sh
VS=(CAND_LINE=pinned CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0)
trun e2e_p_eager CAND_LINE=pinned -- vllm/bench/e2e_tokens.py --arm eager --out vllm/out/e2e_p_eager.json
trun e2e_p_fullnc CAND_LINE=pinned -- vllm/bench/e2e_tokens.py --arm fullnc --out vllm/out/e2e_p_fullnc.json
trun e2e_p_fullnc_full CAND_LINE=pinned -- vllm/bench/e2e_tokens.py --arm fullnc_full --out vllm/out/e2e_p_fullnc_full.json
trun e2e_p_vstock "${VS[@]}" -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_p_vstock.json
W=(--batch-size 1 8 64 128 --prefill-lens 64 512 --mixed 4 --async-decode 1 8 64 128)
for r in 1 2; do
  trun t_dnc_fullnc_r$r CAND_LINE=pinned -- vllm/bench/drive.py --arm fullnc "${W[@]}" --out vllm/out/t_dnc_fullnc_r$r.json
  trun t_dnc_fullnc_full_r$r CAND_LINE=pinned -- vllm/bench/drive.py --arm fullnc_full "${W[@]}" --out vllm/out/t_dnc_fullnc_full_r$r.json
  trun t_dnc_vstock_r$r "${VS[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_dnc_vstock_r$r.json
  trun t_dnc_default_r$r CAND_LINE=pinned HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${W[@]}" --out vllm/out/t_dnc_default_r$r.json
done
echo ALLDONE
