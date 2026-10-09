#!/usr/bin/env bash
# Candidate 2 tokens-per-step sweep (GPU 0, one hold): one step of exactly T tokens = nd decoding requests + prefill prompts
# filling the rest, T in 1k..32k, nd in {0, 64, 256}; engine max_num_batched_tokens 32768, max_num_seqs 512 (both arms).
# default and V cpp closed (2 interleaved rounds); eager at 2 points.
source $(dirname $0)/timing_lib.sh
E=(--max-batched 32768 --max-seqs 512 --batch-size 1 --prefill-lens 64 --mixed 0)
S=(--sweep-tokens 1024 2048 4096 8192 16384 32768 --sweep-decode 0 64 256)
for r in ${ROUNDS:-1 2}; do
  trun t_sweep_default_r$r CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${E[@]}" "${S[@]}" --out vllm/out/t_sweep_default_r$r.json
  trun t_sweep_Vcpp_closed_r$r CAND_LINE=cpp -- vllm/bench/drive.py --arm V "${E[@]}" "${S[@]}" --out vllm/out/t_sweep_Vcpp_closed_r$r.json
done
trun t_sweep_eager_r1 CAND_LINE=cpp -- vllm/bench/drive.py --arm eager "${E[@]}" --sweep-tokens 2048 16384 --sweep-decode 0 --out vllm/out/t_sweep_eager_r1.json
echo ALLDONE
