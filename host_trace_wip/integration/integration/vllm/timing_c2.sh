#!/usr/bin/env bash
# Candidate 2 timing (GPU 0, one hold): drive.py arms x 2 interleaved rounds. Aligned and non-aligned sizes (vLLM's
# capture sizes 1,2,4,8,16,24,..): decode bs 1 8 64 5 100, prefill 64 512 2048 300 100, 4x128 / 8x128, mixed 8x128 + specs.
source $(dirname $0)/timing_lib.sh
W=(--batch-size 1 8 64 5 100 --prefill-lens 64 512 2048 300 100 --prefill-reqs 4 8 --mixed-spec 32x512 4x2048 64x64 128x16)
for r in ${ROUNDS:-1 2}; do
  trun t_eager_r$r CAND_LINE=cpp -- vllm/bench/drive.py --arm eager "${W[@]}" --out vllm/out/t_eager_r$r.json
  trun t_default_r$r CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm default "${W[@]}" --out vllm/out/t_default_r$r.json
  trun t_Vcpp_closed_r$r CAND_LINE=cpp -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_closed_r$r.json
  trun t_Vcpp_open_r$r CAND_LINE=cpp "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_open_r$r.json
  trun t_Vpy_closed_r$r CAND_LINE=py -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vpy_closed_r$r.json
  trun t_Vpy_open_r$r CAND_LINE=py "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vpy_open_r$r.json
  [ "${PAD_ARMS:-1}" = 1 ] && trun t_Vcpp_closed_pad_r$r CAND_LINE=cpp ARMV_PAD=1 -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_closed_pad_r$r.json
  [ "${PAD_ARMS:-1}" = 1 ] && trun t_Vcpp_open_pad_r$r CAND_LINE=cpp ARMV_PAD=1 "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_open_pad_r$r.json
done
echo ALLDONE
