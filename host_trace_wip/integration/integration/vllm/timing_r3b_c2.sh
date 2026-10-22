#!/usr/bin/env bash
# Round 3 arms lost to a foreign GPU 0 job on 10-07 14:13-14:56 (their contended JSONs are *.json.contended).
source $(dirname $0)/timing_lib.sh
W=(--batch-size 1 8 64 5 100 --prefill-lens 64 512 2048 300 100 --prefill-reqs 4 8 --mixed-spec 32x512 4x2048 64x64 128x16)
trun t_Vcpp_open_r3 CAND_LINE=cpp "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vcpp_open_r3.json
trun t_Vpy_closed_r3 CAND_LINE=py -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vpy_closed_r3.json
trun t_Vpy_open_r3 CAND_LINE=py "${OPEN[@]}" -- vllm/bench/drive.py --arm V "${W[@]}" --out vllm/out/t_Vpy_open_r3.json
echo ALLDONE
