#!/usr/bin/env bash
# Stock max_seq_len: (1) open path (trtllm fork) with max_seq_len symbolic (ARMV_DECODE_MAX_SEQ=sym): --check vs stock eager,
# traces/variants per split region (drive + range); (2) closed path: trtllm decode bindings across max_seq_len values
# (exact keys, vLLM's values) to see what max_seq_len changes in the harvested launch.
source $(dirname $0)/timing_lib.sh
SYM=(CAND_LINE=cpp CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_DECODE_MAX_SEQ=sym ARMV_PREFILL_MAX_KV=sym ARMV_BOUND=0)
trun c2_fork_sym_Vcheck "${SYM[@]}" -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 4x2048 64x64 128x16 --out vllm/out/c2_fork_sym_Vcheck.json
trun c2_fork_sym_range "${SYM[@]}" ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_fork_sym_range.json
trun c2_fork_model_range CAND_LINE=cpp CAND_FORK=1 ARMV_TRTLLM_FORK=1 ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_fork_model_range.json
trun diag_closed_trtllm_keys CAND_LINE=cpp ARMV_FAMILY=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 ARMV_DUMP_TRTLLM=1 -- vllm/bench/drive.py --arm V --batch-size 8 --prefill-lens 64 --mixed 0 --out vllm/out/diag_closed_trtllm_keys.json
echo ALLDONE
