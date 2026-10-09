#!/usr/bin/env bash
# Follow-ups (GPU 0, one hold): open NaN with eager memory (is the run-buffer plan involved?); stock max_seq_len cost
# (range --check with ARMV_DECODE_MAX_SEQ/PREFILL_MAX_KV=actual); Vc feasibility (compile ON, cudagraph NONE, V around it).
source $(dirname $0)/timing_lib.sh
D=(--arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0)
trun diag_fork_silu_memeager CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=silu_and_mul ARMV_MEMORY=eager -- vllm/bench/drive.py "${D[@]}" --out vllm/out/diag_fork_silu_memeager.json
trun c2_cpp_closed_maxseq_range CAND_LINE=cpp ARMV_STATIC_SHAPES=1 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual -- vllm/bench/rangesweep.py --check --out vllm/out/c2_cpp_closed_maxseq_range.json
trun c2_cpp_closed_maxseq_Vcheck CAND_LINE=cpp ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --out vllm/out/c2_cpp_closed_maxseq_Vcheck.json
trun c2_Vc_check CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm Vc --check --batch-size 1 8 64 --prefill-lens 64 --mixed 2 --out vllm/out/c2_Vc_check.json
trun t_Vc_r1 CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm Vc --batch-size 1 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/t_Vc_r1.json
trun t_compnc_r1 CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive.py --arm compnc --batch-size 1 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/t_compnc_r1.json
echo ALLDONE
