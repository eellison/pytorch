#!/usr/bin/env bash
# GPU 1 correctness chain 2 (gpu1_run.sh: lock per run, retry on foreign-memory init failures, explicit KV when shared).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
R=vllm/gpu1_run.sh
OPEN=(CAND_FORK=1 ARMV_TRACED_IMPLS=1 ARMV_TRTLLM_FORK=1)
bash $R c2_cpp_VP2_Vcheck CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/drive.py --arm V --check --batch-size 1 8 64 128 --prefill-lens 64 --mixed 2 --out vllm/out/c2_cpp_VP2_Vcheck.json
bash $R e2e_VP2 CAND_LINE=cpp ARMV_VP=1 -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_VP2.json
bash $R e2e_V CAND_LINE=cpp -- vllm/bench/e2e_tokens.py --arm V --out vllm/out/e2e_V.json
bash $R e2e_VAf CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/e2e_tokens.py --arm VAf --out vllm/out/e2e_VAf.json
bash $R e2e_default CAND_LINE=cpp HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/e2e_tokens.py --arm default --out vllm/out/e2e_default.json
bash $R diag_both_nopdl CAND_LINE=cpp "${OPEN[@]}" ARMV_HT_SET=torch.cuda._host_trace.trace_pdl=0,torch.cuda._host_trace_capture.keep_programmatic_edges=0 -- vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/diag_both_nopdl.json
bash $R diag_fork_cache CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=reshape_and_cache_flash -- vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/diag_fork_cache.json
bash $R diag_fork_nocache CAND_LINE=cpp "${OPEN[@]}" ARMV_TRACED_IMPLS_NAMES=rms_norm,fused_add_rms_norm,rotary_embedding,silu_and_mul -- vllm/bench/drive.py --arm V --check --batch-size 5 8 64 --prefill-lens 64 --mixed 0 --out vllm/out/diag_fork_nocache.json
bash $R c2_cpp_closed_shrink_Vcheck CAND_LINE=cpp -- vllm/bench/drive.py --arm V --check --batch-size 1 --prefill-lens 64 --mixed 0 --shrink 64 128 --out vllm/out/c2_cpp_closed_shrink_Vcheck.json
bash $R c2_py_open_Vcheck CAND_LINE=py "${OPEN[@]}" -- vllm/bench/drive.py --arm V --check --prefill-reqs 4 8 --mixed-spec 32x512 4x2048 64x64 128x16 --out vllm/out/c2_py_open_Vcheck.json
bash $R c2_py_open_range CAND_LINE=py "${OPEN[@]}" ARMV_STATIC_SHAPES=1 -- vllm/bench/rangesweep.py --check --out vllm/out/c2_py_open_range.json
echo ALLDONE
