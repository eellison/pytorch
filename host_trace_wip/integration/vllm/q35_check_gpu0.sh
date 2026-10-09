#!/usr/bin/env bash
# Qwen3.5-9B bring-up. V-stock attention: the TRT-LLM C++ launcher path (CAND_LINE=trtcpp, ARMV_TRTLLM_CPP=1; the fork refuses head dim 256). Recipe (vllm-project/recipes models/Qwen/Qwen3.5-9B.yaml @ fc2e815, copy in vllm/recipes/):
# --trust-remote-code (+ opt-in --language-model-only, used: text-only), no attention backend set -> vLLM's own pick (auto).
# Step 1 (approved 10-08: the bf16 head-256 trtllm-gen cubins, once): an eager run over every planned shape with FLASHINFER_NO_DOWNLOAD
# unset (non-fork env, cache = land/scratch/vllm/cache/.cache/flashinfer); the files it adds go to logs/q35_cubins_fetched.txt.
# Everything after runs with FLASHINFER_NO_DOWNLOAD=1. Step 2: default smoke; step 3: V-stock --check vs stock eager + counts.
source $(dirname $0)/timing_lib.sh
export UTIL_CAP=0.9
M=/data/eellison/models/Qwen3.5-9B; CB=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm/cache/.cache/flashinfer
VS=(CAND_LINE=trtcpp ARMV_TRTLLM_CPP=1 ARMV_STOCK=1 ARMV_METADATA=eager ARMV_HOSTCUTS=0 ARMV_DECODE_MAX_SEQ=actual ARMV_PREFILL_MAX_KV=actual ARMV_BOUND=0 FLASHINFER_NO_DOWNLOAD=1)
C=(--model $M --trust-remote-code --language-model-only --max-model-len 16384)
IX=(--ix-mixed 1024x64@1024-2048 1024x128@1024-2048 2048x64@1024-2048 2048x128@1024-2048 1024x64@6000-8000 1024x128@6000-8000 2048x64@6000-8000 2048x128@6000-8000
    4096x64@4096-8000 8192x32@4096-8000 8192x64@4096-8000 8192x128@4096-8000 --ix-decode 64:6000-8000 128:6000-8000 256:6000-8000 --ix-reps 3)
unset FLASHINFER_NO_DOWNLOAD
find $CB -type f | sort > vllm/logs/q35_cubins_before.txt
trun q35_eager_fetch CAND_LINE=attn -- vllm/bench/drive_q35.py --arm eager "${C[@]}" --batch-size 1 8 64 --prefill-lens 64 512 --prefill-reqs 4 8 --mixed 4 --mixed-spec 32x512 64x64 --async-decode 1 8 "${IX[@]}" --out vllm/out/q35_eager_fetch.json
find $CB -type f | sort > vllm/logs/q35_cubins_after.txt
comm -13 vllm/logs/q35_cubins_before.txt vllm/logs/q35_cubins_after.txt | tee vllm/logs/q35_cubins_fetched.txt | xargs -r du -cb | tail -1 > vllm/logs/q35_cubins_fetched_bytes.txt
echo "fetched $(wc -l < vllm/logs/q35_cubins_fetched.txt) files, $(cat vllm/logs/q35_cubins_fetched_bytes.txt)"
{ echo "- $(date '+%m-%d %H:%M') Qwen3.5-9B cubin fetch (user-approved, once; FLASHINFER_NO_DOWNLOAD=1 for everything after): $(wc -l < vllm/logs/q35_cubins_fetched.txt) files, $(cut -f1 vllm/logs/q35_cubins_fetched_bytes.txt) bytes, into land/scratch/vllm/cache/.cache/flashinfer:"
  sed 's|.*/flashinfer/|    |' vllm/logs/q35_cubins_fetched.txt; } >> vllm/STATUS.md
grep -h "Using .* backend\|attention backend" vllm/logs/q35_eager_fetch.log | sort | uniq -c | head -5
export FLASHINFER_NO_DOWNLOAD=1
trun q35_default_smoke CAND_LINE=trtcpp FLASHINFER_NO_DOWNLOAD=1 HOSTTRACE_TORCH_DISABLE_CACHES=0 -- vllm/bench/drive_q35.py --arm default "${C[@]}" --batch-size 1 8 --prefill-lens 64 --mixed 0 --out vllm/out/q35_default_smoke.json
trun q35_vstock_Vcheck "${VS[@]}" -- vllm/bench/drive_q35.py --arm V --check "${C[@]}" --batch-size 1 8 64 --prefill-lens 64 512 --prefill-reqs 4 8 --mixed 4 --mixed-spec 32x512 64x64 --out vllm/out/q35_vstock_Vcheck.json
echo ALLDONE
