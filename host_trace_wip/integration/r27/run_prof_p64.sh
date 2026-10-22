#!/usr/bin/env bash
# Profile one vLLM default prefill-64 step (compile + FULL_AND_PIECEWISE; it dispatches PIECEWISE) and one default-nc step: chrome traces
# out/prof/p64/<arm>_1x64.json (execute_model + sample_tokens annotated), for GPU busy / kernel count / host time (scripts/prof_one.py)
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install FI_HT_BUILD=$R/build_c12 TORCH_EXTENSIONS_DIR=$R/torch_ext_c12 DRIVE_STEP_WITH_SAMPLE=1
export R27_OVERLAY=$S INTEG_GPU_UTIL=0.5 INTEG_KV_BYTES=$(( 40 << 30 )) DRIVE_PROFILE_DIR=$R/out/prof/p64 DRIVE_PROFILE_CLASSES=1:64 DRIVE_PROFILE_AT=6
for arm in default fullnc; do
  DRIVE_PROFILE_TAG=$arm HOSTTRACE_TORCH_DISABLE_CACHES=0 bash $R/python_r27.sh $R/bench/drive_sweep27.py --arm $arm --model /data/eellison/models/Qwen3.8-27B-NVFP4 --language-model-only \
    --max-model-len 16384 --batch-size 1 --decode-tokens 8 --prefill-lens 64 --mixed 0 --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 128, "gdn_prefill_backend": "triton"}' \
    --out $R/out/prof/p64/$arm.json 2>&1 | grep -E "^prefill|Error|rc="
done
ls -la $R/out/prof/p64
