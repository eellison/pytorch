#!/usr/bin/env bash
# (b) diagnosis: which eager call takes a scalar derived from a tape allocation base (ARMV_SYM_ORIGINS SYMCALL/SYMDECL lines)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/../../scratch/integration
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/step318 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp11/install FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/build_c11 TORCH_EXTENSIONS_DIR=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/torch_ext_c11 ARMV_SYM_ORIGINS=1
export ARMV_PORTS=1 ARMV_CHECK_TWICE=1
bash /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/run_check.sh symdiag_c11 --batch-size 1 --prefill-lens 64 --ix-mixed 1024x8@1024-2048 --ix-reps 2 --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 64, "gdn_prefill_backend": "triton"}'
