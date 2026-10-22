#!/usr/bin/env bash
# c5s prefill / mixed replay check (Triton GDN, V-stock eager forward, no ARMV_EAGER_KINDS): does symti remove the state-pool index eager steps?
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/../../scratch/integration
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/snap/step318 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6/build_cpp11/install FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/build_c11 TORCH_EXTENSIONS_DIR=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/torch_ext_c11 ARMV_PIN_UNPINNED=1 ARMV_SYM_ORIGINS=1
export ARMV_PORTS=1 ARMV_CHECK_TWICE=1
bash /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/run_check.sh pm_c11 --batch-size 1 8 --prefill-lens 64 512 --ix-mixed 1024x8@1024-2048 2048x8@6000-8000 --ix-decode --ix-mixed 1024x8@1024-2048 2048x8@6000-8000 1024x64@1024-2048 --ix-reps 6 --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 64, "gdn_prefill_backend": "triton"}'
