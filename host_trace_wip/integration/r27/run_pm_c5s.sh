#!/usr/bin/env bash
# c5s prefill / mixed replay check (Triton GDN, V-stock eager forward, no ARMV_EAGER_KINDS): does symti remove the state-pool index eager steps?
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/../../scratch/integration
export R27_WT=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/symti/wt314 R27_INSTALL=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/symti/install15 FI_HT_BUILD=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/build_c5s TORCH_EXTENSIONS_DIR=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/torch_ext_c5s
export ARMV_PORTS=1 ARMV_CHECK_TWICE=1
bash /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27/run_check.sh pm_c5s --batch-size 1 8 --prefill-lens 64 512 --ix-mixed 1024x8@1024-2048 2048x8@6000-8000 --ix-decode --ix-reps 5 --llm-kw '{"kv_cache_dtype": "fp8", "max_num_seqs": 64, "gdn_prefill_backend": "triton"}'
