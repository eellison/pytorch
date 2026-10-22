#!/usr/bin/env bash
# both forms of repro_fp4_foreign.py (launched by gpu1_cmds.sh's launcher)
L=serve/python_vllm_cand3.sh
FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_repro bash $L serve/repro/repro_fp4_foreign.py; echo "rc=$? (not observed: the GEMM is a foreign TVM-FFI eager step)"
REPRO_OBSERVE=1 FLASHINFER_WORKSPACE_BASE=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/serve/fi_ws_ht bash $L serve/repro/repro_fp4_foreign.py; echo "rc=$? (observed + fi_ws_ht workspace, whose mm_fp4 objects were compiled and exported under observation: CuTe DSL route)"
