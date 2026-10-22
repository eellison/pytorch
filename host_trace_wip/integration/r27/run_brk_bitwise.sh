#!/usr/bin/env bash
# report-only: vLLM breakable graphs' captured prefill (Qwen GDN inside the segments) vs vLLM's eager prefill (default-nc), lengths 33-80
R=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
I=/data/eellison/src/pytorch/agent_space/paramgraph/land/int6
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang/site
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
export R27_WT=$I/snap/step324 R27_INSTALL=$I/build_cpp12/install R27_OVERLAY=$S TORCH_EXTENSIONS_DIR=$R/torch_ext_c12
VLLM_USE_BREAKABLE_CUDAGRAPH=1 bash $R/python_r27.sh $R/bench/brk_bitwise.py $R/out/brk_bitwise_breakable.json breakable 2>&1 | grep -E "^done|Error" 
bash $R/python_r27.sh $R/bench/brk_bitwise.py $R/out/brk_bitwise_fullnc.json fullnc 2>&1 | grep -E "^done|Error"
python3 $R/bench/brk_bitwise.py --compare $R/out/brk_bitwise_breakable.json $R/out/brk_bitwise_fullnc.json
