cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
M="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 131072 --attention-backend trtllm_mha"
GPU=1 FLUCT_GH=1 FLUCT_ARM=Fc FLUCT_ADAPTER_DIR=$G/sglang_stock/adapter FLUCT_N=256 FLUCT_BASE=4 FLUCT_REPS=1 MODEL_ARGS="$M" bash run_s.sh gh_fluct1 $G/bsdiag/fluct.py --batch-size 1
