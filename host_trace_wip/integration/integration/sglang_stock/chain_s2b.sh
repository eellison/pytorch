cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while pgrep -f "chain_s23.sh" > /dev/null; do sleep 15; done
M="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 131072 --attention-backend trtllm_mha"
FLUCT_ARM=Fc FLUCT_ADAPTER_DIR=$G/sglang_stock/adapter FLUCT_N=256 FLUCT_BASE=4 FLUCT_REPS=3 MODEL_ARGS="$M" bash run_s.sh s2b_fluct $G/bsdiag/fluct.py --batch-size 1
