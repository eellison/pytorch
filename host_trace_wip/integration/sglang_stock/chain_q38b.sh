cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
source ./q38_args.sh
# one Flash-Next hold (<= 25 min, coordinator 10-09): load once, stock eager smoke + structure dump
GPU=${Q38_GPU:-0} TMO=1500 MODEL_ARGS="$Q38_ARGS" bash run_s.sh q38_probe1 /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock/q38_probe.py --batch-size 1
