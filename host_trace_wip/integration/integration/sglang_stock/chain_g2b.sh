cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while pgrep -f "bash chain_g2.sh" > /dev/null; do sleep 20; done
# the generic hook adapter on Qwen3-8B (attn line + fork), eager vs hooked in one load
GPU=2 bash run_s.sh h1_seq $G/sglang_stock/seqcheck.py --batch-size 1 --bs 1 8 64
