# Flash-Next stock eager: degenerate output diagnosis (q38_probe2: prefill argmax 0, identical logits digests for
# prefill 64 and 512, decode bs 1 all token 0). Two loads in sequence, each its own <= 25 min hold on GPU 2 (idle-checked).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
P=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock/q38_probe.py
( export Q38_MOE="" Q38_LIN=triton; source ./q38_args.sh; GPU=2 TMO=1500 MODEL_ARGS="$Q38_ARGS" bash run_s.sh q38_probe3a $P --batch-size 1 )
( unset Q38_MOE; source ./q38_args.sh; GPU=2 TMO=1500 MODEL_ARGS="$Q38_ARGS" bash run_s.sh q38_probe3b $P --batch-size 1 )
