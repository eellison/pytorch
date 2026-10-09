cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
M="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 131072 --attention-backend trtllm_mha"
FLUCT_ARM=Fc FLUCT_ADAPTER_DIR=$G/sglang_stock/adapter FLUCT_N=256 FLUCT_BASE=4 FLUCT_REPS=1 MODEL_ARGS="$M" bash run_s.sh s2_fluct $G/bsdiag/fluct.py --batch-size 1
MIXM_F=1 MODEL_ARGS="$M" bash run_s.sh s3_mixF $G/sglang_stock/mixed_stock.py --enable-mixed-chunk --batch-size 1 8 32 64 128 --profile-dir $G/sglang_stock/out/prof_s3_mixF
MIXM_F=0 MODEL_ARGS="$M" bash run_s.sh s3_mixD $G/sglang_stock/mixed_stock.py --enable-mixed-chunk --batch-size 1 8 32 64 128 --profile-dir $G/sglang_stock/out/prof_s3_mixD
