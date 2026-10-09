# GPU 1 was held by long other-lane jobs (int6 candidate4 tests, 1 h+, 24 waiters); these short correctness jobs take
# turns on GPU 0 instead, one gpu0.lock hold each (<= 10 min).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
CUDA_VISIBLE_DEVICES=0 flock -x $G/../eager/gpu0.lock timeout 300 taskset -c 0-71 bash python_sgl_attn.sh repro/core_gaps_sglang.py > logs/repro3.log 2>&1
echo "repro3 rc=$? $(date +%T)"
M="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 131072 --attention-backend trtllm_mha"
GPU=0 FLUCT_ARM=Fc FLUCT_ADAPTER_DIR=$G/sglang_stock/adapter FLUCT_N=256 FLUCT_BASE=4 FLUCT_REPS=3 MODEL_ARGS="$M" bash run_s.sh s2b_fluct $G/bsdiag/fluct.py --batch-size 1
source ./q38_args.sh
GPU=0 SGL_FORK=0 MODEL_ARGS="$Q38_ARGS" bash run_s.sh q38_D0 bench/drive.py --batch-size 1 8 --prefill-lens 64 512 --reps 1 --decode-steps 4 --profile-prefill --profile-decode
