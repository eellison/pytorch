cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
M="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 131072 --attention-backend trtllm_mha"
GPU=1 FLUCT_GH=1 FLUCT_ARM=Fc FLUCT_ADAPTER_DIR=$G/sglang_stock/adapter FLUCT_N=256 FLUCT_BASE=4 FLUCT_REPS=0 MODEL_ARGS="$M" bash run_s.sh gh_fluct2 $G/bsdiag/fluct.py --batch-size 1
# the RD2 + C++ launcher line on Qwen3-8B: the step-1 check and the decode sweep
GPU=1 SGL_LINE=rd2cpp ARMF_ADAPTER_DIR=$G/sglang_stock/adapter bash run_s.sh r1_F armF/driveF_cpp.py --batch-size 1 8 64 --prefill-lens 64 512 2048 --reps 2 --decode-steps 8 --profile-dir $G/sglang_stock/out/prof_r1_F
GPU=1 SGL_LINE=rd2cpp FLUCT_GH=1 FLUCT_ARM=Fc FLUCT_ADAPTER_DIR=$G/sglang_stock/adapter FLUCT_N=256 FLUCT_BASE=4 FLUCT_REPS=3 MODEL_ARGS="$M" bash run_s.sh r2_fluct $G/bsdiag/fluct.py --batch-size 1
