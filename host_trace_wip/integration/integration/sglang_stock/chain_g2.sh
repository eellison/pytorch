# after the Flash-Next diagnosis loads: short Qwen3-8B / torch jobs on GPU 2 (weekend rule; GPU 1 queue ~18 deep)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch
while pgrep -f "bash chain_q38c.sh" > /dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=2 flock -x $X/eager/gpu2.lock bash -c "$X/eager/gpu_idle_ok.sh 2 || exit 75; exec timeout 300 taskset -c 108-125 bash python_sgl_attn.sh repro/guards_hold_retrace2.py" > logs/repro_gh2.log 2>&1
echo "repro_gh2 rc=$? $(date +%T)"
M="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 131072 --attention-backend trtllm_mha"
GPU=2 SGL_LINE=rd2cpp ARMF_ADAPTER_DIR=$G/sglang_stock/adapter bash run_s.sh r1_F armF/driveF_cpp.py --batch-size 1 8 64 --prefill-lens 64 512 2048 --reps 2 --decode-steps 8 --profile-dir $G/sglang_stock/out/prof_r1_F
GPU=2 SGL_LINE=rd2cpp FLUCT_GH=1 FLUCT_ARM=Fc FLUCT_ADAPTER_DIR=$G/sglang_stock/adapter FLUCT_N=256 FLUCT_BASE=4 FLUCT_REPS=3 MODEL_ARGS="$M" bash run_s.sh r2_fluct $G/bsdiag/fluct.py --batch-size 1
