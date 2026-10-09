cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/sglang_stock
source ./q38_args.sh
# stock SGLang eager (no hook), stock FlashInfer (no fork): does the model load and run in our env?
SGL_FORK=0 MODEL_ARGS="$Q38_ARGS" bash run_s.sh q38_D0 bench/drive.py --batch-size 1 8 --prefill-lens 64 512 --reps 1 --decode-steps 4 --profile-prefill --profile-decode
