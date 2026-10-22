#!/usr/bin/env bash
# run_s.sh TAG DRIVER [args...]: one SGLang job on python_sgl_attn.sh. GPU=1: gpu1.lock, CPUs 72-107 (correctness); GPU=0:
# gpu0.lock, CPUs 0-71 (timing). The GPU lock is taken inside the process by lockwrap.py after the heavy imports (no GPU
# use before it) and held to the end; a watchdog ends the run TMO (default 600) s after the lock is taken.
# Default (LOCKWRAP=0): flock -x around the whole process (visible as a flock process), timeout TMO. LOCKWRAP=1: the lock taken in-process after the imports (fcntl on the same file; not visible as a flock process). Qwen3-8B unless MODEL_ARGS is set; graphs
# disabled unless GRAPHS=1. LOCKWRAP_ADAPTER=adapter_cpp pre-imports the adapter (ARMF_ADAPTER_DIR) before sglang.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
GPU=${GPU:-1}; case $GPU in 0) CPUS=0-71;; 1) CPUS=72-107;; 2) CPUS=108-125;; 3) CPUS=126-143;; esac
# GPUs 2/3 (weekend rule, LANES.md): only when gpu_idle_ok.sh says no other user's process is on them, checked inside the hold
IDLE=""; [ "$GPU" -ge 2 ] && IDLE="$S/../eager/gpu_idle_ok.sh $GPU || { echo NOT_IDLE; exit 75; }; "
LOCK=$S/../eager/gpu$GPU.lock
tag=$1; drv=$2; shift 2
DIS="--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled"; [ "${GRAPHS:-0}" = 1 ] && DIS=""
M=${MODEL_ARGS:---model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 32768 --attention-backend trtllm_mha}
cd $S
if [ "${LOCKWRAP:-0}" = 1 ]; then
  CUDA_VISIBLE_DEVICES=$GPU LOCKWRAP_LOCK=$LOCK LOCKWRAP_TMO=${TMO:-600} timeout 21600 taskset -c $CPUS \
    bash $G/sglang_stock/python_sgl_attn.sh $G/sglang_stock/lockwrap.py $drv $M $DIS "$@" --out $G/sglang_stock/out/$tag.json > $G/sglang_stock/logs/$tag.log 2>&1
else
  CUDA_VISIBLE_DEVICES=$GPU flock -x $LOCK bash -c "$IDLE"'echo "LOCKED $(date +%T) load $(cut -d" " -f1 /proc/loadavg)"; exec "$@"' _ \
    timeout ${TMO:-600} taskset -c $CPUS bash $G/sglang_stock/python_sgl_attn.sh $drv $M $DIS "$@" --out $G/sglang_stock/out/$tag.json > $G/sglang_stock/logs/$tag.log 2>&1
fi
echo "$tag rc=$? $(date +%T)"
