#!/usr/bin/env bash
# Decode-step timing for the P A/B on GPU 0 (exclusive), int6 candidate 2 cpp top (python_sgl_cand2_even.sh), trtllm_mha,
# the bench_one_batch path (not the server scheduler). Metrics as armF/prep/run_time.sh: per decode step the pre-launch
# host time (stamps_prep.py + launchshim.so -> PREP_STAMPS, us), the synced step (R["decode"][bs]["step_median_ms"]),
# the pipelined step (PREP_ASYNC: R["async"][bs]), and one profiled decode step per bs (--profile-decode; count GPU ops
# outside graphs with adapters_c2/af/count_outside.py on prof_<job>/decode_<bs>.json). PREP_ASYNC_PROF chrome traces too.
# One gpu0.lock hold per job (the whole run), CUDA_VISIBLE_DEVICES=0, taskset -c 0-71; inside the hold an idle check (no
# foreign compute process on GPU 0); while not idle the lock is released and the job retries after a wait.
# Usage: ROUND=r1 run_prep_time_c2.sh <arm>...; arms:
#   A  (stock SGLang graphs, the model's own path: ARMF_OFF=1 under this driver)
#   Af (stock graphs + P's fused prep kernel eagerly: adapters_c2/af)
#   F  (graphs off, host trace, prep eager: ARMF_PREP=0)
#   P  (graphs off, prep inside the trace: ARMF_PREP=1)
#   Pc (P on the C++ bound entry: PREP_CPP=1, adapters_c2/adapter_prep_cpp.py)
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $S
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
L=$X/python_sgl_cand2_even.sh
O=$X/time_c2/prep; G=$O/logs; mkdir -p $O $G
R=${ROUND:-r1}
BS=${BS:-"1 8 13 64"}
BASE="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens 32768 --attention-backend trtllm_mha"
COMMON="--batch-size $BS --prefill-lens 64 --profile-prefill --profile-decode $BS"
DIS="--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled"
for a in "$@"; do
  case $a in
    A)  X2=""; ENVX=(ARMF_OFF=1);;
    Af) X2=""; ENVX=(ARMF_ADAPTER_DIR=$X/adapters_c2/af);;
    F)  X2="$DIS"; ENVX=(ARMF_PREP=0);;
    P)  X2="$DIS"; ENVX=(ARMF_PREP=1);;
    Pc) X2="$DIS"; ENVX=(ARMF_PREP=1 PREP_CPP=1 ARMF_ADAPTER_DIR=$X/adapters_c2);;
    *) echo "unknown arm $a"; continue;;
  esac
  N=c2_even_trtllm_${a}_$R
  for try in $(seq 30); do
    echo "=== $a $R try=$try $(date +%T)"
    env "${ENVX[@]}" PREP_ASYNC=1 PREP_ASYNC_PROF=$O/aprof_$N LD_PRELOAD=$S/armF/prep/launchshim.so PREP_STAMPS=$O/$N.stamps.json \
      CUDA_VISIBLE_DEVICES=0 flock -x $LOCK bash -c '
      apps=$(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader)
      if [ -n "$apps" ]; then echo "NOT_IDLE apps=[$apps]"; exit 75; fi
      echo "IDLE $(date +%T) load=$(cut -d" " -f1 /proc/loadavg)"; exec "$@"' _ \
      timeout ${TMO:-3600} taskset -c 0-71 bash $L armF/prep/stamps_prep.py armF/prep/driveF_prep.py $BASE $X2 $COMMON \
      --out $O/$N.json --profile-dir $O/prof_$N > $G/$N.log 2>&1
    rc=$?
    if [ $rc = 75 ]; then echo "  $(head -1 $G/$N.log)"; sleep 120; continue; fi
    echo "rc=$rc $(date +%T)"; break
  done
done
