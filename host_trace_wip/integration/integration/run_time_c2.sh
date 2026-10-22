#!/usr/bin/env bash
# Timing on GPU 0 (exclusive): one gpu0.lock hold per job (the whole run), CUDA_VISIBLE_DEVICES=0, taskset -c 0-71, and
# inside the hold an idle check (no foreign compute process on GPU 0, load average recorded; LOADMAX gates on it if set) before the run starts;
# while not idle the lock is released and the job retries after a wait, so the vLLM lane can take turns.
# NOLOCK=1: the caller holds gpu0.lock for a group of jobs (run_time_group.sh); the idle check then waits inside the hold.
# Usage: LINE=even|odd run_time_c2.sh <job>...; jobs:
#   steady (bench/drive.py timing; decode bs aligned 1 8 64 and non-aligned 13 50 130, prefill T 64 512 2048 + ragged):
#     tA tD tF tFc tP tPc (closed: cuBLAS) | tAT tDT tFT tM (open: SGLang's bf16 GEMM backend, TGV)
#   mixed (driveM.py, chunks 64/512/2048 x decodes 8/32/128, ext-decode bs 1..128): mA mD mF | mAT mDT mFT
#   startup (startup_sgl.py): sA sD sF sFc sP sPc sFT sM sAT  (suffix W: warm, STARTUP_LOAD of the arm's saved bindings)
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $S
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
LINE=${LINE:-even}; BACKEND=${BACKEND:-trtllm_mha}; L=$X/python_sgl_cand2_$LINE.sh
O=$X/time_c2; G=$X/time_c2/logs; mkdir -p $O $G $O/bindings
R=${ROUND:-r1}; T=c2_${LINE}_${BACKEND%%_*}
DIS="--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled"
STEADY="--batch-size 1 8 13 50 64 130 --prefill-lens 64 512 2048 --profile-prefill --profile-decode"
MIX="--enable-mixed-chunk --batch-size 1 8 32 64 128"
SU="--batch-size 1 8 13 64 --prefill-lens 64 512 2048"
BF16=$S/armF/tgv/drive_bf16.py
TGVENV=(ARMF_ADAPTER_DIR=$X/adapters_c2/tgv ARMF_TGV=1 ARMF_STATIC=1)
for j in "$@"; do
  ENVX=(); PRE=(); X2="$DIS"; MAX=65536; OUT=(--out $O/${T}_${j}_$R.json)
  case $j in
    tA)  DRV=bench/drive.py; X2=""; ARGS="$STEADY";;
    tD)  DRV=bench/drive.py; ARGS="$STEADY";;
    tF)  DRV=armF/driveF.py; ARGS="$STEADY";;
    tFc) DRV=armF/driveF_cpp.py; ARGS="$STEADY";;
    tP)  DRV=armF/prep/driveF_prep.py; ENVX=(ARMF_PREP=1); ARGS="$STEADY";;
    tPc) DRV=armF/prep/driveF_prep.py; ENVX=(ARMF_PREP=1 PREP_CPP=1 ARMF_ADAPTER_DIR=$X/adapters_c2); ARGS="$STEADY";;
    tAT) PRE=($BF16); DRV=bench/drive.py; X2=""; ARGS="$STEADY";;
    tDT) PRE=($BF16); DRV=bench/drive.py; ARGS="$STEADY";;
    tFT) PRE=($BF16); DRV=armF/driveF.py; ENVX=("${TGVENV[@]}"); ARGS="$STEADY";;
    tM)  DRV=armF/prep/driveF_prep.py; ENVX=(ARMF_PREP=1 PREP_CPP=1 PREP_M=1 ARMF_ADAPTER_DIR=$X/adapters_c2); ARGS="$STEADY";;
    mA)  DRV=armF/mixed/driveM.py; ENVX=(MIXM_F=0); X2=""; ARGS="$MIX"; MAX=262144;;
    mD)  DRV=armF/mixed/driveM.py; ENVX=(MIXM_F=0); ARGS="$MIX"; MAX=262144;;
    mF)  DRV=armF/mixed/driveM.py; ENVX=(MIXM_F=1); ARGS="$MIX"; MAX=262144;;
    mAT) DRV=$X/adapters_c2/mixed_open.py; ENVX=(MIXM_F=0); X2=""; ARGS="$MIX"; MAX=262144;;
    mDT) DRV=$X/adapters_c2/mixed_open.py; ENVX=(MIXM_F=0); ARGS="$MIX"; MAX=262144;;
    mFT) DRV=$X/adapters_c2/mixed_open.py; ENVX=(MIXM_F=1 "${TGVENV[@]}"); ARGS="$MIX"; MAX=262144;;
    s*)  a=${j#s}; W=""; [ "${a%W}" != "$a" ] && W=1 && a=${a%W}
         DRV=$X/startup_sgl.py; ARGS="$SU"; ENVX=(STARTUP_ARM=$a)
         [ "$a" = A ] || [ "$a" = AT ] && X2=""
         B=$O/bindings/${T}_$a.pkl
         if [ -n "$W" ]; then ENVX+=(STARTUP_LOAD=$B); else ENVX+=(STARTUP_SAVE=$B); fi;;
    *) echo "unknown job $j"; continue;;
  esac
  BASE="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens $MAX --attention-backend $BACKEND"
  for try in $(seq 30); do
    echo "=== $j $R try=$try $(date +%T)"
    FL=(flock -x $LOCK); [ -n "${NOLOCK:-}" ] && FL=()
    env "${ENVX[@]}" CUDA_VISIBLE_DEVICES=0 "${FL[@]}" bash -c '
      apps=$(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader)
      l=$(cut -d" " -f1 /proc/loadavg)
      if [ -n "$apps" ] || [ "${l%.*}" -ge "${LOADMAX:-100000}" ]; then echo "NOT_IDLE load=$l apps=[$apps]"; exit 75; fi
      echo "IDLE $(date +%T) load=$l"; export STARTUP_T0=$(date +%s.%N); exec "$@"' _ \
      timeout ${TMO:-3600} taskset -c 0-71 bash $L "${PRE[@]}" $DRV $BASE $X2 $ARGS "${OUT[@]}" > $G/${T}_${j}_$R.log 2>&1
    rc=$?
    if [ $rc = 75 ]; then echo "  $(head -1 $G/${T}_${j}_$R.log)"; sleep ${IDLE_WAIT:-120}; continue; fi
    echo "rc=$rc $(date +%T)"; break
  done
done
