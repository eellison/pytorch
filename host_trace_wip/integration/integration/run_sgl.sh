#!/usr/bin/env bash
# SGLang counts/correctness on int6. LINE=even|odd (python_sgl_int6_$LINE.sh), BACKEND=trtllm_mha|triton, GPU=1|0.
# Arms: D (eager, bench/drive.py), F (armF/driveF.py), Fc (driveF_cpp.py, even only), MD/MF (mixed, driveM.py),
# SD/SP/SPc (prep/driveF_prep.py PREP_SEQCHECK: eager ARMF_OFF=1 / P ARMF_PREP=1 / Pc PREP_CPP=1, even only).
# KV pool capped by --max-total-tokens (GPUs are shared); counts and bitwise do not depend on the pool size.
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $S
GPU=${GPU:-1}
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu$GPU.lock
CPUS=$([ "$GPU" = 1 ] && echo 72-107 || echo 0-71)
LINE=${LINE:-even}; BACKEND=${BACKEND:-trtllm_mha}; L=$X/python_sgl_int6_$LINE.sh
T=${TAG:-$LINE}_${BACKEND%%_*}
BASE="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 --max-total-tokens ${MAXTOK:-32768} --attention-backend $BACKEND"
DIS="--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled"
for a in "$@"; do
  ENVX=(); OUT=(--out $X/out/${T}_$a.json --profile-dir $X/out/prof_${T}_$a)
  case $a in
    D) DRV=bench/drive.py; ARGS="--batch-size 1 8 64 --prefill-lens 64 512 2048";;
    F) DRV=armF/driveF.py; ARGS="--batch-size 1 8 64 --prefill-lens 64 512 2048";;
    Fc) DRV=armF/driveF_cpp.py; ARGS="--batch-size 1 8 64 --prefill-lens 64 512 2048";;
    MD) DRV=armF/mixed/driveM.py; ENVX=(MIXM_F=0); ARGS="--enable-mixed-chunk --batch-size 1 8 32 64 128";;
    MF) DRV=armF/mixed/driveM.py; ENVX=(MIXM_F=1); ARGS="--enable-mixed-chunk --batch-size 1 8 32 64 128";;
    SD|SP|SPc|SDL|SPL|SPcL)
      DRV=armF/prep/driveF_prep.py; ARGS="--batch-size 1 8 64 --prefill-lens 64"; OUT=(--out /dev/null)
      case $a in SD*) ENVX=(ARMF_OFF=1);; SPc*) ENVX=(ARMF_PREP=1 PREP_CPP=1);; SP*) ENVX=(ARMF_PREP=1);; esac
      case $a in *L) ENVX+=(PREP_SEQ_LEN=250);; esac
      ENVX+=(PREP_SEQCHECK=$X/out/seq_${T}_$a.json);;
    *) echo "unknown arm $a"; continue;;
  esac
  echo "=== $a $(date +%T) free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader)"
  for try in 1 2 3; do
    env "${ENVX[@]}" CUDA_VISIBLE_DEVICES=$GPU flock -x $LOCK timeout ${TMO:-2700} taskset -c $CPUS bash $L $DRV $BASE $DIS $ARGS "${OUT[@]}" > $X/logs/${T}_$a.log 2>&1
    rc=$?; echo "rc=$rc try=$try $(date +%T)"
    grep -q EADDRINUSE $X/logs/${T}_$a.log || break
  done
done
