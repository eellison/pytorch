#!/usr/bin/env bash
# SGLang correctness/counts on int6 candidate 2. LINE=even|odd (python_sgl_cand2_$LINE.sh), BACKEND=trtllm_mha|triton, GPU=1|0.
# Closed (cuBLAS GEMMs, one_batch's default):
#   A (SGLang default graphs, bench/drive.py), D (eager), F (armF/driveF.py), Fc (driveF_cpp.py, even only),
#   MD/MF (mixed, driveM.py), SD/SP/SPc (prep/driveF_prep.py PREP_SEQCHECK: eager / P / Pc, even only for Pc),
#   SDL/SPL/SPcL (the same with PREP_SEQ_LEN=250: page crossings).
# Open (SGLang's bf16 GEMM backend initialized as its scheduler does: TGV for small M):
#   AT/DT (drive_bf16.py + drive.py), FT (drive_bf16.py + driveF.py, adapters_c2/tgv, ARMF_TGV=1 ARMF_STATIC=1),
#   MDT/MFT (adapters_c2/mixed_open.py), SDT (seq eager, PREP_TGV=1), SM (seq, PREP_M=1 PREP_CPP=1, adapters_c2, even only).
# KVA/KVD: A / D with no --max-total-tokens (KV capacity at --mem-fraction-static 0.8), one small case.
# NOLOCK=1: no per-arm flock (the caller holds gpu$GPU.lock for a group of arms: run_c2_group.sh).
# KV pool capped by --max-total-tokens (GPUs are shared); counts and bitwise do not depend on the pool size.
S=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/sglang
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
cd $S
GPU=${GPU:-1}
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu$GPU.lock
CPUS=$([ "$GPU" = 1 ] && echo 72-107 || echo 0-71)
LINE=${LINE:-even}; BACKEND=${BACKEND:-trtllm_mha}; L=$X/python_sgl_cand2_$LINE.sh
T=c2_${TAG:-$LINE}_${BACKEND%%_*}
O=$X/out_c2; G=$X/logs_c2; mkdir -p $O $G
DIS="--cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled"
STD="--batch-size 1 8 64 --prefill-lens 64 512 2048"
MIX="--enable-mixed-chunk --batch-size 1 8 32 64 128"
SEQ="--batch-size 1 8 64 --prefill-lens 64"
BF16=$S/armF/tgv/drive_bf16.py
for a in "$@"; do
  ENVX=(); PRE=(); X2="$DIS"; MAX=${MAXTOK:-32768}; OUT=(--out $O/${T}_$a.json --profile-dir $O/prof_${T}_$a)
  case $a in
    A)   DRV=bench/drive.py; X2=""; ARGS="$STD";;
    D)   DRV=bench/drive.py; ARGS="$STD";;
    F)   DRV=armF/driveF.py; ARGS="$STD";;
    Fc)  DRV=armF/driveF_cpp.py; ARGS="$STD";;
    AT)  PRE=($BF16); DRV=bench/drive.py; X2=""; ARGS="$STD";;
    DT)  PRE=($BF16); DRV=bench/drive.py; ARGS="$STD";;
    FT)  PRE=($BF16); DRV=armF/driveF.py; ARGS="$STD"; ENVX=(ARMF_ADAPTER_DIR=$X/adapters_c2/tgv ARMF_TGV=1 ARMF_STATIC=1);;
    MD)  DRV=armF/mixed/driveM.py; ENVX=(MIXM_F=0); ARGS="$MIX"; MAX=${MAXTOK:-262144};;
    MF)  DRV=armF/mixed/driveM.py; ENVX=(MIXM_F=1); ARGS="$MIX"; MAX=${MAXTOK:-262144};;
    MDT) DRV=$X/adapters_c2/mixed_open.py; ENVX=(MIXM_F=0); ARGS="$MIX"; MAX=${MAXTOK:-262144};;
    MFT) DRV=$X/adapters_c2/mixed_open.py; ENVX=(MIXM_F=1 ARMF_ADAPTER_DIR=$X/adapters_c2/tgv ARMF_TGV=1 ARMF_STATIC=1); ARGS="$MIX"; MAX=${MAXTOK:-262144};;
    KVA) DRV=bench/drive.py; X2=""; ARGS="--batch-size 1 --prefill-lens 64 --reps 2 --decode-steps 4 --profile-prefill --profile-decode"; MAX="";;
    KVD) DRV=bench/drive.py; ARGS="--batch-size 1 --prefill-lens 64 --reps 2 --decode-steps 4 --profile-prefill --profile-decode"; MAX="";;
    SD|SP|SPc|SDL|SPL|SPcL|SDT|SM)
      DRV=armF/prep/driveF_prep.py; ARGS="$SEQ"; OUT=(--out /dev/null)
      case $a in
        SD*) ENVX=(ARMF_OFF=1);;
        SPc*) ENVX=(ARMF_PREP=1 PREP_CPP=1);;
        SP*) ENVX=(ARMF_PREP=1);;
        SM) ENVX=(ARMF_PREP=1 PREP_CPP=1 PREP_M=1 ARMF_ADAPTER_DIR=$X/adapters_c2);;
      esac
      case $a in SDT) ENVX+=(PREP_TGV=1);; esac
      case $a in *L) ENVX+=(PREP_SEQ_LEN=250);; esac
      ENVX+=(PREP_SEQCHECK=$O/seq_${T}_$a.json);;
    *) echo "unknown arm $a"; continue;;
  esac
  CAP=""; [ -n "$MAX" ] && CAP="--max-total-tokens $MAX"
  BASE="--model-path /data/eellison/models/Qwen3-8B --mem-fraction-static 0.8 $CAP --attention-backend $BACKEND"
  echo "=== $a $(date +%T) free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader)"
  for try in 1 2 3; do
    FL=(flock -x $LOCK); [ -n "${NOLOCK:-}" ] && FL=()
    env "${ENVX[@]}" CUDA_VISIBLE_DEVICES=$GPU "${FL[@]}" bash -c 'echo "LOCKED $(date +%T) free=$(nvidia-smi -i '$GPU' --query-gpu=memory.free --format=csv,noheader)"; exec "$@"' _ \
      timeout ${TMO:-2700} taskset -c $CPUS bash $L "${PRE[@]}" $DRV $BASE $X2 $ARGS "${OUT[@]}" > $G/${T}_$a.log 2>&1
    rc=$?; echo "rc=$rc try=$try $(date +%T)"
    grep -q EADDRINUSE $G/${T}_$a.log || break
  done
done
