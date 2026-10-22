#!/usr/bin/env bash
# Like gpu1_cmds.sh, on the first of GPUs 1/2/3 whose lock is free (correctness only; GPUs 2/3 only when
# land/scratch/eager/gpu_idle_ok.sh N passes, before taking it and before each job; user rule 10-12): bash gpu_any_cmds.sh TAG "cmd" ...
# The job sees CUDA_VISIBLE_DEVICES=N and GPU_IDX=N (nvidia-smi -i) and runs on that GPU's CPUs (1: 72-107, 2: 108-125,
# 3: 126-143). One hold per call: keep each call to ~20 min (one job per call; queue several calls instead of a chain).
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
E=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager
TAG=$1; shift
cd $G
export CMD_TIMEOUT=${CMD_TIMEOUT:-1800} CAND_LINE=cpp FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date +%T) waiting for any of gpu1/2/3"
while :; do
  for g in ${GPUS:-1 2 3}; do
    [ $g != 1 ] && ! bash $E/gpu_idle_ok.sh $g && continue
    case $g in 1) cpus=72-107 ;; 2) cpus=108-125 ;; 3) cpus=126-143 ;; esac
    exec 9>$E/gpu$g.lock
    if flock -n -x 9; then
      echo "$(date +%T) holding gpu$g.lock"
      for c in "$@"; do
        [ $g != 1 ] && ! bash $E/gpu_idle_ok.sh $g && { echo "gpu$g taken by another user: stop"; break; }
        echo "== $(date +%T) gpu$g $c"; CUDA_VISIBLE_DEVICES=$g GPU_IDX=$g timeout $CMD_TIMEOUT taskset -c $cpus bash $c; echo "rc=$?"
      done > serve/logs/$TAG.log 2>&1
      echo "$(date +%T) done on gpu$g"
      exit 0
    fi
    exec 9>&-
  done
  sleep 20
done
