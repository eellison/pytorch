#!/usr/bin/env bash
# One attn_open timing job on GPU 0 under gpu0.lock: bash attn_open/gpu0.sh TAG [ENV=..] -- script args. CPUs 0-71.
# After the lock: idle_check (no foreign compute process on GPU 0); a contended run is skipped and logged.
G=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock
TAG=$1; shift; envs=()
while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
cd $G
echo "$(date +%T) queued $TAG" >> attn_open/logs/jobs.log
until [ $(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | tr -d , | awk '{print $1 - $2}') -ge 150000 ]; do sleep 30; done
flock -x $LOCK bash -c '
  source vllm/timing_lib.sh >/dev/null; v=$(IDLE_MAX=300 idle_check | tail -1)
  echo "$(date +%T) start '$TAG' $v load $(cut -d" " -f1 /proc/loadavg)" >> attn_open/logs/jobs.log
  case "$v" in "idle_check ok"*) ;; *) echo "$(date +%T) SKIPPED_CONTENDED '$TAG'" >> attn_open/logs/jobs.log; exit 3 ;; esac
  exec env "$@"' _ "${envs[@]}" CUDA_VISIBLE_DEVICES=0 CAND_LINE=${CAND_LINE:-pinned} FLASHINFER_NO_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  INTEG_ARMV=$G/attn_open/armV_open timeout ${JOB_TIMEOUT:-2400} taskset -c 0-71 bash python_vllm_cand2.sh "$@" > attn_open/logs/$TAG.log 2>&1
echo "$(date +%T) end $TAG rc=$?" >> attn_open/logs/jobs.log
