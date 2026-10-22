#!/usr/bin/env bash
# One gpu0.lock hold for a group of run_time_c2.sh jobs (a whole timing job: e.g. every steady arm), idle-checked per job.
# Usage: LINE=.. ROUND=.. run_time_group.sh <jobs...>
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
exec flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu0.lock env NOLOCK=1 IDLE_WAIT=60 bash $X/run_time_c2.sh "$@"
