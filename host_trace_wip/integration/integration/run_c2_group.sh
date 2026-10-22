#!/usr/bin/env bash
# One gpu$GPU.lock hold for a group of run_c2.sh arms (each still bounded by its own timeout), so a group of short
# correctness runs does not re-enter the lock queue per arm. Usage: LINE=.. [BACKEND=..] run_c2_group.sh <arms...>
X=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
GPU=${GPU:-1}
exec flock -x /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu$GPU.lock env NOLOCK=1 bash $X/run_c2.sh "$@"
