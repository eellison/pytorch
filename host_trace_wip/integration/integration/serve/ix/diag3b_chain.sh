#!/usr/bin/env bash
# gb300-box GPU 0 (10-14): diag3 vcsl rerun (r1 failed: set_log.so shadowed set_log.py) after the old chain (PID $1) exits
# (its bswalk8k vrs run was stopped by me to put this first), then bswalk8k vrs again. GPU 0 goes to the 27B lane after this chain.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 2; done
ARM=vcsl GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_diag3.sh > serve/logs/diag3_vcsl_r2.log 2>&1
echo "$(date +%T) DIAG3_VCSL_R2_DONE" >> serve/logs/swp_timeline.log
TAG=_vrs ARM=V GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_bswalk8k.sh > serve/logs/bswalk8k_vrs.log 2>&1
echo "$(date +%T) BSWALK8K_VRS_DONE" >> serve/logs/swp_timeline.log
