#!/usr/bin/env bash
# gb300-box GPU 0 (replaces bswalk8k_gpu0.sh after its vcsl item, 10-14): wait for the running bswalk8k vcsl job (gpuN_job PID $1),
# then main's untimed diagnostics (diag3 vcsl, diag3 default), the held-memset spot check if land/core/memset/check_held_memsets.py
# exists by then (serve/ix/memset_spot.sh), then bswalk8k vrs. GPU 0 goes to the 27B lane after this chain.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while kill -0 $1 2>/dev/null; do sleep 15; done
echo "$(date +%T) bswalk8k vcsl job $1 ended" >> serve/logs/swp_timeline.log
ARM=vcsl GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_diag3.sh > serve/logs/diag3_vcsl.log 2>&1
echo "$(date +%T) DIAG3_VCSL_DONE" >> serve/logs/swp_timeline.log
ARM=default GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_diag3.sh > serve/logs/diag3_default.log 2>&1
echo "$(date +%T) DIAG3_DEFAULT_DONE" >> serve/logs/swp_timeline.log
if [ -f serve/ix/memset_spot.sh ]; then
  GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/ix/memset_spot.sh > serve/logs/memset_spot.log 2>&1
  echo "$(date +%T) MEMSET_SPOT_DONE" >> serve/logs/swp_timeline.log
fi
TAG=_vrs ARM=V GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_bswalk8k.sh > serve/logs/bswalk8k_vrs.log 2>&1
echo "$(date +%T) BSWALK8K_DONE" >> serve/logs/swp_timeline.log
