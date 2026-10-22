#!/usr/bin/env bash
# GPU 0 / CPUs 0-35 after the current queue (incl. the flagged bnc r1 rerun): (a) the q8vcs and bs-walk checks again (sections policy:
# a run that used any section entry is invalid; the summaries record e.sections), then (c) swph1 vrs r1/r2 again (the earlier ones,
# made before the section policy, kept as t_swph1sec_vrs_r*).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
T=serve/logs/swp_timeline.log
while ! grep -q FLAGGED_RERUNS_DONE $T; do sleep 60; done
TAG=_c5h_sec0 VC_LINE=c5h GPU_IDX=0 GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_q8vcs_check.sh > serve/logs/q8vcs_c5h_sec0.log 2>&1
TAG=_c5h_bswalk_sec0 VC_LINE=c5h GPU_IDX=0 GPU=0 CPUS=0-35 bash vllm/gpuN_job.sh serve/repro/run_q8vcs_bswalk.sh > serve/logs/q8vcs_c5h_bswalk_sec0.log 2>&1
echo "$(date +%T) CHECKS_SEC0_DONE" >> $T
for r in 1 2; do
  [ -f vllm/out/t_swph1_vrs_r$r.json ] && mv vllm/out/t_swph1_vrs_r$r.json vllm/out/t_swph1sec_vrs_r$r.json && mv vllm/logs/t_swph1_vrs_r$r.log vllm/logs/t_swph1sec_vrs_r$r.log
done
SET=swph1 ROUND=vllm/round_sweep2_line.sh GPU=0 CPUS=0-35 LINE=c5h ARMS="vrs" bash serve/ix/set_chain.sh
echo "$(date +%T) AFTER_REST_DONE" >> $T
