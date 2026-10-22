#!/usr/bin/env bash
# After run_s5.sh: arm V with ARMV_DECODE_MAX_SEQ=actual (vLLM's per-step max_seq_len), decode + one prefill, for the trace count and step cost.
V=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/vllm
cd $V
until grep -q ALLDONE bench/logs/run_s5.log; do sleep 20; done
ARMV_DECODE_MAX_SEQ=actual TAG=actual DRIVE_ARGS="--prefill-lens 64 --mixed 0 --dump-decode bench/out/dump_Vactual.pt" bash bench/run_arms.sh V
echo ALLDONE
