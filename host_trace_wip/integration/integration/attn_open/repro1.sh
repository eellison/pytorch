#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
JOB_TIMEOUT=900 bash attn_open/gpu1.sh repro_fa4_1 -- attn_open/repro_fa4.py
