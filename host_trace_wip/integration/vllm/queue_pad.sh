#!/usr/bin/env bash
# Starts the GPU 0 padding A/B once the padded --check run on GPU 1 is bitwise on all real rows.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
until [ -f vllm/out/c2_cpp_closed_pad_Vcheck.json ]; do sleep 60; done
ok=$(python3 -c "
import json; c=json.load(open('vllm/out/c2_cpp_closed_pad_Vcheck.json'))['V']['checks']; print(int(len(c) > 0 and all(x['bitwise'] for x in c)))")
echo "$(date +%T) pad check ok=$ok"
[ "$ok" = 1 ] && exec bash vllm/gpu0_job.sh vllm/timing_pad_c2.sh
