#!/usr/bin/env bash
# Ends startup_c2.sh after its round 1 (a whole 3-round job holds GPU 0 ~4 h at today's import/model-load times).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm
until grep -q "start s_default_r2" logs/startup_c2.log; do sleep 5; done
kill $(pgrep -f "vllm/startup_c2.sh") $(pgrep -f "out/s_default_r2.json") 2>/dev/null
echo "$(date +%T) stopped startup_c2 after round 1" >> logs/startup_c2.log
