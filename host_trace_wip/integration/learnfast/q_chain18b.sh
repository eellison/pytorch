#!/usr/bin/env bash
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
./q_ms19.sh > logs/q_ms19.out 2>&1
touch logs/q_ms19.done
./q_prof19.sh > logs/q_prof19.out 2>&1
