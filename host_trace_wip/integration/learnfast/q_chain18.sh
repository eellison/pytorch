#!/usr/bin/env bash
# q_chain18.sh: keydiff (GDN template keys per layer, ms16, quick) -> q_ms19 -> q_prof19
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
./q_keydiff.sh > logs/q_keydiff.out 2>&1
touch logs/q_tpldiag.done
./q_ms19.sh > logs/q_ms19.out 2>&1
./q_prof19.sh > logs/q_prof19.out 2>&1
