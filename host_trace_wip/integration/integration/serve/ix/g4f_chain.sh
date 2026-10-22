#!/usr/bin/env bash
# GPU 0: after the c5g cache warm-up, the full-step set g4f (vllm/round_g4f_line.sh, 6 arms, DRIVE_STEP_WITH_SAMPLE=1) on candidate 5
# gated (CAND_LINE=c5g), refilled until 3 runs per arm; contended runs are skipped by trun's idle check.
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration
while ! grep -q "done on" serve/logs/g4fwarm.outer 2>/dev/null; do sleep 30; done
have() { ls vllm/out/t_g4f_{default,fullnc,vreg,vc,vcs,vrs}_r{1,2,3}.json 2>/dev/null | wc -l; }
while [ $(have) -lt 18 ]; do
  echo "$(date +%T) g4f have $(have)/18"
  STOCK_LINE=c5g ROUNDS=3 SET=g4f ROUND_SCRIPT=round_g4f_line GATE_MAX=1200 bash vllm/rounds_gpu0.sh >> vllm/logs/rounds_g4f.out 2>&1
  [ $(have) -lt 18 ] && sleep 120
done
echo "$(date +%T) G4F_DONE"
