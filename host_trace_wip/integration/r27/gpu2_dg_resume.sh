#!/usr/bin/env bash
# GPU 2 after the 03:35 reboot (coordinator 10-16): a post-reboot sanity smoke (vcvp, R27_DG_SMOKE, set n27dgs2), then n27dg round 3
# (rounds 1-2 on disk are skipped) and the flagged-point reruns; own-core / PSI gates before the lock and the own-core gate in the hold
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
R27_DG_SMOKE=1 PSI_GATE=100 ROUNDS=1 R27_KV_GIB=72 R27_TIMEOUT=1800 R27_ARMS="vcvp" bash timing_gpuN4.sh round_27dg_line.sh n27dgs2 2 36-71 >> logs/timing_n27dgs2.out 2>&1 < /dev/null
if ! python3 -c "import json,sys; j=json.load(open('out/timing/t_n27dgs2_vcvp_r1.json')); sys.exit(0 if j['dcurve'] and not j.get('window_violations') else 1)"; then
  echo "$(date +%T) post-reboot smoke FAILED: not resuming"; exit 1
fi
echo "$(date +%T) smoke ok; n27dg round 3"
bash gpu2_dg_chain2.sh < /dev/null
bash gpu2_dg_flag.sh < /dev/null
echo "$(date +%T) all done"
