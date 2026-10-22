#!/usr/bin/env bash
# chain_g1x.sh: overnight extra w rounds on gb300-box GPU 1 (coordinator 10-15) after chain_g1 (PID $1): g1w rounds 4, 5, ... until
# every arm has >= 2 PSI-clean windows in every valid row (scripts/fn_clean.py --check 2) or 07:00. No round starts after 06:20, and
# whatever runs at 07:00 is stopped (timeout on the round's process group) so GPU 1 is free in the morning.
set -u
D=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
echo $$ > $D/logs/mon/chain_g1x.pid
while [ -n "$(ps -o args= -p $1 2>/dev/null | grep chain_g1.sh)" ]; do sleep 60; done
export HT_GPU=1 HT_CPUS=72-107 FN_FI_TAG=f1 FN_LOCK=/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/eager/gpu1.lock FN_YIELD_PATTERN=
END=$(date -d "07:00" +%s); [ $END -le $(date +%s) ] && END=$(date -d "tomorrow 07:00" +%s)
for r in $(seq 4 20); do
  python3 $D/scripts/fn_clean.py --check 2 g1w && { echo "$(date +%T) chain_g1x: >= 2 clean windows everywhere"; break; }
  now=$(date +%s); [ $((END - now)) -lt 2400 ] && { echo "$(date +%T) chain_g1x: too close to 07:00 for a round"; break; }
  echo "$(date +%T) chain_g1x: g1w R=$r"
  env SET=g1w KIND=w R=$r ARMS="fullnc breakable v" timeout -s TERM -k 60 $((END - now)) bash $D/round_fn_line3.sh
  echo "$(date +%T) chain_g1x: round $r rc $?"
done
echo "$(date +%T) chain_g1x done"
