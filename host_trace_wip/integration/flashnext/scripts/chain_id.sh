#!/usr/bin/env bash
# chain_id.sh: functional (untimed) dry run of round_fn_line.sh KIND=id on gb300-box GPU 3, one gpu3.lock hold per arm:
# validates the unified FlashInfer autotune file (fullnc primes it, the others must hit it), the drive, the profiles, and
# gives the kernel-identity comparison before the dev-box timing.
F=/data/eellison/src/pytorch/agent_space/paramgraph/land/core/flashnext
cd $F
for a in fullnc breakable v; do
  FN_SKIP_IDLE=1 OWN_BUSY_FRAC=1.0 SET=fnid KIND=id R=1 ARMS=$a HT_GPU=3 HT_CPUS=108-143 FN_TIMEOUT=1150 CMD_TIMEOUT=1200 \
    bash gpu3.sh fnid_$a bash round_fn_line.sh
  sleep 60
done
python3 scripts/kernel_identity.py out/timing/prof_fnid_fullnc out/timing/prof_fnid_breakable out/timing/prof_fnid_v
