#!/usr/bin/env bash
# re-freeze A/B (coordinator 10-16; untimed for the slides): Vc+VP with / without ARMV_REFREEZE at 1K and 8K, V+VP with / without at
# 1K; bs 1 / 16 / 64 / 128, 16 warm + 128 measured steps per point, 3 rounds interleaved (set n27ab), the n27dg gates and config
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
R27_DG_CTX="1024 8192" R27_DG_CTX_V="1024" R27_DG_BS="1 16 64 128" ROUNDS=3 R27_KV_GIB=72 R27_TIMEOUT=3600 R27_ARMS="vcvp vcvpnr vsvp vsvpnr" \
  bash timing_gpuN4.sh round_27dg_line.sh n27ab 2 36-71 >> logs/timing_n27ab.out 2>&1 < /dev/null
echo "$(date +%T) n27ab done"
