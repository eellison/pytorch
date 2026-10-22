#!/usr/bin/env bash
# 27B Delta 1 + fla len() overlay (DIAG27_FLA_SYMLEN): Triton GDN --check parts a/b, then n27wf (FlashInfer GDN) trace count
cd $(dirname $0)
while ! grep -q "^done" logs/b27tb.log 2>/dev/null; do sleep 30; done
for spec in "l27sa a chk triton" "l27sb b chk triton" "l27sw all run fi"; do
  set -- $spec
  g=; [ "$4" = triton ] && g=triton
  DIAG27_FLA_SYMLEN=1 TAG=$1 PART=$2 MODE=$3 GDN=$g G=3 CPUS=108-143 ./q_lift27.sh
done
