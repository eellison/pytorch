#!/usr/bin/env bash
# 27B --check, Triton GDN (the lanes' correctness config), n27wf shapes split in 2 parts; LIFT arm, then baseline
cd $(dirname $0)
for spec in "l27ta a 0" "l27tb b 0" "b27ta a 1" "b27tb b 1"; do
  set -- $spec
  TAG=$1 PART=$2 BASE=$3 GDN=triton MODE=chk G=3 CPUS=108-143 ./q_lift27.sh
done
