#!/usr/bin/env bash
# dev-box GPU 3 window (inside dev3_run.sh, which holds <path> for the chain; between MiMo's w and cb
# sets, 10-14): the n27cbt arms whose process also ran KID profiling (moved to out/timing/invalid_kid/), rerun without KID; then, if
# the breakable+gdnbreak verification passed (out/kid/GDNBREAK_OK pushed from gb300-box), n27wtg and n27cbtg (3 rounds each).
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
mkdir -p out/timing/invalid_kid logs/timing
for t in breakable_r2 breakable_r3 default_r2 default_r3 fullnc_r2 vcvp_r2 vcvp_r3 vsvp_r2 vsvp_r3; do
  [ -f out/timing/t_n27cbt_$t.json ] && mv out/timing/t_n27cbt_$t.json out/timing/invalid_kid/
done
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp breakable vsvp" bash timing_gpuN3.sh round_27cbt_line.sh n27cbt 3 108-143 >> logs/timing_n27cbt_rerun.out 2>&1
if [ -f out/kid/GDNBREAK_OK ]; then
  R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="breakablegb" bash timing_gpuN3.sh round_27wtg_line.sh n27wtg 3 108-143 >> logs/timing_n27wtg.out 2>&1
  R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="breakablegb" bash timing_gpuN3.sh round_27cbtg_line.sh n27cbtg 3 108-143 >> logs/timing_n27cbtg.out 2>&1
else
  echo "$(date +%T) no out/kid/GDNBREAK_OK: gdnbreak sets skipped" >> logs/timing_n27cbt_rerun.out
fi
echo "dev3_window done $(date +%T)" >> logs/timing_n27cbt_rerun.out
