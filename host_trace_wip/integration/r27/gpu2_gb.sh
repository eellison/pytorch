#!/usr/bin/env bash
# GPU 2 after the n27cbf reruns: timed breakable+gdnbreak sets n27wfg (breakable's PIECEWISE sizes) and n27cbfg (churn), 3 rounds each,
# only once the gdnbreak verification passed: kidw fi breakablegb identical to V+VP at every size and the Triton logit probe bitwise
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "^bash gpu2_cbf_rerun.sh" > /dev/null || pgrep -f "^bash gpu1_q3.sh" > /dev/null; do sleep 60; done
ok=1
grep -q "breakablegb vs vsvp: .*differ with vLLM padding 0, differ otherwise 0" out/kid/KIDW_fi.txt || ok=0
[ -s out/brk3/CMP_tr_gb.txt ] && ! grep -q DIFFER out/brk3/CMP_tr_gb.txt || ok=0
if [ $ok = 0 ]; then echo "$(date +%T) gdnbreak verification not passed: not timing it"; exit 1; fi
echo "$(date +%T) gdnbreak verified; n27wfg"
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="breakablegb" bash timing_gpuN3.sh round_27wfg_line.sh n27wfg 2 36-71 >> logs/timing_n27wfg.out 2>&1
echo "$(date +%T) n27cbfg"
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="breakablegb" bash timing_gpuN3.sh round_27cbfg_line.sh n27cbfg 2 36-71 >> logs/timing_n27cbfg.out 2>&1
echo "$(date +%T) done"
