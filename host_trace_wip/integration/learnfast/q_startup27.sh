#!/usr/bin/env bash
# 27B fair startup on GPU 0 (after the running MiMo cbl16): seed + learn (ours' fast-learner bindings), then runs default, fullnc,
# vc (ours as shipped), vcl (ours + Delta 1 + fla overlay)
cd $(dirname $0)
while ! grep -q "^done" logs/cbl16.log 2>/dev/null && pgrep -f "cbrc.sh cbl16" > /dev/null; do sleep 30; done
G=0; E="LF_GPU=$G LF_CPUS=0-35 LF_CPU0=0"
STEP_OFF=1 ./gpulock.sh $G 36000 env $E ./startup27.sh seed vc > logs/st27_seed.log 2>&1
STEP_OFF=1 ./gpulock.sh $G 36000 env $E ./startup27.sh learn vc > logs/st27_learn.log 2>&1
for arm in default fullnc vc vcl; do
  STEP_OFF=1 ./gpulock.sh $G 36000 env $E ./startup27.sh run $arm 1 > logs/st27_${arm}_r1.log 2>&1
done
echo done >> logs/st27_vcl_r1.log
