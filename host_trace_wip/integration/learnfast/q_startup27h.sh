#!/usr/bin/env bash
# 27B fair startup on the head (step_ms8 / build_cpp_ms8), GPU 0, after the c5h chain: seed + learn, then default, fullnc, vc, vcl
cd $(dirname $0)
while pgrep -f "^bash ./q_startup27.sh" > /dev/null; do sleep 30; done
G=0; E="HEAD=1 LF_GPU=$G LF_CPUS=0-35 LF_CPU0=0"
STEP_OFF=1 ./gpulock.sh $G 36000 env $E ./startup27.sh seed vc > logs/st27h_seed.log 2>&1
STEP_OFF=1 ./gpulock.sh $G 36000 env $E ./startup27.sh learn vc > logs/st27h_learn.log 2>&1
for arm in vcl vc default fullnc; do
  STEP_OFF=1 ./gpulock.sh $G 36000 env $E ./startup27.sh run $arm 1 > logs/st27h_${arm}_r1.log 2>&1
done
echo done >> logs/st27h_fullnc_r1.log
