#!/usr/bin/env bash
cd $(dirname $0)
while pgrep -f "^bash ./q_startup27h.sh" > /dev/null; do sleep 30; done
G=0; E="HEAD=1 LF_GPU=$G LF_CPUS=0-35 LF_CPU0=0"
STEP_OFF=1 ./gpulock.sh $G 36000 env $E ./startup27.sh run vclt 1 > logs/st27h_vclt_r1.log 2>&1
echo done >> logs/st27h_vclt_r1.log
