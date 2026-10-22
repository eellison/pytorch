#!/usr/bin/env bash
# 10-14 op-dependency classification (opdeps.py via DIAG27_OPDEPS, no redo_context, adapters as shipped), lowest GPU 3
# priority: od9 (Qwen3.5-9B) -> odm (MiMo, bs 1 2) -> od27 (27B) -> od8 (Qwen3-8B)
cd $(dirname $0)
B="DIAG27_OPDEPS=1 LF_GPU=3 LF_CPUS=108-143"
./gpulock.sh 3 36000 env ARMV_SECTIONS=1 MODEL=/data/eellison/models/Qwen3.5-9B MODEL_EXTRA=--trust-remote-code KV_GIB=4 INTEG_GPU_UTIL=0.08 $B LEARN_TIMEOUT=900 ./run27m.sh od9 60 62 > logs/od9.log 2>&1
echo done >> logs/od9.log
./gpulock.sh 3 36000 env $B BS="1 2" RUN_TIMEOUT=1000 ./mimorc.sh odm dc > logs/odm.log 2>&1
echo done >> logs/odm.log
./gpulock.sh 3 36000 env ARMV_SECTIONS=1 $B LEARN_TIMEOUT=900 ./run27.sh od27 60 62 > logs/od27.log 2>&1
echo done >> logs/od27.log
./gpulock.sh 3 36000 env MODEL=/data/eellison/models/Qwen3-8B KV_GIB=4 INTEG_GPU_UTIL=0.08 $B LEARN_TIMEOUT=900 ./run27m.sh od8 60 62 > logs/od8.log 2>&1
echo done >> logs/od8.log
