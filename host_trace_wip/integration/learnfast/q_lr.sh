#!/usr/bin/env bash
cd $(dirname $0)
flock -w 3600 -x ../../scratch/eager/gpu3.lock timeout 900 env LF_GPU=3 CPU0=108 ./learn_range.sh out/seed8.pkl out/lr8_cbhrange.pkl 1446 160 151936 > logs/lr8.log 2>&1
./py.sh -c "
import pickle
a={k for k,_ in pickle.load(open('out/lr8_cbhrange.pkl','rb'))['entries']}
c={k for k,_ in pickle.load(open('../../scratch/integration/vllm/out/bindings_cbh.pkl','rb'))['entries']}
print('learned', len(a), 'cbh3', len(c), 'cbh3 subset', c <= a, 'extra', len(a - c))" >> logs/lr8.log 2>&1
./py.sh compare.py ../../scratch/integration/vllm/out/bindings_cbh.pkl out/lr8_cbhrange.pkl >> logs/lr8.log 2>&1
echo done >> logs/lr8.log
