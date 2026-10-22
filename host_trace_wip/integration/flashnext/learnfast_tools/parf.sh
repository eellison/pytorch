#!/usr/bin/env bash
# parf.sh TAG N KEYS [fastlearn flags]: N fastlearn processes on contiguous shards (CPUs split over ${CPU0:-108}..+35), merge; wall printed.
cd $(dirname $0)
TAG=$1 N=$2 K=$3; shift 3
t0=$(date +%s.%N)
for i in $(seq 0 $((N-1))); do
  c0=$((${CPU0:-108} + (i * 36 / N))); c1=$((${CPU0:-108} + ((i+1) * 36 / N) - 1)); [ $c1 -lt $c0 ] && c1=$c0
  OMP_NUM_THREADS=1 LF_CPUS=$c0-$c1 ${PY:-./py.sh} ${SCRIPT:-fastlearn.py} --keys $K --out out/${TAG}_$i.pkl --json out/${TAG}_$i.json --shard $i/$N "$@" > logs/${TAG}_$i.log 2>&1 &
done
wait
t1=$(date +%s.%N)
if [ $N -gt 1 ]; then ${PY:-./py.sh} merge.py out/${TAG}.pkl $(for i in $(seq 0 $((N-1))); do echo out/${TAG}_$i.pkl; done) > /dev/null; else cp out/${TAG}_0.pkl out/${TAG}.pkl; fi
t2=$(date +%s.%N)
python3 -c "
import json
d=[json.load(open('out/${TAG}_%d.json'%i)) for i in range($N)]
print('$TAG N=$N wall %.1f s (procs %.1f, merge %.1f) learn_wall %s learn_cpu %s total %s full %s fail %d load %s' % ($t2-$t0, $t1-$t0, $t2-$t1, [x['learn_wall_s'] for x in d], [x['learn_cpu_s'] for x in d], [x['total_s'] for x in d], [x['full_templates'] for x in d], sum(x.get('failed', x.get('refused', 0)) for x in d), open('/proc/loadavg').read().split()[:3]))"
