#!/usr/bin/env bash
# par.sh TAG N SHARD_BY KEYS: N harness processes on shards of KEYS (CPUs split over LF_CPUS 108-143), then merge; wall printed.
cd $(dirname $0)
TAG=$1 N=$2 BY=$3 K=$4
t0=$(date +%s.%N)
lo=108; per=$(( 36 / N )); [ $per -lt 1 ] && per=1
for i in $(seq 0 $((N-1))); do
  c0=$((lo + (i * 36 / N))); c1=$((lo + ((i+1) * 36 / N) - 1)); [ $c1 -lt $c0 ] && c1=$c0
  OMP_NUM_THREADS=1 LF_CPUS=$c0-$c1 ./py.sh harness.py --keys $K --out out/${TAG}_$i.pkl --json out/${TAG}_$i.json --shard $i/$N --shard-by $BY > logs/${TAG}_$i.log 2>&1 &
done
wait
t1=$(date +%s.%N)
./py.sh merge.py out/${TAG}.pkl $(for i in $(seq 0 $((N-1))); do echo out/${TAG}_$i.pkl; done)
t2=$(date +%s.%N)
python3 -c "
import json
d=[json.load(open('out/${TAG}_%d.json'%i)) for i in range($N)]
print('$TAG N=$N by=$BY wall_s %.1f (procs %.1f, merge %.1f) per-proc learn_s %s total_s %s full %s load %s' % ($t2-$t0, $t1-$t0, $t2-$t1, [round(x['learn_wall_s'],1) for x in d], [round(x['total_s'],1) for x in d], [x['full_templates'] for x in d], open('/proc/loadavg').read().split()[:3]))"
