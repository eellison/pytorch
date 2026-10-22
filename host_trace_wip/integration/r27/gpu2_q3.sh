#!/usr/bin/env bash
# GPU 2 after gpu2_sync_q.sh: verify the unsynced-window fix (churn sync check, default + vcvp: no harness sync, device syncs ~0),
# then rerun the churn sets as n27cbn (default, fullnc, vcvp, vsvp x 3; plain breakable is INVALID in churn) and n27cbng (breakable+gdnbreak x 3)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "^bash gpu2_sync_q.sh" > /dev/null; do sleep 30; done
mkdir -p out/synccheck_fixed
sed 's#out/synccheck/#out/synccheck_fixed/#g; s#logs/synccheck_#logs/synccheck_fixed_#g' run_synccheck.sh > run_synccheck_fixed.sh
ARMS="default vcvp" bash run_synccheck_fixed.sh
ok=$(python3 -c "
import json
ok = True
for a in ('default', 'vcvp'):
    d = json.load(open('out/synccheck_fixed/' + a + '.json'))['cb']['64']
    print(a, d.get('sync_per_step'), 'harness', d.get('harness_syncs_in_window'), flush=True)
    ok &= d.get("harness_syncs_in_window") == 0 and d.get('sync_per_step', {}).get('cudaDeviceSynchronize', 0) < 0.1
print('OK' if ok else 'FAIL')" | tee -a logs/gpu2_q3.out | tail -1)
if [ "$ok" != OK ]; then echo "$(date +%T) fix not verified: no rerun"; exit 1; fi
echo "$(date +%T) fix verified; n27cbn"
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="default fullnc vcvp vsvp" bash timing_gpuN3.sh round_27cbn_line.sh n27cbn 2 36-71 >> logs/timing_n27cbn.out 2>&1
echo "$(date +%T) n27cbng"
R27_KV_GIB=64 R27_TIMEOUT=7200 R27_ARMS="breakablegb" bash timing_gpuN3.sh round_27cbng_line.sh n27cbng 2 36-71 >> logs/timing_n27cbng.out 2>&1
echo "$(date +%T) done"
