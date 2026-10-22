#!/usr/bin/env bash
# GPU 2: after run_synccheck.sh, the call-site rerun of the churn sync check (default, vcvp) then the per-shape sync check
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/r27
while pgrep -f "^bash run_synccheck.sh" > /dev/null; do sleep 20; done
mkdir -p out/synccheck_sites
sed 's#out/synccheck/#out/synccheck_sites/#g; s#logs/synccheck_#logs/synccheck_sites_#g' run_synccheck.sh > run_synccheck_sites.sh
ARMS="default vcvp" bash run_synccheck_sites.sh
python3 -c "
import json
for a in ('default','vcvp'):
    d=json.load(open('out/synccheck_sites/'+a+'.json'))['cb']['64']; print(a, d.get('sync_per_step')); [print('   ',n,k) for k,n in sorted(d.get('sync_sites',{}).items(), key=lambda x:-x[1])[:8]]"
bash run_syncstep.sh
