#!/usr/bin/env bash
# rsum.sh TAG...: per-run summary (ready, ready-to-steady, warm-up phases, slow-path timers, GC, XVR)
cd /data/eellison/src/pytorch/agent_space/paramgraph/land/core/learnfast
for t in "$@"; do echo "== $t"; python3 breakdown27.py out/startup27_h3t_vt_rw_sec_r$t.json logs/rw_vt$t.log | grep -E "warm-up|READY|stream first|TTFT|steady" | cut -c1-150 | sed 's/  */ /g'; python3 -c "
import json
for l in open('out/startup27_h3t_vt_rw_sec_r$t.diag.jsonl'):
  d=json.loads(l)
  if d.get('ev')=='end': print('ACC', {k:v for k,v in d['acc'].items() if v}, 'gcu_n', d['cnt'].get('gc_untrack'))
  if d.get('ev')=='gc_time': print('GC', d)
  if d.get('ev')=='xvr': print('XVR', {k:v for k,v in d['stats'].items() if not k.startswith('tpl_')}, d['secs'])
d=json.load(open('out/startup27_h3t_vt_rw_sec_r$t.json'))['startup']; print('mem', [v['max_alloc_gib'] for v in d['range_warm']['phases'].values()], d['stream_max_alloc_gib'], d['stream_max_reserved_gib'], d['events_total'])
"; grep -h "ARMV_GC_FREEZE" logs/rw_vt$t.log | cut -c1-200; done
