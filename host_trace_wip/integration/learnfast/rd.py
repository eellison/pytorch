import json,sys,collections
c=collections.Counter()
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d['ev']=='trace': print('TRACE', d['s'], d['toks'][-1])
    if d['ev']=='redispatch_refused': print('RR', d['toks'][-1], d['op'].split('.')[1], d['why'][:300], (d['detail'] or {}).get('redo_eager_reasons'))
    if d['ev']=='miss': print('MISS', d['toks'][-1], d['traces_added'], 'redispatch', d['cnt'].get('redispatch'), 'acc', d['acc'])
    if d['ev']=='redo_body_error': print('ERR', d['err'][:200])
    if d['ev']=='end': print('END', d)
