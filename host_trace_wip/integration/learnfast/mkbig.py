import pickle, sys
d = pickle.load(open('out/keys8_16k.pkl', 'rb'))
Ms = [int(x) for x in sys.argv[2].split(',')]
sel = [k for k, _ in d['entries'] if k[3][0][0] in Ms and k[3][1][1] != 151936]
pickle.dump({'meta': d['meta'], 'entries': [(k, None) for k in sel]}, open(sys.argv[1], 'wb')); print(len(sel))
