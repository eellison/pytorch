# seedsub.py IN.pkl OUT.pkl M1,M2,...: a seed with only the keys at these token counts (a short traced run's harvest)
import pickle, sys
d = pickle.load(open(sys.argv[1], "rb")); Ms = {int(x) for x in sys.argv[3].split(",")}
e = [(k, b) for k, b in d["entries"] if k[3][0][0] in Ms]
pickle.dump({"meta": d["meta"], "entries": e}, open(sys.argv[2], "wb")); print(len(e))
