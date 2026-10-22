# Condenses a missreason log: one line per miss (key, family, events with learn refusals counted by op).
import collections, sys
cur = None; rows = []
for l in open(sys.argv[1]):
    if l.startswith("MISS"):
        cur = [l.split(" cands")[0].replace("MISS ", ""), l.split("before ")[1].strip(), collections.Counter()]; rows.append(cur)
    elif l.startswith("    (") and cur:
        t = eval(l.strip())
        if t[0] == "_learn":
            cur[2][("learn", t[1], t[5]["op"].split(".")[-2] if t[4] is None else "ok")] += 1
        else:
            cur[2][(t[0], t[1], t[2], t[4])] += 1
    elif l.startswith(("secs", "decode {", "prefill {", "mixed {")):
        print(l.strip()[:220])
for m, b, c in rows:
    print(m, "before", b, "|", "; ".join(f"{k} x{v}" for k, v in c.items() if not (k[0] == "learn" and k[2] == "ok")), "| learn ok", sum(v for k, v in c.items() if k[0] == "learn" and k[2] == "ok"))
