# Offline (10-14, main): K-slot LRU exec cache per entry kind (decode / mixed / prefill), keyed by the step's size (reqs, tokens),
# simulated over cbh3's per-step logs of ours (vcsl r1-r3). A step "misses" today (K=1) when its key differs from the previous step of
# the same entry; with K slots it hits when its key is among the last K distinct keys. Saved per converted miss: the phase-clock full
# re-patch increment (patch 650 + upload 207 + evaluate 272 us, swph1 alternating vs fixed) and, for decode, the measured entry us of
# that step minus the decode steady median.
import collections, json, statistics as st, sys

FULL = 650 + 207 + 272


def kind(r):
    return "decode" if r[0] == r[1] else ("prefill" if r[0] == 1 else "mixed")


def sim(L, K):
    caches = collections.defaultdict(list)
    hits = collections.Counter()
    n = collections.Counter()
    for r in L:
        k, key = kind(r), (r[0], r[1])
        c = caches[k]
        n[k] += 1
        if key in c:
            hits[k] += 1
            c.remove(key)
        c.insert(0, key)
        del c[K:]
    return n, hits


out = {}
for C in ("16", "64", "128"):
    rows = []
    for r_ in (1, 2, 3):
        R = json.load(open(f"vllm/out/t_cbh3_vcsl_r{r_}.json"))
        L = R["cb"][C]["steps_log"]
        dec = [x for x in L if x[0] == x[1]]
        steady = st.median(x[8] for x in dec if x[5] <= 50)
        res = {}
        for K in (1, 2, 4):
            n, h = sim(L, K)
            res[K] = (n, h)
        # measured saving for decode steps that miss at K=1 but hit at K=2 / 4
        meas = {}
        for K in (2, 4):
            caches1, cachesK = [], []
            s_ = 0.0
            for x in dec:
                key = x[0]
                m1 = key not in caches1
                mK = key not in cachesK
                if m1 and not mK:
                    s_ += max(0.0, x[8] - steady)
                for c, KK in ((caches1, 1), (cachesK, K)):
                    if key in c:
                        c.remove(key)
                    c.insert(0, key)
                    del c[KK:]
            meas[K] = s_
        rows.append((len(L), res, meas, steady, st.fmean(x[2] for x in L)))
    out[C] = rows
print("| C | entry | steps | hit rate K=1 (today) | K=2 | K=4 | saved us/step K=2 (fixed-cost model) | K=4 | decode saved us/step K=2 (measured entry) | K=4 | mean step ms (synced) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for C, rows in out.items():
    for k in ("decode", "mixed"):
        N = sum(r[1][1][0][k] for r in rows)
        h = {K: sum(r[1][K][1][k] for r in rows) for K in (1, 2, 4)}
        steps = sum(r[0] for r in rows)
        sav = {K: (h[K] - h[1]) * FULL / steps for K in (2, 4)}
        ms = {K: sum(r[2][K] for r in rows) / steps for K in (2, 4)} if k == "decode" else {2: None, 4: None}
        f = lambda x: f"{x:.0f}" if x is not None else "-"
        print(f"| {C} | {k} | {N} | {h[1] / N:.3f} | {h[2] / N:.3f} | {h[4] / N:.3f} | {sav[2]:.0f} | {sav[4]:.0f} | {f(ms[2])} | {f(ms[4])} | {st.fmean(r[4] for r in rows):.2f} |")
