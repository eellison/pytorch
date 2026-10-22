# python3 bench/servetable_sr.py [RUN ...]: step 5 serving table. RUN = TAG_ARM (default sr1_eagerm sr1_default sr1_V
# sr2_V sr3_V), files bench/out/serve/RUN_{c1,c8,c64}.json. =ref: requests whose generated text equals sr1_eagerm's
# (eager plus V's max_seq_len policy, V's bitwise reference); =V1: equals sr1_V's (a V-vs-V batching floor).
import json, os, sys

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "serve")
RUNS = sys.argv[1:] or ["sr1_eagerm", "sr1_default", "sr1_V", "sr2_V", "sr3_V"]


def load(run, w):
    f = os.path.join(D, f"{run}_{w}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def same(a, b):
    return "-" if a is None or b is None else f"{sum(x == y for x, y in zip(a['generated_texts'], b['generated_texts']))}/{len(a['generated_texts'])}"


print("work run          ttft med/p99 ms | itl med/p99 ms | tpot med | out tok/s | dur s | =ref | =V1")
for w in ("c1", "c8", "c64"):
    ref, v1 = load("sr1_eagerm", w), load("sr1_V", w)
    for run in RUNS:
        r = load(run, w)
        if r is None:
            continue
        print(f"{w:4} {run:12} {r['median_ttft_ms']:8.1f} {r['p99_ttft_ms']:8.1f} | {r['median_itl_ms']:6.2f} {r['p99_itl_ms']:7.2f} | "
              f"{r['median_tpot_ms']:6.2f} | {r['output_throughput']:8.0f} | {r['duration']:6.1f} | {same(r, ref)} | {same(r, v1)}")
for run in RUNS:
    for s in ("warm", "served"):
        f = os.path.join(D, f"{run}_armv_{s}.json")
        if os.path.exists(f):
            d = json.load(open(f))
            print(run, s, {k: d.get(k) for k in ("forward_calls", "traces", "relowers", "variants", "learned", "harvests", "eager", "reserved_mib")})
