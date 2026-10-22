# python3 armV/servetable.py [TAG]: bench/out/serve/TAG_{eager,default,V}_p{1,2}_{w}.json as one table per pass.
# =eager / =eagerm: requests whose generated text equals that arm's (same pass); eagerm (eager plus V's max_seq_len policy) is
# V's bitwise reference. Eager p1 vs p2 (on eager's rows) is the batching noise floor.
import json, os, sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "s1"
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "out", "serve")


def load(arm, run):
    f = os.path.join(D, f"{TAG}_{arm}_{run}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def same(a, b):
    return f"{sum(x == y for x, y in zip(a['generated_texts'], b['generated_texts']))}/{len(a['generated_texts'])}"


for p in (1, 2):
    print(f"pass {p}: ttft med/p99 ms | itl med/p99 ms | tpot med ms | out tok/s | dur s | =eager | =eagerm")
    for w in ("r1", "r8", "r64", "m8", "m64"):
        run, e = f"p{p}_{w}", load("eager", f"p{p}_{w}")
        m = load("eagerm", run)
        for arm in ("eager", "eagerm", "default", "V", "Vd"):
            r = load(arm, run)
            if r is None:
                continue
            s = same(r, e) if e is not None and arm != "eager" else (same(r, load("eager", f"p1_{w}")) + " (p1)" if p == 2 and load("eager", f"p1_{w}") else "-")
            s += " | " + (same(r, m) if m is not None and arm != "eagerm" else "-")
            print(f"  {w:4} {arm:7} {r['median_ttft_ms']:8.1f} {r['p99_ttft_ms']:8.1f} | {r['median_itl_ms']:6.2f} {r['p99_itl_ms']:7.2f} | "
                  f"{r['median_tpot_ms']:6.2f} | {r['output_throughput']:8.0f} | {r['duration']:6.1f} | {s}")
