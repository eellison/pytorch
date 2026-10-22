# diag3 analysis (main's questions 10-14). Timelines (serve/out/prof_diag3/<arm>_{fixed63,alt63_65}.json): per profiled step
# (PH_execute_model .. PH_sample_tokens), host time from execute_model's start to the first GPU work, GPU busy (union of kernel /
# memset / memcpy intervals) vs the step's wall, idle gaps, graph launches; kernel identity (name, grid, block) multiset per step;
# per-kernel durations of ours' alternating bs-63 steps vs fixed bs-63 steps (same kernels, same order). Setlog (ours): the
# alternating steps' node sets split by what a per-size exec cache would still patch. GC steps: garbage per step class.
#   python3 serve/ix/diag3_analyze.py
import collections, json, os, statistics as st

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PD = os.path.join(HERE, "serve", "out", "prof_diag3")


def steps_of(path):
    ev = json.load(open(path))["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]
    ex0 = sorted([e for e in X if e["name"] == "PH_execute_model"], key=lambda e: e["ts"])
    ex = [e for e in ex0 if not any(o is not e and o["ts"] <= e["ts"] and e["ts"] + e["dur"] <= o["ts"] + o["dur"] for o in ex0)]  # outermost only
    sm = sorted([e for e in X if e["name"] == "PH_sample_tokens"], key=lambda e: e["ts"])
    gpu = sorted([e for e in X if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
    rt = [e for e in X if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    out = []
    for i, e in enumerate(ex):
        t0 = e["ts"]
        s = next((x for x in sm if x["ts"] >= t0), None)
        t1 = (s["ts"] + s["dur"]) if s else e["ts"] + e["dur"]
        g = [x for x in gpu if t0 <= x["ts"] <= t1 + 50]
        launches = [x for x in rt if t0 <= x["ts"] <= t1 and "GraphLaunch" in x["name"]]
        busy, cur = 0.0, None
        for x in g:
            a, b = x["ts"], x["ts"] + x["dur"]
            if cur is None or a > cur[1]:
                if cur:
                    busy += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur:
            busy += cur[1] - cur[0]
        gaps = sorted(((g[j + 1]["ts"] - (g[j]["ts"] + g[j]["dur"])) for j in range(len(g) - 1)), reverse=True)
        out.append({"wall_us": t1 - t0, "exec_us": e["dur"], "sample_us": s["dur"] if s else None, "first_gpu_us": (g[0]["ts"] - t0) if g else None,
                    "first_launch_us": (launches[0]["ts"] - t0) if launches else None, "launch_call_us": [round(x["dur"], 1) for x in launches],
                    "gpu_busy_us": busy, "gpu_span_us": (g[-1]["ts"] + g[-1]["dur"] - g[0]["ts"]) if g else None, "n_gpu": len(g),
                    "top_gaps_us": [round(x, 1) for x in gaps[:4]],
                    "kernels": [(x["name"][:90], str(x.get("args", {}).get("grid")), str(x.get("args", {}).get("block")), x["dur"]) for x in g if x.get("cat") == "kernel"],
                    "other_gpu": [(x["cat"], x["name"][:40], x.get("args", {}).get("bytes"), x["dur"]) for x in g if x.get("cat") != "kernel"]})
    return out


PDS = [PD, os.path.join(HERE, "serve", "out", "prof_setlog")]


def show(arm, name, pd=PD):
    p = os.path.join(pd, f"{arm}_{name}.json")
    if not os.path.exists(p):
        return None
    S = steps_of(p)
    print(f"\n#### {arm} {name}: {len(S)} steps\n")
    print("| step | wall us | execute_model us | sample us | host to first GPU work us | to first graph launch us | graph launch call us | GPU busy us | GPU span us | GPU ops | largest gaps us |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(S):
        print(f"| {i} | {s['wall_us']:.0f} | {s['exec_us']:.0f} | {s['sample_us'] or 0:.0f} | {s['first_gpu_us'] or 0:.0f} | {s['first_launch_us'] or 0:.0f} | "
              f"{s['launch_call_us']} | {s['gpu_busy_us']:.0f} | {s['gpu_span_us'] or 0:.0f} | {s['n_gpu']} | {s['top_gaps_us']} |")
    return S


def kid(S):
    return collections.Counter((k[0], k[1], k[2]) for k in S["kernels"])


res = {}
for arm in ("vcsl", "default"):
    for name in ("fixed37", "fixed63", "fixed65", "alt63_65"):
        S = show(arm, name)
        if S:
            res[(arm, name)] = S
# setlog job (10-14): the ~1.1K curve step at bs 63 next to fixed 63 at the alternating KV, both arms (prof_setlog)
for arm in ("vcsl", "default"):
    for name in ("dec1100_63", "fixed63", "alt63_65", "alt37_50"):
        S = show(arm, name, pd=PDS[1])
        if S:
            res[(arm, "sl_" + name)] = S
for a_, b_ in (("sl_dec1100_63", "sl_fixed63"),):
    for arm in ("vcsl", "default"):
        if (arm, a_) in res and (arm, b_) in res:
            x, y = res[(arm, a_)][-1], res[(arm, b_)][-1]
            print(f"\n{arm}: curve bs 63 GPU busy {x['gpu_busy_us']:.0f} us vs alternating-KV fixed 63 {y['gpu_busy_us']:.0f} us")
            tx, ty = collections.defaultdict(float), collections.defaultdict(float)
            for k in x["kernels"]:
                tx[k[0][:50]] += k[3]
            for k in y["kernels"]:
                ty[k[0][:50]] += k[3]
            for k in sorted(set(tx) | set(ty), key=lambda k: -abs(tx.get(k, 0) - ty.get(k, 0)))[:6]:
                print(f"   {k}: curve {tx.get(k, 0):.0f} us, alternating-KV {ty.get(k, 0):.0f} us")
for f in ("vllm/out/t_marks_vcsl_r1.json",):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        print("\n#### marks (GPU ms per replay, 50 back-to-back replays, CUDA events)\n", json.dumps(json.load(open(p)).get("marks"), indent=1))
# kernel identity at fixed bs 63 (second profiled step of each), ours vs default
if ("vcsl", "fixed63") in res and ("default", "fixed63") in res:
    o, d = kid(res[("vcsl", "fixed63")][-1]), kid(res[("default", "fixed63")][-1])
    print("\n#### Kernel identity at fixed bs 63 (ours vs default; (name, grid, block) multiset)\n")
    print(f"ours {sum(o.values())} kernels, default {sum(d.values())}; only ours: {sum((o - d).values())}, only default: {sum((d - o).values())}")
    for k, n in (o - d).most_common(12):
        print(f"  only ours x{n}: {k}")
    for k, n in (d - o).most_common(12):
        print(f"  only default x{n}: {k}")
    tO = collections.defaultdict(float)
    tD = collections.defaultdict(float)
    for k in res[("vcsl", "fixed63")][-1]["kernels"]:
        tO[k[0][:50]] += k[3]
    for k in res[("default", "fixed63")][-1]["kernels"]:
        tD[k[0][:50]] += k[3]
    print("\n| kernel (first 50 chars) | ours us | default us | ours - default |")
    print("|---|---|---|---|")
    for k in sorted(set(tO) | set(tD), key=lambda k: -abs(tO.get(k, 0) - tD.get(k, 0)))[:12]:
        print(f"| {k} | {tO.get(k, 0):.1f} | {tD.get(k, 0):.1f} | {tO.get(k, 0) - tD.get(k, 0):+.1f} |")
# GPU-side cost of the alternating updates: ours' alternating bs-63 steps (0, 2) vs fixed bs-63 steps, kernel by kernel
if ("vcsl", "alt63_65") in res and ("vcsl", "fixed63") in res:
    A = [res[("vcsl", "alt63_65")][i] for i in (0, 2) if i < len(res[("vcsl", "alt63_65")])]
    F = res[("vcsl", "fixed63")]
    ka = [[k for k in s["kernels"]] for s in A]
    kf = [[k for k in s["kernels"]] for s in F]
    same = all(len(x) == len(kf[0]) for x in ka + kf) and all([k[:3] for k in x] == [k[:3] for k in kf[0]] for x in ka + kf)
    print(f"\n#### Ours: alternating bs-63 steps vs fixed bs-63 steps (same kernel sequence: {same})\n")
    if same:
        n = len(kf[0])
        da = [st.mean(x[j][3] for x in ka) for j in range(n)]
        df = [st.mean(x[j][3] for x in kf) for j in range(n)]
        print(f"kernel time per step: alternating {sum(da):.0f} us, fixed {sum(df):.0f} us, difference {sum(da) - sum(df):+.0f} us over {n} kernels")
        by = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for j in range(n):
            nm = kf[0][j][0][:50]
            by[nm][0] += da[j]
            by[nm][1] += df[j]
            by[nm][2] += 1
        print("\n| kernel (first 50 chars) | count | alternating us | fixed us | difference |")
        print("|---|---|---|---|---|")
        for nm, (x, y, c) in sorted(by.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:12]:
            print(f"| {nm} | {c} | {x:.1f} | {y:.1f} | {x - y:+.1f} |")
    for nm, S in (("alternating", A), ("fixed", F)):
        print(f"{nm}: GPU busy {[round(s['gpu_busy_us']) for s in S]} us, host to first GPU work {[round(s['first_gpu_us'] or 0) for s in S]} us, "
              f"wall {[round(s['wall_us']) for s in S]} us")
# setlog: alternating node sets vs what a per-size ready exec would still patch
for f in ("vllm/out/t_diag3_vcsl_r2.json", "vllm/out/t_setlog_vcsl_r1.json"):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        continue
    R = json.load(open(p))
    print("\nasync_scheduling:", R.get("async_scheduling"))
    SL = R.get("setlog", {})
    print("setlog install rc:", SL.get("install_rc"))
    if SL.get("after_warmup_execs"):
        print("execs after warm-up (nodes set / with a launch-config change / with a function change):",
              [(x["nodes_set"], x["nodes_with_launch_config_change"], x["nodes_with_function_change"]) for x in SL["after_warmup_execs"][:12]])
    for key in [k for k in SL if k.startswith(("alt", "fixed")) and isinstance(SL[k], dict)]:
        E = SL[key]["entries"]
        names = SL[key]["names"]
        bystep = collections.defaultdict(list)
        for e in E:
            bystep[e["step"]].append(e)
        steps = sorted(bystep)
        print(f"\n#### setlog {key}: steps {steps[:12]}, sets per step {[len(bystep[s]) for s in steps]}\n")
        hist = {}
        for i, s in enumerate(steps):
            cur = {(e["exec"], e["node"]): e for e in bystep[s]}
            prev2 = {(e["exec"], e["node"]): e for e in bystep[steps[i - 2]]} if i >= 2 else None
            prev1 = {(e["exec"], e["node"]): e for e in bystep[steps[i - 1]]} if i >= 1 else None
            if prev2 is None:
                continue
            c = collections.Counter()
            kinds = collections.Counter()
            for k, e in cur.items():
                c["sets"] += 1
                c["function changed vs previous step"] += bool(e["func_changed"])
                c["grid/block/smem changed vs previous step"] += bool(e["cfg_changed"]) and not e["func_changed"]
                c["arguments only vs previous step"] += bool(e["bytes_changed"]) and not e["cfg_changed"] and not e["func_changed"]
                c["identical to previous set (no-op set)"] += not (e["bytes_changed"] or e["cfg_changed"] or e["func_changed"])
                q = prev2.get(k)
                if q is None:
                    c["not set 2 steps ago (same size)"] += 1
                    continue
                same_cfg = q["func"] == e["func"] and q["dims"] == e["dims"]
                same_args = q["arg_hash"] == e["arg_hash"]
                if same_cfg and same_args:
                    c["same as 2 steps ago: a per-size exec would skip"] += 1
                elif same_cfg:
                    c["args differ from 2 steps ago: still patched with a per-size exec"] += 1
                    nm = names.get(str(e["func"])) or "?"
                    kinds[nm[:60]] += 1
                else:
                    c["config differs from 2 steps ago"] += 1
            hist[s] = (c, kinds)
        for s, (c, kinds) in list(hist.items())[:4]:
            print(f"step {s}: " + "; ".join(f"{k} {v}" for k, v in c.items()))
            print("   still patched by kernel: " + "; ".join(f"{k} x{v}" for k, v in kinds.most_common(8)))
    GS = R.get("gc_steps")
    if GS:
        print("\n#### GC per step class (ours, vcsl)\n", json.dumps(GS.get("by_class"), indent=1), "\ntracked", GS.get("tracked0"), "->", GS.get("tracked_end"), "\nper point (whole point incl. unmeasured steps):", GS.get("per_point"))
        for x in GS.get("steps", [])[:40]:
            print("  ", x["cls"], "garbage", x["garbage"], "marker", x["marker"], "d(traces,harvests,bindings,slow)", x["d_traces_harvests_bindings_slow"], "tracked", x["tracked"], x["top"][:3])
p = os.path.join(HERE, "vllm/out/t_diag3_default_r1.json")
if os.path.exists(p):
    R = json.load(open(p))
    print("\ndefault async_scheduling:", R.get("async_scheduling"))
    GS = R.get("gc_steps")
    if GS:
        print("#### GC per step class (default)\n", json.dumps(GS.get("by_class"), indent=1), "\ntracked", GS.get("tracked0"), "->", GS.get("tracked_end"))
