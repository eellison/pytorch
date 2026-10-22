# Tables for NUMBERS_cand2.md from vllm/out/*.json: python c2tables.py [section...]  (counts, timing, memory, startup, pad)
import glob, json, os, re, statistics, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
load = lambda f: json.load(open(os.path.join(OUT, f)))
have = lambda f: os.path.exists(os.path.join(OUT, f))
sections = sys.argv[1:] or ["counts", "stock", "timing", "vp", "sweep", "shrink", "memory", "startup", "pad"]


def counts():
    print("| run | checks bitwise | traces | variants | boundaries (serving) | segments | eager first calls | keys harvested | bound hits | model fwds init/bench | json |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(OUT, "c2_*_Vcheck.json")) + glob.glob(os.path.join(OUT, "smoke_pad2.json"))):
        R = json.load(open(f)); V = R["V"]; ch = V["checks"]
        bnd, seg = {}, {}
        for e in V["entries"].values():
            for v in e["per_variant"]:
                if not v["learns"]:
                    bnd[v["boundaries"]] = bnd.get(v["boundaries"], 0) + 1
                    seg[v["segments"]] = seg.get(v["segments"], 0) + 1
        print(f"| {os.path.basename(f)[:-5]} | {sum(c['bitwise'] for c in ch)}/{len(ch)} | {V['traces']} | {V['variants']} | {bnd} | {seg} | {V['eager']} | {V['harvests']} | "
              f"{(V.get('bound') or {}).get('hits', '-')} | {R['model_forwards']['init']}/{R['model_forwards']['bench']} | out/{os.path.basename(f)} |")
    print()
    print("| range run | checks bitwise | traces dec/pre/mix | variants | boundaries/variant | relowers | keys learned | eager entry | first-call median ms dec/pre/mix | json |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(OUT, "c2_*_range.json"))):
        R = json.load(open(f)); S = R["summary"]; C = R["census"]
        tr = "/".join(str(S[p].get("traces", 0)) for p in ("decode", "prefill", "mixed"))
        print(f"| {os.path.basename(f)[:-5]} | {R['checks']['bitwise']}/{R['checks']['n']} | {tr} = {sum(S[p].get('traces', 0) for p in S)} | {C['variants']} | {dict(C['boundaries'])} | "
              f"{sum(S[p].get('relowers', 0) for p in S)} | {sum(C['learned_by_op'].values())} | {sum(S[p].get('eager_entry', 0) for p in S)} | "
              f"{'/'.join(str(S[p].get('first_ms_med')) for p in ('decode', 'prefill', 'mixed'))} | out/{os.path.basename(f)} |")


def stock():
    # Pinned build (land/core/pinned). default-nc = compile off + vLLM's own FULL_DECODE_ONLY graphs; its runs with cudagraph_mode FULL
    # are pooled in, since vLLM downgrades FULL to FULL_DECODE_ONLY with FlashInfer (compilation.py:1418, UNIFORM_BATCH support).
    print("Arm V-stock (the default reported arm) next to the shipped default and default-nc (vLLM with compile off and its own full decode graphs;")
    print("prefill and mixed run eager). Pinned build (land/core/pinned).")
    print()
    print("Counts (V-stock; c2_vstock_* on the pinned build, c2_attn_vstock_* on CAND_LINE=attn = pinned + A1 + F1 (choice() present but unused)):")
    for f in ("c2_vstock_Vcheck.json", "c2_vstock_range.json", "c2_attn_vstock_Vcheck.json", "c2_attn_vstock_range.json", "q35_vstock_Vcheck.json"):  # q35: TRT-LLM C++ launcher path
        if have(f):
            R = load(f)
            if "V" in R:
                V = R["V"]; ch = V["checks"]
                rd = "/".join(str(sum((e.get("redispatches") or 0) for k, e in V["entries"].items() if k.startswith(p))) for p in ("prefill", "decode", "mixed"))
                print(f"- {f[:-5]}: {sum(c['bitwise'] for c in ch)}/{len(ch)} bitwise vs stock eager, {V['traces']} traces, {V['variants']} variants, redispatches prefill/decode/mixed {rd}, model forwards init/bench {R['model_forwards']['init']}/{R['model_forwards']['bench']} (out/{f})")
            else:
                S = R["summary"]
                print(f"- {f[:-5]}: {R['checks']['bitwise']}/{R['checks']['n']} bitwise, {sum(S[p].get('traces', 0) for p in S)} traces (out/{f})")
    E = {a: load(f"e2e_p_{a}.json")["tokens"] for a in ("eager", "fullnc", "fullnc_full", "vstock") if have(f"e2e_p_{a}.json")}
    if have("e2e_a_vstock.json"):
        E["a_vstock"] = load("e2e_a_vstock.json")["tokens"]
    E.update({a: load(f"e2e_{a}.json")["tokens"] for a in ("default", "V") if have(f"e2e_{a}.json")})
    if "eager" in E:
        print()
        print("Greedy tokens (e2e_tokens.py: seed-0 random 128-token prompts, 32 tokens, 2 repeats per bs): sequences equal to eager per bs.")
        print("| arm | " + " | ".join(f"bs {b}" for b in ("1", "8", "64", "128")) + " | json |")
        print("|---|---|---|---|---|---|")
        names = {"fullnc": "default-nc", "fullnc_full": "default-nc (FULL requested)", "vstock": "V-stock", "a_vstock": "V-stock, attn line", "default": "default (int6 run)", "V": "template V (int6 run)"}
        for a, t in E.items():
            if a == "eager":
                continue
            eq = [sum(x == y for x, y in zip(t[f"{b}_0"], E["eager"][f"{b}_0"])) for b in ("1", "8", "64", "128")]
            js = f"out/e2e_p_{a}.json" if a in ("fullnc", "fullnc_full", "vstock") else f"out/e2e_{a}.json"
            js = "out/e2e_a_vstock.json" if a == "a_vstock" else js
            print(f"| {names[a]} | " + " | ".join(f"{e}/{int(b)}" for e, b in zip(eq, ("1", "8", "64", "128"))) + f" | {js} |")
    arms_of = lambda st: {"default": [f"t_{st}_default_r*"], "default-nc": [f"t_{st}_fullnc_r*"], "V-stock": [f"t_{st}_vstock_r*"]}
    ix_rows = [("decode synced", b, lambda R, b=b: R["decode"][b]["step_ms"], None) for b in ("1", "8")] + \
              [("decode async", b, lambda R, b=b: R["async"][b]["step_ms"], None) for b in ("1", "8")] + \
              [(f"mixed {p}-token chunk +", f"{n} decodes", lambda R, k=f"mixed_{p}x{n}": R["ix"][k]["step_ms"], f"mixed_{p}x{n}") for p, n in ((8192, 32), (8192, 64), (8192, 128), (4096, 64))] + \
              [("decode bs", b, lambda R, k=f"decode_{b}": R["ix"][k]["step_ms"], f"decode_{b}") for b in ("64", "128", "256")]
    small_rows = lambda tile: [("decode synced", b, lambda R, b=b: R["decode"][b]["step_ms"], None) for b in ("1", "8", "64", "128")] + \
                              [("decode async", b, lambda R, b=b: R["async"][b]["step_ms"], None) for b in ("1", "8", "64", "128")] + \
                              [("prefill", t, lambda R, t=t: R["prefill"][t]["step_ms"], None) for t in ("64", "512")] + [("mixed", "8x128" + tile, lambda R: R["mixed"]["step_ms"], None)]
    g4_rows = [(f"{lab} mixed: {p}-token chunk +", f"{n} decodes", lambda R, k=f"mixed_{p}x{n}@{kv}": R["ix"][k]["step_ms"], f"mixed_{p}x{n}@{kv}")
               for lab, kv in (("1k1k-shaped", "1024-2048"), ("8k1k prompt-tail", "6000-8000")) for p, n in ((1024, 64), (1024, 128), (2048, 64), (2048, 128))]
    q1_rows = ix_rows[:4] + g4_rows + [(f"8k1k mixed: {p}-token chunk +", f"{n} decodes", lambda R, k=f"mixed_{p}x{n}@4096-8000": R["ix"][k]["step_ms"], f"mixed_{p}x{n}@4096-8000")
                                       for p, n in ((4096, 64), (8192, 32), (8192, 64), (8192, 128))] + ix_rows[-3:]
    main_rows = {"g3": ix_rows, "g4": g4_rows, "q1": q1_rows}
    sets = [(st, arms_of(st), main_rows.get(st, ix_rows)) for st in os.environ.get("STOCK_MAIN_SETS", "g3,g4,q1").split(",")]
    sets += [(st, arms_of(st), small_rows(" (pre-fix: known tile-choice pin)" if st == "g1" else "")) for st in os.environ.get("STOCK_SETS", "g1").split(",") if st not in main_rows]
    sets.append((None, {"default": ["t_dnc_default_r*", "t_pdefault_r*"], "default-nc": ["t_dnc_fullnc_r*", "t_dnc_fullnc_full_r*"], "V-stock": ["t_dnc_vstock_r*", "t_vstock_r*"]}, small_rows("")))
    appendix = False
    for st, arms, rows in sets:
        runs = {a: [json.load(open(f)) for g in gs for f in sorted(glob.glob(os.path.join(OUT, g + ".json")))] for a, gs in arms.items()}
        if not any(runs.values()):
            continue
        if rows not in main_rows.values() and not appendix:
            appendix = True
            print()
            print("#### Diagnostics appendix: small-step microbenchmarks (not InferenceX shapes)")
            print("prefill 64/512 and mixed 8x128 are too small to stand for serving; kept as raw data only.")
        print()
        if st:
            line = {"g1": "pinned build; V-stock attention = FlashInfer trtllm fork (before F1); mixed before the attention fix, 1 measured mixed step per round",
                    "q1": "Qwen3.5-9B (GDN hybrid; recipe args --trust-remote-code --language-model-only, vLLM's own attention backend); V-stock attention = TRT-LLM C++ launcher path (CAND_LINE=trtcpp: attn snap s5 Python on the pinned install, ARMV_TRTLLM_CPP=1); max_model_len 16384, util 0.9",
                    "g4": "1k1k-shaped and 8k1k prompt-tail mixed steps; V-stock attention = FlashInfer trtllm fork (F1), CAND_LINE=attn = A1 + F1 (choice() present but unused); max_model_len 16384, util 0.9; 4 measured steps per row",
                    "g3": "InferenceX-shaped single steps; V-stock attention = FlashInfer trtllm fork (F1), CAND_LINE=attn = A1 + F1 (choice() present but unused); max_model_len 16384, util 0.9; 4 measured steps per mixed row, 16 per decode row"}.get(st, "")
            print(f"Set {st} ({line}): load-gated rounds (rounds_gpu0.sh, one gpu0.lock hold per round, arm order rotated, CPUs 0-71).")
            log = os.path.join(os.path.dirname(OUT), "logs", f"rounds_{st}.log")
            if os.path.exists(log):
                print(f"Per-round load gate (logs/rounds_{st}.log; per-run load in logs/timing_idle.log):")
                print("".join(f"- {l}" for l in open(log) if " round " in l))
        else:
            print("Earlier ungated runs (10-08 afternoon; load avg 83-155, up to 20% round-to-round spread).")

        def cell(a, get):
            v = []
            for R in runs[a]:
                try:
                    x = get(R)
                except (KeyError, TypeError):
                    x = None
                if x is not None:
                    v.append(x)
            return (statistics.median(v), min(v), max(v)) if v else None
        print("Step ms (execute_model, synced unless async), median over rounds [min-max]; async = (wall(48 tok) - wall(16 tok)) / 32 under async scheduling.")
        kvcol = rows in main_rows.values()
        print("| workload | " + " | ".join(arms) + " | default / V-stock | default-nc / V-stock |" + (" decoder KV min/med/max |" if kvcol else ""))
        print("|---|---|---|---|---|---|" + ("---|" if kvcol else ""))
        for sec, k, get, ixk in rows:
            m = {a: cell(a, get) for a in arms}
            f = lambda c: f"{c[0]:.2f} [{c[1]:.2f}-{c[2]:.2f}]" if c else "-"
            sp = lambda a: f"{m[a][0] / m['V-stock'][0]:.2f}" if m[a] and m["V-stock"] else "-"
            kv = ""
            if kvcol:
                kvs = [R["ix"][ixk].get("kv_min_med_max") for rs in runs.values() for R in rs if ixk and ixk in R.get("ix", {})]
                kvs = [x for x in kvs if x]
                kv = f" {min(x[0] for x in kvs)}/{int(statistics.median(x[1] for x in kvs))}/{max(x[2] for x in kvs)} |" if kvs else " - |"
            print(f"| {sec} {k} | " + " | ".join(f(m[a]) for a in arms) + f" | {sp('default')} | {sp('default-nc')} |" + kv)
        for name, get in (("KV tokens (k)", lambda R: R["mem"]["kv_cache_tokens"] / 1e3), ("bench peak over init MiB", lambda R: R["mem"]["bench_peak_over_init_mib"]),
                          ("init s (engine ready; V-stock traces at first call, not at init)", lambda R: R["init_s"]), ("model forwards at init", lambda R: R["model_forwards"]["init"])):
            print(f"| {name} | " + " | ".join(f"{c[0]:.0f}" if (c := cell(a, get)) else "-" for a in arms) + " | | |" + (" |" if kvcol else ""))
        print("rounds:", {a: len(r) for a, r in runs.items()}, "| json:", "; ".join(f"out/{g}.json" for gs in arms.values() for g in gs))


def med(arm, sec, key):
    vals = []
    for f in sorted(glob.glob(os.path.join(OUT, f"t_{arm}_r*.json"))):
        R = json.load(open(f))
        d = R[sec].get(key) if sec != "mixed" else (R["mixed"] if key == "8x128" else R["mixed_specs"].get(key))
        if d and d.get("step_ms") is not None:
            vals.append(d["step_ms"])
    return statistics.median(vals) if vals else None, len(vals)


ARMS = ["eager", "default", "Vcpp_closed", "Vcpp_open", "Vpy_closed", "Vpy_open", "Vcpp_closed_pad", "Vcpp_open_pad"]


def timing():
    rows = [("decode", k) for k in ("1", "5", "8", "64", "100")] + [("prefill", k) for k in ("64", "100", "300", "512", "2048", "4x128", "8x128")] + \
           [("mixed", k) for k in ("8x128", "32x512", "4x2048", "64x64", "128x16")]
    arms = [a for a in ARMS if glob.glob(os.path.join(OUT, f"t_{a}_r*.json"))]
    print("Step ms (execute_model, synced; median over steps, then median over rounds); speedup = default / arm.")
    print("| workload | " + " | ".join(arms) + " | " + " | ".join(f"{a} x" for a in arms if a.startswith("V")) + " |")
    print("|---|" + "---|" * (len(arms) + sum(a.startswith("V") for a in arms)))
    for sec, k in rows:
        m = {a: med(a, sec, k) for a in arms}
        d = m.get("default", (None, 0))[0]
        cells = [f"{m[a][0]:.2f}" if m[a][0] else "-" for a in arms]
        sp = [f"{d / m[a][0]:.2f}" if d and m[a][0] else "-" for a in arms if a.startswith("V")]
        print(f"| {sec} {k} | " + " | ".join(cells) + " | " + " | ".join(sp) + " |")
    print("rounds:", {a: len(glob.glob(os.path.join(OUT, f't_{a}_r*.json'))) for a in arms})


def memory():
    print("| arm | util | KV tokens | after init MiB | reserved after init | reserved end | bench peak over init MiB | V graph growth (reserved end - after init) | default capture GiB | json |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for f in sorted(glob.glob(os.path.join(OUT, f"t_{a}_r*.json")))[:2]:
            R = json.load(open(f)); m = R["mem"]
            cap = ""
            log = os.path.join(os.path.dirname(OUT), "logs", os.path.basename(f)[:-5] + ".log")
            if a == "default" and os.path.exists(log):
                cap = ", ".join(re.findall(r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB", open(log, errors="replace").read()))
            print(f"| {os.path.basename(f)[2:-5]} | {R.get('util')} | {m['kv_cache_tokens']} | {m['after_init_mib']:.0f} | {m['reserved_after_init_mib']:.0f} | {m['reserved_mib']:.0f} | "
                  f"{m['bench_peak_over_init_mib']:.0f} | {m['reserved_mib'] - m['reserved_after_init_mib']:.0f} | {cap} | out/{os.path.basename(f)} |")


def startup():
    groups = {}
    for f in sorted(glob.glob(os.path.join(OUT, "s_*.json"))):
        name = os.path.basename(f)[2:-5]
        base = re.sub(r"_(r\d+|prewarm)$", "", name)
        groups.setdefault(base, []).append((name, json.load(open(f))))
    print("| run | imports s | LLM() s (after imports) | model load s | compile s | capture s (GiB) | V install s | V warm-up s | ready - imports s | warm-up traces / harvests / restored | probes first-call s (traces, harvests) | json |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for base, runs in groups.items():
        for name, R in runs:
            st, vl = R["stamps"], R["vllm_log"]
            w = R.get("warmup", {})
            wt = (sum(x.get("traces", 0) for x in w.values()), sum(x.get("harvests", 0) for x in w.values()), sum(x.get("restored", 0) for x in w.values())) if w else "-"
            pr = "; ".join(f"{k} {v['s']:.2f}" + (f" ({v.get('traces', 0)},{v.get('harvests', 0)})" if "traces" in v else "") for k, v in R["probes"].items() if not k.endswith("_again"))
            cap = " + ".join(f"{s:.0f} ({g})" for s, g in vl.get("capture", []))
            inst = st["installed"] - st["llm_init"] if "installed" in st else None
            warm = st["ready"] - st["installed"] if "installed" in st else None
            print(f"| {name} | {st['imports']:.1f} | {st['llm_init'] - st['imports']:.1f} | {vl.get('model_load_s', 0):.1f} | {vl.get('compile_s', '-')} | {cap or '-'} | "
                  f"{f'{inst:.1f}' if inst is not None else '-'} | {f'{warm:.1f}' if warm is not None else '-'} | {st['ready'] - st['imports']:.1f} | {wt} | {pr} | out/s_{name}.json |")


def pad():
    rows = [("decode", k) for k in ("1", "5", "8", "64", "100")] + [("prefill", k) for k in ("64", "100", "300", "512", "2048", "4x128", "8x128")] + \
           [("mixed", k) for k in ("8x128", "32x512", "4x2048", "64x64", "128x16")]
    arms = ["Vcpp_closed_ab", "Vcpp_closed_pad", "Vcpp_open_ab", "Vcpp_open_pad"]
    if not glob.glob(os.path.join(OUT, "t_Vcpp_closed_pad_r*.json")):
        print("(not run yet)"); return
    print("Step ms, ARMV_PAD=0 (exact sizes) vs 1 (vLLM-default padding); interleaved rounds. pad/nopad > 1 = padding costs.")
    print("| workload | " + " | ".join(arms) + " | closed pad/nopad | open pad/nopad |")
    print("|---|---|---|---|---|---|---|")
    for sec, k in rows:
        m = {a: med(a, sec, k)[0] for a in arms}
        r = lambda x, y: f"{m[x] / m[y]:.2f}" if m[x] and m[y] else "-"
        print(f"| {sec} {k} | " + " | ".join(f"{m[a]:.2f}" if m[a] else "-" for a in arms) + f" | {r('Vcpp_closed_pad', 'Vcpp_closed_ab')} | {r('Vcpp_open_pad', 'Vcpp_open_ab')} |")
    for f in sorted(glob.glob(os.path.join(OUT, "t_Vcpp_*pad_r1.json"))):
        R = json.load(open(f))
        print(f"- {os.path.basename(f)}: padded rows {R['V']['pad']['rows']}, kernels per serving variant",
              {k[:24]: [v["kernels"] for v in e["per_variant"]] for k, e in R["V"]["entries"].items()})


def shrink():
    fs = sorted(glob.glob(os.path.join(OUT, "t_shrink_*_r*.json")))
    if not fs:
        print("(not run yet)"); return
    print("Fluctuating decode batch: N requests, one finishing per step (bs N..1). Mean step ms over the pass (pass 2 = settled), and pass-2 median.")
    print("| run | 64 pass1 mean | 64 pass2 mean | 64 pass2 median | 128 pass1 mean | 128 pass2 mean | 128 pass2 median | json |")
    print("|---|---|---|---|---|---|---|---|")
    for f in fs:
        S = json.load(open(f)).get("shrink", {})
        c = lambda k: f"{S[k]['mean_ms']:.2f}" if k in S else "-"
        m = lambda k: f"{S[k]['step_ms']:.2f}" if k in S else "-"
        print(f"| {os.path.basename(f)[8:-5]} | {c('64_pass1')} | {c('64_pass2')} | {m('64_pass2')} | {c('128_pass1')} | {c('128_pass2')} | {m('128_pass2')} | out/{os.path.basename(f)} |")




def sweep():
    def vals(arm):
        out = {}
        for f in sorted(glob.glob(os.path.join(OUT, f"t_sweep_{arm}_r*.json"))):
            for k, v in json.load(open(f)).get("sweep", {}).items():
                if v.get("step_ms") is not None:
                    out.setdefault(k, []).append(v["step_ms"])
        return {k: statistics.median(v) for k, v in out.items()}
    d, v, e = vals("default"), vals("Vcpp_closed"), vals("eager")
    if not d and not v:
        print("(not run yet)"); return
    print("One step of exactly T tokens (nd decoding requests + prefill prompts <= 4000 tokens filling T - nd); engine max_num_batched_tokens 32768,")
    print("max_num_seqs 512 in every arm. Step ms, median of reps 3-6, median over rounds. Kernel-bound where speedup -> 1.")
    print("| T | nd | default | V cpp closed | speedup | eager |")
    print("|---|---|---|---|---|---|")
    keys = sorted(set(d) | set(v), key=lambda k: (int(k.split("_")[1]), int(k.split("_")[0])))
    for k in keys:
        T, nd = k.split("_")
        sp = f"{d[k] / v[k]:.2f}" if k in d and k in v else "-"
        print(f"| {T} | {nd} | {d.get(k, float('nan')):.2f} | {v.get(k, float('nan')):.2f} | {sp} | {e.get(k, float('nan')):.2f} |")



def vp():
    import subprocess
    arms = ["default", "VAf", "V", "VP"]
    def med2(arm, sec, k):
        vals = [json.load(open(f))[sec].get(k, {}).get("step_ms") for f in sorted(glob.glob(os.path.join(OUT, f"t_vp_{arm}_r*.json")))]
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None
    print("Decode step, arm VP (forward + sampler + post_update in one entry; VP_UVA stopgap: staging and token outputs via pinned UVA views).")
    print("synced = execute_model synced step ms; async = (wall(48 tok) - wall(16 tok)) / 32 under vLLM's async scheduling; pre-launch = execute_model entry -> graph launch (us, launch shim);")
    print("outside = GPU ops outside the graph per decode step (profile, bs 1 and 128 for default/V; all bs for VAf/VP).")
    print("| bs | " + " | ".join(f"{a} synced" for a in arms) + " | " + " | ".join(f"{a} async" for a in arms) + " | " + " | ".join(f"{a} pre-launch" for a in arms) + " |")
    print("|---|" + "---|" * 12)
    ls = {a: json.load(open(os.path.join(OUT, f"t_vpls_{a}.json")))["decode"] if os.path.exists(os.path.join(OUT, f"t_vpls_{a}.json")) else {} for a in arms}
    for b in ("1", "8", "64", "128"):
        sy = [med2(a, "decode", b) for a in arms]
        asy = [med2(a, "async", b) for a in arms]
        pl = [ls[a].get(b, {}).get("pre_launch_us") for a in arms]
        f = lambda v, p=2: f"{v:.{p}f}" if v is not None else "-"
        print(f"| {b} | " + " | ".join(f(v) for v in sy) + " | " + " | ".join(f(v) for v in asy) + " | " + " | ".join(f(v, 0) for v in pl) + " |")
    for d in ("prof_default4", "prof_VAf", "prof_V4", "prof_VP"):
        for b in (1, 8, 64, 128):
            p = os.path.join(OUT, d, f"decode_{b}.json")
            if os.path.exists(p):
                line = subprocess.run([sys.executable, os.path.join(os.path.dirname(OUT), "prof_list.py"), p], capture_output=True, text=True).stdout.splitlines()[1]
                print(f"- {d} bs{b}: {line}")
    print("Correctness: c2_cpp_VP4_Vcheck 127/127 bitwise, 108/108 sampled tokens equal to vLLM's sampler on eager logits, 0 boundaries;")
    print("e2e greedy tokens VP == V at bs 1/8/64/128 (e2e_VP4 vs e2e_V); VAf == default (e2e_VAf vs e2e_default).")


for s in sections:
    print(f"\n### {s}\n")
    globals()[s]()
