"""breakdown27.py RUN.json RUN.log [OFFLINE_LEARN_S]: where a 27B startup run's cold-start time goes (coordinator 10-15): vLLM's own
phases from its log lines, ours' install / bindings / warm-up from the driver's JSON (DRIVE_STARTUP), first-use costs in the stream."""
import datetime
import json
import re
import sys

d = json.load(open(sys.argv[1]))
st = d["startup"]
log = open(sys.argv[2], errors="replace").read().splitlines()


def ts(line):
    m = re.search(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)", line) or None
    m2 = re.search(r"\d{4}-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d),(\d+)", line)
    if m2:
        mo, dd, h, mi, se, ms = map(int, m2.groups())
        return datetime.datetime(2026, mo, dd, h, mi, se, ms * 1000).timestamp()
    if m:
        mo, dd, h, mi, se = map(int, m.groups())
        return datetime.datetime(2026, mo, dd, h, mi, se).timestamp()
    return None


def grab(pat, conv=float):
    for line in log:
        m = re.search(pat, line)
        if m:
            return conv(m.group(1)), line
    return None, None


rows = []
w, _ = grab(r"Loading weights took ([\d.]+) seconds")
mi, mline = grab(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds")
rows.append(("weight load", w))
rows.append(("model init incl. weights", mi))
c, _ = grab(r"torch.compile (?:takes|took) ([\d.]+) s in total")
rows.append(("torch.compile", c or 0.0))
kv_line = next((ln for ln in log if "GPU KV cache size" in ln), None)
caps = [(ts(ln), float(re.search(r"finished in ([\d.]+) secs", ln).group(1))) for ln in log if "Graph capturing finished in" in ln]
in_kv = [c for t, c in caps if kv_line and mline and ts(mline) <= t <= ts(kv_line)]
kv_s = (ts(kv_line) - ts(mline) - sum(in_kv)) if kv_line and mline else None
rows.append(("KV profiling / cache init (minus a capture inside it)", kv_s))
rows.append(("CUDA graph capture inside memory profiling", sum(in_kv)))
starts = [ts(ln) for ln in log if "Autotuning process starts" in ln]
ends = [ts(ln) for ln in log if "Autotuning process ends" in ln]
cache = "; ".join(sorted({re.sub(r".*(Skipping FlashInfer autotuning for ops [^)]*\)|Saved \d+ configs|Loaded \d+ configs|Using FlashInfer autotune cache file).*", r"\1", ln)
                          for ln in log if re.search(r"Skipping FlashInfer autotuning|Saved \d+ configs|Loaded \d+ configs|Using FlashInfer autotune cache file", ln)}))
rows.append((f"FlashInfer autotune [{cache}]", sum(e - s for s, e in zip(starts, ends)) if starts else 0.0))
rows.append(("CUDA graph capture", sum(c for t, c in caps) - sum(in_kv)))
ie, _ = grab(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s")
rows.append(("vLLM init engine total (profile, KV, warmup, capture)", ie))
rows.append(("LLM() total", st.get("init_s")))
v = d.get("V") or {}
rows.append(("ours: install (incl. bindings load)", st.get("install_s")))
rows.append(("ours: bindings load", v.get("bindings_load_s")))
rw = st.get("range_warm")
if rw:
    for name, ph in rw["phases"].items():
        a, n = ph.get("acc", {}), ph.get("cnt", {})
        rows.append((f"ours warm-up {name}: traces {n.get('trace', 0)} ({a.get('trace', 0):.1f} s) builds {a.get('build', 0):.1f} s, "
                     f"redispatch attempts {n.get('redispatch', 0)} ({a.get('redispatch', 0):.1f} s; confirm {ph.get('tpl', {}).get('confirm_s', 0):.1f} s)", ph["s"]))
else:
    rows.append((f"ours: minimal warm-up (adapter: {v.get('traces', 0)} traces" + (f", trace time {v['trace_time_s']:.1f} s)" if v.get("trace_time_s") else ")") if v else "ours: minimal warm-up", st.get("min_warm_s")))
known = sum(x for x in (st.get("init_s"), st.get("min_warm_s"), st.get("install_s")) if x)
rows.append(("other: pre-LLM imports / observers" + ("" if st.get("install_s") is not None else " + ours' install"), (st.get("engine_ready_s") or 0) - known))
rows.append(("ENGINE READY (process start)", st.get("engine_ready_s")))
a0, a1 = st.get("acc_at_ready") or {}, st.get("acc_at_end") or {}
n0, n1 = st.get("cnt_at_ready") or {}, st.get("cnt_at_end") or {}
ev = dict(zip(st.get("events_fields", ["traces", "harvests", "bindings", "redispatches"]), st.get("events_total", [])))
if a1:
    rows.append((f"stream first-use: traces {ev.get('traces')} ({a1.get('trace', 0) - a0.get('trace', 0):.1f} s), builds {a1.get('build', 0) - a0.get('build', 0):.1f} s, "
                 f"redispatches {ev.get('redispatches')} ({a1.get('redispatch', 0) - a0.get('redispatch', 0):.1f} s), harvests {ev.get('harvests')}", None))
rows.append(("TTFT first 8 (median ms)", sorted(x for x in st["ttft_first8_ms"] if x is not None)[4] if st.get("ttft_first8_ms") else None))
# steady: after the last stream step with a trace, a harvest (learn) or a redispatch; first-use binds of stored keys excluded
tl = [e for e in st.get("event_steps", []) if e[2][0] > 0 or e[2][1] > 0 or (len(e[2]) > 3 and e[2][3] > 0)]
quiet = st.get("steps", 0) - 1 - tl[-1][0] if tl else st.get("steps", 0)
rows.append((f"ready to steady (last trace / learn / redispatch step; {quiet} quiet steps after)", (st.get("engine_ready_s") or 0) + (tl[-1][1] if tl else 0.0)))
rows.append(("stream tok/s (all) / quiet tail", f"{st.get('stream_tok_s')} / {st.get('steady_tok_s')}"))
if len(sys.argv) > 3:
    rows.append(("offline cuBLAS learn (seed + learn_range, not in engine ready)", float(sys.argv[3])))
for k, x in rows:
    print(f"{k:110s}\t{x if isinstance(x, str) else ('-' if x is None else f'{x:8.1f}')}")
