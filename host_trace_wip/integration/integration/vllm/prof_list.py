# Per-step GPU activity list from a prof_decode.py chrome trace: every kernel / memcpy / memset of the middle profiled
# engine step (ENGINE_STEP_1), in GPU order, with its launching runtime call (correlation id), whether it ran inside a
# graph replay, and the innermost Python frames around the launch; then host time per Python phase of that step.
#   python prof_list.py out/prof_default/decode_8.json [--step 1] [--frames 3]
import argparse, bisect, collections, json

p = argparse.ArgumentParser()
p.add_argument("trace")
p.add_argument("--step", type=int, default=1)
p.add_argument("--frames", type=int, default=3)
p.add_argument("--phases", nargs="*", default=["execute_model", "prepare_inputs", "prepare_attn", "prepare_pos_seq_lens", "combine_sampled_and_draft_tokens",
                                                "build_slot_mappings_by_layer", "async_copy_to_gpu", "apply_staged_writes", "update_requests", "add_requests",
                                                "gather_batch_req_state", "sample_tokens", "sample", "compute_logits", "run_fullgraph", "run_pw_graph",
                                                "forward", "_forward_deferred", "schedule", "update_from_output", "process_outputs", "step_with_batch_queue"])
a = p.parse_args()
ev = json.load(open(a.trace))["traceEvents"]
step = next(e for e in ev if e.get("name") == f"ENGINE_STEP_{a.step}" and e.get("ph") == "X" and e.get("cat") in ("user_annotation", "cpu_op"))
t0, t1 = step["ts"], step["ts"] + step["dur"]
gpu_step = next((e for e in ev if e.get("name") == f"ENGINE_STEP_{a.step}" and e.get("cat") == "gpu_user_annotation"), None)
runtime = {e["args"]["correlation"]: e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
py = sorted((e for e in ev if e.get("cat") == "python_function" and e.get("ph") == "X" and t0 <= e["ts"] <= t1), key=lambda e: e["ts"])
starts = [e["ts"] for e in py]


def frames(ts, n):
    """The innermost n Python frames whose span contains ts (outermost first)."""
    out = [e for e in py[:bisect.bisect_right(starts, ts)] if e["ts"] + e["dur"] >= ts]
    names = [e["name"].split(": ", 1)[-1] if e["name"].startswith("<built-in") else e["name"] for e in out]
    names = [x for x in names if not x.startswith(("torch/", "<built-in", "contextlib", "threading", "triton/", "vllm/utils/", "functools"))] or names
    return " < ".join(reversed(names[-n:]))


gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
rows = []
for e in gpu:
    rt = runtime.get(e.get("args", {}).get("correlation"))
    if rt is None or not (t0 <= rt["ts"] <= t1):
        continue
    rows.append((e["ts"], e, rt))
rows.sort(key=lambda r: r[0])
in_graph = collections.Counter()
outside = []
for ts, e, rt in rows:
    if "GraphLaunch" in rt["name"]:
        in_graph[rt["args"]["correlation"]] += 1
    else:
        outside.append((ts, e, rt))
graph_launches = sorted({rt["args"]["correlation"] for _, _, rt in rows if "GraphLaunch" in rt["name"]})
first_graph = min((ts for ts, _, rt in rows if "GraphLaunch" in rt["name"]), default=None)
print(f"# {a.trace} step {a.step}: host {step['dur']:.0f} us" + (f", GPU span {gpu_step['dur']:.0f} us" if gpu_step else ""))
print(f"graph launches: {len(graph_launches)} ({sum(in_graph.values())} kernels inside); outside graphs: {len(outside)} "
      f"(kernels {sum(e['cat'] == 'kernel' for _, e, _ in outside)}, memcpy {sum(e['cat'] == 'gpu_memcpy' for _, e, _ in outside)}, "
      f"memset {sum(e['cat'] == 'gpu_memset' for _, e, _ in outside)})")
print("| # | when | GPU us | kind | name | runtime call (corr) | host us into step | Python frames |")
print("|---|---|---|---|---|---|---|---|")
for i, (ts, e, rt) in enumerate(outside):
    when = "before graph" if first_graph is None or ts < first_graph else "after graph"
    kind = {"kernel": "kernel", "gpu_memcpy": e["name"].replace("Memcpy ", ""), "gpu_memset": "memset"}[e["cat"]]
    name = e["name"] if e["cat"] == "kernel" else e["args"].get("bytes", "")
    print(f"| {i} | {when} | {e['dur']:.1f} | {kind} | {str(name)[:70]} | {rt['name']} ({rt['args']['correlation']}) | {rt['ts'] - t0:.0f} | {frames(rt['ts'], a.frames)[:160]} |")
print("\nhost phases (inclusive us within the step; count):")
tot = collections.defaultdict(lambda: [0.0, 0])
open_until = {}
for e in py:  # outermost frame of each phase name only (wrappers share names)
    nm = e["name"]
    for ph in a.phases:
        if nm.endswith(f"): {ph}"):
            if open_until.get(ph, -1) >= e["ts"]:
                continue
            open_until[ph] = e["ts"] + e["dur"]
            tot[ph][0] += e["dur"]
            tot[ph][1] += 1
for ph in a.phases:
    if ph in tot:
        print(f"- {ph}: {tot[ph][0]:.0f} us ({tot[ph][1]})")
