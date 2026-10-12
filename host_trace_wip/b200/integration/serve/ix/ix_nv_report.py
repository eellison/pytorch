# NVFP4 InferenceX report (serve/ix/nv_sweep.sh tags TAG_r1, TAG_r2, ...): one row per SKU x shape x conc, repeats merged
# (median, with min-max when there are several). Counts first, then InferenceX's metrics (build_result, pinned clone e0315d2a).
#   bash ../python_vllm_cand2.sh serve/ix/ix_nv_report.py --master vllm/ix_master_attnlc.yaml nv1_r1 nv1_r2 nv1_r3
# Eager steps (V-stock): from each point's GET /armv/summary (cumulative over the server's life: warm-up + timed run, one
# server per point): per entry, the eager calls inside its variants (EagerCall records by name) times its replays. They are
# reported per replayed step, split into mm_fp4 (FlashInfer's CuTe NVFP4 dense GEMM, the kernel module mm_fp4's cute-dsl
# backend compiles: dense_blockscaled_gemm) and other. Fully eager steps = the timed run's steps that did not replay
# (trace calls, entry eager calls, the adapter's eager paths).
import argparse
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SERVE, "inferencex", "inferencex-e2e"))
from infx.results.fixed_sequence import build_result  # noqa: E402

import yaml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--master", action="append", default=[])
ap.add_argument("tags", nargs="+")
ARGS = ap.parse_args()
MASTER = yaml.safe_load(open(os.path.join(HERE, "master.yaml")))
for m in ARGS.master:
    MASTER.update(yaml.safe_load(open(m)))
MM_FP4 = "dense_blockscaled_gemm"  # report label only; exact names stay in the summaries


def env_of(sku, isl, osl):
    e = MASTER[sku]
    prec = e.get("precision") or MASTER[e["inherit"]].get("precision") or MASTER[MASTER[e["inherit"]].get("inherit", e["inherit"])].get("precision")
    return {"RUNNER_TYPE": "gb300-local", "FRAMEWORK": "vllm", "PRECISION": prec or "fp4", "SPEC_DECODING": "none", "ISL": str(isl), "OSL": str(osl),
            "DISAGG": "false", "MODEL_PREFIX": e["model-prefix"], "IMAGE": "local vllm 0.29.0 / torch candidate 4", "TP": "1", "EP_SIZE": "1", "DP_ATTENTION": "false"}


def eager_calls(summary):
    """(mm_fp4, other, replays, other op counter) eager-call executions inside replays, from a point's summary."""
    mm = other = reps = 0.0
    names = collections.Counter()
    for e in (summary or {}).get("entries", {}).values():
        pv = e.get("per_variant") or []
        if not pv or not e.get("replays"):
            continue
        n = e["replays"]
        reps += n
        for v in pv:
            for op, c in v.get("eager_ops", {}).items():
                w = n * c / len(pv)
                if MM_FP4 in op:
                    mm += w
                else:
                    other += w
                    names[op] += w
    return mm, other, reps, names


points = collections.defaultdict(list)
for tag in ARGS.tags:
    for p in sorted(glob.glob(os.path.join(SERVE, "out", "ix", tag, "*_point.json"))):
        d = json.load(open(p))
        res = p.replace("_point.json", ".json")
        if d["rc"] != 0 or not os.path.exists(res):
            points[(d["sku"], d["scenario"], d["conc"])].append({"rc": d["rc"], "tag": tag})
            continue
        isl, osl = (int(x) * 1024 for x in d["scenario"].lower().split("k")[:2])
        r = build_result(json.load(open(res)), env_of(d["sku"], isl, osl))
        b0, b1 = d["breakdown_before"] or {}, d["breakdown_after"] or {}
        delta = lambda k: (b1.get(k) or 0) - (b0.get(k) or 0)
        ours = "replays" in b1
        steps = max(delta("steps"), 1)
        x = {"rc": 0, "tag": tag, "r": r, "steps": steps, "wall": d["client_wall_s"], "prof": (d.get("profile") or {}).get("per_step")}
        if ours:
            sp = p.replace("_point.json", "_summary.json")
            mm, other, reps, names = eager_calls(json.load(open(sp)) if os.path.exists(sp) else None)
            x.update(replays=delta("replays"), traces=delta("traces"), learns=delta("learns"), learn_s=delta("learn_s"), keys=delta("exact_keys"),
                     redisp=delta("redispatches"), redisp_s=delta("redispatch_s"),
                     mm=mm / reps if reps else None, other=other / reps if reps else None, names=names)
        else:
            x["full"] = (b1.get("cg_modes", {}).get("FULL", 0) - b0.get("cg_modes", {}).get("FULL", 0)) / steps
        points[(d["sku"], d["scenario"], d["conc"])].append(x)


def agg(xs, get, n=1, scale=1.0):
    vals = [get(x) for x in xs]
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return "-"
    med = statistics.median(vals)
    return f"{med:.{n}f}" + (f" [{min(vals):.{n}f}-{max(vals):.{n}f}]" if len(vals) > 1 else "")


print("NVFP4 InferenceX-method sweep (serve lane), candidate 4 (int6 snap/step290 + build_cpp9). InferenceX client (fixed_seq.py "
      "client_argv via `infx.bench fixed-seq point`) and result conversion (build_result); fresh server per point; repeats merged "
      "as median [min-max].")
print("Deviations from InferenceX: local fixed KV pool (--kv-cache-memory-bytes: 27B 100 GiB, Flash-Next 40 GiB; shared GPUs) instead of "
      "--gpu-memory-utilization; Flash-Next bf16 KV (vLLM 0.29.0 rejects the recipe's fp8 KV for QSA) and the recipe's max-num-seqs 32 "
      "(conc 64 queues at 32); default-nc = -cc.mode none -cc.cudagraph_mode full_decode_only (Flash-Next's recipe default already is); "
      "V-stock adds --enforce-eager, the worker extension + endpoint plugin and an HTTP range warm-up after install (startup); "
      "--seed 0; a profiler window per point on a separate burst (not in the numbers).")
print("Attention: 27B FlashInfer trtllm-gen (fp8 KV; V-stock: the adapter's dispatcher-op shim, a keyed site); Flash-Next vLLM QSA "
      "Triton + _C top-k. MoE (Flash-Next): vLLM's default FLASHINFER_TRTLLM in both arms.")
print()
print("| sku | shape | conc | runs | steps | fully eager steps | eager calls / replayed step: mm_fp4 + other | learns (s) | traces | "
      "redispatches (s) | FULL steps / launches outside graphs per step | tok/s/GPU | output tok/s/GPU | ITL p50 / p90 ms | "
      "TPOT p50 / p90 ms | TTFT p50 / p90 ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
others = collections.defaultdict(collections.Counter)
for (sku, sc, conc), xs in sorted(points.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
    ok = [x for x in xs if x["rc"] == 0]
    runs = f"{len(ok)}" + (f" (+{len(xs) - len(ok)} failed)" if len(ok) < len(xs) else "")
    if not ok:
        print(f"| {sku} | {sc} | {conc} | {runs} |" + " - |" * 12)
        continue
    ours = "replays" in ok[0]
    m = lambda k, s=1e3: (lambda x: None if x["r"].get(k) is None else s * x["r"][k])
    if ours:
        for x in ok:
            others[sku.split("-")[0]].update(x["names"])
        fe = agg(ok, lambda x: x["steps"] - x["replays"], 0)
        ec = f"{agg(ok, lambda x: x['mm'], 1)} + {agg(ok, lambda x: x['other'], 1)}"
        le = f"{agg(ok, lambda x: x['learns'], 0)} ({agg(ok, lambda x: x['learn_s'], 1)})"
        tr = agg(ok, lambda x: x["traces"], 0)
        rd = f"{agg(ok, lambda x: x['redisp'], 0)} ({agg(ok, lambda x: x['redisp_s'], 1)})"
        cov = f"- / {agg(ok, lambda x: (x['prof'] or {}).get('kernel_launches_outside_graphs'), 0)}"
    else:
        fe = ec = le = tr = rd = "-"
        cov = f"{agg(ok, lambda x: x['full'], 2)} / {agg(ok, lambda x: (x['prof'] or {}).get('kernel_launches_outside_graphs'), 0)}"
    print(f"| {sku} | {sc} | {conc} | {runs} | {agg(ok, lambda x: x['steps'], 0)} | {fe} | {ec} | {le} | {tr} | {rd} | {cov} | "
          f"{agg(ok, m('tput_per_gpu', 1))} | {agg(ok, m('output_tput_per_gpu', 1))} | "
          f"{agg(ok, m('median_itl'), 2)} / {agg(ok, m('p90_itl'), 2)} | {agg(ok, m('median_tpot'), 2)} / {agg(ok, m('p90_tpot'), 2)} | "
          f"{agg(ok, m('median_ttft'), 0)} / {agg(ok, m('p90_ttft'), 0)} |")
for model, c in others.items():
    print(f"\n{model}: top 'other' eager ops (executions summed over its V-stock points): " + "; ".join(f"{op[:100]} {n:.0f}" for op, n in c.most_common(8)))
