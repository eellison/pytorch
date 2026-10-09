# Morning report table for ix_run.py tags, with InferenceX's own result conversion (infx.results.fixed_sequence.build_result,
# pinned clone e0315d2a) applied to each point's benchmark_serving JSON. Counts first, then InferenceX's metrics, then memory.
#   bash ../python_vllm_cand2.sh serve/ix/ix_report.py TAG [TAG ...]      (needs the clone on the path: done here)
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SERVE, "inferencex", "inferencex-e2e"))
from infx.results.fixed_sequence import build_result  # noqa: E402

import yaml  # noqa: E402

MASTER = yaml.safe_load(open(os.path.join(HERE, "master.yaml")))
LINE = {"qwen38bvs-bf16-gb300-vllm": "pinned + trtllm-gen fork", "qwen38bp-bf16-gb300-vllm": "pinned"}


def env_of(sku, isl, osl):
    e = MASTER[sku]
    return {"RUNNER_TYPE": "gb300-box", "FRAMEWORK": "vllm", "PRECISION": MASTER.get(e.get("inherit", sku), e).get("precision", "bf16"),
            "SPEC_DECODING": "none", "ISL": str(isl), "OSL": str(osl), "DISAGG": "false", "MODEL_PREFIX": e["model-prefix"],
            "IMAGE": f"local vllm 0.29.0 / torch {LINE.get(sku, '?')}", "TP": "1", "EP_SIZE": "1", "DP_ATTENTION": "false"}


rows = []
for tag in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(SERVE, "out", "ix", tag, "*_point.json"))):
        d = json.load(open(p))
        res_path = p.replace("_point.json", ".json")
        if not os.path.exists(res_path):
            continue
        bench = json.load(open(res_path))
        isl, osl = (int(x) * 1024 for x in d["scenario"].lower().split("k")[:2])
        r = build_result(bench, env_of(d["sku"], isl, osl))
        b0, b1 = d["breakdown_before"] or {}, d["breakdown_after"] or {}
        delta = lambda k: (b1.get(k) or 0) - (b0.get(k) or 0)
        ours = "replays" in b1
        steps = max(delta("steps"), 1)
        kv = None
        log = p.replace("_point.json", "_server.log")
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)  # bench reporting only
                if m:
                    kv = int(m.group(1).replace(",", ""))
        eager = (delta("entry_eager") + sum(b1.get("eager_path_calls", {}).values()) - sum(b0.get("eager_path_calls", {}).values())) if ours else None
        rows.append(dict(sku=d["sku"], sc=d["scenario"], conc=d["conc"], steps=steps, keys=delta("exact_keys") if ours else None,
                         learns=delta("learns") if ours else None, learn_share=delta("learn_s") / d["client_wall_s"] if ours else None,
                         traces=delta("traces") if ours else None, eager=eager,
                         redisp=delta("redispatches") if "redispatches" in b1 else None, redisp_s=delta("redispatch_s") if "redispatches" in b1 else None,
                         replayed=delta("replays") / steps if ours else None, wall=d["client_wall_s"],
                         full=(b1.get("cg_modes", {}).get("FULL", 0) - b0.get("cg_modes", {}).get("FULL", 0)) / steps,
                         r=r, kv=kv, rc=d["rc"]))
f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}" if isinstance(x, float) else str(x)
ms = lambda r, k: None if r.get(k) is None else 1e3 * r[k]
print("Provisional, pre-core-done (learn-cost follow-ups, attention fixes, entry binding pending). InferenceX client "
      "(fixed_seq.py client_argv via `infx.bench fixed-seq point`) and result conversion (build_result), fresh server per point.")
print("Deviations from InferenceX: Qwen3-8B is not an InferenceX model (no recipe): server args = vLLM defaults + "
      "--attention-backend FLASHINFER, --no-enable-prefix-caching, --max-model-len isl+osl+256 and --max-num-seqs max(conc,16) "
      "(the qwen3.827b PRs' rules), --seed 0; local: fixed KV pool (--kv-cache-memory-bytes 120 GiB, util 0.3 for the "
      "startup check) because GPUs are shared; ours adds --enforce-eager, the worker extension + endpoint plugin, and an HTTP "
      "range warm-up after install (startup); no chat template for 1k1k either; one run per point; profiler window after each "
      "point on a separate burst (not in the numbers).")
print()
print("| sku (line) | shape | conc | steps | new keys | learns (share of run) | traces | eager steps | redispatches (host s, share of run) | replayed / FULL | "
      "tok/s/GPU total | output tok/s/GPU | TPOT p50 / p90 ms | ITL p50 / p90 ms | TTFT p50 / p90 ms | E2E p50 / p90 ms | KV tokens | rc |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in sorted(rows, key=lambda x: (x["sc"], x["conc"], x["sku"])):
    r = x["r"]
    share = "" if x["learn_share"] is None else f" ({100 * x['learn_share']:.0f}%{' LEARN-DOMINATED' if x['learn_share'] > 0.25 else ''})"
    rd = "-" if x["redisp"] is None else f"{x['redisp']} ({x['redisp_s']:.1f} s = {100 * x['redisp_s'] / x['wall']:.0f}%{' REDISPATCH-DOMINATED' if x['redisp_s'] > 0.25 * x['wall'] else ''})"
    cov = f"{x['replayed']:.2f}" if x["replayed"] is not None else f"FULL {x['full']:.2f}"
    print(f"| {x['sku']} ({LINE.get(x['sku'], '?')}) | {x['sc']} | {x['conc']} | {x['steps']} | {f(x['keys'])} | {f(x['learns'])}{share} | {f(x['traces'])} | "
          f"{f(x['eager'])} | {rd} | {cov} | {f(r['tput_per_gpu'], 1)} | {f(r['output_tput_per_gpu'], 1)} | "
          f"{f(ms(r, 'median_tpot'))} / {f(ms(r, 'p90_tpot'))} | {f(ms(r, 'median_itl'))} / {f(ms(r, 'p90_itl'))} | "
          f"{f(ms(r, 'median_ttft'), 1)} / {f(ms(r, 'p90_ttft'), 1)} | {f(ms(r, 'median_e2el'), 0)} / {f(ms(r, 'p90_e2el'), 0)} | {f(x['kv'])} | {x['rc']} |")
