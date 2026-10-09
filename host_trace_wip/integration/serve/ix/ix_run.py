# InferenceX-style fixed-seq-len sweep of one SKU on this box: reads ix/master.yaml + the SKU's recipe, and per
# concurrency point launches `vllm serve` (recipe args + the per-point --max-model-len / --max-num-seqs rules of
# InferenceX's qwen3.827b scripts), installs the arm extension over HTTP, runs InferenceX's own client
# (`python3 -m infx.bench fixed-seq point` from the pinned clone, i.e. infx.bench_serving.benchmark_serving with their
# client policy: random dataset, range ratio, 10x conc prompts, 2x conc warm-ups, request rate inf, ignore EOS), and
# records the per-win breakdown (GET /armv/breakdown before/after; optional profiler window on a separate short burst).
# One server per point, as InferenceX restarts per concurrency (--reuse-server keeps one server for the whole sweep).
# Run under the GPU's lock with python_vllm_cand2.sh:
#   bash ../python_vllm_cand2.sh serve/ix/ix_run.py --sku qwen38bht-bf16-gb300-vllm --scenario 1k1k --gpu 0 --tag t3
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.path.dirname(HERE)
G = os.path.dirname(SERVE)  # land/scratch/integration
IX = os.path.join(SERVE, "inferencex", "inferencex-e2e")


def entry(master, key):
    """A master entry with `inherit` resolved; x-args / x-env / x-armv merge over the parent's."""
    e = dict(master[key])
    parent = e.pop("inherit", None)
    if parent is None:
        return e
    base = entry(master, parent)
    for k in ("x-args", "x-env", "x-armv"):
        merged = {**base.get(k, {}), **e.get(k, {})}
        if merged:
            e[k] = merged
    return {**base, **e}


def http(method, url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def serve_cmd(e, recipe, isl, osl, conc, port):
    args = dict(recipe["base"]["roles"]["agg"]["args"])
    args.update(e.get("x-args", {}))
    args.update({k: v for k, v in e.get("x-local", {}).items()})
    args["max-model-len"] = isl + osl + 256  # InferenceX qwen3.827b: serve the matrix context
    if "max-num-seqs" not in args:  # a recipe's own value wins (Flash-Next's single-GB300 32)
        args["max-num-seqs"] = max(conc, 16)  # InferenceX qwen3.827b: GDN/Mamba blocks; size the batch to the point
    argv = ["-m", "vllm.entrypoints.cli.main", "serve", e["model"], "--served-model-name", e["model"], "--port", str(port),
            "--worker-extension-cls", "armv_serve.ArmVWorkerExt", "--seed", "0"]
    for k, v in args.items():
        if v is True:
            argv.append(f"--{k}")
        elif v is not False and v is not None:
            argv += [f"--{k}", str(v)]
    # raw flags as the recipe writes them (vLLM's -cc.<field> form), recipe first, then the SKU's
    argv += [str(x) for x in recipe["base"]["roles"]["agg"].get("raw-args", [])] + [str(x) for x in e.get("x-raw-args", [])]
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=os.path.join(HERE, "master.yaml"))
    p.add_argument("--sku", required=True)
    p.add_argument("--scenario", required=True, help="e.g. 8k1k / 1k1k (isl k osl k)")
    p.add_argument("--concs", default=None, help="comma list; default the master's conc-list")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--reuse-server", action="store_true")
    p.add_argument("--profile-steps", type=int, default=0, help="profiler window (steps) on a separate 2x conc burst per point")
    p.add_argument("--warm", action="store_true", help="ours: serve_warm2.py range warm-up after install (counted as startup)")
    a = p.parse_args()
    master = yaml.safe_load(open(a.master))
    e = entry(master, a.sku)
    recipe = yaml.safe_load(open(os.path.join(HERE, e["srt-recipe"])))
    isl, osl = (int(x) * 1024 for x in a.scenario.lower().split("k")[:2])
    scen = next(s for s in e["scenarios"]["fixed-seq-len"] if s["isl"] == isl and s["osl"] == osl)
    concs = [int(c) for c in a.concs.split(",")] if a.concs else scen["search-space"][0]["conc-list"]
    rr = recipe["base"]["benchmark"]["env"]["RANDOM_RANGE_RATIO"]
    port = 18900 + a.gpu
    scpu, ccpu = ("0-63", "64-71") if a.gpu == 0 else ("72-101", "102-107")
    out = os.path.join(SERVE, "out", "ix", a.tag)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update(recipe["base"]["roles"]["agg"].get("env", {}))
    env.update(e.get("x-env", {}))
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), VLLM_EXTRA_PYTHONPATH=os.path.join(SERVE, "plugin"),
               VLLM_PLUGINS="armv_endpoints,lora_filesystem_resolver,lora_hf_hub_resolver")
    launcher = os.path.join(G, "python_vllm_cand2.sh")
    base = f"http://127.0.0.1:{port}"
    srv = None

    def start(conc):
        cmd = serve_cmd(e, recipe, isl, osl, conc, port)
        log = open(os.path.join(out, f"{a.sku}_{a.scenario}_c{conc}_server.log"), "w")
        t0 = time.time()
        proc = subprocess.Popen(["taskset", "-c", scpu, "bash", launcher, *cmd], env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                urllib.request.urlopen(f"{base}/health", timeout=5)
                break
            except OSError:
                if proc.poll() is not None or time.time() - t0 > 3600:
                    raise SystemExit(f"server failed (see {log.name})")
                time.sleep(2)
        ready = time.time() - t0
        inst = http("POST", f"{base}/armv/install", {"mode": e.get("x-armv", {}).get("install", "observe")}, timeout=600)
        warm = None
        if a.warm and e.get("x-armv", {}).get("install") == "trace":
            t1 = time.time()
            # the range warm-up within this server's limits: lengths below max-model-len, decode / mixed batches up to max-num-seqs
            top, seqs = isl + osl + 256 - 8, max(conc, 16)
            lens = sorted({n for n in (1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
                                       2047, 2048, 2049, 4095, 4096, 4097, 8191, 8192) if n < top - 1}) + [top - 1]
            subprocess.run(["taskset", "-c", ccpu, "bash", launcher, os.path.join(SERVE, "serve_warm2.py"), "--port", str(port), "--model", e["model"],
                            "--dec-bs", *map(str, range(1, seqs + 1)), "--pre-lens", *map(str, lens), "--pre-multi", "2x8", "2x100", f"{min(8, seqs)}x64",
                            "--mixed-nd", *map(str, sorted({1, 2, 3, 8, min(16, seqs - 1), seqs - 1} - {0})), "--mixed-lens", *map(str, [n for n in (1, 2, 7, 16, 17, 40, 128, 500, 1000, 4000, 8000) if n < top - 1] + [top - 1])],
                           env=env, check=False, stdout=subprocess.DEVNULL)
            warm = time.time() - t1
        return proc, {"cmd": cmd, "ready_s": ready, "install": inst, "warm_s": warm}

    def stop(proc):
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def client(conc, n, result):
        # python_vllm_cand2.sh rebuilds PYTHONPATH and appends VLLM_EXTRA_PYTHONPATH: the pinned InferenceX clone goes there
        cenv = dict(env, VLLM_EXTRA_PYTHONPATH=IX, PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1")
        cmd = ["taskset", "-c", ccpu, "bash", launcher, "-m", "infx.bench", "fixed-seq", "point", "--base-url", base, "--model", e["model"],
               "--backend", "vllm", "--tokenizer", e["model"], "--isl", str(isl), "--osl", str(osl), "--random-range-ratio", rr,
               "--conc", str(conc), "--num-prompts", str(n), "--result", result]
        with open(result + ".log", "w") as log:
            return subprocess.run(cmd, env=cenv, cwd=out, stdout=log, stderr=subprocess.STDOUT).returncode

    try:
        for conc in concs:
            if srv is None:
                srv = start(conc)
            proc, info = srv
            name = f"{a.sku}_{a.scenario}_c{conc}"
            b0 = http("GET", f"{base}/armv/breakdown")
            t0 = time.time()
            rc = client(conc, 10 * conc, os.path.join(out, name + ".json"))
            wall = time.time() - t0
            b1 = http("GET", f"{base}/armv/breakdown")
            prof = None
            if a.profile_steps and rc == 0:
                http("POST", f"{base}/armv/profile", {"steps": a.profile_steps})
                client(conc, 2 * conc, os.path.join(out, name + "_prof.json"))
                prof = http("GET", f"{base}/armv/profile")
            rec = {"sku": a.sku, "scenario": a.scenario, "conc": conc, "rc": rc, "client_wall_s": wall, "server": info,
                   "reuse_server": a.reuse_server, "breakdown_before": b0, "breakdown_after": b1, "profile": prof,
                   "counters": http("GET", f"{base}/armv/counters") if e.get("x-armv", {}).get("install") == "trace" else None}
            json.dump(rec, open(os.path.join(out, name + "_point.json"), "w"), indent=1)
            print(f"{time.strftime('%T')} {name} rc={rc} wall={wall:.0f}s", flush=True)
            if not a.reuse_server:
                stop(proc)
                srv = None
    finally:
        if srv is not None:
            stop(srv[0])


if __name__ == "__main__":
    main()
