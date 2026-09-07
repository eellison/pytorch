import argparse, json, os, subprocess, sys, time, threading, queue
from pathlib import Path
HERE = Path(__file__).parent
SHAPES = [(1024, 4096), (2048, 3072), (8192, 4096), (65536, 2048)]
GEMM_SHAPES = [(2048, 3072, 4096), (8192, 4096, 4096)]
OUR_MODES = ["default", "cd", "persistent", "persistent_cd", "mk"]
def cells():
    for M, K in SHAPES:
        yield ("rmsnorm", M, K, None, [("ours", m) for m in ("default", "cd")] + [("quack", "-"), ("fi", "-")])
        for fmt in ("nvfp4", "mxfp4"):
            yield (f"rmsnorm_{fmt}", M, K, None, [("ours", m) for m in OUR_MODES] + [("fi_fused", "-"), ("quack_fi", "-"), ("fi_composed", "-")])
        yield ("rmsnorm_mxfp8", M, K, None, [("ours", m) for m in OUR_MODES] + [("quack_fi", "-"), ("fi_composed", "-")])
        for fmt in ("nvfp4", "mxfp4", "mxfp8"):
            yield (f"quant_{fmt}", M, K, None, [("ours", m) for m in ("default", "cd")] + [("fi", "-")])
    for M, K, N in GEMM_SHAPES:
        for fmt in ("nvfp4", "mxfp8"):
            yield (f"gemm_{fmt}", M, K, N, [("ours", m) for m in ("default", "cd")] + [("quack", "-"), ("mm_bf16", "-"), ("quack_bf16", "-")])
def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=HERE / "results.jsonl"); p.add_argument("--gpus", default="2,3,4,5,6,7")
    p.add_argument("--only", default=None); p.add_argument("--timeout", type=int, default=1200); a = p.parse_args()
    done = set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["workload"], r["M"], r["K"], r.get("N"), r["impl"], r["mode"]))
    q = queue.Queue()
    for g in cells():
        if a.only is None or a.only in g[0]: q.put(g)
    lock = threading.Lock()
    env = dict(os.environ, LD_LIBRARY_PATH="/home/eellison/.conda/envs/pytorch-3.12/lib", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
    def worker(gpu):
        while True:
            try: wl, M, K, N, variants = q.get_nowait()
            except queue.Empty: return
            for impl, mode in variants:
                key = (wl, M, K, N, impl, mode)
                if key in done: continue
                argv = [sys.executable, str(HERE / "peak_cell.py"), wl, str(M), str(K)] + ([str(N)] if N else []) + [impl, mode]
                t0 = time.time()
                try:
                    proc = subprocess.run(argv, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), capture_output=True, text=True, timeout=a.timeout, cwd=str(HERE.parent.parent))
                    lines = [l for l in proc.stdout.splitlines() if l.startswith("CELL_RESULT ")]
                    r = json.loads(lines[-1][12:]) if lines else {"workload": wl, "M": M, "K": K, "N": N, "impl": impl, "mode": mode, "error": (proc.stderr or proc.stdout)[-800:]}
                except subprocess.TimeoutExpired:
                    r = {"workload": wl, "M": M, "K": K, "N": N, "impl": impl, "mode": mode, "error": "timeout"}
                r["gpu"] = gpu; r["wall_s"] = round(time.time() - t0, 1)
                with lock:
                    with a.output.open("a") as f: f.write(json.dumps(r) + "\n")
                    msg = f"{r['median_us']:.2f}us k={r.get('kernels')} n={r.get('nested')}" if "median_us" in r else f"ERROR {r['error'][:150]}"
                    print(f"[gpu{gpu}] {wl} {M}x{K}{'x'+str(N) if N else ''} {impl}/{mode}: {msg}", flush=True)
                time.sleep(0.5)
    threads = [threading.Thread(target=worker, args=(int(g),)) for g in a.gpus.split(",")]
    for t in threads: t.start()
    for t in threads: t.join()
if __name__ == "__main__": main()
