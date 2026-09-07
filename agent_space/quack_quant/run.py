"""Dispatch cells across GPUs, one process per cell, resumable JSONL."""
import argparse, json, os, subprocess, sys, time, threading, queue
from pathlib import Path

HERE = Path(__file__).parent
FORMATS = ["mxfp8_e4m3", "mxfp8_e5m2", "mxfp4", "mxfp4_byte", "mxfp6_e2m3", "mxfp6_e3m2",
           "mxfp6_e2m3_packed", "mxfp6_e3m2_packed", "nvfp4"]
SHAPES = [(2048, 3072), (8192, 4096), (65536, 2048)]

def cells():
    for M, K in SHAPES:
        for f in FORMATS:
            yield (f"rmsnorm+{f}", M, K, ["eager", "off", "on", "on_mk"])
            yield (f, M, K, ["eager", "on"])
        yield ("mxfp8_e4m3", M, K, ["on_dynamic", "off_dynamic"])
        yield ("rmsnorm+mxfp8_e4m3", M, K, ["on_dynamic", "off_dynamic"])
        yield ("mxfp8_dim0", M, K, ["eager", "on"])
        yield ("rmsnorm+mxfp8_dim0", M, K, ["eager", "off", "on"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=HERE / "results.jsonl")
    p.add_argument("--gpus", default="2,3,4,5,6,7")
    p.add_argument("--cooldown", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--only", default=None, help="substring filter on case")
    a = p.parse_args()
    done = set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["case"], r["M"], r["K"], r["mode"]))
    groups = [g for g in cells() if a.only is None or a.only in g[0]]
    q = queue.Queue()
    for g in groups: q.put(g)
    lock = threading.Lock()
    env = dict(os.environ, LD_LIBRARY_PATH="/home/eellison/.conda/envs/pytorch-3.12/lib", TORCHINDUCTOR_FORCE_DISABLE_CACHES="1")
    def worker(gpu):
        while True:
            try: case, M, K, modes = q.get_nowait()
            except queue.Empty: return
            for mode in modes:
                key = (case, M, K, mode)
                if key in done: continue
                t0 = time.time()
                try:
                    proc = subprocess.run([sys.executable, str(HERE / "cell.py"), case, str(M), str(K), mode],
                                          env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), capture_output=True, text=True, timeout=a.timeout, cwd=str(HERE.parent.parent))
                    lines = [l for l in proc.stdout.splitlines() if l.startswith("CELL_RESULT ")]
                    if lines: r = json.loads(lines[-1][len("CELL_RESULT "):])
                    else: r = {"case": case, "M": M, "K": K, "mode": mode, "error": (proc.stderr or proc.stdout)[-800:]}
                except subprocess.TimeoutExpired:
                    r = {"case": case, "M": M, "K": K, "mode": mode, "error": "timeout"}
                r["gpu"] = gpu; r["wall_s"] = round(time.time() - t0, 1)
                with lock:
                    with a.output.open("a") as f: f.write(json.dumps(r) + "\n")
                    print(f"[gpu{gpu}] {case} {M}x{K} {mode}: " + (f"{r['median_us']:.2f}us k={r.get('kernels')} n={r.get('nested')}" if "median_us" in r else f"ERROR {r['error'][:120]}"), flush=True)
                time.sleep(a.cooldown)
    threads = [threading.Thread(target=worker, args=(int(g),)) for g in a.gpus.split(",")]
    for t in threads: t.start()
    for t in threads: t.join()

if __name__ == "__main__":
    main()
