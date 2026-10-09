# lockwrap.py DRIVER [args]: run DRIVER (runpy) holding LOCKWRAP_LOCK (flock -x) only from after the heavy imports to
# the end, with a watchdog that ends the process LOCKWRAP_TMO seconds after the lock is taken. The imports (torch, the
# adapter when ARMF_ADAPTER_DIR / LOCKWRAP_ADAPTER name one, sglang) take several minutes on a CPU-saturated host and use
# no GPU, so the GPU hold covers model load + the run only. CUDA is not initialized before the lock.
import fcntl
import os
import runpy
import sys
import threading
import time

t0 = time.time()
driver, rest = sys.argv[1], sys.argv[2:]
import torch  # noqa: E402,F401

if os.environ.get("ARMF_ADAPTER_DIR") and os.environ.get("LOCKWRAP_ADAPTER"):
    sys.path.insert(0, os.environ["ARMF_ADAPTER_DIR"])
    __import__(os.environ["LOCKWRAP_ADAPTER"])  # before sglang: arms the CuTe route before any CuTe compile
import sglang.benchmark.one_batch  # noqa: E402,F401

# FlashInfer's gdn_kernels read device properties at import (torch.cuda lazy init); what matters for the lock is
# whether this process holds a context (GPU memory) before taking it
import subprocess  # noqa: E402

apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
mine = [l for l in apps.splitlines() if l.split(",")[0].strip() == str(os.getpid())]
print(f"LOCKWRAP before lock: cuda initialized {torch.cuda.is_initialized()}, own GPU context {mine or 'none'}", flush=True)
if mine and os.environ.get("LOCKWRAP_STRICT", "1") == "1":
    raise SystemExit("LOCKWRAP: a GPU context exists before the lock")
t1 = time.time()
lock = open(os.environ["LOCKWRAP_LOCK"], "a")
fcntl.flock(lock, fcntl.LOCK_EX)
t2 = time.time()
print(f"LOCKWRAP imports {t1 - t0:.0f} s, lock wait {t2 - t1:.0f} s, locked {time.strftime('%H:%M:%S')} load "
      f"{os.getloadavg()[0]:.0f}", flush=True)
tmo = float(os.environ.get("LOCKWRAP_TMO", "600"))


def _expire():
    print(f"LOCKWRAP timeout after {tmo:.0f} s of hold", flush=True)
    os._exit(124)


if tmo > 0:
    timer = threading.Timer(tmo, _expire)
    timer.daemon = True
    timer.start()
sys.argv = [driver] + rest
sys.path.insert(0, os.path.dirname(os.path.abspath(driver)))  # as `python DRIVER` would
rc = 0
try:
    runpy.run_path(driver, run_name="__main__")
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException:
    import traceback

    traceback.print_exc()
    rc = 1
print(f"LOCKWRAP held {time.time() - t2:.0f} s rc {rc}", flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(rc)  # the lock goes with the process; sglang's threads do not hold it open
