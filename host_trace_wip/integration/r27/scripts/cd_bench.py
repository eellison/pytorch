# cd_bench.py [ht]: per-call cost of torch.cuda.current_device() / _lazy_init() / triton's driver get_current_device after CUDA init;
# with "ht", the host-trace modules imported first (no trace active)
import sys, timeit
import torch
if "ht" in sys.argv:
    import torch.cuda._host_trace  # noqa: F401
    import torch.cuda._host_trace_replay  # noqa: F401
    import torch.cuda._host_trace_cute  # noqa: F401
torch.cuda.init()
torch.empty(1, device="cuda")
from triton.runtime.driver import driver
gcd = driver.active.get_current_device
n = 200000
for name, fn in (("current_device", torch.cuda.current_device), ("_lazy_init", torch.cuda._lazy_init), ("triton get_current_device", gcd),
                 ("python no-op call", lambda: None)):
    t = min(timeit.repeat(fn, number=n, repeat=5)) / n
    print(f"{'ht' if 'ht' in sys.argv else 'plain'} {torch.__version__} {name}: {t * 1e6:.3f} us/call", flush=True)
