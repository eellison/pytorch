# per-step cost of ARMV_PIN_UNPINNED's pin_memory() of GDN's nums (int64, one element per prefill request)
import time, torch
torch.cuda.init()
for n in (1, 9, 65):
    qsl = torch.arange(n + 1, dtype=torch.int32) * 100
    nums = -(-qsl.diff() // 8)
    for _ in range(200):
        nums.pin_memory()
    t = time.perf_counter()
    for _ in range(2000):
        nums.pin_memory()
    print(n, "requests: pin_memory", round((time.perf_counter() - t) / 2000 * 1e6, 2), "us", nums.dtype, nums.is_pinned())
