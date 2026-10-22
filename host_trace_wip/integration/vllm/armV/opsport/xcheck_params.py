# Per-T launch cross-check: trace each case at every T of the sweep; the launcher's witness compares its launch
# (function, name, grid, block, smem, attributes, PDL, parameter layout and bytes) with the real op's, captured at the
# same T, and declines on any difference. A KernelLaunch record at T means byte-identical at T.
#   ... python_vllm_cpp.sh armV/opsport/xcheck_params.py [CASE...]
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_trace as T  # noqa: E402

import vllm_ops_trace  # noqa: E402
from torch.cuda._host_trace_launch import KernelLaunch  # noqa: E402
from torch.cuda._host_trace_tape import trace  # noqa: E402

SWEEP = sorted({*range(1, 257), *range(257, 8193, 61), *(p + d for p in (2**i for i in range(9, 14)) for d in (-1, 0, 1) if p + d <= 8192)})


def main():
    names = sys.argv[1:] or list(T.OPS)
    vllm_ops_trace.install()
    summary = {}
    for n in names:
        case, f = T.OPS[n]
        good, bad = 0, Counter()
        t0 = time.time()
        for t in SWEEP:
            args, _ = case(t)
            try:
                recs = [r for _, r in trace(f, args).launches]
            except Exception as e:
                bad[f"{type(e).__name__}: {str(e)[:200]}"] += 1
                continue
            if len(recs) == 1 and isinstance(recs[0], KernelLaunch):
                good += 1
            else:
                bad["; ".join(str(getattr(r, "reason", type(r).__name__))[:300] for r in recs)] += 1
        summary[n] = (good, len(SWEEP))
        print(f"XCHECK {n} {good}/{len(SWEEP)} equal ({time.time() - t0:.0f} s)", flush=True)
        for why, c in bad.most_common(5):
            print(f"  {c}x {why}", flush=True)
    print("XCHECK", summary)


if __name__ == "__main__":
    main()
