# Guards and launch records of one trace per case at T (debug): python_vllm_cpp.sh dbg_guards.py T [CASE...]
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_trace as T  # noqa: E402

import vllm_ops_trace  # noqa: E402
from torch.cuda._host_trace_tape import trace  # noqa: E402

vllm_ops_trace.install()
t = int(sys.argv[1])
for n in sys.argv[2:] or list(T.OPS):
    case, f = T.OPS[n]
    tape = trace(f, case(t)[0])
    print(n, "T", t, "guards:", [str(g) for g in tape.guards])
    for _, r in tape.launches:
        print("  ", getattr(r, "name", None), getattr(r, "grid", None), getattr(r, "block", None), str(getattr(r, "reason", ""))[:3000])
