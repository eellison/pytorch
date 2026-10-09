# Runs this lane's GPU tests in one process sequence (one lock hold): the trace tests (Qwen3-8B context/decode, R1,
# the guard-region sweep) and the parity grid. Each file's result is printed; exit status is nonzero if any failed.
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
rc = 0
for f in sys.argv[1:] or ["test_trtllm_cpp_trace.py", "test_trtllm_cpp_parity.py"]:
    print(f"==== {f}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, f), "-v"])
    print(f"==== {f} rc={r.returncode}", flush=True)
    rc |= r.returncode
sys.exit(rc)
