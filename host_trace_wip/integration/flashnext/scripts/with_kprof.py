"""Kernel identity, every size (user, 10-14: "don't sample batch sizes"): run a driver (argv[1], its arguments after) with
one Kineto-profiled engine step per step class, the GPU kernel-name multiset of each saved to FN_KPROF_OUT (JSON).
Step class = (kind, requests, scheduled tokens, context bucket): kind decode (every request schedules 1 token), prefill (one
request, no decoders) or mixed; context bucket = the largest computed-token count among the step's requests, rounded down to
a power of two (separates the 1K and 8K decode curves). Each class is profiled at its FN_KPROF_NTH-th occurrence (default 3,
past first-call work), once. Used with the d (full bs 1-128 ladder, both contexts, few steps per point) and w (every prefill
and mixed size) workloads on every arm; scripts/kernel_identity_all.py compares the files. GPUModelRunner.execute_model is
wrapped outside the driver's own timing wrapper; the profiled steps are not timed (KIND=idall runs are untimed).
    python_fn.sh scripts/with_kprof.py DRIVER.py ARGS..."""
import collections
import json
import os
import runpy
import sys

OUT = os.environ["FN_KPROF_OUT"]
NTH = int(os.environ.get("FN_KPROF_NTH", "3"))
SEEN: collections.Counter = collections.Counter()
RESULT: dict = {}


def _install():
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    orig = GPUModelRunner.execute_model

    def step_class(so):
        n = len(so.num_scheduled_tokens)
        tokens = so.total_num_scheduled_tokens
        if n == 0 or tokens == 0:
            return None
        computed = [r.num_computed_tokens for r in so.scheduled_new_reqs] + list(so.scheduled_cached_reqs.num_computed_tokens)
        ctx = max(computed) if computed else 0
        bucket = 1 << (ctx.bit_length() - 1) if ctx > 0 else 0
        kind = "decode" if tokens == n else "prefill" if n == 1 else "mixed"
        return f"{kind}|reqs{n}|tok{tokens}|ctx{bucket}"

    def execute_model(self, so, *args, **kw):
        key = None if kw.get("dummy_run") else step_class(so)
        if key is None or key in RESULT:
            return orig(self, so, *args, **kw)
        SEEN[key] += 1
        if SEEN[key] != NTH:
            return orig(self, so, *args, **kw)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = orig(self, so, *args, **kw)
            torch.cuda.synchronize()
        names = collections.Counter()
        other = collections.Counter()
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                (other if e.name.startswith(("Memcpy", "Memset")) else names)[e.name] += 1
        RESULT[key] = {"kernels": dict(names), "memops": dict(other)}
        with open(OUT + ".tmp", "w") as f:
            json.dump(RESULT, f)
        os.replace(OUT + ".tmp", OUT)
        return out

    GPUModelRunner.execute_model = execute_model


_install()
print(f"FN_KPROF_OUT {OUT} (class profiled at occurrence {NTH})", flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
