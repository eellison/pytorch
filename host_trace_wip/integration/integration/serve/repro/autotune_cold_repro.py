# Core repro (serve lane, 10-12): an Inductor kernel whose FIRST call ever happens inside a host trace (as a vLLM compile range
# first used after the entry's first, eager, call) on a cold cache. Its CachingAutotuner has several configs (max_autotune_pointwise)
# and tunes at its first call. Under the trace the core's Inductor seam (_InductorKernel, installed as vllm_ports/inductor_hosts.py
# does) declines it with retry=True; the call runs eagerly; the next call of the class should trace with the one tuned launcher.
# Two artifacts compiled with standalone_compile (no Dynamo under the trace, as vLLM's aot_compiled forward); the step picks one by
# its input's size (a guard). Prints per call: traces / replays / eager, each autotuner's launcher count, the declines.
#   CUDA_VISIBLE_DEVICES=N CAND_LINE=c5g bash python_vllm_cand2.sh serve/repro/autotune_cold_repro.py
import os
import sys
import tempfile

os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="ind_cold_")  # cold: nothing tuned on disk
import torch
import torch._inductor.config as icfg
import torch.cuda._host_trace_replay as R
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

sys.path.insert(0, "/data/eellison/src/pytorch/agent_space/paramgraph/land/scratch/integration/vllm/vllm_ports")
import inductor_hosts  # noqa: E402

icfg.max_autotune_pointwise = True  # several configs: tuned at the first call
icfg.force_disable_caches = False
print("seams", inductor_hosts.install(), flush=True)
TUNERS = {}
_init = CachingAutotuner.__init__


def _track(self, *a, **k):
    _init(self, *a, **k)
    TUNERS[id(self)] = self


CachingAutotuner.__init__ = _track


def f(x, w):
    return (torch.sin(x) * w + torch.cos(x)).relu()


def artifact(n):
    x, w = torch.randn(n, 4096, device="cuda"), torch.randn(4096, device="cuda")
    gm = torch.fx.symbolic_trace(f)
    return torch._inductor.standalone_compile(gm, [x, w], dynamic_shapes="from_example_inputs")


A, B = artifact(64), artifact(128)  # compiled (not run) before any trace


def step(x, w):
    return ((A if x.shape[0] == 64 else B)(x, w)[0] * 1,)


entry = R.HostTraceReplay(step, memory="run_buffer")
w = torch.randn(4096, device="cuda")
for i, n in enumerate((64, 64, 128, 128, 128, 128, 64)):
    x = torch.randn(n, 4096, device="cuda")
    (got,) = entry(x, w)
    print(f"call {i} n={n}: traces {entry.traces} replays {entry.replays} eager {entry.eager} variants {len(entry.variants)} "
          f"launchers per autotuner {[len(t.launchers) for t in TUNERS.values()]} bitwise {torch.equal(got, f(x, w))} "
          f"declines {sorted({str(d)[:160] for d in entry.declines})[:3]}", flush=True)
