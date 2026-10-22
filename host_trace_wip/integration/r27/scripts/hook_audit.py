# hook_audit.py: what a vLLM default-arm process has installed before any trace: Triton launch hooks, host-trace modules,
# patched torch.cuda functions
import sys
import torch
from triton import knobs

def state(tag):
    ht = sorted(m for m in sys.modules if "host_trace" in m or "cudagraph_host_trace" in m)
    print(tag, "| launch_enter_hook:", knobs.runtime.launch_enter_hook, "| launch_exit_hook:", knobs.runtime.launch_exit_hook,
          "| jit_post_compile_hook:", knobs.runtime.jit_post_compile_hook, "| host-trace modules:", ht,
          "| current_device patched:", torch.cuda.current_device.__module__ != "torch.cuda")

state("after torch+triton")
import vllm  # noqa: F401
from vllm import LLM  # noqa: F401
state("after vllm")
import vllm.v1.worker.gpu.model_runner  # noqa: F401
state("after model_runner")
for name in ("flashinfer", "flashinfer.gdn_prefill"):
    try:
        __import__(name)
    except Exception as e:
        print(name, "import failed", type(e).__name__)
state("after flashinfer")
