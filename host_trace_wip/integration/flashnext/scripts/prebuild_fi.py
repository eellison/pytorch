"""CPU-only: FlashInfer's own JIT builds, in this lane's workspace, of the modules the stock runs load (the copy of
trtllm_moe_cpp/fi_ws rebuilds, since build.ninja carries the workspace path). MAX_JOBS from the launcher."""
import sys
import time

from flashinfer.fused_moe.core import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.jit.fp4_quantization import gen_fp4_quantization_sm103_module
from flashinfer.jit.sampling import gen_sampling_module
from flashinfer.jit.tllm_utils import gen_trtllm_utils_module

GENS = {"moe": lambda: gen_trtllm_gen_fused_moe_sm100_module(enable_rubin=False), "fp4q": gen_fp4_quantization_sm103_module,
        "sampling": gen_sampling_module, "utils": gen_trtllm_utils_module}
for w in sys.argv[1:] or list(GENS):
    t = time.time()
    spec = GENS[w]()
    spec.build()
    print(w, spec.get_library_path(), f"{time.time() - t:.0f}s", flush=True)
