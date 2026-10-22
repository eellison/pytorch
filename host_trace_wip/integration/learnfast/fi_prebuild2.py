"""CPU prebuild of the other FlashInfer JIT modules the Flash-Next ladder loads (paths moved: their build.ninja differs)."""
import time
from flashinfer.jit.sampling import gen_sampling_module
from flashinfer.jit.tllm_utils import gen_trtllm_utils_module
from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module

for gen in (gen_sampling_module, gen_trtllm_utils_module, gen_trtllm_gen_fmha_module):
    t = time.time()
    spec = gen()
    spec.build(verbose=False, need_lock=True)
    print("built", spec.get_library_path(), round(time.time() - t, 1), "s", flush=True)
