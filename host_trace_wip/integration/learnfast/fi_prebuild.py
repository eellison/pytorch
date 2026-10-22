"""Build the FlashInfer trtllm-gen fused-MoE JIT module into FLASHINFER_WORKSPACE_BASE (learnfast/fncache/fi_ws) on the
CPU (no GPU), under FlashInfer's own lock, so the GPU runs find it built."""
import time
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

t = time.time()
spec = gen_trtllm_gen_fused_moe_sm100_module(enable_rubin=False)
spec.build(verbose=False, need_lock=True)
print("built", spec.get_library_path(), round(time.time() - t, 1), "s", flush=True)
