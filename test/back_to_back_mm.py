import torch
import torch.utils.flop_counter
from torch._dynamo.testing import rand_strided
from torch.utils.flop_counter import FlopCounterMode

def foo(inp, w1, w2):
    breakpoint()
    return torch.nn.functional.gelu(inp @ w1) @ w2


inp = rand_strided((3136, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
w1 = rand_strided((512, 2048), (1, 512), device='cuda:0', dtype=torch.bfloat16)
w2 = rand_strided((2048, 512), (1, 2048), device='cuda:0', dtype=torch.bfloat16)

foo(inp, w1, w2)

# def get_flops(dtype):
#     from triton.testing import get_max_simd_tflops, get_max_tensorcore_tflops

#     assert dtype in (torch.float16, torch.bfloat16, torch.float32)
#     if dtype in (torch.float16, torch.bfloat16):
#         return get_max_tensorcore_tflops(dtype)

#     if torch.backends.cuda.matmul.allow_tf32:
#         return get_max_tensorcore_tflops(torch.float32)
#     else:
#         return get_max_simd_tflops(torch.float32)


# flop_counter = FlopCounterMode()
# with flop_counter:
#     out = inp @ w1

# with flop_counter:
#     out @ w2
#     # foo(inp, w1, w2)

# import torch._inductor
# from torch._inductor.utils import do_bench

# time = (do_bench(lambda: inp @ w1))
# flops = 6576.669 * 1000000 / (time)

# total_flops = get_flops(torch.bfloat16) * 1000000000

# import torch._inductor.fx_passes.pad_mm
# torch._inductor.fx_passes.pad_mm.is_mm_compute_bound(3136, 512, 2048, torch.float16)

# breakpoint()
# print("hi")
