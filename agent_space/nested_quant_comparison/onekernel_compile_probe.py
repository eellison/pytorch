import torch

from bench_main_cuda import compile_main
from onekernel_probe import pad_input


x = torch.randn(129, 4096, dtype=torch.bfloat16, device="cuda")
weight = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
_, output, metadata = compile_main(
    lambda a, b: pad_input(a, b, 16, "e4m3"),
    x,
    weight,
    coordinate_descent=False,
)
print(metadata, tuple(value.shape for value in output))
