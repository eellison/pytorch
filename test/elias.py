import torch
import torch._inductor
from torch._inductor import config

config.max_autotune_gemm_backends = "TRITON"
import torch.utils.benchmark as benchmark

# config.benchmark_kernel = True
config.benchmark_fusion = True


def benchmark_torch_function_in_microseconds(func, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


@torch.compile(mode="max-autotune-no-cudagraphs")
def foo(m, inp):
    return torch.nn.functional.gelu(m(inp))

# with torch.no_grad():
#     m = torch.nn.Linear(1028, 1028).half().cuda()
#     inp = torch.rand([4096, 1028]).half().cuda()

#     print(benchmark_torch_function_in_microseconds(foo, m, inp))

@torch.compile()
def f(a, b):

    a = torch.cat([torch.softmax(a, dim=-1) + 1, torch.softmax(a, dim=-1) + 1]) + 1
    a = a[0:10, 0:10]
    return torch.cat([a, b])


f(torch.rand([10, 10], device="cuda"), torch.rand([10, 10], device="cuda"))


# torch._dynamo.reset()

# config.epilogue_fusion = False

# print(benchmark_torch_function_in_microseconds(foo, m, inp))