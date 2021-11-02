import torch
@torch.jit.script
def foo(x):
    return x + 3

print(foo.graph)