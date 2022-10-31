import torch

import torch._dynamo


@torch._dynamo.optimize("inductor")
def foo(x):
    return x + x + 1


inps = torch.rand([4]).cuda()
print(foo(inps))