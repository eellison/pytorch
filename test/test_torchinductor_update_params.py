import torch
from torch import empty_strided, as_strided
import torch._dynamo


@torch._dynamo.optimize()
def foo(x, y):
    return x + y + 1 + 1


inps = [torch.rand([20, 20]).cuda() for _ in range(2)]

foo(*inps)

# def model(a, b):
#     out = torch.matmul(a, b)
#     return out

# static_args = [torch.rand([1024, 1024]).cuda() for _ in range(2)]
# static_outputs = model(*static_args)

# stream = torch.cuda.Stream()
# stream.wait_stream(torch.cuda.current_stream())
# # warm up
# with torch.cuda.stream(stream):
#     for _ in range(3):
#         static_outputs = model(*static_args)
# torch.cuda.current_stream().wait_stream(stream)

# graph = torch.cuda.CUDAGraph()
# with torch.cuda.graph(graph, stream=stream):
#     static_outputs = model(*static_args)

# graph.replay()
# torch.cuda.synchronize()
# new_args = [torch.rand([1024, 1024]).cuda() for _ in range(3)]
# graph.update_params(static_args + [static_outputs], new_args)
