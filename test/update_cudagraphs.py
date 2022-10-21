
import torch
from torch import empty_strided, as_strided

def model(a, b, c):
    out = ((a + b) / c)
    return out @ out

static_args = [torch.rand([3, 3]).cuda() for _ in range(3)]
model(*static_args)
torch.cuda.synchronize()
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
# copy static_inputs because it will be cleared in model
with torch.cuda.stream(stream):
    model(*static_args)
stream.synchronize()
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
abc = torch.cuda.graph(graph, stream=stream)
with abc:
    static_outputs = model(*static_args)

graph.replay()
new_args = [torch.rand([3, 3]).cuda() for _ in range(3)]
graph.update_params(static_args, new_args)
