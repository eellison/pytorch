import torch
from torch import empty_strided, as_strided
from torch._inductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx

torch._inductor.config.triton.cudagraphs = False  

def model(a, b):
    out = torch.add(a, b)
    return (out,)

static_args = [torch.rand([100, 100]).cuda() for _ in range(2)]

fn_fx = make_fx(model)(static_args[0], static_args[1])
fn_compiled = compile_fx_inner(fn_fx, static_args)
# def fn_compiled(args):
#     return model(*args)


stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
# warm up
with torch.cuda.stream(stream):
    for _ in range(3):
        static_outputs = fn_compiled(list(static_args))
torch.cuda.current_stream().wait_stream(stream)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    static_outputs = fn_compiled(list(static_args))

graph.replay()
torch.cuda.synchronize()
new_args = [torch.ones([100, 100]).cuda() for _ in range(2)]
print("New args: ", new_args[0].data_ptr())
graph.update_params(static_args + list(static_outputs), new_args)
graph.replay()
print(static_outputs)
