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