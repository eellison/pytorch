import torch

@torch.jit.script
def foo(x, i: int):
    y = x.view([-1])
    a = torch.nn.functional.gelu(torch.tanh(y))
    return torch.mul(a, y)

torch._C._jit_pass_inline(foo.graph)
torch._C._jit_pass_constant_propagation(foo.graph)
torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
with torch.jit._hide_source_ranges():
    print(foo.graph)