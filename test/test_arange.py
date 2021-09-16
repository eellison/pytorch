import torch
from torch import nn

mod = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
mod = torch.jit.freeze(torch.jit.script(mod.eval()))

with torch.jit._hide_source_ranges():
    inps = list(mod.graph.inputs())
    inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))
    g = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
    print(mod.graph)
    func = torch._C._create_function_from_graph("partial_eval_graph", g)
    print(func.code)
    print(g)
