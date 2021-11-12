import torch

@torch.jit.script
def foo(x, y):
    return (x + .5 * y).exp()


x = torch.randn(4, 4, dtype=torch.float)
y = torch.randn(4, 4, dtype=torch.float)
for _ in range(10):
    foo(x, y)


# @torch.jit.script
# def foo(x):
#     y = torch.zeros([4])
#     return x + y

# torch._C._jit_pass_constant_propagation(foo.graph)
# inp = next(foo.graph.inputs())
# inp.setType(inp.type().with_sizes([None]))
# out = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(foo.graph)
# g = out.partial_eval_shape_graph()
# g.makeMultiOutputIntoTuple()
# func = torch._C._create_function_from_graph("partial_eval_graph", g)
# print(func.code)
# print(foo.graph)
