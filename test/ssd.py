# run with PYTORCH_JIT_TYPE_VERBOSITY=4 python example_resnet.py

import torch
import torchvision

resnet = torchvision.models.resnet18()

resnet_frozen = torch.jit.freeze(torch.jit.script(resnet.eval()))
# make it so ops like add_ can be propagated (until https://github.com/pytorch/pytorch/pull/65729 lands)
torch._C._jit_pass_remove_mutation(resnet_frozen.graph)

# until https://github.com/pytorch/pytorch/issues/65643 lands to clean up control flow..
torch._C._jit_pass_propagate_shapes_on_graph(resnet_frozen.graph)
torch._C._jit_pass_peephole(resnet_frozen.graph)
torch._C._jit_pass_constant_propagation(resnet_frozen.graph)

inps = list(resnet_frozen.graph.inputs())
# None creates a new dynamic dimension,
# alternative inputs:
# [None, 3, 255, 255] - batch dimension not specified
# sym_shape = torch._C._new_symbolic_shape_symbol(); [1, 3, sym_shape, sym_shape] - same width/height dimension
inps[1].setType(inps[1].type().with_sizes([None, 3, None, None]))
shape_compute_graph = torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(resnet_frozen.graph)

g = shape_compute_graph.partial_eval_shape_graph()
print("RESNET GRAPH \n\n")
print(resnet_frozen.graph)
print("Shape Compute Graph \n\n")
# here is the encoding of shape arithmetic from the input, represnted as TS graph
print(g)
print("Mapping from Sym Dim to Shape Compute Graph Output")
for key, value in shape_compute_graph.graph_output_to_symbolic_shape_dim().items():
    print(str(value) + " :", key)


# to execute jit function it must have a single output
g.makeMultiOutputIntoTuple()
func = torch._C._create_function_from_graph("partial_eval_graph", g)
print("Calculating dims from input", func([1, 3, 255, 29]))