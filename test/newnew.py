import torch

import torchvision.models as models
resnet18 = models.resnet18()


mod = torch.jit.freeze(torch.jit.script(resnet18.eval()))
torch._C._jit_pass_remove_mutation(mod.graph)
torch._C._jit_pass_propagate_shapes_on_graph(mod.graph)
torch._C._jit_pass_peephole(mod.graph)
torch._C._jit_pass_constant_propagation(mod.graph)
torch._C._jit_pass_constant_pooling(mod.graph)
g = (torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph))

with torch.jit._hide_source_ranges():
	g = (torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph, convs[0]))
	func = torch._C._create_function_from_graph("forward", g)
	import pdb; pdb.set_trace()
	inputs = list(mod.graph.inputs())
	inputs[1].setType(inputs[1].type().with_sizes([1, 3, 224, 224]))
	torch._C._jit_pass_propagate_shapes_on_graph(mod.graph)
	import pdb; pdb.set_trace()
	print(g)