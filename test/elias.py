import torch
from typing import List
from torch import nn


class MaxPool2dGenerator(nn.Module):
	__constants__ = ["kernel_len", "stirde_len", "padding_len", "dilation_len", "ceil_mode"]

	def __init__(self, kernel_len, stride_len, padding_len, dilation_len, ceil_mode):
		super(MaxPool2dGenerator, self).__init__()

		self.kernel_len = kernel_len
		self.stride_len = stride_len
		self.padding_len = padding_len
		self.dilation_len = dilation_len
		self.ceil_mode = ceil_mode

	def rand_intarray(self, length: int):
		return [int(torch.randint(0, 10)) for i in range(self.kernel_len)]

	def forward(self, input):
		kernel = self.rand_intarray(self.kernel_len)
		stride = self.rand_intarray(self.stride_len)
		padding = self.rand_intarray(self.padding_len)
		dilation = self.rand_intarray(self.dilation_len)
		ceil_mode = self.ceil_mode
		return torch.max_pool2d(input, kernel, stride, padding, dilation, ceil_mode)

@torch.jit.script
def foo(input, kernel: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool):
	return torch.max_pool2d(input, kernel, stride, padding, dilation, ceil_mode)

shape_compute_graph = torch._C._jit_get_shape_compute_graph_for_node(foo.graph.findNode("aten::max_pool2d"))
# import pdb; pdb.set_trace()
torch._C._augment_with_length(shape_compute_graph, 0, 4)
for i in range(1, 5):
	torch._C._augment_with_length(shape_compute_graph, i, 2)
print(shape_compute_graph)
# import pdb; pdb.set_trace()

changed = True
while changed:
	changed = torch._C._jit_pass_shape_graph_cleanup_passes(shape_compute_graph)
print(shape_compute_graph)
# import pdb; pdb.set_trace()


@torch.jit.script
def foo(input, list_int: List[int]):
	return input.expand(list_int)

shape_compute_graph = torch._C._jit_get_shape_compute_graph_for_node(foo.graph.findNode("aten::expand"))
# import pdb; pdb.set_trace()
torch._C._augment_with_length(shape_compute_graph, 0, 4)
for i in range(1, 2):
	torch._C._augment_with_length(shape_compute_graph, i, 2)
print(shape_compute_graph)
# import pdb; pdb.set_trace()

changed = True
while changed:
	changed = torch._C._jit_pass_shape_graph_cleanup_passes(shape_compute_graph)
print(shape_compute_graph)

import pdb; pdb.set_trace()

mod = torch.jit.script(MaxPool2dGenerator(2, 2, 2, 2, True))
mod = torch.jit.freeze(mod.eval())
torch._C._jit_pass_constant_loop_unrolling(mod.graph)
torch._C._jit_pass_remove_mutation(mod.graph)

inps = list(mod.graph.inputs())
inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))

g = (torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph, mod.graph.findNode("aten::max_pool2d")))
print(torch._C._create_function_from_graph("forward", g).code)
import pdb; pdb.set_trace()
print(g)

