import torch
@torch.jit.script
def foo(x: bool, y: bool):
	return x != y

print(foo.graph)