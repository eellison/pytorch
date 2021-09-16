import torch
import torchvision


resnet = torchvision.models.resnet18()
resnet = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool, resnet.layer1)
frozen = torch.jit.freeze(torch.jit.script(resnet.eval()))
torch._C._jit_pass_remove_mutation(frozen.graph)
torch._C._jit_pass_propagate_shapes_on_graph(frozen.graph)
torch._C._jit_pass_peephole(frozen.graph)
torch._C._jit_pass_constant_pooling(frozen.graph)
torch._C._jit_pass_constant_propagation(frozen.graph)
with torch.jit._hide_source_ranges():
    print(frozen.graph)
    inps = list(frozen.graph.inputs())
    inps[1].setType(inps[1].type().with_sizes([1, 3, 224, 224]))
    torch._C._jit_pass_propagate_shapes_on_graph(frozen.graph)
    print(frozen.graph)