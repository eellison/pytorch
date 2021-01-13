import torch
from typing import List
from torch import nn

@torch.jit.script
def conv2d_output_shape(input: List[int], weight: List[int], stride: List[int], padding: List[int], dilation: List[int], groups: int):
    n = input[0]
    out_channels = weight[0]
    hin = input[2]
    kernel_size_0 = weight[2]
    kernel_size_1 = weight[3]
    win = input[3]

    # compute
    hout = ((hin + 2 * padding[0] - dilation[0] * (kernel_size_0 - 1) - 1) // 2) + 1
    wout = ((win + 2 * padding[1] - dilation[1] * (kernel_size_1 - 1) - 1) // stride[1]) + 1
    return [n, out_channels, hout, wout]


# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
# m = torch.jit.script(m)
# torch._C._jit_pass_inline(m.graph)
# # torch._C._jit_pass_symbolic_shape_analysis(m.graph.findNode("aten::conv2d"), conv2d_output_shape.graph)
# m = torch.jit.freeze(m.eval())
# torch._C._jit_pass_symbolic_shape_analysis(m.graph.findNode("aten::conv2d"), conv2d_output_shape.graph)
# print(m.graph)

def broadcast_dim(dim1: int, dim2: int):
    if dim1 == dim2:
        return dim2
    if dim1 == 1:
        return dim2
    elif dim2 == 1:
        return dim1
    else:
        raise Exception("Dims could not be broadcast, {}, {}".format(dim1, dim2))

@torch.jit.script
def broadcast(x: List[int], y: List[int]):
    x_len = len(x)
    y_len = len(y)
    z : List[int] = []
    max_len = max(x_len, y_len)
    min_len = min(x_len, y_len)
    for i in range(min_len):
        z.append(broadcast_dim(x[-(1 + i)], y[-(1 + i)]))
    for i in range(len(x) - len(y)):
        z.append(x[-(1 + i + len(y))])
    for i in range(len(y) - len(x)):
        z.append(y[-(1 + i + len(x))])

    z_reversed : List[int] = []
    for i in range(max_len):
        z_reversed.append(z[-(1 + i)])

    return z_reversed

@torch.jit.script
def foo(x, y):
    return x * y

# print(broadcast([8, 1, 6, 1], [7, 1, 5]))
# torch._C._jit_pass_symbolic_shape_analysis(foo.graph.findNode("aten::mul"), broadcast.graph)
torch._C._jit_pass_inline(broadcast.graph)
inputs = list(foo.graph.inputs())
inputs[0].setType(inputs[0].type().with_sizes([8, None, 6, 1]))
inputs[1].setType(inputs[1].type().with_sizes([7, 1, 5]))
torch._C._jit_pass_symbolic_shape_analysis(foo.graph.findNode("aten::mul"), broadcast.graph)

print(foo.graph)


# input = torch.randn(20, 16, 50, 100)
# output = m(input)
# print(output.size())
# print("HELELO")
# print(foo.graph)
