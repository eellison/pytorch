import torch

inp = torch.randn(2**10, requires_grad=True)

def gelu_bias(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

f = torch.jit.script(gelu_bias)
# f = torch.jit.trace(gelu_bias, (inp, inp))
print(f.graph)