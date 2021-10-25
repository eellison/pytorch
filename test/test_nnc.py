import torch

def mul_mul_add(a, b, c):
    return (a * b) + (a * c)

jit_mul_mul_add = torch.jit.script(mul_mul_add)

a = torch.randn(64,8,256,162)
b = torch.randn(256,162)
c = torch.randn(256,162)

jit_mul_mul_add(a, b, c)
jit_mul_mul_add(a, b, c)
jit_mul_mul_add(a, b, c)