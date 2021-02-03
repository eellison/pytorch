
@torch.jit.script
def foo(x):
    return x + x + x

foo(torch.rand([4, 4], device='cuda'))
