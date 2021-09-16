
import torch

@torch.jit.script
def broadcast_to_one(x: int):
    if x == 1:
        return 1
    else:
        return x

print(broadcast_to_one.graph)
torch._C._jit_pass_refine_integer_values(broadcast_to_one.graph)
torch._C._jit_pass_dce(broadcast_to_one.graph)
print(broadcast_to_one.graph)


# x: List[int] = []
# x.append(y)
# x.append(z)
# ->
# x = [y, z]

# x: List[int] = [y, z]
# a = x[0]
# ->
# a = y


# mod nn.Conv2d(16, 33, 3, stride=2)), torch.randn(20, 16, 5, 10))

# def foo(x, y):
#     for _ in range(len(y)):
#         x = x * x
#     return x

# # torch.jit.trace(foo, (torch.rand([10, 10]), torch.rand([2, 2])))

# with torch.jit._hide_source_ranges():
#     print(torch.jit.trace(foo, (torch.rand([10, 10]), torch.rand([2, 2]))).graph)

