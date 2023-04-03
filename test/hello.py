import torch

@torch.compile
def foo(x):
    return torch.nn.functional.dropout((x + x), .5) * 10


inp = torch.rand([20, 20], requires_grad=True, device="cuda")

out = torch.rand([20, 20], requires_grad=True, device="cuda")
# foo(inp).backward(out)



# @torch.compile()
# def foo(x, y, z):
#     out = z.add_(1)
#     a = x[y]
#     b = x + 3
#     return a + 4, b, out

# b = torch.tensor([4], device="cuda")
# foo(inp, b, torch.rand([20], device="cuda"))


@torch.compile()
def foo(x):
    return x + 5

foo(torch.rand([4], device="cuda"))

# import torch
# import torchvision

# resnet = torchvision.models.resnet18().cuda()

# @torch.compile
# def foo(m, x):
#     return m(x)

# inp = torch.rand([1, 3, 255, 255], requires_grad=True, device=)
# foo(resnet, inp).sum().backward()

# # inp = torch.rand([20, 20], requires_grad=True, device="cuda")
# # out = torch.rand([20, 20], requires_grad=True, device="cuda")
# # foo(inp).backward(out)

