import torch


@torch.jit.script
def foo():
	x = torch.tensor([2, 2])
	x.add_(1)
	x.add_(3)
	y = x + 4
	return x, y

torch._C._jit_pass_remove_mutation(foo.graph)
with torch.jit._hide_source_ranges():
	print(foo.code)


# @torch.jit.script
# def foo(i: int):
# 	x = [i, i]
# 	y = [i, i]
# 	y.append(3)
# 	out_sum = sum(x) + sum(y)
# 	return out_sum

# torch._C._jit_pass_cse(foo.graph)

# %output : Tensor = aten::view(%self, %size)

# view(Tensor(a) self, int[] size) -> Tensor(a)

# @torch.jit.script
# def foo(a : Tensor, b : Tensor):
#   c = 2 * b
#   d = c + 1
#   a += 1
#   if a.item() > 4:
#     r = a[0]
#   else:
#     r = b[0]
#   ret = [d, r]
#   return ret


# graph(%a.1 : Tensor,
#       %b.1 : Tensor):
#   %2 : int = prim::Constant[value=2]()
#   %6 : int = prim::Constant[value=1]()
#   %14 : int = prim::Constant[value=4]()
#   %17 : int = prim::Constant[value=0]()
#   %c.1 : Tensor = aten::mul(%b.1, %2)
#   %d.1 : Tensor = aten::add(%c.1, %6, %6)
#   %a.3 : Tensor = aten::add_(%a.1, %6, %6)
#   %13 : Scalar = aten::item(%a.3)
#   %15 : bool = aten::gt(%13, %14)
#   %r : Tensor = prim::If(%15)
#     block0():
#       %r.1 : Tensor = aten::select(%a.3, %17, %17)
#       -> (%r.1)
#     block1():
#       %r.2 : Tensor = aten::select(%b.1, %17, %17)
#       -> (%r.2)
#   %ret.1 : Tensor[] = prim::ListConstruct(%d.1, %r)
#   return (%ret.1)

# @torch.jit.script
# def foo(a : Tensor, b : Tensor):
#   c = 2 * b
#   d = c + 1
#   a += 1
#   if a.item() > 4:
#     r = a[0]
#   else:
#     r = b[0]
#   ret = [d, r]
#   return ret



# # 	x = torch.zeros([i, i, i, i])
# # 	y = x[0]
# # 	y.add_(3)
# # 	return x.sum()

# # graph(%i.1 : int):
# #   %13 : int = prim::Constant[value=1]()
# #   %6 : NoneType = prim::Constant()
# #   %12 : int = prim::Constant[value=3]()
# #   %5 : int[] = prim::ListConstruct(%i.1, %i.1, %i.1, %i.1)
# #   %x.1 : Tensor = aten::zeros(%5, %6, %6, %6, %6)
# #   %14 : Tensor = aten::add_(%x.1, %12, %13)
# #   %17 : Tensor = aten::sum(%x.1, %6)
# #   return (%17)



# with torch.jit._hide_source_ranges():
# 	print(foo.graph)


# # @torch.jit.script
# # def foo(x: int):
# # 	for i in range(x - 4):
# # 		x *= 3
# # 		if x > 40:
# # 			return x
# # 	return x

# # def foo(x: int) -> int:
# #   _0 = uninitialized(int)
# #   _1 = torch.sub(x, 4)
# #   _2 = False
# #   _3 = _0
# #   x0 = x
# #   _4 = 0
# #   _5 = torch.gt(_1, 0)
# #   while _5:
# #     x1 = torch.mul(x0, 3)
# #     if torch.gt(x1, 40):
# #       _6, _7, _8 = False, True, x1
# #     else:
# #       _6, _7, _8 = True, False, _0
# #     _9 = torch.add(_4, 1)
# #     _5, _2, _3, x0, _4 = torch.__and__(torch.lt(_9, _1), _6), _7, _8, x1, _9
# #   if _2:
# #     _10 = _3
# #   else:
# #     _10 = torch.add(x0, 4)



# # # graph(%x.1 : Tensor):
# #   %7 : bool = prim::Constant[value=1]()
# #   %3 : int = prim::Constant[value=0]()
# #   %4 : int = aten::size(%x.1, %3)
# #   %z : Tensor = prim::Loop(%4, %7, %x.1)
# #     block0(%i : int, %z.11 : Tensor):
# #       %z.5 : Tensor = aten::mul(%z.11, %z.11)
# #       -> (%7, %z.5)
# #   return (%z)
	
# with torch.jit._hide_source_ranges():
# 	print(foo.code)


# # @torch.jit.script
# # def f(a, b, c):
# #     d = a + b
# #     if c:
# #         e = d + d
# #     else:
# #         e = b + d
# #     return e

# # graph(%a.1 : Tensor,
# #       %b.1 : Tensor,
# #       %c.1 : Tensor):
# #   %5 : int = prim::Constant[value=1]()
# #   %d.1 : Tensor = aten::add(%a.1, %b.1, %5)
# #   %9 : bool = aten::Bool(%c.1)
# #   %e : Tensor = prim::If(%9)
# #     block0():
# #       %e.1 : Tensor = aten::add(%d.1, %d.1, %5)
# #       -> (%e.1)
# #     block1():
# #       %e.3 : Tensor = aten::add(%b.1, %d.1, %5)
# #       -> (%e.3)
# #   return (%e)





