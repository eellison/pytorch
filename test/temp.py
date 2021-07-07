import torch
# import pdb; pdb.set_trace()
# from typing import List

# def broadcast(a: List[int], b: List[int]):
#     dimsA = len(a)
#     dimsB = len(b)
#     ndim = max(dimsA, dimsB)
#     expandedSizes : List[int] = []

#     for i in range(ndim):
#     offset = ndim - 1 - i
#     dimA = dimsA - 1 - offset
#     dimB = dimsB - 1 - offset
#     sizeA = a[dimA] if (dimA >= 0) else 1
#     sizeB = b[dimB] if (dimB >= 0) else 1

#     if sizeA != sizeB and sizeA != 1 and sizeB != 1:
#         # TODO: only assertion error is bound in C++ compilation right now
#         raise AssertionError("The size of tensor a {} must match the size of tensor b ("
#                         "{}) at non-singleton dimension {}".format(sizeA, sizeB, i))

#     expandedSizes.append(sizeB if sizeA == 1 else sizeA)

#     return expandedSizes

# def adaptive_avg_pool2d(self: List[int], out: List[int]):
#     # TODO: return out directly, list len refiner would need to
#     # annotate the List Type with len directly in IR
#     assert len(out) == 2
#     return [out[0], out[1]]

# # TODO: maybe make it customary that extra arguments are unused ?
# # TODO: return self directly
# def unary_two_unused_inputs(self: List[int], inp0: Any, inp1: Any):
#     out: List[int] = []
#     for elem in self:
#     out.append(elem)
#     return out

# def broadcast_one_unused_input(self: List[int], other: List[int], unused: Any):
#     return broadcast(self, other)

# def mm(self: List[int] , mat2: List[int]):
#     assert len(self) == 2, "self must be a matrix"
#     assert len(mat2) == 2, "mat2 must be a matrix"

#     assert self[1] == mat2[0]
#     return [self[0], mat2[1]]

# def dot(self: List[int], tensor: List[int]):
#     assert len(self) == 1 and len(tensor) == 1
#     assert self[0] == tensor[0]
#     # TODO: return self
#     return [self[0]]

# def mv(self: List[int], vec: List[int]):
#     assert len(self) == 2 and len(vec) == 1
#     assert self[1] == vec[0]
#     # TODO: return self
#     return [self[0]]

# # TODO: optional dim, then expose as a registered shape function
# def unsqueeze(li: List[int], dim: int):
#     out: List[int] = []
#     for i in range(len(li)):
#     if i == dim:
#         out.append(1)
#     out.append(li[i])
#     return out

# # TODO: optional dim, then expose as a registered shape function
# def squeeze(li: List[int], dim: int):
#     out: List[int] = []
#     for i in range(len(li)):
#     if i == dim:
#         if li[i] != 1:
#         out.append(li[i])
#     else:
#         out.append(li[i])
#     return out

# def matmul(tensor1: List[int] , tensor2: List[int]):
#     dim_tensor1 = len(tensor1)
#     dim_tensor2 = len(tensor2)
#     if dim_tensor1 == 1 and dim_tensor2 == 1:
#     return dot(tensor1, tensor2)
#     elif dim_tensor1 == 2 and dim_tensor2 == 1:
#     return mv(tensor1, tensor2)
#     elif dim_tensor1 == 1 and dim_tensor2 == 2:
#     return squeeze(mm(unsqueeze(tensor1, 0), tensor2), 0)
#     elif dim_tensor1 == 2 and dim_tensor2 == 2:
#     return mm(tensor1, tensor2)
#     elif dim_tensor1 >= 1 and dim_tensor2 >=1:
#     # We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
#     # we track m1 vs m2 separately even though they must match for nicer error messages
#     n = tensor1[-2] if dim_tensor1 > 1 else 1
#     m1 = tensor1[-1]
#     batch_tensor1 : List[int] = []
#     # TODO: handling of slice
#     for i in range(dim_tensor1 - 2):
#         batch_tensor1.append(tensor1[i])
#     m2 = tensor2[-1] if dim_tensor2 > 1 else 1
#     p = tensor2[-1]
#     batch_tensor2 : List[int] = []
#     # TODO: handling of slice
#     for i in range(dim_tensor2 - 2):
#         batch_tensor2.append(tensor2[i])

#     # expand the batch portion (i.e. cut off matrix dimensions and expand rest)
#     expand_batch_portion = broadcast(batch_tensor1, batch_tensor2)

#     # todo: copy ?
#     output_shape = expand_batch_portion
#     if dim_tensor1 > 1:
#         output_shape.append(n)

#     if dim_tensor2 > 1:
#         output_shape.append(p)

#     return output_shape
#     else:
#     assert False, "both  arguments to matmul need to be at least 1D"

# def t(self: List[int]):
#     assert len(self) <= 2
#     self_len = len(self)
#     if self_len == 0:
#     out: List[int] = []
#     return out
#     elif self_len == 1:
#     return [self[0]]
#     else:
#     return [self[1], self[0]]

# def linear(input: List[int], weight: List[int], bias: Optional[List[int]]):
#     out = matmul(input, t(weight))
#     if bias is not None:
#     assert broadcast(bias, out) == out
#     return out

# def check_non_negative(array: List[int]) -> bool:
#     # TODO: look into rewriting with early return and getting loop unrolling to fire
#     non_negative = False
#     for val in array:
#     if val < 0:
#         non_negative = True
#     return non_negative

# def check_shape_forward(input: List[int], weight_sizes: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], groups: int):
#     k = len(input)
#     weight_dim = len(weight_sizes)

#     # TODO: assertions could be expanded with the error messages
#     assert not check_non_negative(padding)
#     assert not check_non_negative(stride)

#     assert weight_dim == k
#     assert weight_sizes[0] >= groups
#     assert (weight_sizes[0] % groups) == 0
#     # only handling not transposed
#     assert input[1] == weight_sizes[1] * groups
#     assert bias is None or (len(bias) == 1 and bias[0] == weight_sizes[0])

#     for i in range(2, k):
#     assert (input[i] + 2 * padding[i - 2]) >= (dilation[i - 2] * (weight_sizes[i] - 1) + 1)

# # this is not handling transposed convolution yet
# def conv_output_size(input_size: List[int], weight_size: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], groups: int):
#     check_shape_forward(input_size, weight_size, bias, stride, padding, dilation, groups)

#     has_dilation = len(dilation) > 0
#     dim = len(input_size)
#     output_size: List[int] = []
#     input_batch_size_dim = 0
#     weight_output_channels_dim = 0
#     output_size.append(input_size[input_batch_size_dim])
#     output_size.append(weight_size[weight_output_channels_dim])

#     for d in range(2, dim):
#     dilation_ = dilation[d - 2] if has_dilation else 1
#     kernel = dilation_ * (weight_size[d] - 1) + 1
#     output_size.append((input_size[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1)
#     return output_size

# def conv1d(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], groups: int):
#     assert len(weight) == 3
#     assert len(input) == 3
#     return conv_output_size(input, weight, bias, stride, padding, dilation, groups)

# def conv2d(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], groups: int):
#     assert len(weight) == 4
#     assert len(input) == 4
#     return conv_output_size(input, weight, bias, stride, padding, dilation, groups)

# def conv3d(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], groups: int):
#     assert len(weight) == 5
#     assert len(input) == 5
#     return conv_output_size(input, weight, bias, stride, padding, dilation, groups)

# def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
#     if dim_post_expr <= 0:
#     assert wrap_scalar
#     dim_post_expr = 1
#     min = -dim_post_expr
#     max = dim_post_expr - 1
#     assert not (dim < min or dim > max)
#     if dim < 0:
#     dim += dim_post_expr
#     return dim

# def multiply_integers(li: List[int]):
#     out = 1
#     for elem in li:
#     out = out * elem
#     return out

# def flatten(input: List[int], start_dim: int, end_dim: int):
#     start_dim = maybe_wrap_dim(start_dim, len(input))
#     end_dim = maybe_wrap_dim(end_dim, len(input))
#     assert start_dim <= end_dim
#     if len(input) == 0:
#     return [1]
#     if (start_dim == end_dim):
#     # TODO: return self
#     out: List[int] = []
#     for elem in input:
#         out.append(elem)
#     return out
#     slice_numel = multiply_integers(input[start_dim:end_dim - start_dim + 1])
#     shape: List[int] = []
#     for i in range(start_dim):
#     shape.append(input[i])
#     shape.append(slice_numel)
#     for i in range(end_dim + 1, len(input)):
#     shape.append(input[i])
#     return shape

@torch.jit.script
def test_elias(x, y):
    return x[-1:0]

print(test_elias.graph)

# test_elias(-9, 2)