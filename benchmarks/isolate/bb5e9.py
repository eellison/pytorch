
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.14.0a0+gitb05e2e0
# torch cuda version: 11.6
# torch git version: b05e2e0a970d489b404568ed0cb2e09248253e44


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Thu_Feb_10_18:23:41_PST_2022 
# Cuda compilation tools, release 11.6, V11.6.112 
# Build cuda_11.6.r11.6/compiler.30978841_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg74_1, arg151_1, arg377_1, add_66, mul_67, view_43):
        mul_77 = torch.ops.aten.mul.Tensor(view_43, arg74_1);  view_43 = arg74_1 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, 1024)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_77, [3], True)
        mul_79 = torch.ops.aten.mul.Tensor(mul_77, mul_67);  mul_77 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_79, [3], True);  mul_79 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_67, sum_10);  mul_67 = sum_10 = None
        sub_38 = torch.ops.aten.sub.Tensor(mul_78, sum_9);  mul_78 = sum_9 = None
        sub_39 = torch.ops.aten.sub.Tensor(sub_38, mul_80);  sub_38 = mul_80 = None
        div_1 = torch.ops.aten.div.Tensor(arg377_1, 1024);  arg377_1 = None
        mul_81 = torch.ops.aten.mul.Tensor(div_1, sub_39);  div_1 = sub_39 = None
        permute_84 = torch.ops.aten.permute.default(mul_81, [0, 3, 1, 2]);  mul_81 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(permute_84, add_66, arg151_1, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_84 = add_66 = arg151_1 = None
        getitem = convolution_backward[0];  convolution_backward = None
        return (getitem,)
        
args = [((1024,), (1,), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((32, 7, 7, 1024), (50176, 7, 1, 49), torch.float32, 'cuda'), ((32, 7, 7, 1024), (50176, 7168, 1024, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
