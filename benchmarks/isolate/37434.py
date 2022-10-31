
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

    
    
    def forward(self, arg151_1, add_64, mul_66, permute_84):
        add_66 = torch.ops.aten.add.Tensor(mul_66, add_64);  mul_66 = add_64 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(permute_84, add_66, arg151_1, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_84 = add_66 = arg151_1 = None
        getitem = convolution_backward[0];  convolution_backward = None
        return (getitem,)
        
args = [((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
