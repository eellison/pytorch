
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

    
    
    def forward(self, arg375_1, arg376_1, arg377_1, arg387_1, add_64, mul_66, mul_76, getitem):
        add_66 = torch.ops.aten.add.Tensor(mul_66, add_64);  mul_66 = add_64 = None
        permute_70 = torch.ops.aten.permute.default(arg375_1, [0, 2, 3, 1]);  arg375_1 = None
        sub_35 = torch.ops.aten.sub.Tensor(permute_70, arg376_1);  permute_70 = arg376_1 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_35, arg377_1);  sub_35 = arg377_1 = None
        view_42 = torch.ops.aten.view.default(mul_76, [1568, 4096]);  mul_76 = None
        mm_5 = torch.ops.aten.mm.default(view_42, arg387_1);  view_42 = arg387_1 = None
        view_43 = torch.ops.aten.view.default(mm_5, [32, 7, 7, 1024]);  mm_5 = None
        return (getitem,)
        
args = [((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
