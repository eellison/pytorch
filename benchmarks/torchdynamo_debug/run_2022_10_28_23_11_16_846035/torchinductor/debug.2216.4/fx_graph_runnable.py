import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.14.0a0+gitd13b678
# torch cuda version: 11.6
# torch git version: d13b6781d8b7353919ee06378636773f762b880e


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

    
    
    def forward(self, arg0_1):
        unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 1);  arg0_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(unsqueeze_1, -2)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
        mul = torch.ops.aten.mul.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        _to_copy = torch.ops.aten._to_copy.default(mul, dtype = torch.uint8);  mul = None
        return (_to_copy,)
        
args = [((4, 512), (512, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
