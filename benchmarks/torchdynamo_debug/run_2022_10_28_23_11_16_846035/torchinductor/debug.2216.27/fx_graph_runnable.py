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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        add = torch.ops.aten.add.Tensor(primals_3, primals_4);  primals_3 = primals_4 = None
        mean = torch.ops.aten.mean.dim(add, [-1], True)
        sub = torch.ops.aten.sub.Tensor(add, mean);  add = mean = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
        sqrt = torch.ops.aten.sqrt.default(add_1);  add_1 = None
        div = torch.ops.aten.div.Tensor(sub, sqrt)
        mul = torch.ops.aten.mul.Tensor(primals_1, div);  div = None
        add_2 = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
        return [add_2, primals_1, sub, sqrt]
        
args = [((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((4, 512, 768), (393216, 768, 1), torch.float32, 'cuda'), ((4, 512, 768), (393216, 768, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
