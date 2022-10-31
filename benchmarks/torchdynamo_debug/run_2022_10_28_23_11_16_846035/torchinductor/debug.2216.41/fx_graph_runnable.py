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
        self.register_buffer('_tensor_constant0', torch.randn([], dtype=torch.float32))

    
    
    def forward(self, arg0_1):
        empty_like = torch.ops.aten.empty_like.default(arg0_1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  arg0_1 = None
        alias = torch.ops.aten.alias.default(empty_like);  empty_like = None
        rand_like = torch.ops.aten.rand_like.default(alias, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        alias_1 = torch.ops.aten.alias.default(rand_like);  rand_like = None
        lt = torch.ops.aten.lt.Scalar(alias_1, 0.9);  alias_1 = None
        copy_ = torch.ops.aten.copy_.default(alias, lt);  alias = lt = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub = torch.ops.aten.sub.Tensor(lift_fresh_copy, copy_);  lift_fresh_copy = copy_ = None
        _to_copy = torch.ops.aten._to_copy.default(sub, dtype = torch.bool);  sub = None
        return (_to_copy,)
        
args = [((4, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
