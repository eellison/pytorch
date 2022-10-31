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
        self.register_buffer('_tensor_constant1', torch.randn([], dtype=torch.float32))

    
    
    def forward(self, arg0_1, arg1_1):
        _to_copy = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.bool);  arg1_1 = None
        bitwise_not = torch.ops.aten.bitwise_not.default(_to_copy);  _to_copy = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        _to_copy_1 = torch.ops.aten._to_copy.default(lift_fresh_copy, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  lift_fresh_copy = None
        where = torch.ops.aten.where.self(bitwise_not, _to_copy_1, arg0_1);  _to_copy_1 = arg0_1 = None
        amax = torch.ops.aten.amax.default(where, [-1], True)
        sub = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        where_1 = torch.ops.aten.where.self(bitwise_not, lift_fresh_copy_1, div);  bitwise_not = lift_fresh_copy_1 = None
        copy_ = torch.ops.aten.copy_.default(div, where_1);  div = where_1 = None
        return (copy_,)
        
args = [((4, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32, 'cuda'), ((4, 1, 512, 512), (262144, 1048576, 512, 1), torch.uint8, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
