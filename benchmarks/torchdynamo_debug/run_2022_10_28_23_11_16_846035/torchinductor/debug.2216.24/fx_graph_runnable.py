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

    
    
    def forward(self, primals_1, primals_2, primals_3):
        permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        view = torch.ops.aten.view.default(primals_3, [2048, 768]);  primals_3 = None
        addmm = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = None
        view_1 = torch.ops.aten.view.default(addmm, [4, 512, 3072]);  addmm = None
        mul = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1 = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
        sign = torch.ops.aten.sign.default(mul_1)
        abs_1 = torch.ops.aten.abs.default(mul_1);  mul_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(abs_1, 0.3275911)
        add = torch.ops.aten.add.Tensor(mul_2, 1.0);  mul_2 = None
        reciprocal = torch.ops.aten.reciprocal.default(add);  add = None
        mul_3 = torch.ops.aten.mul.Tensor(reciprocal, 1.0);  reciprocal = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 1.061405429)
        add_1 = torch.ops.aten.add.Tensor(mul_4, -1.453152027);  mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(add_1, mul_3);  add_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_5, 1.421413741);  mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_2, mul_3);  add_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_6, -0.284496736);  mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_3, mul_3);  add_3 = None
        add_4 = torch.ops.aten.add.Tensor(mul_7, 0.254829592);  mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(add_4, mul_3);  add_4 = mul_3 = None
        neg = torch.ops.aten.neg.default(abs_1)
        mul_9 = torch.ops.aten.mul.Tensor(neg, abs_1);  neg = abs_1 = None
        exp = torch.ops.aten.exp.default(mul_9);  mul_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_8, exp);  mul_8 = exp = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        sub = torch.ops.aten.sub.Tensor(lift_fresh_copy, mul_10);  lift_fresh_copy = None
        mul_11 = torch.ops.aten.mul.Tensor(sign, sub);  sub = None
        add_5 = torch.ops.aten.add.Tensor(mul_11, 1);  mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul, add_5);  mul = add_5 = None
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        sub_1 = torch.ops.aten.sub.Tensor(lift_fresh_copy_1, mul_10);  lift_fresh_copy_1 = mul_10 = None
        mul_23 = torch.ops.aten.mul.Tensor(sign, sub_1);  sign = sub_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_23, 1);  mul_23 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_1, view_1)
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, -0.5);  mul_25 = None
        exp_2 = torch.ops.aten.exp.default(mul_26);  mul_26 = None
        mul_27 = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_1, mul_27);  view_1 = mul_27 = None
        add_12 = torch.ops.aten.add.Tensor(mul_24, mul_28);  mul_24 = mul_28 = None
        permute_1 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return [mul_12, view, add_12, permute_1]
        
args = [((3072, 768), (768, 1), torch.float32, 'cuda'), ((3072,), (1,), torch.float32, 'cuda'), ((4, 512, 768), (393216, 768, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
