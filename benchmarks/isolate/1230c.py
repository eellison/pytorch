
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

    
    
    def forward(self, arg71_1, arg73_1, arg74_1, arg75_1, arg76_1, arg148_1, arg150_1, arg151_1, arg362_1, arg368_1, arg374_1, arg375_1, arg376_1, arg377_1, arg381_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg497_1):
        _unsafe_view_33 = torch.ops.aten._unsafe_view.default(arg368_1, [32, 7, 7, 1024]);  arg368_1 = None
        add_63 = torch.ops.aten.add.Tensor(_unsafe_view_33, arg148_1);  _unsafe_view_33 = arg148_1 = None
        permute_67 = torch.ops.aten.permute.default(add_63, [0, 3, 1, 2]);  add_63 = None
        view_33 = torch.ops.aten.view.default(arg71_1, [1, 1024, 1, 1]);  arg71_1 = None
        mul_64 = torch.ops.aten.mul.Tensor(permute_67, view_33);  permute_67 = view_33 = None
        add_64 = torch.ops.aten.add.Tensor(mul_64, arg362_1);  mul_64 = arg362_1 = None
        _unsafe_view_34 = torch.ops.aten._unsafe_view.default(arg374_1, [32, 7, 7, 1024]);  arg374_1 = None
        add_65 = torch.ops.aten.add.Tensor(_unsafe_view_34, arg150_1);  _unsafe_view_34 = arg150_1 = None
        permute_69 = torch.ops.aten.permute.default(add_65, [0, 3, 1, 2]);  add_65 = None
        view_34 = torch.ops.aten.view.default(arg73_1, [1, 1024, 1, 1]);  arg73_1 = None
        mul_66 = torch.ops.aten.mul.Tensor(permute_69, view_34);  permute_69 = view_34 = None
        add_66 = torch.ops.aten.add.Tensor(mul_66, add_64);  mul_66 = add_64 = None
        permute_70 = torch.ops.aten.permute.default(arg375_1, [0, 2, 3, 1]);  arg375_1 = None
        sub_35 = torch.ops.aten.sub.Tensor(permute_70, arg376_1);  permute_70 = arg376_1 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_35, arg377_1);  sub_35 = arg377_1 = None
        view_35 = torch.ops.aten.view.default(arg75_1, [1, 1024, 1, 1]);  arg75_1 = None
        mm = torch.ops.aten.mm.default(arg497_1, arg383_1);  arg497_1 = arg383_1 = None
        view_37 = torch.ops.aten.view.default(mm, [32, 1024, 1, 1]);  mm = None
        permute_75 = torch.ops.aten.permute.default(view_37, [0, 2, 3, 1]);  view_37 = None
        mul_68 = torch.ops.aten.mul.Tensor(permute_75, arg76_1);  permute_75 = arg76_1 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, 1024)
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_68, [3], True)
        mul_70 = torch.ops.aten.mul.Tensor(mul_68, arg381_1);  mul_68 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_70, [3], True);  mul_70 = None
        mul_71 = torch.ops.aten.mul.Tensor(arg381_1, sum_3);  arg381_1 = sum_3 = None
        sub_36 = torch.ops.aten.sub.Tensor(mul_69, sum_2);  mul_69 = sum_2 = None
        sub_37 = torch.ops.aten.sub.Tensor(sub_36, mul_71);  sub_36 = mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(arg384_1, sub_37);  arg384_1 = sub_37 = None
        permute_76 = torch.ops.aten.permute.default(mul_72, [0, 3, 1, 2]);  mul_72 = None
        squeeze = torch.ops.aten.squeeze.dim(permute_76, 3);  permute_76 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        new_zeros = torch.ops.aten.new_zeros.default(squeeze_1, [32768])
        as_strided_scatter = torch.ops.aten.as_strided_scatter.default(new_zeros, squeeze_1, [32, 1024], [1024, 1], 0);  new_zeros = squeeze_1 = None
        as_strided = torch.ops.aten.as_strided.default(as_strided_scatter, [32, 1024, 1, 1], [1024, 1, 1, 1], 0);  as_strided_scatter = None
        expand = torch.ops.aten.expand.default(as_strided, [32, 1024, 7, 7]);  as_strided = None
        div = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        mul_75 = torch.ops.aten.mul.Tensor(div, view_35);  div = view_35 = None
        permute_77 = torch.ops.aten.permute.default(mul_75, [0, 2, 3, 1]);  mul_75 = None
        clone = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        _unsafe_view_36 = torch.ops.aten._unsafe_view.default(clone, [1568, 1024]);  clone = None
        mm_3 = torch.ops.aten.mm.default(_unsafe_view_36, arg385_1);  _unsafe_view_36 = arg385_1 = None
        view_40 = torch.ops.aten.view.default(mm_3, [32, 7, 7, 4096]);  mm_3 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_40, arg386_1);  view_40 = arg386_1 = None
        view_42 = torch.ops.aten.view.default(mul_76, [1568, 4096]);  mul_76 = None
        return (view_42,)
        
args = [((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024,), (1,), torch.float32, 'cuda'), ((1024, 1, 7, 7), (49, 49, 7, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 1, 7168, 1024), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((1568, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1024, 7, 7), (50176, 49, 7, 1), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 7, 7, 1), (49, 7, 1, 1568), torch.float32, 'cuda'), ((32, 1, 1, 1024), (1024, 32768, 32768, 1), torch.float32, 'cuda'), ((1000, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1, 1, 1), (1, 32, 32, 32), torch.float32, 'cuda'), ((1024, 4096), (4096, 1), torch.float32, 'cuda'), ((32, 7, 7, 4096), (200704, 28672, 4096, 1), torch.float32, 'cuda'), ((4096, 1024), (1024, 1), torch.float32, 'cuda'), ((32, 1000), (1000, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._dynamo.debug_utils import inductor_fails

if inductor_fails(mod, args):
    exit(1)
else:
    exit(0)
