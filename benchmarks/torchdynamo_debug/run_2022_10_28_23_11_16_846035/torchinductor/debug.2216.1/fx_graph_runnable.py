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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7):
        slice_1 = torch.ops.aten.slice.Tensor(primals_5, 0, 0, 9223372036854775807);  primals_5 = None
        embedding = torch.ops.aten.embedding.default(primals_3, primals_6, 0);  primals_3 = None
        embedding_1 = torch.ops.aten.embedding.default(primals_4, slice_1);  primals_4 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1)
        mean = torch.ops.aten.mean.dim(add, [-1], True)
        sub = torch.ops.aten.sub.Tensor(add, mean);  add = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
        sqrt = torch.ops.aten.sqrt.default(add_1);  add_1 = None
        div = torch.ops.aten.div.Tensor(sub, sqrt);  sub = None
        mul = torch.ops.aten.mul.Tensor(primals_1, div);  div = None
        add_2 = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_7, 2)
        mul_1 = torch.ops.aten.mul.Tensor(add_2, unsqueeze);  add_2 = unsqueeze = None
        view_3 = torch.ops.aten.view.default(slice_1, [512]);  slice_1 = None
        view_5 = torch.ops.aten.view.default(primals_6, [2048]);  primals_6 = None
        return [mul_1, primals_1, primals_7, embedding, embedding_1, mean, sqrt, view_3, view_5]
        
args = [((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((50265, 768), (768, 1), torch.float32, 'cuda'), ((512, 768), (768, 1), torch.float32, 'cuda'), ((1, 512), (512, 1), torch.int64, 'cuda'), ((4, 512), (512, 1), torch.int64, 'cuda'), ((4, 512), (512, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
