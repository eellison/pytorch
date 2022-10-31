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

    
    
    def forward(self, primals_1, primals_2):
        expand = torch.ops.aten.expand.default(primals_1, [4, 12, 512, 512]);  primals_1 = None
        view = torch.ops.aten.view.default(expand, [48, 512, 512]);  expand = None
        expand_1 = torch.ops.aten.expand.default(primals_2, [4, 12, 512, 64]);  primals_2 = None
        clone = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [48, 512, 64]);  clone = None
        bmm = torch.ops.aten.bmm.default(view, _unsafe_view)
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(bmm, [4, 12, 512, 64]);  bmm = None
        permute = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 1, 3]);  _unsafe_view_1 = None
        clone_1 = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1 = torch.ops.aten.view.default(clone_1, [4, 512, -1]);  clone_1 = None
        permute_2 = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        permute_3 = torch.ops.aten.permute.default(_unsafe_view, [0, 2, 1]);  _unsafe_view = None
        return [view_1, permute_2, permute_3]
        
args = [((4, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32, 'cuda'), ((4, 12, 512, 64), (393216, 64, 768, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
