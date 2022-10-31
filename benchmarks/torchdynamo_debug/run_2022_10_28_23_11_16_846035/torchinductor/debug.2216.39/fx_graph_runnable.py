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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        permute = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        view = torch.ops.aten.view.default(primals_4, [2048, 768]);  primals_4 = None
        mm = torch.ops.aten.mm.default(view, permute)
        _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [4, 512, 2304]);  mm = None
        view_1 = torch.ops.aten.view.default(_unsafe_view, [4, 512, 12, -1]);  _unsafe_view = None
        permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        split = torch.ops.aten.split.Tensor(permute_1, 64, -1);  permute_1 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2];  split = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        slice_1 = torch.ops.aten.slice.Tensor(unsqueeze_1, 2, 0, 9223372036854775807);  unsqueeze_1 = None
        view_2 = torch.ops.aten.view.default(slice_1, [1, 1, 12, -1]);  slice_1 = None
        permute_2 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        add = torch.ops.aten.add.Tensor(getitem, permute_2);  getitem = permute_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_2, 0);  primals_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        slice_2 = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, 9223372036854775807);  unsqueeze_3 = None
        view_3 = torch.ops.aten.view.default(slice_2, [1, 1, 12, -1]);  slice_2 = None
        permute_3 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        add_1 = torch.ops.aten.add.Tensor(getitem_2, permute_3);  getitem_2 = permute_3 = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        mul = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = None
        sqrt = torch.ops.aten.sqrt.default(mul);  mul = None
        div = torch.ops.aten.div.Tensor(add, sqrt);  add = sqrt = None
        permute_4 = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand = torch.ops.aten.expand.default(div, [4, 12, 512, 64]);  div = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone, [48, 512, 64]);  clone = None
        expand_1 = torch.ops.aten.expand.default(permute_4, [4, 12, 64, 512]);  permute_4 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_1, [48, 64, 512]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(_unsafe_view_1, _unsafe_view_2)
        _unsafe_view_3 = torch.ops.aten._unsafe_view.default(bmm, [4, 12, 512, 512]);  bmm = None
        permute_5 = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 1]);  _unsafe_view_1 = None
        permute_6 = torch.ops.aten.permute.default(_unsafe_view_2, [0, 2, 1]);  _unsafe_view_2 = None
        permute_13 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return [_unsafe_view_3, add_1, view, permute_5, permute_6, permute_13]
        
args = [((768,), (1,), torch.float32, 'cuda'), ((768,), (1,), torch.float32, 'cuda'), ((2304, 768), (768, 1), torch.float32, 'cuda'), ((4, 512, 768), (393216, 768, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
