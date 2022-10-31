
import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# torch version: 1.14.0a0+git455ba86
# torch cuda version: 11.6
# torch git version: 455ba8615dfc73064860d805f99e6b6c6364439d


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Thu_Feb_10_18:23:41_PST_2022
# Cuda compilation tools, release 11.6, V11.6.112
# Build cuda_11.6.r11.6/compiler.30978841_0

# GPU Hardware Info:
# NVIDIA A100-SXM4-40GB : 1


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, arg3_1, addmm, sigmoid, view, permute_1):
        mul_4 = torch.ops.aten.mul.Tensor(addmm, sigmoid);  addmm = sigmoid = None
        addmm_1 = torch.ops.aten.addmm.default(arg3_1, mul_4, permute_1);  arg3_1 = mul_4 = permute_1 = None
        var = torch.ops.aten.var.correction(view, [2, 3], correction = 0, keepdim = True);  view = None
        return (var,)

args = [((512,), (1,), torch.float32, 'cuda'), ((16, 512), (512, 1), torch.float32, 'cuda'), ((16, 512), (512, 1), torch.float32, 'cuda'), ((16, 32, 4, 16384), (2097152, 65536, 16384, 1), torch.float32, 'cuda'), ((512, 512), (1, 512), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro().to(device="cuda"))(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
compiled(args)
