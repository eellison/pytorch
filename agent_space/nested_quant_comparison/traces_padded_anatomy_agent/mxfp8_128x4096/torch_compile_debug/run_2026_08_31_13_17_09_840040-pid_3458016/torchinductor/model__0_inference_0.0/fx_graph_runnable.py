
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['PYTORCH_BUILD_ROOT'] = '/data/eellison/build'
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.3'
os.environ['_PYTORCH_DEV_OLD_PYTHONPATH'] = '/usr/local/lib/oilfs'
os.environ['_PYTORCH_DEV_OLD_LD_LIBRARY_PATH'] = '/opt/rh/gcc-toolset-13/root/usr/lib64:/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib:/data/eellison/cuda-13-libs'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/data/users/eellison/pytorch/agent_space/nested_quant_comparison/traces_padded_anatomy_agent/mxfp8_128x4096'
os.environ['TORCH_LOGS_FORMAT'] = '[%(levelname)s]:%(message)s'
os.environ['PYTORCH_ROOT'] = '/data/eellison/src/pytorch'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_eellison'
os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)
os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.fx_graph_cache = False
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.force_pointwise_cat = False
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.nested_reduction = True
torch._inductor.config.emulate_precision_casts = True
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.enabled = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False




isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.15.0a0+gitcdd22ad
# torch cuda version: 13.0
# torch git version: cdd22ade2948699188c3e2d0d80b5a396a8489ae


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Wed_Aug_20_01:58:59_PM_PDT_2025 
# Cuda compilation tools, release 13.0, V13.0.88 
# Build cuda_13.0.r13.0/compiler.36424714_0 

# GPU Hardware Info: 
# NVIDIA B200 : 8 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [1], True);  pow_1 = None
        add = torch.ops.aten.add.Scalar(mean, 1.1920928955078125e-07);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        view = torch.ops.aten.view.default(convert_element_type_1, [128, 128, 32]);  convert_element_type_1 = None
        abs_1 = torch.ops.aten.abs.default(view)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(abs_1, torch.float32);  abs_1 = None
        amax = torch.ops.aten.amax.default(convert_element_type_2, [-1]);  convert_element_type_2 = None
        div = torch.ops.aten.div.Tensor(amax, 448.0);  amax = None
        clamp_min = torch.ops.aten.clamp_min.default(div, 1.1754943508222875e-38);  div = None
        inductor_cvt_e8m0_rceil = torch.ops.prims.inductor_cvt_e8m0_rceil.default(clamp_min);  clamp_min = None
        full_default = torch.ops.aten.full.default([128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(inductor_cvt_e8m0_rceil, torch.int32)
        sub = torch.ops.aten.sub.Tensor(convert_element_type_3, 127);  convert_element_type_3 = None
        ldexp = torch.ops.aten.ldexp.Tensor(full_default, sub);  full_default = sub = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        unsqueeze = torch.ops.aten.unsqueeze.default(ldexp, -1);  ldexp = None
        div_1 = torch.ops.aten.div.Tensor(convert_element_type_4, unsqueeze);  convert_element_type_4 = unsqueeze = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(div_1, -448.0);  div_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min_1, 448.0);  clamp_min_1 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max, torch.float8_e4m3fn);  clamp_max = None
        view_1 = torch.ops.aten.view.default(convert_element_type_5, [128, 4096]);  convert_element_type_5 = None
        view_2 = torch.ops.aten.view.default(inductor_cvt_e8m0_rceil, [1, 128, 32, 4]);  inductor_cvt_e8m0_rceil = None
        permute = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(permute, [32, 4, 32, 4]);  permute = None
        permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        clone = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4 = torch.ops.aten.view.default(clone, [16384]);  clone = None
        return (view_1, view_4)

def load_args(reader):
    buf0 = reader.storage(None, 1048576, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (128, 4096), dtype=torch.bfloat16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (4096,), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)