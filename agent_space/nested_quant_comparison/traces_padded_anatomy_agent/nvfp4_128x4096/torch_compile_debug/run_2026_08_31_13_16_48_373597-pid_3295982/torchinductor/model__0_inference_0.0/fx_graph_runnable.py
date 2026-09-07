
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['PYTORCH_BUILD_ROOT'] = '/data/eellison/build'
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.3'
os.environ['_PYTORCH_DEV_OLD_PYTHONPATH'] = '/usr/local/lib/oilfs'
os.environ['_PYTORCH_DEV_OLD_LD_LIBRARY_PATH'] = '/opt/rh/gcc-toolset-13/root/usr/lib64:/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib:/data/eellison/cuda-13-libs'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/data/users/eellison/pytorch/agent_space/nested_quant_comparison/traces_padded_anatomy_agent/nvfp4_128x4096'
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
        view = torch.ops.aten.view.default(convert_element_type_1, [128, 256, 16]);  convert_element_type_1 = None
        abs_1 = torch.ops.aten.abs.default(view)
        amax = torch.ops.aten.amax.default(abs_1, [-1]);  abs_1 = None
        div = torch.ops.aten.div.Tensor(amax, 6.0);  amax = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        clamp_min = torch.ops.aten.clamp_min.default(convert_element_type_2, 1e-12);  convert_element_type_2 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 448.0);  clamp_min = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(clamp_max, torch.bfloat16);  clamp_max = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float8_e4m3fn);  convert_element_type_3 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32)
        reciprocal = torch.ops.aten.reciprocal.default(convert_element_type_5);  convert_element_type_5 = None
        view_1 = torch.ops.aten.view.default(view, [128, 256, 8, 2]);  view = None
        select = torch.ops.aten.select.int(view_1, 3, 0)
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(select, torch.float32);  select = None
        unsqueeze = torch.ops.aten.unsqueeze.default(reciprocal, -1)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_6, unsqueeze);  convert_element_type_6 = unsqueeze = None
        select_1 = torch.ops.aten.select.int(view_1, 3, 1);  view_1 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(select_1, torch.float32);  select_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(reciprocal, -1);  reciprocal = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_7, unsqueeze_1);  convert_element_type_7 = unsqueeze_1 = None
        inline_asm_elementwise = torch.ops.higher_order.inline_asm_elementwise(mul_2, mul_3, asm_str = '{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', constraints = '=r,f,f', dtype = torch.int32, is_pure = True, pack = 1);  mul_2 = mul_3 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(inline_asm_elementwise, torch.uint8);  inline_asm_elementwise = None
        view_2 = torch.ops.aten.view.default(convert_element_type_8, [128, 2048]);  convert_element_type_8 = None
        view_3 = torch.ops.aten.view.default(convert_element_type_4, [1, 128, 64, 4]);  convert_element_type_4 = None
        permute = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4 = torch.ops.aten.view.default(permute, [64, 4, 32, 4]);  permute = None
        permute_1 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_5 = torch.ops.aten.view.default(clone, [32768]);  clone = None
        return (view_2, view_5)

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