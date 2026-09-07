# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_eellison/tmpf19_yk1s/md/cmd6hvhfnipcxlrkwuf5cx5u44xrot4lp725ty5ji3x47d2rdp6s.py
# Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
# Source node to ATen node mapping:
#   rms_norm => convert_element_type, mean, pow_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[8192, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type : Tensor "f32[8192, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[8192, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2), kwargs = {})
#   %mean : Tensor "f32[8192, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   return %buf0
triton_per_fused__fused_rms_norm_0 = async_compile.triton('triton_per_fused__fused_rms_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr2': '*fp8e4nv', 'out_ptr4': '*u8', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__fused_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 65536, 'r0_': 67108864}, 'min_rblock': 16, 'backend_hash': 'B49614C0BBA23CA71245E046F2A6ABFCFD211E8FB0FFF5563191062F952B4CD3', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__fused_rms_norm_0(in_ptr0, in_ptr1, out_ptr2, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = 4096
    nested_R0_LOCAL_REDUCTION_SIZE: tl.constexpr = 16
    nested_R0_REDUCED_BLOCK: tl.constexpr = R0_BLOCK // nested_R0_LOCAL_REDUCTION_SIZE
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    reduced_r0_index = r0_offset // nested_R0_LOCAL_REDUCTION_SIZE + tl.arange(0, nested_R0_REDUCED_BLOCK)[None, :]
    reduced_r0_index_mask = reduced_r0_index < 256
    r0_4 = reduced_r0_index
    lane2_r0_index = r0_offset // 2 + tl.arange(0, R0_BLOCK // 2)[None, :]
    lane2_r0_index_mask = lane2_r0_index < 2048
    r0_7 = lane2_r0_index
    tmp12 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.full([1, 1], 4096.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-06, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp1 * tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl_math.abs(tmp15)
    tmp17 = tl.reshape(tmp16, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])
    tmp18 = triton_helpers.max2(tmp17, 2)
    tmp19 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.full([1, 1], 1e-12, tl.float32)
    tmp23 = tl.maximum(tmp21, tmp22, tl.PropagateNan.ALL)
    tmp24 = tl.full([1, 1], 448.0, tl.float32)
    tmp25 = tl.minimum(tmp23, tmp24, tl.PropagateNan.ALL)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26.to(tl.float8e4nv)
    tmp28, tmp29 = tl.split(tl.reshape(tmp0, [XBLOCK, (R0_BLOCK//2), 2]))
    tmp30, tmp31 = tl.split(tl.reshape(tmp12, [1, (R0_BLOCK//2), 2]))
    tmp32 = tmp28.to(tl.float32)
    tmp33 = tmp32 * tmp10
    tmp34 = tmp30.to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp27.to(tl.float32)
    tmp39 = tl.full([1, 1], 1.0, tl.float32)
    tmp40 = (tmp39 / tmp38)
    tmp41 = tl.reshape(tl.broadcast_to(tmp40[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, (nested_R0_LOCAL_REDUCTION_SIZE//2)]), [XBLOCK, (R0_BLOCK//2)])
    tmp42 = tmp37 * tmp41
    tmp43 = tmp29.to(tl.float32)
    tmp44 = tmp43 * tmp10
    tmp45 = tmp31.to(tl.float32)
    tmp46 = tmp44 * tmp45
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 * tmp41
    tmp50 = tl.inline_asm_elementwise('{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}', '=r,f,f', [tmp42, tmp49], dtype=tl.int32, is_pure=True, pack=1)
    tmp51 = tmp50.to(tl.uint8)
    tl.store(out_ptr2 + (4*(((x0 // 32) % 4)) + 16*((x0 % 32)) + 512*(r0_4 // 4) + 32768*(x0 // 128) + ((r0_4 % 4))), tmp27, reduced_r0_index_mask)
    tl.store(out_ptr4 + (r0_7 + 2048*x0), tmp51, lane2_r0_index_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride_grouped((arg0_1, arg1_1), ((8192, 4096), (4096, )), ((4096, 1), (1, )), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf4 = empty_strided_cuda((4096, 32, 4, 4), (512, 16, 4, 1), torch.float8_e4m3fn)
            buf3 = empty_strided_cuda((8192, 256, 8), (2048, 8, 1), torch.uint8)
            # Topologically Sorted Source Nodes: [rms_norm], Original ATen: [aten._fused_rms_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__fused_rms_norm_0.run(arg0_1, arg1_1, buf4, buf3, 8192, 4096, stream=raw_stream0)
            del arg0_1
            del arg1_1
        return (reinterpret_tensor(buf3, (8192, 2048), (2048, 1), 0), reinterpret_tensor(buf4, (2097152, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((8192, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
