# AOT ID: ['1_inference']
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


# kernel path: /data/users/eellison/pytorch/agent_space/tmp/torchinductor_eellison/mb/cmbe3nxxhfb5uytqrfnzq4a7higcgpeba3ognytg536u3sfm3rto.py
# Topologically Sorted Source Nodes: [float_1, float_2, summed_f, mul, mean, summed], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   mean => mean
#   mul => mul
#   summed => convert_element_type_2
#   summed_f => add
# Graph fragment:
#   %arg0_1 : Tensor "bf16[128, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[128, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f32[128, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[128, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %add : Tensor "f32[128, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %mul : Tensor "f32[128, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %add), kwargs = {})
#   %mean : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul, [-1], True), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[128, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.bfloat16), kwargs = {})
#   return %buf0,%convert_element_type_2
triton_red_fused__to_copy_add_mean_mul_0 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr4': '*fp8e4nv', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 6299648}, 'min_rblock': 128, 'backend_hash': '8D2EAD21100904E7D92B765D580AEF108246FE970468EF0946AEBA51E4679C50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 4096
    nested_R0_LOCAL_REDUCTION_SIZE: tl.constexpr = 128
    nested_R0_REDUCED_BLOCK: tl.constexpr = R0_BLOCK // nested_R0_LOCAL_REDUCTION_SIZE
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.bfloat16)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.bfloat16)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp3 + tmp7
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp13 = tmp8.to(tl.bfloat16)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14.to(tl.bfloat16)
        tmp16 = tmp15.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 4096*x0), tmp16, r0_mask & xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        reduced_r0_index = r0_offset // nested_R0_LOCAL_REDUCTION_SIZE + tl.arange(0, nested_R0_REDUCED_BLOCK)[None, :]
        reduced_r0_index_mask = reduced_r0_index < 32
        r0_4 = reduced_r0_index
        tmp17 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp17.to(tl.bfloat16)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.bfloat16)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp20 + tmp24
        tmp26 = tl.full([1, 1], 4096.0, tl.float32)
        tmp27 = (tmp11 / tmp26)
        tmp28 = tl.full([1, 1], 1e-06, tl.float32)
        tmp29 = tmp27 + tmp28
        tmp30 = libdevice.rsqrt(tmp29)
        tmp31 = tmp25 * tmp30
        tmp33 = tmp32.to(tl.bfloat16)
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp31 * tmp35
        tmp37 = tmp36.to(tl.bfloat16)
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp38.to(tl.bfloat16)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp40.to(tl.bfloat16)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tl_math.abs(tmp43)
        tmp45 = tl.reshape(tmp44, [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE])
        tmp46 = triton_helpers.max2(tmp45, 2)
        tmp47 = tl.reshape(tl.broadcast_to(tmp46[:, :, None], [XBLOCK, nested_R0_REDUCED_BLOCK, nested_R0_LOCAL_REDUCTION_SIZE]), [XBLOCK, R0_BLOCK])
        tmp48 = tl.full([1, 1], 0.0001, tl.float32)
        tmp49 = tl.maximum(tmp47, tmp48, tl.PropagateNan.ALL)
        tmp50 = tl.full([1, 1], 0.002232142857142857, tl.float32)
        tmp51 = tmp49 * tmp50
        tmp52 = (tmp43 / tmp51)
        tmp53 = tl.full([1, 1], -448.0, tl.float32)
        tmp54 = tl.maximum(tmp52, tmp53, tl.PropagateNan.ALL)
        tmp55 = tl.full([1, 1], 448.0, tl.float32)
        tmp56 = tl.minimum(tmp54, tmp55, tl.PropagateNan.ALL)
        tmp57 = tmp56.to(tl.float8e4nv)
        tmp58 = tl.maximum(tmp46, tmp48, tl.PropagateNan.ALL)
        tmp59 = tmp58 * tmp50
        tl.store(out_ptr2 + (r0_1 + 4096*x0), tmp40, r0_mask & xmask)
        tl.store(out_ptr4 + (r0_1 + 4096*x0), tmp57, r0_mask & xmask)
        tl.store(out_ptr5 + (x0 + 128*r0_4), tmp59, reduced_r0_index_mask & xmask)
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        assert_size_stride_grouped((arg0_1, arg1_1, arg2_1), ((128, 4096), (128, 4096), (4096, )), ((4096, 1), (4096, 1), (1, )), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf1 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
            buf3 = empty_strided_cuda((128, 32, 128), (4096, 128, 1), torch.float8_e4m3fn)
            buf4 = empty_strided_cuda((32, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [float_1, float_2, summed_f, mul, mean, summed], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.mean]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_0.run(arg0_1, arg1_1, arg2_1, buf5, buf1, buf3, buf4, 128, 4096, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
        return (reinterpret_tensor(buf3, (128, 4096), (4096, 1), 0), buf4, buf1, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((128, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
