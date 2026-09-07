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


# kernel path: /tmp/torchinductor_eellison/lj/cljddpbi36ckmtl6c54664jqk5qpw5z3u7obvul2ijuovnsp2syz.py
# Topologically Sorted Source Nodes: [float_1, float_2, summed_f, pow_1, mean, add_1, rms, mul, to_1, normed, abs_1, amax, float_3, truediv, scale, truediv_1, clamp, quant, summed], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.div, aten.clamp_min, aten.clamp]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   add_1 => add_1
#   amax => amax
#   clamp => clamp_max, clamp_min_1
#   float_1 => convert_element_type
#   float_2 => convert_element_type_1
#   float_3 => convert_element_type_4
#   mean => mean
#   mul => mul
#   normed => mul_1
#   pow_1 => pow_1
#   quant => convert_element_type_5
#   rms => rsqrt
#   scale => clamp_min
#   summed => convert_element_type_2
#   summed_f => add
#   to_1 => convert_element_type_3
#   truediv => div
#   truediv_1 => div_1
# Graph fragment:
#   %arg0_1 : Tensor "bf16[4096, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[4096, 4096][4096, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f32[4096, 1][1, 4096]cuda:0" = PlaceHolder[target=buf0]
#   %arg2_1 : Tensor "bf16[4096][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %abs_1 : Tensor "bf16[4096, 4096][4096, 1]cuda:0" = PlaceHolder[target=abs_1]
#   %amax : Tensor "bf16[4096, 1][1, 4096]cuda:0" = PlaceHolder[target=amax]
#   %clamp_min : Tensor "f32[4096, 1][1, 1]cuda:0" = PlaceHolder[target=clamp_min]
#   %div_1 : Tensor "f32[4096, 4096][4096, 1]cuda:0" = PlaceHolder[target=div_1]
#   %convert_element_type : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %add : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %pow_1 : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : Tensor "bf16[4096, 4096][4096, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %arg2_1), kwargs = {})
#   %abs_1 : Tensor "bf16[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%mul_1,), kwargs = {})
#   %amax : Tensor "bf16[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%abs_1, [-1], True), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%amax, torch.float32), kwargs = {})
#   %div : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, 448.0), kwargs = {})
#   %clamp_min : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div, 4.359654017857143e-06), kwargs = {})
#   %div_1 : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_1, %clamp_min), kwargs = {})
#   %clamp_min_1 : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_1, -448.0), kwargs = {})
#   %clamp_max : Tensor "f32[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 448.0), kwargs = {})
#   %convert_element_type_5 : Tensor "f8e4m3fn[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.float8_e4m3fn), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[4096, 4096][4096, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.bfloat16), kwargs = {})
#   return %buf0,%abs_1,%amax,%clamp_min,%div_1,%convert_element_type_5,%convert_element_type_2
triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr3': '*fp32', 'out_ptr5': '*fp8e4nv', 'out_ptr6': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': False, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 3, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 32768, 'r0_': 167780352}, 'add_persistent_rblock': True, 'backend_hash': '8D2EAD21100904E7D92B765D580AEF108246FE970468EF0946AEBA51E4679C50', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr5, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    _tmp46 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.bfloat16)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.bfloat16)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp16 + tmp20
        tmp22 = tl.full([1, 1], 4096.0, tl.float32)
        tmp23 = (tmp11 / tmp22)
        tmp24 = tl.full([1, 1], 1e-06, tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp27.to(tl.bfloat16)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29.to(tl.bfloat16)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31.to(tl.bfloat16)
        tmp33 = tmp32.to(tl.float32)
        tmp35 = tmp34.to(tl.bfloat16)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp33 * tmp36
        tmp38 = tmp37.to(tl.bfloat16)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp39.to(tl.bfloat16)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tl_math.abs(tmp41)
        tmp43 = tmp42.to(tl.bfloat16)
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK, R0_BLOCK])
        tmp47 = tl.maximum(_tmp46, tmp45, tl.PropagateNan.ALL)
        _tmp46 = tl.where(r0_mask, tmp47, _tmp46)
    tmp46 = triton_helpers.max2(_tmp46, 1)[:, None]
    tmp48 = tmp46.to(tl.bfloat16)
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tl.full([1, 1], 0.002232142857142857, tl.float32)
    tmp52 = tmp50 * tmp51
    tmp53 = tl.full([1, 1], 4.359654017857143e-06, tl.float32)
    tmp54 = tl.maximum(tmp52, tmp53, tl.PropagateNan.ALL)
    tl.store(out_ptr3 + (x0), tmp54, None)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp55 = tl.load(in_ptr0 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp59 = tl.load(in_ptr1 + (r0_1 + 4096*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp76 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp55.to(tl.bfloat16)
        tmp57 = tmp56.to(tl.float32)
        tmp58 = tmp57.to(tl.float32)
        tmp60 = tmp59.to(tl.bfloat16)
        tmp61 = tmp60.to(tl.float32)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp58 + tmp62
        tmp64 = tl.full([1, 1], 4096.0, tl.float32)
        tmp65 = (tmp11 / tmp64)
        tmp66 = tl.full([1, 1], 1e-06, tl.float32)
        tmp67 = tmp65 + tmp66
        tmp68 = libdevice.rsqrt(tmp67)
        tmp69 = tmp63 * tmp68
        tmp70 = tmp69.to(tl.bfloat16)
        tmp71 = tmp70.to(tl.float32)
        tmp72 = tmp71.to(tl.bfloat16)
        tmp73 = tmp72.to(tl.float32)
        tmp74 = tmp73.to(tl.bfloat16)
        tmp75 = tmp74.to(tl.float32)
        tmp77 = tmp76.to(tl.bfloat16)
        tmp78 = tmp77.to(tl.float32)
        tmp79 = tmp75 * tmp78
        tmp80 = tmp79.to(tl.bfloat16)
        tmp81 = tmp80.to(tl.float32)
        tmp82 = tmp81.to(tl.bfloat16)
        tmp83 = tmp82.to(tl.float32)
        tmp84 = tmp83.to(tl.float32)
        tmp85 = (tmp84 / tmp54)
        tmp86 = tl.full([1, 1], -448.0, tl.float32)
        tmp87 = tl.maximum(tmp85, tmp86, tl.PropagateNan.ALL)
        tmp88 = tl.full([1, 1], 448.0, tl.float32)
        tmp89 = tl.minimum(tmp87, tmp88, tl.PropagateNan.ALL)
        tmp90 = tmp89.to(tl.float8e4nv)
        tmp91 = tmp63.to(tl.bfloat16)
        tmp92 = tmp91.to(tl.float32)
        tmp93 = tmp92.to(tl.bfloat16)
        tmp94 = tmp93.to(tl.float32)
        tl.store(out_ptr5 + (r0_1 + 4096*x0), tmp90, r0_mask)
        tl.store(out_ptr6 + (r0_1 + 4096*x0), tmp94, r0_mask)
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
        assert_size_stride_grouped((arg0_1, arg1_1, arg2_1), ((4096, 4096), (4096, 4096), (4096, )), ((4096, 1), (4096, 1), (1, )), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            arg1_1 = copy_if_misaligned(arg1_1)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf3 = empty_strided_cuda((4096, 1), (1, 1), torch.float32)
            buf5 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float8_e4m3fn)
            buf6 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, float_2, summed_f, pow_1, mean, add_1, rms, mul, to_1, normed, abs_1, amax, float_3, truediv, scale, truediv_1, clamp, quant, summed], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.abs, aten.amax, aten.div, aten.clamp_min, aten.clamp]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_abs_add_amax_clamp_clamp_min_div_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg2_1, buf3, buf5, buf6, 4096, 4096, stream=raw_stream0)
            del arg0_1
            del arg1_1
            del arg2_1
        return (buf5, buf3, buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
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
