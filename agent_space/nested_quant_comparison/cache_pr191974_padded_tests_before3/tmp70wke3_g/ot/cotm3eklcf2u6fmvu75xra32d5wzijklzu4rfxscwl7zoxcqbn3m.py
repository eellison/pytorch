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


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_pr191974_padded_tests_before3/tmp70wke3_g/rt/crtinpwqj5tq3cdbwxdvnkdowxkzgc2rz3arpzfjv37uqsb76d22.py
# Topologically Sorted Source Nodes: [linear, linear_1, linear_2, linear_3, row_chunk, mul_5, row_outer, mul_6, add_4, row_lane, logical_row, ge, col_outer, mul_7, col_inner, logical_col, ge_1, padding_mask, padding, index_put_default], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.remainder, aten.add, aten.ge, aten.bitwise_or, aten.zeros, aten.index_put]
# Source node to ATen node mapping:
#   add_4 => add_4
#   col_inner => remainder_3
#   col_outer => remainder_6
#   ge => ge
#   ge_1 => ge_1
#   index_put_default => index_put_1
#   linear => iota_2
#   linear_1 => div_3
#   linear_2 => div_4
#   linear_3 => div_5
#   logical_col => add_6
#   logical_row => add_5
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   padding => full
#   padding_mask => bitwise_or
#   row_chunk => div_6
#   row_lane => remainder_5
#   row_outer => remainder_4
# Graph fragment:
#   %index_put : Tensor "u8[16384][1]cuda:0" = PlaceHolder[target=index_put]
#   %iota_2 : Tensor "i64[16384][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (16384,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %div_3 : Tensor "i64[16384][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%iota_2, 4), kwargs = {rounding_mode: floor})
#   %div_4 : Tensor "i64[16384][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_3, 4), kwargs = {rounding_mode: floor})
#   %div_5 : Tensor "i64[16384][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_4, 32), kwargs = {rounding_mode: floor})
#   %div_6 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%div_5, 32), kwargs = {rounding_mode: floor})
#   %mul_5 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, 128), kwargs = {})
#   %remainder_4 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_3, 4), kwargs = {})
#   %mul_6 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_4, 32), kwargs = {})
#   %add_4 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %remainder_5 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_4, 32), kwargs = {})
#   %add_5 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %remainder_5), kwargs = {})
#   %ge : Tensor "b8[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_5, 19), kwargs = {})
#   %remainder_6 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_5, 32), kwargs = {})
#   %mul_7 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_6, 4), kwargs = {})
#   %remainder_3 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%iota_2, 4), kwargs = {})
#   %add_6 : Tensor "i64[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %remainder_3), kwargs = {})
#   %ge_1 : Tensor "b8[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%add_6, 128), kwargs = {})
#   %bitwise_or : Tensor "b8[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%ge, %ge_1), kwargs = {})
#   %full : Tensor "u8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.uint8, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : Tensor "u8[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [%bitwise_or], %full), kwargs = {})
#   return %buf1
triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_0 = async_compile.triton('triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*u8', 'out_ptr0': '*u8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 49152}, 'backend_hash': 'B49614C0BBA23CA71245E046F2A6ABFCFD211E8FB0FFF5563191062F952B4CD3', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp9 = tl.load(in_ptr0 + (x0), None)
    tmp0 = (32*(((x0 // 4) % 4)) + (((x0 // 16) % 32))).to(tl.int64)
    tmp1 = (tmp0).to(tl.int64)
    tmp2 = tl.full([1], 19, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = (4*(x0 // 512) + ((x0 % 4))).to(tl.int64)
    tmp5 = (tmp4).to(tl.int64)
    tmp6 = tl.full([1], 128, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp3 | tmp7
    tmp10 = tl.full([1], 0, tl.uint8)
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tl.store(out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /data/users/eellison/pytorch/agent_space/nested_quant_comparison/cache_pr191974_padded_tests_before3/tmp70wke3_g/ge/cge4hwj5qwhqgj3vn4d3lgw6ql7ihjxz4pgsis6v6tqbxaczzf2i.py
# Topologically Sorted Source Nodes: [arange, row, floordiv, mul, offset, arange_1, col, floordiv_1, mul_2, offset_1, mod, mul_3, offset_2, floordiv_2, mod_1, mul_4, offset_3, mod_2, offset_4, output_1], Original ATen: [aten.arange, aten.unsqueeze, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten._unsafe_index_put]
# Source node to ATen node mapping:
#   arange => iota
#   arange_1 => iota_1
#   col => unsqueeze_1
#   floordiv => div
#   floordiv_1 => div_1
#   floordiv_2 => div_2
#   mod => remainder
#   mod_1 => remainder_1
#   mod_2 => remainder_2
#   mul => mul
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   offset => mul_1
#   offset_1 => add
#   offset_2 => add_1
#   offset_3 => add_2
#   offset_4 => add_3
#   output_1 => index_put
#   row => unsqueeze
# Graph fragment:
#   %arg0_1 : Tensor "u8[19, 128][128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %index_put_1 : Tensor "u8[16384][1]cuda:0" = PlaceHolder[target=index_put_1]
#   %iota : Tensor "i64[19][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (19,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 1), kwargs = {})
#   %div : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze, 128), kwargs = {rounding_mode: floor})
#   %mul : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 32), kwargs = {})
#   %mul_1 : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 512), kwargs = {})
#   %iota_1 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, 0), kwargs = {})
#   %div_1 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze_1, 4), kwargs = {rounding_mode: floor})
#   %mul_2 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, 512), kwargs = {})
#   %add : Tensor "i64[19, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %remainder : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze, 32), kwargs = {})
#   %mul_3 : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder, 16), kwargs = {})
#   %add_1 : Tensor "i64[19, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_3), kwargs = {})
#   %div_2 : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%unsqueeze, 32), kwargs = {rounding_mode: floor})
#   %remainder_1 : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%div_2, 4), kwargs = {})
#   %mul_4 : Tensor "i64[19, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%remainder_1, 4), kwargs = {})
#   %add_2 : Tensor "i64[19, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_4), kwargs = {})
#   %remainder_2 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%unsqueeze_1, 4), kwargs = {})
#   %add_3 : Tensor "i64[19, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %remainder_2), kwargs = {})
#   %index_put : Tensor "u8[16384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%empty, [%add_3], %arg0_1), kwargs = {})
#   return %buf1
triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_1 = async_compile.triton('triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*u8', 'out_ptr0': '*u8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 7296}, 'backend_hash': 'B49614C0BBA23CA71245E046F2A6ABFCFD211E8FB0FFF5563191062F952B4CD3', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (16*x1 + 512*(x0 // 4) + ((x0 % 4))), tmp0, xmask)
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
        arg0_1, = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((16384, ), (1, ), torch.uint8)
            buf2 = empty_strided_cuda((16384, ), (1, ), torch.uint8)
            # Topologically Sorted Source Nodes: [linear, linear_1, linear_2, linear_3, row_chunk, mul_5, row_outer, mul_6, add_4, row_lane, logical_row, ge, col_outer, mul_7, col_inner, logical_col, ge_1, padding_mask, padding, index_put_default], Original ATen: [aten.arange, aten.floor_divide, aten.mul, aten.remainder, aten.add, aten.ge, aten.bitwise_or, aten.zeros, aten.index_put]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_arange_bitwise_or_floor_divide_ge_index_put_mul_remainder_zeros_0.run(buf0, buf2, 16384, stream=raw_stream0)
            del buf0
            assert_size_stride(arg0_1, (19, 128), (128, 1), 'input')
            arg0_1 = copy_if_misaligned(arg0_1)
            # Topologically Sorted Source Nodes: [arange, row, floordiv, mul, offset, arange_1, col, floordiv_1, mul_2, offset_1, mod, mul_3, offset_2, floordiv_2, mod_1, mul_4, offset_3, mod_2, offset_4, output_1], Original ATen: [aten.arange, aten.unsqueeze, aten.floor_divide, aten.mul, aten.add, aten.remainder, aten._unsafe_index_put]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_index_put_add_arange_floor_divide_mul_remainder_unsqueeze_1.run(arg0_1, buf2, 2432, stream=raw_stream0)
            del arg0_1
        return (buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((19, 128), (128, 1), device='cuda:0', dtype=torch.uint8)
    return [arg0_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
