// The reductions that opt into the traced sibling iterator (the entries in
// ReduceSumProdKernel.cu, ReduceMomentKernel.cu, ReduceMaxValuesKernel.cu).
// Each entry does what the op's CUDA kernel host does
// around gpu_reduce_kernel (ReduceOps.cpp's make_reduction, the reduce
// kernel's dispatch) on the sibling iterator; the trace mode calls it in
// place of the real op, and outside a trace it runs the same launches in
// ordinary mode, which is how the parity test compares it with the real op.
// v1: one input of one dtype reduced into an output of the same dtype
// (sum / mean promote nothing; amax never does); dims=[] reduces every dim.
// sum's out= form (sum.IntList_out, what at::sum_out issues) writes the
// caller's tensor when it already has the result's shape and dtype.
#pragma once
#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

#include <optional>

namespace at::cuda::host_trace::ti {

TORCH_CUDA_CU_API Tensor sum_traced(const Tensor& self, IntArrayRef dims, bool keepdim, const std::optional<Tensor>& out = std::nullopt);
TORCH_CUDA_CU_API Tensor mean_traced(const Tensor& self, IntArrayRef dims, bool keepdim);
TORCH_CUDA_CU_API Tensor amax_traced(const Tensor& self, IntArrayRef dims, bool keepdim);
// all / any (ReduceLogicKernel.cu): the input in its own dtype reduced into
// a bool result, the input's dims=[] form for all.default / any.default
TORCH_CUDA_CU_API Tensor allany_traced(const Tensor& self, IntArrayRef dims, bool keepdim, bool all_of, const std::optional<Tensor>& out = std::nullopt);

} // namespace at::cuda::host_trace::ti
