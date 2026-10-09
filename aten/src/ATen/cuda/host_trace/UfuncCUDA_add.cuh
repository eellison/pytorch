// The traced host of add.Tensor. Only the generated UfuncCUDA_add.cu, which
// holds the CUDAFunctor_add kernels, includes this, so add() is defined once.
#pragma once
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/cuda/host_trace/Ops.h>

namespace at::cuda::host_trace {

TensorBase add(Recorder& rec, const TensorBase& a, const TensorBase& b, const Scalar& alpha) {
  auto iter = TensorIteratorSym::binary_op(rec, a, b);
  if (isComplexType(iter.common_dtype())) {
    decline("complex add");
  }
  // alpha_check; BinaryOps.h can't be included here (it redeclares add_stub)
  if (!isFloatingType(iter.common_dtype()) && !alpha.isIntegral(false)) {
    decline("an integral add of a floating alpha");
  }
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, iter.common_dtype(), "ufunc_add_CUDA", [&]() {
    using opmath_t = at::opmath_type<scalar_t>;
    gpu_kernel(rec, iter, at::native::CUDAFunctor_add<scalar_t>(alpha.to<opmath_t>()));
  });
  return iter.output();
}

} // namespace at::cuda::host_trace
