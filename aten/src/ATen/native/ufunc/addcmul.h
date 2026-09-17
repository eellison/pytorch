#pragma once

#include <c10/macros/Macros.h>
#include <ATen/native/cuda/DeviceAddCmulCdiv.cuh>

#include <functional>

namespace at::native::ufunc {

// PointwiseOpsKernel.cu addcmul_cuda_kernel's real-type device body
// (CUDAFunctor_addcmul): the value in acc_type<scalar_t, true>, the arithmetic
// pointwise_op_impl's
template <typename T, typename A>
C10_HOST_DEVICE C10_ALWAYS_INLINE T addcmul(T a, T b, T c, A alpha) {
  return at::native::pointwise_op_impl<A>(a, b, c, alpha, std::multiplies<A>());
}

} // namespace at::native::ufunc
