#pragma once

#include <c10/macros/Macros.h>

namespace at::native::ufunc {

// BinaryMiscOpsKernels.cu mse_kernel_cuda's device body (CUDAFunctor_mse_loss):
// (self, target)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T mse_loss(T a, T b) {
  auto diff = a - b;
  return diff * diff;
}

} // namespace at::native::ufunc
