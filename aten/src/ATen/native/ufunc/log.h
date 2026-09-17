#pragma once

#include <c10/macros/Macros.h>

#include <cmath>

namespace at::native::ufunc {

// UnaryLogKernels.cu log_kernel_cuda's real-type device body (CUDAFunctor_log)
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T log(T a) {
  return ::log(a);
}

} // namespace at::native::ufunc
