#pragma once

#include <c10/macros/Macros.h>

namespace at::native::ufunc {

// ActivationHardtanhKernel.cu hardtanh_backward_kernel's device body
// (CUDAFunctor_hardtanh_backward): (grad_output, self), the bounds in opmath_t
template <typename T, typename O>
C10_HOST_DEVICE C10_ALWAYS_INLINE T hardtanh_backward(T a, T b, O min_val, O max_val) {
  O aop = static_cast<O>(a);
  O bop = static_cast<O>(b);
  return (bop <= min_val) || (bop >= max_val) ? O(0) : aop;
}

} // namespace at::native::ufunc
