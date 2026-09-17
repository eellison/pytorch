#pragma once

#include <c10/macros/Macros.h>
#include <ATen/native/Lerp.h>

namespace at::native::ufunc {

// Lerp.cu lerp_scalar_kernel's and lerp_tensor_kernel's real-type device
// bodies (CUDAFunctor_lerp_Scalar / CUDAFunctor_lerp_Tensor): Lerp.h's lerp
// over (self, end) with the weight in opmath_t (the Scalar form) or scalar_t
// (the Tensor form)
template <typename T, typename W>
C10_HOST_DEVICE C10_ALWAYS_INLINE T lerp(T self_val, T end_val, W weight_val) {
  return at::native::lerp(self_val, end_val, weight_val);
}

} // namespace at::native::ufunc
