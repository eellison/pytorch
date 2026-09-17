#pragma once

#include <c10/macros/Macros.h>

namespace at::native::ufunc {

// ActivationThresholdKernel.cu threshold_kernel_impl's device body
// (CUDAFunctor_threshold_backward, the forward threshold's too) over (self,
// grad): the threshold and the value, rounded to the tensor dtype by the host,
// are held in opmath_t (the comparison and the returned value convert as the
// scalar_t captures did)
template <typename T, typename O>
C10_HOST_DEVICE C10_ALWAYS_INLINE T threshold_backward(T self, T grad_output, O threshold, O value) {
  return self <= threshold ? static_cast<T>(value) : grad_output;
}

} // namespace at::native::ufunc
