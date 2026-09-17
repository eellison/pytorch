#pragma once

#include <c10/macros/Macros.h>

namespace at::native::ufunc {

// Dropout.cu masked_scale_kernel's device body for a bool mask
// (CUDAFunctor_native_dropout_backward): (grad, mask), the scale in
// acc_type<scalar_t, true>
template <typename T, typename A>
C10_HOST_DEVICE C10_ALWAYS_INLINE T native_dropout_backward(T src_val, bool mask_val, A scale) {
  return (float)mask_val * src_val * scale;
}

} // namespace at::native::ufunc
