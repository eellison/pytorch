#pragma once

#include <c10/macros/Macros.h>

#include <type_traits>

namespace at::native::ufunc {

// MaxMinElementwiseKernel.cu minimum_kernel_cuda's device body (the generated
// CUDAFunctor_minimum): bool is `and`, an integer type ::min, a floating type
// ::min with either nan propagated
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T minimum(T a, T b) {
  if constexpr (std::is_same_v<T, bool>) {
    return a && b;
  } else if constexpr (std::is_integral_v<T>) {
    return ::min(a, b);
  } else {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return ::min(a, b);
    }
  }
}

} // namespace at::native::ufunc
