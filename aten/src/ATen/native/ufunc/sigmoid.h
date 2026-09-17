#pragma once

#include <c10/macros/Macros.h>
#include <ATen/OpMathType.h>

#include <cmath>

namespace at::native::ufunc {

// UnarySpecialOpsKernel.cu sigmoid_kernel_cuda's real-type device body:
// the generated CUDAFunctor_sigmoid eager's host and the traced sibling launch
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T sigmoid(T a) {
  using opmath_t = at::opmath_type<T>;
  const auto one = opmath_t{1};
  return static_cast<T>(one / (one + std::exp(-opmath_t{a})));
}

} // namespace at::native::ufunc
