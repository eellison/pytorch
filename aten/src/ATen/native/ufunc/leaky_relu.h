#pragma once

#include <c10/macros/Macros.h>
#include <ATen/OpMathType.h>

namespace at::native::ufunc {

// ActivationLeakyReluKernel.cu leaky_relu_kernel's device body
// (CUDAFunctor_leaky_relu): negval in opmath_t
template <typename T, typename O>
C10_HOST_DEVICE C10_ALWAYS_INLINE T leaky_relu(T a, O negval) {
  O aop = static_cast<O>(a);
  return aop > O(0) ? aop : aop * negval;
}

} // namespace at::native::ufunc
