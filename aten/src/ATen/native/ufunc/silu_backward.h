#pragma once

#include <c10/macros/Macros.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace at::native::ufunc {

// ActivationSiluKernel.cu silu_backward_kernel's device body
// (CUDAFunctor_silu_backward): (dy, x), the sigmoid and the product in opmath
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T silu_backward(T dy, T x) {
  using opmath_t = at::opmath_type<T>;
  const opmath_t dy_acc = static_cast<opmath_t>(dy);
  const opmath_t x_acc = static_cast<opmath_t>(x);
  const opmath_t s_acc =
      opmath_t(1) / (opmath_t(1) + c10::cuda::compat::exp(-x_acc));
  return dy_acc * s_acc * (opmath_t(1) + x_acc * (opmath_t(1) - s_acc));
}

} // namespace at::native::ufunc
