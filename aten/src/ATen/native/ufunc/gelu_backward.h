#pragma once

#include <c10/macros/Macros.h>
#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <cmath>

namespace at::native::ufunc {

// ActivationGeluKernel.cu GeluBackwardCUDAKernelImpl's two device bodies
// (CUDAFunctor_gelu_backward_none / _tanh) over (dy, x): the erf form and the
// tanh approximation
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T gelu_backward_none(T dy, T x) {
  using opmath_t = at::opmath_type<T>;
  constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
  constexpr opmath_t kAlpha = M_SQRT1_2;
  const opmath_t cdf =
      opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
  const opmath_t pdf =
      c10::cuda::compat::exp(
          opmath_t(-0.5) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x)) *
      kBeta;
  return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T gelu_backward_tanh(T dy, T x) {
  using opmath_t = at::opmath_type<T>;
  constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
  constexpr opmath_t kKappa = 0.044715;
  auto x_sq = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
  auto x_cube = x_sq * static_cast<opmath_t>(x);
  auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
  auto tanh_inner = c10::cuda::compat::tanh(inner);

  auto left = opmath_t(0.5) * static_cast<opmath_t>(x);
  auto right = opmath_t(1) + tanh_inner;

  auto left_derivative = opmath_t(0.5) * right;

  auto tanh_derivative = opmath_t(1) - tanh_inner * tanh_inner;
  auto inner_derivative = kBeta * (opmath_t(1) + opmath_t(3) * kKappa * x_sq);
  auto right_derivative = left * tanh_derivative * inner_derivative;

  return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);
}

} // namespace at::native::ufunc
