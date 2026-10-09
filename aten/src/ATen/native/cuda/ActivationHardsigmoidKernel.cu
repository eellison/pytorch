#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {
namespace {

template <typename scalar_t, typename opmath_t>
struct HardsigmoidFunctor {
  __device__ scalar_t operator()(scalar_t self_val) const {
    const opmath_t zero(0.0f);
    const opmath_t one_sixth(1.0f / 6.0f);
    const opmath_t three(3.0f);
    const opmath_t six(6.0f);
    opmath_t x = static_cast<opmath_t>(self_val);
    return std::min<opmath_t>(std::max<opmath_t>(x + three, zero), six) * one_sixth;
  }
};

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(iter, HardsigmoidFunctor<scalar_t, opmath_t>());
      });
}

template <typename scalar_t, typename opmath_t>
struct HardsigmoidBackwardFunctor {
  __device__ scalar_t operator()(scalar_t grad_val_, scalar_t self_val_) const {
    const opmath_t zero(0.0f);
    const opmath_t three(3.0f);
    const opmath_t neg_three(-3.0f);
    const opmath_t one_sixth(1.0f / 6.0f);
    opmath_t grad_val = static_cast<opmath_t>(grad_val_);
    opmath_t self_val = static_cast<opmath_t>(self_val_);
    return (self_val > neg_three && self_val < three)
        ? grad_val * one_sixth
        : zero;
  }
};

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(iter, HardsigmoidBackwardFunctor<scalar_t, opmath_t>());
      });
}

} // namespace

REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)
REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel)

} // namespace at::native
