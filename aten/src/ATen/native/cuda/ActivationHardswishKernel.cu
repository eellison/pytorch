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
struct HardswishFunctor {
  __device__ scalar_t operator()(scalar_t self_val) const {
    const opmath_t zero(0.0f);
    const opmath_t one_sixth(1.0f / 6.0f);
    const opmath_t three(3.0f);
    const opmath_t six(6.0f);
    opmath_t x = static_cast<opmath_t>(self_val);
    return x * std::min(std::max(x + three, zero), six) * one_sixth;
  }
};

void hardswish_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardswish_cuda", [&]() {
    using opmath_t = at::opmath_type<scalar_t>;
    gpu_kernel(iter, HardswishFunctor<scalar_t, opmath_t>());
  });
}

template <typename scalar_t, typename opmath_t>
struct HardswishBackwardFunctor {
  __device__ scalar_t operator()(scalar_t grad_val_, scalar_t self_val_) const {
    const opmath_t zero(0.0f);
    const opmath_t three(3.0f);
    const opmath_t neg_three(-3.0f);
    const opmath_t one_half(0.5f);
    opmath_t grad_val = static_cast<opmath_t>(grad_val_);
    opmath_t self_val = static_cast<opmath_t>(self_val_);
    if (self_val <= neg_three) {
      return zero;
    } else if (self_val < three) {
      return grad_val * ((self_val / three) + one_half);
    } else {
      return grad_val;
    }
  }
};

void hardswish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardswish_backward_cuda", [&]() {
    using opmath_t = at::opmath_type<scalar_t>;
    gpu_kernel(iter, HardswishBackwardFunctor<scalar_t, opmath_t>());
  });
}
} // namespace

REGISTER_DISPATCH(hardswish_stub, &hardswish_kernel)
REGISTER_DISPATCH(hardswish_backward_stub, &hardswish_backward_kernel)

} // namespace at::native
