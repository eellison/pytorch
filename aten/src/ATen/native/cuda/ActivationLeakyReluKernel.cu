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
struct LeakyReluFunctor {
  opmath_t negval;
  __device__ scalar_t operator()(scalar_t a) const {
    opmath_t aop = static_cast<opmath_t>(a);
    return aop > opmath_t(0) ? aop : aop * negval;
  }
  auto host_trace_fields() const {
    return std::tie(negval);
  }
};

template <typename scalar_t, typename opmath_t>
struct LeakyReluBackwardFunctor {
  opmath_t negval;
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);
    return aop > opmath_t(0) ? bop : bop * negval;
  }
  auto host_trace_fields() const {
    return std::tie(negval);
  }
};

void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "leaky_relu_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negval = negval_.to<opmath_t>();
        gpu_kernel(iter, LeakyReluFunctor<scalar_t, opmath_t>{negval});
      });
}

void leaky_relu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "leaky_relu_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negval = negval_.to<opmath_t>();
        gpu_kernel(iter, LeakyReluBackwardFunctor<scalar_t, opmath_t>{negval});
      });
}
} // namespace

REGISTER_DISPATCH(leaky_relu_stub, &leaky_relu_kernel)
REGISTER_DISPATCH(leaky_relu_backward_stub, &leaky_relu_backward_kernel)

} // namespace at::native
