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
#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/cuda/host_trace/Ops.h>
#endif

namespace at::native {
namespace {

template <typename scalar_t>
struct GeluTanhFunctor {
  __device__ scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
    constexpr opmath_t kKappa = 0.044715;
    auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
    auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
    return opmath_t(0.5) * static_cast<opmath_t>(x) * (opmath_t(1) + c10::cuda::compat::tanh(inner));
  }
};

template <typename scalar_t>
struct GeluErfFunctor {
  __device__ scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    constexpr opmath_t kAlpha = M_SQRT1_2;
    return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
  }
};

} // namespace

void GeluCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, it.dtype(), "GeluCUDAKernelImpl", [&]() {
      gpu_kernel(it, GeluTanhFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, it.dtype(), "GeluCUDAKernelImpl", [&]() {
      gpu_kernel(it, GeluErfFunctor<scalar_t>());
    });
  }
}

void GeluBackwardCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        it.dtype(), "GeluBackwardCUDAKernelImpl", [&]() {
          gpu_kernel(it, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
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
        });
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        it.dtype(), "GeluBackwardCUDAKernelImpl", [&]() {
          gpu_kernel(it, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
            constexpr opmath_t kAlpha = M_SQRT1_2;
            const opmath_t cdf =
                opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
            const opmath_t pdf =
                c10::cuda::compat::exp(
                    opmath_t(-0.5) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x)) *
                kBeta;
            return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
          });
        });
  }
}

} // namespace at::native

#if !defined(USE_ROCM)
// Traced host (ATen/cuda/host_trace/Ops.h)
namespace at::cuda::host_trace {

TensorBase gelu(Recorder& rec, const TensorBase& a, std::string_view approximate) {
  const bool tanh = at::native::get_gelutype_enum(approximate) == at::native::GeluType::Tanh;
  auto iter = TensorIteratorSym::unary_op(rec, a);
  if (isComplexType(iter.dtype(0))) {
    decline("complex gelu");
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(0), "GeluCUDAKernelImpl", [&]() {
    if (tanh) {
      gpu_kernel(rec, iter, at::native::GeluTanhFunctor<scalar_t>());
    } else {
      gpu_kernel(rec, iter, at::native::GeluErfFunctor<scalar_t>());
    }
  });
  return iter.output();
}

} // namespace at::cuda::host_trace
#endif
