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
struct HardtanhBackwardFunctor {
  opmath_t min_val;
  opmath_t max_val;
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    opmath_t aop = static_cast<opmath_t>(a);
    opmath_t bop = static_cast<opmath_t>(b);
    return (bop <= min_val) || (bop >= max_val) ? opmath_t(0) : aop;
  }
  auto host_trace_fields() const {
    return std::tie(min_val, max_val);
  }
};

void hardtanh_backward_kernel(
    TensorIterator& iter,
    const Scalar& min,
    const Scalar& max) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "hardtanh_backward_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto min_val = min.to<opmath_t>();
        auto max_val = max.to<opmath_t>();
        gpu_kernel(iter, HardtanhBackwardFunctor<scalar_t, opmath_t>{min_val, max_val});
      });
}
} // namespace

REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel)

} // namespace at::native
