#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Normalization.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include <ATen/Dispatch.h>

namespace at::native {
namespace {

template <typename scalar_t>
struct RenormScaleFactorFunctor {
  scalar_t maxnorm_s;
  __device__ scalar_t operator()(scalar_t norm) const {
    const auto eps = static_cast<scalar_t>(1e-7);
    const auto one = static_cast<scalar_t>(1.0);
    return (norm > maxnorm_s) ?
        maxnorm_s / (norm + eps) : one;
  }
  auto host_trace_fields() const {
    return std::tie(maxnorm_s);
  }
};

void renorm_scale_factor_impl(TensorIteratorBase& iter, double maxnorm) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "renorm_scale_factor_cpu", [&] {
    const auto maxnorm_s = static_cast<scalar_t>(maxnorm);
    gpu_kernel(iter, RenormScaleFactorFunctor<scalar_t>{maxnorm_s});
  });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(renorm_scale_factor_stub, &renorm_scale_factor_impl)

}  // namespace at::native
