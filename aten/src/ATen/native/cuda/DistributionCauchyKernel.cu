#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::cauchy_kernel(iter, median, sigma, generator);
}

REGISTER_DISPATCH(cauchy_stub, &cauchy_kernel)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced entries of cauchy_ (the cauchy_impl_ of
// ATen/native/DistributionTemplates.h on the sibling iterator), compiled here so the entries and
// the real hosts above instantiate the one kernel over the header's functors (DECISIONS E36):
// the tape's launch is eager's function object, not a twin. The philox increment is declared on
// the tape as an expression of the element count (ti/DistributionSym.cuh). Outside a trace an
// entry runs the same launch in ordinary mode, which is how the parity test compares it with the
// real op.
#include <ATen/cuda/host_trace/ti/DistributionSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace::ti {

Tensor& cauchy_traced(Tensor& self, double median, double sigma, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("cauchy_", self, gen_);
  TORCH_CHECK(sigma > 0.0, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "Cauchy distribution is a continuous probability distribution. dtype must be a floating point but you specified ", self.dtype());
  if (is_empty(self)) {
    return self;
  }
  TensorIteratorSym iter = nullary_iterator(self);
  at::native::templates::cuda::cauchy_kernel(iter, median, sigma, gen);
  return self;
}

} // namespace at::cuda::host_trace::ti
