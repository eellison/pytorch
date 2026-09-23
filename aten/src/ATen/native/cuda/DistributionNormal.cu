#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::normal_kernel(self, mean, std, generator);
}

REGISTER_DISPATCH(normal_stub, &normal_kernel)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced entries of normal_ (the normal_impl_ of
// ATen/native/DistributionTemplates.h on the sibling iterator), compiled here so the entries and
// the real hosts above instantiate the one kernel over the header's functors (DECISIONS E36):
// the tape's launch is eager's function object, not a twin. The philox increment is declared on
// the tape as an expression of the element count (ti/DistributionSym.cuh). Outside a trace an
// entry runs the same launch in ordinary mode, which is how the parity test compares it with the
// real op.
#include <ATen/cuda/host_trace/ti/DistributionSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace::ti {

Tensor& normal_traced(Tensor& self, double mean, double std, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("normal_", self, gen_);
  TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);
  if (is_empty(self)) {
    return self;
  }
  if (self.is_complex()) {
    decline("host_trace: normal_ on a complex tensor is not traced (declined)");
  }
  TensorIteratorSym iter = nullary_iterator(self);
  at::native::templates::cuda::normal_kernel(iter, mean, std, gen);
  return self;
}

} // namespace at::cuda::host_trace::ti
