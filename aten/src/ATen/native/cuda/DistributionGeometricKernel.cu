#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void geometric_kernel(TensorIteratorBase& iter, double p_, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::geometric_kernel(iter, p_, generator);
}

REGISTER_DISPATCH(geometric_stub, &geometric_kernel)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced entries of geometric_ (the geometric_impl_ of
// ATen/native/DistributionTemplates.h on the sibling iterator), compiled here so the entries and
// the real hosts above instantiate the one kernel over the header's functors (DECISIONS E36):
// the tape's launch is eager's function object, not a twin. The philox increment is declared on
// the tape as an expression of the element count (ti/DistributionSym.cuh). Outside a trace an
// entry runs the same launch in ordinary mode, which is how the parity test compares it with the
// real op.
#include <ATen/cuda/host_trace/ti/DistributionSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace::ti {

Tensor& geometric_traced(Tensor& self, double p, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("geometric_", self, gen_);
  TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);
  if (is_empty(self)) {
    return self;
  }
  TensorIteratorSym iter = nullary_iterator(self);
  at::native::templates::cuda::geometric_kernel(iter, p, gen);
  return self;
}

} // namespace at::cuda::host_trace::ti
