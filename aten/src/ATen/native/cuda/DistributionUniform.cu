#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());
  templates::cuda::uniform_kernel(iter, from, to, generator);
}

REGISTER_DISPATCH(uniform_stub, &uniform_kernel)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced entries of uniform_ (the uniform_impl_ of
// ATen/native/DistributionTemplates.h on the sibling iterator), compiled here so the entries and
// the real hosts above instantiate the one kernel over the header's functors (DECISIONS E36):
// the tape's launch is eager's function object, not a twin. The philox increment is declared on
// the tape as an expression of the element count (ti/DistributionSym.cuh). Outside a trace an
// entry runs the same launch in ordinary mode, which is how the parity test compares it with the
// real op.
#include <ATen/cuda/host_trace/ti/DistributionSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>

namespace at::cuda::host_trace::ti {

Tensor& uniform_traced(Tensor& self, double from, double to, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("uniform_", self, gen_);
  if (self.is_complex()) {
    decline("host_trace: uniform_ on a complex tensor is not traced (declined)");
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "check_uniform_bounds", [&] {
    [[maybe_unused]] const auto dtype = self.dtype();
    const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
    const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
    TORCH_CHECK(from >= min && from <= max, "from", " is out of bounds for ", dtype);
    TORCH_CHECK(to >= min && to <= max, "to", " is out of bounds for ", dtype);
    TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
    TORCH_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
          "uniform_ expects to-from <= std::numeric_limits<", toString(self.scalar_type()),
          ">::max(), but found to=", to, " and from=", from,
          " which result in to-from to exceed the limit");
    from = std::clamp(from, min, max);
    to = std::clamp(to, min, max);
  });
  if (is_empty(self)) {
    return self;
  }
  TensorIteratorSym iter = nullary_iterator(self);
  at::native::templates::cuda::uniform_kernel(iter, from, to, gen);
  return self;
}

} // namespace at::cuda::host_trace::ti
