#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>

namespace at::native {

void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::random_from_to_kernel(iter, range, base, gen);
}

void random_full_64_bits_range_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::random_full_64_bits_range_kernel(iter, gen);
}

void random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::random_kernel(iter, gen);
}

REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel)
REGISTER_DISPATCH(random_stub, &random_kernel)
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced entries of random_ / random_.from / random_.to (the random_impl and random_from_to_impl of
// ATen/native/DistributionTemplates.h on the sibling iterator), compiled here so the entries and
// the real hosts above instantiate the one kernel over the header's functors (DECISIONS E36):
// the tape's launch is eager's function object, not a twin. The philox increment is declared on
// the tape as an expression of the element count (ti/DistributionSym.cuh). Outside a trace an
// entry runs the same launch in ordinary mode, which is how the parity test compares it with the
// real op.
#include <ATen/cuda/host_trace/ti/DistributionSym.cuh>
#include <ATen/cuda/host_trace/ti/Ops.h>
#include <ATen/native/DistributionTemplates.h>

namespace at::cuda::host_trace::ti {

Tensor& random_from_to_traced(Tensor& self, int64_t from, std::optional<int64_t> to_opt, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("random_", self, gen_);
  TensorIteratorSym iter = nullary_iterator(self);
  uint64_t range = 0;
  if (to_opt.has_value()) {
    // [from, to)
    int64_t to = *to_opt;
    TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_update_from_to", [&] {
        from = at::native::templates::update_from<scalar_t>(from);
        to = at::native::templates::update_to<scalar_t>(to);
        TORCH_CHECK(from < to, "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
      });
    }
    at::native::templates::check_from_to_in_range(from, to - 1, self.dtype());
    if (is_empty(self)) {
      return self;
    }
    range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    at::native::templates::cuda::random_from_to_kernel(iter, range, from, gen);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    int64_t to_inc = 0;
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "random_from_to_range_calc", [&] {
        constexpr int64_t scalar_t_max = static_cast<int64_t>(1) << std::numeric_limits<scalar_t>::digits;
        to_inc = scalar_t_max > std::numeric_limits<int64_t>::max() ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(scalar_t_max);
        from = at::native::templates::update_from<scalar_t>(from);
        TORCH_CHECK(from < to_inc, "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=", from, " > to_inc=", to_inc);
      });
    } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
      AT_DISPATCH_V2(self.scalar_type(), "random_from_to_range_calc", AT_WRAP([&] {
        if constexpr (std::is_same_v<scalar_t, bool>) {
          to_inc = static_cast<int64_t>(true);
        } else {
          to_inc = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
        }
      }), AT_EXPAND(AT_INTEGRAL_TYPES_V2), kBool);
    } else {
      TORCH_CHECK(false, "random_from_to_impl handles only integral, floating-point and boolean types");
    }
    at::native::templates::check_from_to_in_range(from, to_inc, self.dtype());
    if (is_empty(self)) {
      return self;
    }
    range = static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
    at::native::templates::cuda::random_from_to_kernel(iter, range, from, gen);
  } else {
    // [std::numeric_limits<int64_t>::lowest(), std::numeric_limits<int64_t>::max()]
    // range = 2^64
    if (is_empty(self)) {
      return self;
    }
    at::native::templates::cuda::random_full_64_bits_range_kernel(iter, gen);
  }
  return self;
}

Tensor& random_traced(Tensor& self, const std::optional<at::Generator>& gen_) {
  c10::cuda::CUDAGuard device_guard(self.device());
  auto* gen = distribution_generator("random_", self, gen_);
  if (is_empty(self)) {
    return self;
  }
  TensorIteratorSym iter = nullary_iterator(self);
  at::native::templates::cuda::random_kernel(iter, gen);
  return self;
}

} // namespace at::cuda::host_trace::ti
