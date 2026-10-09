#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/Ops.h>
#include <ATen/cuda/host_trace/ReduceSym.cuh>
#endif

#include <thrust/pair.h>

namespace at::native {

template <typename scalar_t, typename out_t=scalar_t>
void std_var_kernel_impl(TensorIterator& iter, double correction, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  using accscalar_t = at::acc_type<scalar_t, true>;
  using ops_t = WelfordOps<scalar_t, accscalar_t, int32_t, thrust::pair<out_t, out_t>>;
  ops_t ops(static_cast<accscalar_t>(correction), take_sqrt);
  gpu_reduce_kernel<scalar_t, out_t, 2>(iter, ops, typename ops_t::acc_t{});
}

static void std_var_kernel_cuda(TensorIterator& iter, double correction, bool take_sqrt) {
  const auto input_dtype = iter.input_dtype();
  if (input_dtype == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::Half, float>(iter, correction, take_sqrt);
  } else if (input_dtype == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::BFloat16, float>(iter, correction, take_sqrt);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    iter.dtype(), "std_cuda", [&]() {
      std_var_kernel_impl<scalar_t>(iter, correction, take_sqrt);
    });
  }
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  //  returns acc_t for all non-complex dtypes and returns T for c10::complex<T>
  constexpr bool is_16_bits = sizeof(scalar_t) == 2;
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor = static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  if constexpr (is_16_bits) {
    gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(iter, MeanOps<scalar_t, acc_t, factor_t, out_t> {factor});
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, MeanOps<scalar_t, acc_t, factor_t, out_t> {factor});
  }
}

static void mean_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::Half, float, float>(iter);
  } else if(iter.dtype() == kBFloat16) {
    mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cuda", [&]() {
      mean_kernel_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda)
REGISTER_DISPATCH(mean_stub, &mean_kernel_cuda)

} // namespace at::native

#if !defined(USE_ROCM)
// Traced host (ATen/cuda/host_trace/Ops.h)
namespace at::cuda::host_trace {

// mean_kernel_impl for a Half, BFloat16 or float self and result of its dtype;
// double's factor is a double, which no program row computes
TensorBase mean(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim) {
  const ScalarType dtype = self.scalar_type();
  if (dtype != kHalf && dtype != kBFloat16 && dtype != kFloat) {
    decline(c10::str("a ", dtype, " mean"));
  }
  TensorBase result;
  auto iter = make_reduction(rec, result, self, dims, keepdim);
  if (iter.numel() == 0) {
    decline("an empty mean");
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "mean_cuda", [&]() {
    using ops_t = at::native::MeanOps<scalar_t, float, float, scalar_t>;
    Param<ops_t> ops;
    // factor = static_cast<float>(num_output_elements) / numel
    ops.set_bits(ops.value().factor, rec.f32_div(iter.num_output_elements(), iter.numel()));
    if constexpr (sizeof(scalar_t) == 2) {
      gpu_reduce_kernel<scalar_t, scalar_t, 4, 8>(rec, iter, ops);
    } else {
      gpu_reduce_kernel<scalar_t, scalar_t>(rec, iter, ops);
    }
  });
  return result;
}

// std_var_kernel_impl for a floating self and result of its dtype
TensorBase std_var(Recorder& rec, const TensorBase& self, IntArrayRef dims, double correction, bool keepdim, bool take_sqrt) {
  const ScalarType dtype = self.scalar_type();
  if (dtype != kHalf && dtype != kBFloat16 && dtype != kFloat && dtype != kDouble) {
    decline(c10::str("a ", dtype, " var"));
  }
  TensorBase result;
  auto iter = make_reduction(rec, result, self, dims, keepdim);
  if (iter.numel() == 0) {
    decline("an empty var");
  }
  // warn_invalid_degrees_of_freedom's warning is eager's
  const double bound = std::floor(correction);
  if (!(bound < 1e18) || iter.numel() / iter.num_output_elements() <= static_cast<int64_t>(bound)) {
    decline("a var of degrees of freedom <= 0");
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, dtype, "std_cuda", [&]() {
    using accscalar_t = at::acc_type<scalar_t, true>;
    using ops_t = at::native::WelfordOps<scalar_t, accscalar_t, int32_t, thrust::pair<scalar_t, scalar_t>>;
    const Param<ops_t> ops(ops_t(static_cast<accscalar_t>(correction), take_sqrt));
    gpu_reduce_kernel<scalar_t, scalar_t, 2>(rec, iter, ops, typename ops_t::acc_t{});
  });
  return result;
}

} // namespace at::cuda::host_trace
#endif
