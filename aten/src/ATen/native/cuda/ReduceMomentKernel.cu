#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

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

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of mean_kernel_impl, compiled here
// so the sibling and the real host above instantiate the one kernel over ReduceOp over MeanOps
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/ti/ReduceOps.h>
#include <ATen/cuda/host_trace/ti/ReduceSym.cuh>

#include <cstring>
#include <limits>

namespace at::cuda::host_trace::ti {

namespace {

template <typename factor_t>
int64_t mean_factor_bits(const int64_t* a, size_t) {
  // mean_kernel_impl: static_cast<factor_t>(num_output_elements) / numel
  const factor_t factor = static_cast<factor_t>(a[0]) / static_cast<factor_t>(a[1]);
  if constexpr (sizeof(factor_t) == 8) {
    int64_t bits = 0;
    std::memcpy(&bits, &factor, sizeof(bits));
    return bits;
  } else {
    uint32_t bits = 0;
    std::memcpy(&bits, &factor, sizeof(bits));
    return static_cast<int64_t>(bits);
  }
}

template <typename scalar_t, typename acc_t, typename factor_t, typename out_t, int vt0, int input_vec_size>
void mean_launch(TensorIteratorSym& iter) {
  using Ops = at::native::MeanOps<scalar_t, acc_t, factor_t, out_t>;
  const c10::SymInt factor_bits = opaque(
      sizeof(factor_t) == 8 ? "mean_factor_bits_f64" : "mean_factor_bits_f32",
      {iter.num_output_elements(), iter.numel()},
      &mean_factor_bits<factor_t>,
      "rebind");
  gpu_reduce_kernel<scalar_t, out_t, vt0, input_vec_size>(
      iter, Ops{factor_t(0)}, 0, [&](auto& reduce) { reduce.ops.factor = factor_bits; });
}

} // namespace

Tensor mean_traced(const Tensor& self, IntArrayRef dims, bool keepdim) {
  const ScalarType dtype = self.scalar_type();
  if (!at::isFloatingType(dtype)) {
    decline(c10::str("host_trace: mean on ", dtype, " is not traced (declined)"));
  }
  ReductionSym r = make_reduction(self, dims, keepdim, dtype);
  if (r.iter.numel() == 0) {
    r.result.fill_(std::numeric_limits<double>::quiet_NaN());
    return r.result;
  }
  switch (dtype) {
    case kHalf:
      mean_launch<at::Half, float, float, at::Half, 4, 8>(r.iter);
      break;
    case kBFloat16:
      mean_launch<at::BFloat16, float, float, at::BFloat16, 4, 8>(r.iter);
      break;
    case kFloat:
      mean_launch<float, float, float, float, 4, 4>(r.iter);
      break;
    case kDouble:
      mean_launch<double, double, double, double, 4, 4>(r.iter);
      break;
    default:
      decline(c10::str("host_trace: mean on ", dtype, " is not traced (declined)"));
  }
  return r.result;
}

} // namespace at::cuda::host_trace::ti
