#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/NumericUtils.h>
#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/Ops.h>
#include <ATen/cuda/host_trace/ReduceSym.cuh>
#endif

#include <thrust/pair.h>

namespace at::native {

template <typename acc_t>
struct MinNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
      return (at::_isnan(a) || a < b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t=scalar_t>
void min_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> (MinNanFunctor<acc_t>()),
    at::numeric_limits<acc_t>::upper_bound());
}

void min_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cuda", [&]() {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

void min_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_cuda", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinOps<scalar_t>{},
      thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::upper_bound(), 0));
  });
}

void min_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_all_cuda", [&] {
    min_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda)

} // namespace at::native

#if !defined(USE_ROCM)
// Traced host (ATen/cuda/host_trace/Ops.h)
namespace at::cuda::host_trace {

// min_values_kernel_cuda_impl
TensorBase amin(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim, const TensorBase& out) {
  TensorBase result = out;
  auto iter = make_reduction(rec, result, self, dims, keepdim);
  if (iter.numel() == 0) {
    decline("an empty amin");
  }
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(0), "min_values_cuda", [&]() {
    Param ops(at::native::func_wrapper<scalar_t>(at::native::MinNanFunctor<scalar_t>()));
    gpu_reduce_kernel<scalar_t, scalar_t>(rec, iter, ops, at::numeric_limits<scalar_t>::upper_bound());
  });
  return result;
}

// min(), min_unary_out: min_all_kernel_impl over self.contiguous()
TensorBase min_all(Recorder& rec, const TensorBase& self, const TensorBase& out) {
  if (self.sym_numel() == 0) {
    decline("a min of an empty tensor");
  }
  if (out.defined() && (out.dim() != 0 || out.scalar_type() != self.scalar_type())) {
    decline("a min into an out= tensor it resizes or of another dtype");
  }
  return amin(rec, contiguous(rec, self), {}, false, out);
}

// minmax_out_impl, min_kernel_impl
std::tuple<TensorBase, TensorBase> min_dim(Recorder& rec, const TensorBase& self, int64_t dim, bool keepdim) {
  TensorBase values;
  TensorBase indices;
  if (auto iter = make_minmax_reduction(rec, values, indices, self, dim, keepdim)) {
    AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter->dtype(2), "min_cuda", [&]() {
      Param ops(at::native::MinOps<scalar_t>{});
      gpu_reduce_kernel<scalar_t, scalar_t>(rec, *iter, ops, thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::upper_bound(), 0));
    });
  }
  return {values, indices};
}

} // namespace at::cuda::host_trace
#endif
