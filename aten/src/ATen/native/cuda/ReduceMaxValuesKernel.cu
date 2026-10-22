#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <ATen/native/cuda/Reduce.cuh>
#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/Ops.h>
#include <ATen/cuda/host_trace/ReduceSym.cuh>
#endif

#include <thrust/pair.h>

namespace at::native {

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a > b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void max_values_kernel_cuda_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      func_wrapper<acc_t>(MaxNanFunctor<acc_t>()),
      at::numeric_limits<acc_t>::lower_bound());
}

void max_values_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cuda", [&]() {
        max_values_kernel_cuda_impl<scalar_t>(iter);
      });
}

void max_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "max_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MaxOps<scalar_t>{},
            thrust::pair<scalar_t, int64_t>(
                at::numeric_limits<scalar_t>::lower_bound(), 0));
      });
}

void max_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "max_all_cuda", [&] {
    max_values_kernel_cuda_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda)

} // namespace at::native

#if !defined(USE_ROCM)
// Traced host (ATen/cuda/host_trace/Ops.h)
namespace at::cuda::host_trace {

// max_values_kernel_cuda_impl
TensorBase amax(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim, const TensorBase& out) {
  TensorBase result = out;
  auto iter = make_reduction(rec, result, self, dims, keepdim);
  if (iter.numel() == 0) {
    decline("an empty amax");
  }
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(0), "max_values_cuda", [&]() {
    Param ops(at::native::func_wrapper<scalar_t>(at::native::MaxNanFunctor<scalar_t>()));
    gpu_reduce_kernel<scalar_t, scalar_t>(rec, iter, ops, at::numeric_limits<scalar_t>::lower_bound());
  });
  return result;
}

// max(), max_unary_out: max_all_kernel_impl over self.contiguous()
TensorBase max_all(Recorder& rec, const TensorBase& self, const TensorBase& out) {
  if (self.sym_numel() == 0) {
    decline("a max of an empty tensor");
  }
  if (out.defined() && (out.dim() != 0 || out.scalar_type() != self.scalar_type())) {
    decline("a max into an out= tensor it resizes or of another dtype");
  }
  return amax(rec, contiguous(rec, self), {}, false, out);
}

// minmax_out_impl, max_kernel_impl
std::tuple<TensorBase, TensorBase> max_dim(Recorder& rec, const TensorBase& self, int64_t dim, bool keepdim) {
  TensorBase values;
  TensorBase indices;
  if (auto iter = make_minmax_reduction(rec, values, indices, self, dim, keepdim)) {
    AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter->dtype(2), "max_cuda", [&]() {
      Param ops(at::native::MaxOps<scalar_t>{});
      gpu_reduce_kernel<scalar_t, scalar_t>(rec, *iter, ops, thrust::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::lower_bound(), 0));
    });
  }
  return {values, indices};
}

} // namespace at::cuda::host_trace
#endif
