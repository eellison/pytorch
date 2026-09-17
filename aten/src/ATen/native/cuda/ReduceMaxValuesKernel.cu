#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
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

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of max_values_kernel_cuda_impl, compiled here
// so the sibling and the real host above instantiate the one kernel over ReduceOp over MaxNanFunctor
// (DECISIONS E36): the tape's launch is eager's function object, not a twin. Outside a trace the
// entry runs the same launches in ordinary mode, which is how the parity test compares it with
// the real op.
#include <ATen/cuda/host_trace/ti/ReduceOps.h>
#include <ATen/cuda/host_trace/ti/ReduceSym.cuh>

#include <ATen/native/ReduceOpsUtils.h>

namespace at::cuda::host_trace::ti {

Tensor amax_traced(const Tensor& self, IntArrayRef dims, bool keepdim) {
  const ScalarType dtype = self.scalar_type();
  if (dtype == kBool || at::isComplexType(dtype)) {
    decline(c10::str("host_trace: amax on ", dtype, " is not traced (declined)"));
  }
  if (self.sym_numel() == 0) {
    at::native::zero_numel_check_dims(self, dims, "amax()");
  }
  ReductionSym r = make_reduction(self, dims, keepdim, dtype);
  if (r.iter.numel() == 0) {
    return r.result;
  }
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, dtype, "amax_traced", [&]() {
    gpu_reduce_kernel<scalar_t, scalar_t>(
        r.iter,
        at::native::func_wrapper<scalar_t>(at::native::MaxNanFunctor<scalar_t>()),
        at::numeric_limits<scalar_t>::lower_bound());
  });
  return r.result;
}

} // namespace at::cuda::host_trace::ti
