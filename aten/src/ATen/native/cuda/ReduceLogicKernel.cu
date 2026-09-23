#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/Dispatch.h>

namespace at::native {

// and_kernel_cuda / or_kernel_cuda's combine lambdas as named functors: the
// traced sibling at the end of this file launches them too (DECISIONS E36)
template <typename scalar_t>
struct AndFunctor {
  __device__ bool operator()(bool acc, scalar_t val) const {
    return (acc && static_cast<bool>(val));
  }
};

template <typename scalar_t>
struct OrFunctor {
  __device__ bool operator()(bool acc, scalar_t val) const {
    return (acc || static_cast<bool>(val));
  }
};

void and_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(iter, func_wrapper<bool>(AndFunctor<scalar_t>{}), true);
      });
}

void or_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_cuda", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(iter, func_wrapper<bool>(OrFunctor<scalar_t>{}), false);
      });
}

REGISTER_DISPATCH(and_stub, &and_kernel_cuda)
REGISTER_DISPATCH(or_stub, &or_kernel_cuda)

} // namespace at::native

// ---- host tracing (ATen/cuda/host_trace): the traced sibling of and_kernel_cuda / or_kernel_cuda,
// compiled here so the sibling and the real hosts above instantiate the one kernel over ReduceOp over
// func_wrapper_t over AndFunctor / OrFunctor (DECISIONS E36): the tape's launch is eager's function
// object, not a twin. Outside a trace the entry runs the same launches in ordinary mode, which is
// how the parity test compares it with the real op.
#undef TORCH_ASSERT_NO_OPERATORS
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/host_trace/ti/Ops.h>
#include <ATen/cuda/host_trace/ti/ReduceOps.h>
#include <ATen/cuda/host_trace/ti/ReduceSym.cuh>

#include <ATen/ops/empty_like.h>

namespace at::cuda::host_trace::ti {

Tensor allany_traced(const Tensor& self, IntArrayRef dims, bool keepdim, bool all_of, const std::optional<Tensor>& out) {
  // ReduceOps.cpp allany_impl: fill_(identity) for an empty input, a copy of
  // a one-element input through bool, else get_allany_iter's reduce_op
  // iterator (the input in its own dtype, the result bool) and the stub
  const ScalarType dtype = self.scalar_type();
  if (dtype == kByte) {
    decline("host_trace: all / any on a Byte input writes a Byte result (the uint8 compatibility rule), which is not traced (declined)");
  }
  if (isComplexType(dtype)) {
    decline(c10::str("host_trace: all / any on ", dtype, " is not traced (declined)"));
  }
  ReductionSym r = make_reduction(self, dims, keepdim, kBool, out);
  if (self.sym_numel() == 0) {
    return fill_traced(r.result, all_of ? 1 : 0);
  }
  if (self.sym_numel() == 1) {
    Tensor v = self.view_as(r.result);
    if (v.scalar_type() != kBool) {
      Tensor converted = at::empty_like(v, v.options().dtype(kBool));
      v = copy_traced(converted, v);
    }
    return copy_traced(r.result, v);
  }
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBFloat16, kBool, dtype, "allany_traced", [&] {
    if (all_of) {
      gpu_reduce_kernel<scalar_t, bool>(r.iter, at::native::func_wrapper<bool>(at::native::AndFunctor<scalar_t>{}), true);
    } else {
      gpu_reduce_kernel<scalar_t, bool>(r.iter, at::native::func_wrapper<bool>(at::native::OrFunctor<scalar_t>{}), false);
    }
  });
  return r.result;
}

} // namespace at::cuda::host_trace::ti
