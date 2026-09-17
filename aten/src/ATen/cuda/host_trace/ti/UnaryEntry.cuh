#pragma once
// The floating-point unary entry of the traced sibling iterator: the real
// ops' real-type branches (a complex or integral input declines by name).
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/TensorIteratorSym.h>

#include <ATen/Dispatch.h>

namespace at::cuda::host_trace::ti {

template <template <typename> class F>
Tensor floating_unary(const Tensor& self, const char* op) {
  TensorIteratorSym iter = TensorIteratorSym::unary_op(Tensor(), self);
  if (!at::isFloatingType(iter.common_dtype())) {
    decline(c10::str("host_trace: ", op, " on ", iter.common_dtype(), " is not traced (declined)"));
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "unary_traced", [&] {
    gpu_kernel(iter, F<scalar_t>());
  });
  return iter.output();
}

} // namespace at::cuda::host_trace::ti
