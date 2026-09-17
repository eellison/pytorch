#pragma once

#include <c10/macros/Macros.h>

namespace at::native::ufunc {

// BinaryMiscBackwardOpsKernels.cu sigmoid_backward_kernel_cuda's real-type
// device body (CUDAFunctor_sigmoid_backward): (grad_output, output),
// arithmetic in the tensor dtype
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T sigmoid_backward(T a, T b) {
  return a * (T(1.) - b) * b;
}

} // namespace at::native::ufunc
