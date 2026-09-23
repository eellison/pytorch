#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace at::native::ufunc {

// BinaryRemainderKernel.cu remainder_kernel_cuda's device bodies
// (CUDAFunctor_remainder_Tensor): Python's remainder, the result taking the
// divisor's sign. NVIDIA hardware returns all ones for an integer modulo by
// zero and ROCm the dividend, so the CUDA build keeps uint8's value explicitly.
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T remainder(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
#if !defined(USE_ROCM)
    if constexpr (std::is_same_v<T, uint8_t>) {
      if (b == 0) {
        return std::numeric_limits<uint8_t>::max();
      }
    }
#endif
    T r = a % b;
    if (r != 0 && c10::signs_differ(r, b)) {
      r += b;
    }
    return r;
  } else {
    auto mod = ::fmod(a, b);
    if (mod != 0 && c10::signs_differ(b, mod)) {
      mod += b;
    }
    return mod;
  }
}

} // namespace at::native::ufunc
