#pragma once
// The vector width an operand's address allows for elements of `size` bytes:
// the largest of 8, 4, 2 (16 on ROCm for 1-byte types) whose vector the
// address is aligned to (aligned_vector<T, vec> is aligned to sizeof(T) * vec),
// else 1. address_t: an integer address, or a CUDA graph host trace's
// symbolic c10::SymInt one (each test a guard). One ladder for CUDALoops.cuh
// (memory::can_vectorize_up_to), the jiterator (at::cuda::jit) and the host
// trace's routing over run-time functor facts.
#include <c10/macros/Macros.h>

#include <cstdint>

namespace at::native {

#pragma nv_exec_check_disable
template <typename address_t>
inline C10_HOST_DEVICE int vector_width_for(int64_t size, const address_t& address) {
#ifdef USE_ROCM
  if (size == 1 && (address % (16 * size) == 0)) {
    return 16;
  } else if (size <= 2 && (address % (8 * size) == 0)) {
    return 8;
  } else
#else
  if (address % (8 * size) == 0) {
    return 8;
  } else
#endif
  if (address % (4 * size) == 0) {
    return 4;
  } else if (address % (2 * size) == 0) {
    return 2;
  }
  return 1;
}

} // namespace at::native
