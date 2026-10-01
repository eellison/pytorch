#pragma once
// ht::launch<&kernel>(h, grid, block, smem, args...): a kernel launch written
// once for both host policies (ATen/native/HostPolicy.h).
//
// The kernel is a template argument, named with its eager parameter types, so
// both policies launch the same instantiation. On EagerHost this is today's
// kernel<<<grid, block, smem, stream>>>(args...) plus the launch check; a
// runtime function pointer did not compile to the same host code. On SymHost
// the arguments are SymInts and twins of the kernel's parameter types, and the
// launch is recorded.
//
// Preview only: not in any build target and not tested. record_launch and
// data_ptr belong to the recorder, which is not included.
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/HostPolicy.h>
#include <c10/cuda/CUDAException.h>

#include <utility>

namespace at::ht {

struct SymDim3 {
  SymDim3(int64_t x) : x(x) {}
  SymDim3(c10::SymInt x) : x(std::move(x)) {}
  SymDim3(dim3 d) : x(d.x), y(d.y), z(d.z) {}
  c10::SymInt x, y = 1, z = 1;
};

// Defined by the recorder. Checks each argument against the kernel's parameter
// type and records the launch with the argument image; a tensor address in it
// becomes an address slot.
template <auto kernel, class... Args>
void record_launch(Recorder& rec, const SymDim3& grid, const SymDim3& block, const c10::SymInt& smem, const Args&... args);

// The trace's placeholder for a tensor's address, as a SymInt. Alignment tests
// on it are kernel-choice guards.
TORCH_API c10::SymInt data_ptr(const TensorBase& t);

template <auto kernel, class G, class B, class... Args>
C10_ALWAYS_INLINE void launch(const EagerHost& /*h*/, G&& grid, B&& block, size_t smem, Args&&... args) {
  kernel<<<std::forward<G>(grid), std::forward<B>(block), smem, at::cuda::getCurrentCUDAStream()>>>(std::forward<Args>(args)...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <auto kernel, class... Args>
void launch(const SymHost& /*h*/, SymDim3 grid, SymDim3 block, c10::SymInt smem, const Args&... args) {
  if (auto* rec = Recorder::current()) {
    record_launch<kernel>(*rec, grid, block, smem, args...);
  }
}

} // namespace at::ht
