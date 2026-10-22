// The SymHost instantiation of gpu_kernel's launch routing (CUDALoops.cuh),
// over a functor's run-time facts: one for every functor.
#if !defined(USE_ROCM)
#include <ATen/native/cuda/Loops.cuh>

namespace at::cuda::host_trace {

void sym_gpu_kernel(TensorIteratorBase& iter, const SymFacts& facts, bool nocast, bool check_cast) {
  if (nocast) {
    at::native::gpu_kernel_body<SymHost, /*nocast=*/true>(iter, facts, SymFunctor{}, check_cast);
  } else {
    at::native::gpu_kernel_body<SymHost, /*nocast=*/false>(iter, facts, SymFunctor{}, check_cast);
  }
}

} // namespace at::cuda::host_trace
#endif
