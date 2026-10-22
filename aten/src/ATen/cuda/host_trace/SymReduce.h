// A reduction's launch (Reduce.cuh's gpu_reduce_kernel) over a symbolic
// TensorIterator: the same set_reduce_config over c10::SymInt, recording the
// reduce_kernel launch eager makes. What is instantiated per ReduceOp is only
// make_reduce_facts (Reduce.cuh): its kernels, layout and bytes.
#pragma once
#include <ATen/TensorIterator.h>
#include <c10/core/SymInt.h>
#include <c10/macros/Export.h>

#include <array>
#include <cstdint>

namespace at::cuda::host_trace {

struct ReduceFacts {
  // reduce_kernel<max_threads / v, v, ReduceOp> for output_vec_size v = 4, 2, 1
  std::array<const void*, 3> kernels{};
  // a jitted reduction's (ReduceJitOp): its kernel for output_vec_size v, compiled as eager's launch does
  void* (*jit_function)(const void* jit, int output_vec_size) = nullptr;
  const void* jit = nullptr;
  // set_reduce_config's ReduceSizes
  int arg_size = 0;
  int scalar_size = 0;
  int input_vec_size = 0;
  int max_threads = 0;
  // ReduceOp's size, alignment and member offsets
  uint32_t size = 0, align = 1;
  uint32_t ops = 0, ident = 0, config = 0, input_calc = 0, output_calc = 0, src = 0, dst = 0, acc_buf = 0, cta_buf = 0,
           semaphores = 0, base_idx = 0, accumulate = 0, final_output = 0, noutputs = 0;
  // the ops functor's and the identity's bytes
  const void* ops_bytes = nullptr;
  uint32_t ops_size = 0;
  const void* ident_bytes = nullptr;
  uint32_t ident_size = 0;
};

TORCH_CUDA_CPP_API void sym_gpu_reduce_kernel(const TensorIteratorBase& iter, const ReduceFacts& facts);

// A member of a reduction's ops functor eager computes from sizes (MeanOps'
// factor): its symbolic bits at that member of the launch while the scope lives
struct TORCH_CUDA_CPP_API OpsFieldScope {
  OpsFieldScope(const void* ops, const void* member, size_t width, c10::SymInt bits);
  ~OpsFieldScope();
  OpsFieldScope(const OpsFieldScope&) = delete;
  OpsFieldScope& operator=(const OpsFieldScope&) = delete;
};

} // namespace at::cuda::host_trace
