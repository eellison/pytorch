// Descriptors of a launch whose parameter types are known at run time (a user
// jiterator's, ATen/cuda/jiterator.cu), for SymHost's recorder.
#pragma once
#include <ATen/cuda/host_trace/SymHost.h>

namespace at::cuda::host_trace {

// an OffsetCalculator<n> parameter of a runtime n
TORCH_CUDA_CPP_API ArgDesc offset_calculator_desc(int n, const SymOffsets& c);

// a kernel's int parameter (the element count)
inline ArgDesc int_desc(const c10::SymInt& v) {
  ArgDesc d;
  d.kind = ArgDesc::Scalar;
  d.size = sizeof(int);
  d.align = alignof(int);
  d.value = &v;
  return d;
}

// a char*[n] parameter
inline ArgDesc pointers_desc(const c10::SymInt* data, size_t n) {
  ArgDesc d;
  d.kind = ArgDesc::Pointers;
  d.size = static_cast<uint32_t>(n * sizeof(char*));
  d.align = alignof(char*);
  d.n = static_cast<uint32_t>(n);
  d.value = data;
  return d;
}

} // namespace at::cuda::host_trace
