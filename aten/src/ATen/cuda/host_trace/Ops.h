// Traced hosts of ATen ops (Recorder.h): the op's output, allocated through the
// dispatcher, and its launches in rec.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>

#include <string_view>

namespace at::cuda::host_trace {

TORCH_CUDA_CU_API TensorBase mul(Recorder& rec, const TensorBase& a, const TensorBase& b);
TORCH_CUDA_CU_API TensorBase silu(Recorder& rec, const TensorBase& a);
TORCH_CUDA_CU_API TensorBase add(Recorder& rec, const TensorBase& a, const TensorBase& b, const c10::Scalar& alpha);
TORCH_CUDA_CU_API TensorBase gelu(Recorder& rec, const TensorBase& a, std::string_view approximate);
TORCH_CUDA_CU_API TensorBase rsqrt(Recorder& rec, const TensorBase& a);
TORCH_CUDA_CU_API TensorBase where(Recorder& rec, const TensorBase& cond, const TensorBase& a, const TensorBase& b);
// copy_ returns dst
TORCH_CUDA_CU_API TensorBase copy_(Recorder& rec, const TensorBase& dst, const TensorBase& src);
TORCH_CUDA_CU_API TensorBase to_copy(Recorder& rec, const TensorBase& src, c10::ScalarType dtype, c10::MemoryFormat memory_format);
// dims empty is all dims
TORCH_CUDA_CU_API TensorBase sum(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim);
TORCH_CUDA_CU_API TensorBase mean(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim);
TORCH_CUDA_CU_API TensorBase amax(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim);

} // namespace at::cuda::host_trace
