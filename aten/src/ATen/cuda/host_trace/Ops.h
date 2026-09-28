// Traced hosts of ATen ops (Recorder.h): the functional op's output, allocated
// through the dispatcher, and its launches in rec.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/core/TensorBase.h>

namespace at::cuda::host_trace {

TORCH_CUDA_CU_API TensorBase mul(Recorder& rec, const TensorBase& a, const TensorBase& b);
TORCH_CUDA_CU_API TensorBase silu(Recorder& rec, const TensorBase& a);

} // namespace at::cuda::host_trace
