// Traced hosts of ATen ops (Recorder.h): the op's output, allocated through the
// dispatcher, and its launches in rec. Each is defined in its kernel's .cu, so
// it names that TU's kernel instantiation, which is the one eager launches.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>

#include <optional>
#include <string_view>
#include <tuple>

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
// var, or std for take_sqrt
TORCH_CUDA_CU_API TensorBase std_var(Recorder& rec, const TensorBase& self, IntArrayRef dims, double correction, bool keepdim, bool take_sqrt);
TORCH_CUDA_CU_API TensorBase softmax(Recorder& rec, const TensorBase& self, int64_t dim, bool half_to_float);
TORCH_CUDA_CU_API TensorBase log_softmax(Recorder& rec, const TensorBase& self, int64_t dim, bool half_to_float);
// (output, mean, rstd) over input's last normalized_ndim dims; an undefined
// weight or bias is none
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase, TensorBase> native_layer_norm(Recorder& rec, const TensorBase& input, int64_t normalized_ndim, const TensorBase& weight, const TensorBase& bias, double eps);
// (output, rstd)
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase> fused_rms_norm(Recorder& rec, const TensorBase& input, int64_t normalized_ndim, const TensorBase& weight, std::optional<double> eps);

} // namespace at::cuda::host_trace
