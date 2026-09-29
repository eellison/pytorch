// Traced hosts of ATen ops (Recorder.h): the op's output, allocated through the
// dispatcher, and its launches in rec. Each is defined in its kernel's .cu, so
// it names that TU's kernel instantiation, which is the one eager launches.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

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
// a pointwise op's gpu_kernel, gpu_kernel_multiple_outputs or
// jitted_gpu_kernel launch as the kernel node `node` of a capture of the op
// launched it (name its kernel's name; empty for no launch): inputs are its
// TensorIterator's inputs, outs its outputs, each where the op writes an
// argument, else undefined; compute_dtype the iterator's common dtype
TORCH_CUDA_CU_API std::vector<TensorBase> pointwise(Recorder& rec, c10::ArrayRef<TensorBase> outs, c10::ArrayRef<c10::ScalarType> out_dtypes, c10::ArrayRef<TensorBase> inputs, c10::ScalarType compute_dtype, std::string_view name, KernelRecord node);
TORCH_CUDA_CU_API TensorBase index_select(Recorder& rec, const TensorBase& self, int64_t dim, const TensorBase& index);
TORCH_CUDA_CU_API TensorBase cat(Recorder& rec, c10::ArrayRef<TensorBase> tensors, int64_t dim);

} // namespace at::cuda::host_trace
