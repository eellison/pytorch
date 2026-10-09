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
TORCH_CUDA_CU_API TensorBase amax(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim, const TensorBase& out = {});
TORCH_CUDA_CU_API TensorBase amin(Recorder& rec, const TensorBase& self, IntArrayRef dims, bool keepdim, const TensorBase& out = {});
TORCH_CUDA_CU_API TensorBase max_all(Recorder& rec, const TensorBase& self, const TensorBase& out = {});
TORCH_CUDA_CU_API TensorBase min_all(Recorder& rec, const TensorBase& self, const TensorBase& out = {});
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase> max_dim(Recorder& rec, const TensorBase& self, int64_t dim, bool keepdim);
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase> min_dim(Recorder& rec, const TensorBase& self, int64_t dim, bool keepdim);
TORCH_CUDA_CU_API TensorBase argmax(Recorder& rec, const TensorBase& self, std::optional<int64_t> dim, bool keepdim);
TORCH_CUDA_CU_API TensorBase argmin(Recorder& rec, const TensorBase& self, std::optional<int64_t> dim, bool keepdim);
// fill_ returns self
TORCH_CUDA_CU_API TensorBase fill_(Recorder& rec, const TensorBase& self, const c10::Scalar& value);
// var, or std for take_sqrt
TORCH_CUDA_CU_API TensorBase std_var(Recorder& rec, const TensorBase& self, IntArrayRef dims, double correction, bool keepdim, bool take_sqrt);
TORCH_CUDA_CU_API TensorBase softmax(Recorder& rec, const TensorBase& self, int64_t dim, bool half_to_float);
TORCH_CUDA_CU_API TensorBase log_softmax(Recorder& rec, const TensorBase& self, int64_t dim, bool half_to_float);
// (output, mean, rstd) over input's last normalized_ndim dims; an undefined
// weight or bias is none
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase, TensorBase> native_layer_norm(Recorder& rec, const TensorBase& input, int64_t normalized_ndim, const TensorBase& weight, const TensorBase& bias, double eps);
// (output, save_mean, save_invstd) in inference; an undefined weight or bias is none
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase, TensorBase> native_batch_norm(Recorder& rec, const TensorBase& input, const TensorBase& weight, const TensorBase& bias, const TensorBase& running_mean, const TensorBase& running_var, bool training, double eps);
// (output, rstd)
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase> fused_rms_norm(Recorder& rec, const TensorBase& input, int64_t normalized_ndim, const TensorBase& weight, std::optional<double> eps);
// a pointwise op's gpu_kernel, gpu_kernel_multiple_outputs or
// jitted_gpu_kernel launch as the kernel node `node` of a capture of the op
// launched it (name its kernel's name; empty for no launch): inputs are its
// TensorIterator's inputs, outs its outputs, each where the op writes an
// argument, else undefined; compute_dtype the iterator's common dtype;
// dynamic for a user jiterator's launch (torch.cuda.jiterator)
TORCH_CUDA_CU_API std::vector<TensorBase> pointwise(Recorder& rec, c10::ArrayRef<TensorBase> outs, c10::ArrayRef<c10::ScalarType> out_dtypes, c10::ArrayRef<TensorBase> inputs, c10::ScalarType compute_dtype, bool dynamic, std::string_view name, KernelRecord node);
TORCH_CUDA_CU_API TensorBase index_select(Recorder& rec, const TensorBase& self, int64_t dim, const TensorBase& index);
// arange(start, start + size * step, step): an integral dtype's integral start and step, a floating one's concrete
TORCH_CUDA_CU_API TensorBase arange(Recorder& rec, const c10::SymInt& size, const c10::Scalar& start, const c10::Scalar& step, c10::ScalarType dtype, c10::Device device);
TORCH_CUDA_CU_API TensorBase triu(Recorder& rec, const TensorBase& self, int64_t k);
// (output, total_weight) of a reduction; an undefined weight is none
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase> nll_loss_forward(Recorder& rec, const TensorBase& self, const TensorBase& target, const TensorBase& weight, int64_t reduction, int64_t ignore_index);
// (output, indices) of a contiguous input
TORCH_CUDA_CU_API std::tuple<TensorBase, TensorBase> max_pool2d_with_indices(Recorder& rec, const TensorBase& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode);
// of a contiguous input
TORCH_CUDA_CU_API TensorBase adaptive_avg_pool2d(Recorder& rec, const TensorBase& input, IntArrayRef output_size);
TORCH_CUDA_CU_API TensorBase cat(Recorder& rec, c10::ArrayRef<TensorBase> tensors, int64_t dim);

} // namespace at::cuda::host_trace
