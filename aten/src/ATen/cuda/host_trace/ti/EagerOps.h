// The eager CUDA hosts with a traced sibling defined beside the real host in
// its own translation unit (the kernels they launch live in that file's
// anonymous namespace): index_select in Indexing.cu, cat in Shape.cu, triu /
// tril in TriangularOps.cu, the fast gather launch in IndexKernelUtils.cu,
// and index_copy_ / index_put_ in IndexKernel.cu, over the named functors
// eager's hosts launch. Called by the trace mode in place of the op
// (torch/cuda/_host_trace_ti.py); outside a trace they run the same launches
// in ordinary mode, which is how the parity tests compare them with the real
// op.
#pragma once
#include <ATen/core/IListRef.h>
#include <ATen/core/Tensor.h>
#include <c10/core/SymInt.h>

#include <vector>

namespace at::cuda::host_trace::ti {

TORCH_CUDA_CU_API Tensor index_select_traced(const Tensor& self, int64_t dim, const Tensor& index);
TORCH_CUDA_CU_API Tensor cat_traced(const ITensorListRef& tensors, int64_t dim);
// index_copy_ into `result` (self, or the copy of it the functional entry
// made): TensorAdvancedIndexing.cpp index_copy_out's iterator over the
// restrided result, index and source, and IndexKernel.cu's index_copy_kernel
TORCH_CUDA_CU_API Tensor& index_copy_traced(Tensor& result, int64_t dim, const Tensor& index, const Tensor& source);
// index_put_ with accumulate=False: the iterator make_index_put_iterator
// builds over the restrided self (stride 0 at the indexed dims), the value
// and the reshaped indices, and IndexKernel.cu's index_put_kernel. The entry
// does make_info's view arithmetic and passes the indexed dims' sizes and
// byte strides.
TORCH_CUDA_CU_API Tensor& index_put_traced(
    Tensor& src,
    const Tensor& value,
    const std::vector<Tensor>& indices,
    const std::vector<c10::SymInt>& indexed_sizes,
    const std::vector<c10::SymInt>& indexed_strides);
// triu / tril (TriangularOps.cu triu_tril_cuda_template) into `result`: self
// for the in-place op, the diagonal as a value
TORCH_CUDA_CU_API Tensor& triu_tril_traced(const Tensor& self, const c10::SymInt& k, bool upper, Tensor& result);
// embedding_dense_backward (Embedding.cu): the feature kernel over a zeroed
// table when the index count is at most 3072 without frequency scaling (the
// real host's route, a guard); the sort-based route declines by name
TORCH_CUDA_CU_API Tensor embedding_dense_backward_traced(
    const Tensor& grad,
    const Tensor& indices,
    const c10::SymInt& num_weights,
    const c10::SymInt& padding_idx,
    bool scale_grad_by_freq);

} // namespace at::cuda::host_trace::ti

namespace at::native {

// IndexKernelUtils.cu's vectorized_gather_kernel_launch with its sizes,
// strides and addresses as SymInts and the launch through the typed helper.
template <int64_t Alignment, typename index_t>
TORCH_CUDA_CU_API void vectorized_gather_kernel_launch_sym(
    const c10::SymInt& out,
    const c10::SymInt& inp,
    const c10::SymInt& idx,
    const c10::SymInt& num_ind,
    const c10::SymInt& slice_size_in_bytes,
    const c10::SymInt& ind_dim_size,
    const c10::SymInt& inp_stride_bytes,
    const c10::SymInt& out_stride_bytes,
    bool allow_neg_indices);

} // namespace at::native
