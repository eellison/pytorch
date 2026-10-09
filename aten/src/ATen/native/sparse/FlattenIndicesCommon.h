#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sparse/Macros.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/SparseTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/tensor.h>
#endif

#ifdef GPUCC
#define NAME "flatten_indices_cuda"
#else
#define NAME "flatten_indices_cpu"
#endif

namespace at::native {

namespace {

template <template <typename func_t> class kernel_t>
struct KernelLauncher {
  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    kernel_t<func_t>::launch(iter, f);
  }
};

template <typename index_t, typename hash_coeffs_t>
struct FlattenIndicesFunctor {
  const index_t* RESTRICT ptr_indices;
  int64_t indices_nnz_stride;
  int64_t sparse_dim;
  hash_coeffs_t hash_coeffs;
  int64_t indices_dim_stride;
  FUNCAPI int64_t operator()(int64_t nnz_idx) const {
    const auto* RESTRICT ptr_indices_dim = ptr_indices + nnz_idx * indices_nnz_stride;
    auto hash = static_cast<int64_t>(0);
    for (int64_t dim = 0; dim < sparse_dim; ++dim) {
      const auto dim_hash_coeff = hash_coeffs[dim];
      const auto dim_index = ptr_indices_dim[dim * indices_dim_stride];
      hash += dim_index * dim_hash_coeff;
    }
    return hash;
  }
  static constexpr bool host_trace_sizes = true;
};

template <
  template <typename func_t> class kernel_t,
  typename index_t,
  int64_t max_static_len = 0>
Tensor _flatten_indices_impl(const Tensor& indices, IntArrayRef size) {
  TORCH_INTERNAL_ASSERT(indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size());

  // Need owning storage in case of the Tensor class.
  const auto hash_coeffs_storage = [&]() -> auto {
    auto strides = c10::contiguous_strides(size);
    return at::sparse::TensorGeometryHolder<max_static_len>(strides, strides, indices.options());
  }();
  const auto hash_coeffs = std::get<0>(*hash_coeffs_storage);

  const auto hash_indices = [&]() -> Tensor {
    const auto sparse_dim = indices.size(0);
    const auto indices_dim_stride = indices.stride(0);
    const auto indices_nnz_stride = indices.stride(1);

    auto hash = at::arange(indices.size(1), indices.options().dtype(kLong));

    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .add_output(hash)
      .add_input(hash)
      .build();

    {
      const auto* RESTRICT ptr_indices = indices.const_data_ptr<index_t>();

      KernelLauncher<kernel_t>::launch(iter,
          FlattenIndicesFunctor<index_t, std::decay_t<decltype(hash_coeffs)>>{ptr_indices, indices_nnz_stride, sparse_dim, hash_coeffs, indices_dim_stride});
    }

    return hash;
  }();

  return hash_indices;
}

template <template <typename func_t> class kernel_t>
Tensor _flatten_indices(const Tensor& indices, IntArrayRef size) {
  TORCH_CHECK(indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size(),
      NAME, "(): the dimensionality of sparse `indices` and the length of `size` must match. ",
            "Got `indices.size(0) == ", indices.size(0), "` != `size.size() == ", size.size(), "`.");
  Tensor flattened_indices;
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), NAME, [&] () {
    constexpr int64_t max_sparse_dims = 8;
    if (indices.size(0) <= max_sparse_dims) {
      flattened_indices = _flatten_indices_impl<kernel_t, index_t, max_sparse_dims>(indices, size);
    } else {
      flattened_indices = _flatten_indices_impl<kernel_t, index_t>(indices, size);
    }
  });
  return flattened_indices;
}

}

} // at::native
