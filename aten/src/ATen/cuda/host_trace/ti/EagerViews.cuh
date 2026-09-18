// Proxies and SymInt helpers for the eager CUDA hosts that opt into tracing
// through a sibling of their host body (index_select in Indexing.cu, cat in
// Shape.cu, triu / tril in TriangularOps.cu). The by-value kernel parameter
// blocks those hosts fill (cuda::detail::TensorInfo, cat's metadata structs)
// are arrays, so their proxies are views over Slots.h; the sibling host writes
// them exactly as the real host writes the POD, and every write is a record of
// the trace.
#pragma once
#include <ATen/cuda/host_trace/ti/Slots.h>

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/core/SymInt.h>

#include <cstddef>
#include <initializer_list>
#include <utility>

namespace at::cuda::host_trace::ti {

inline c10::SymInt ceil_div_sym(const c10::SymInt& a, const c10::SymInt& b) {
  return (a + b - 1) / b;
}

// MemoryAccess.cuh's get_alignment on a symbolic address or byte count: the
// largest of 16 / 8 / 4 / 2 that divides it, each test a guard of the trace.
inline int64_t alignment_of(const c10::SymInt& v) {
  for (int64_t n : {16, 8, 4, 2}) {
    if (v % c10::SymInt(n) == c10::SymInt(0)) {
      return n;
    }
  }
  return 1;
}

// CollapseDims.h's collapse_dims over slot arrays: the same walk, every size
// and stride comparison a guard. Returns (remapped excluded dim, new dims).
template <class Sizes, class Strides>
std::pair<int64_t, int64_t> collapse_dims_sym(
    const Sizes& sizes,
    const Strides& strides,
    int64_t dims,
    const int excludeDim = -1) {
  TORCH_CHECK(
      excludeDim >= -1 && excludeDim < dims,
      "expected excluded dim between -1 and dims - 1");
  const c10::SymInt one(1);
  int64_t stopDim = (excludeDim == -1) ? dims : excludeDim;
  int64_t newIndex = -1;
  int64_t oldIndex = 0;
  int64_t remappedExcludedDim = -1;

  while (oldIndex < dims) {
    for (; oldIndex < stopDim; ++oldIndex) {
      if (c10::SymInt(sizes[oldIndex]) == one) {
        continue;
      }
      ++newIndex;
      sizes[newIndex] = c10::SymInt(sizes[oldIndex]);
      strides[newIndex] = c10::SymInt(strides[oldIndex]);
      ++oldIndex;
      break;
    }
    for (; oldIndex < stopDim; ++oldIndex) {
      if (c10::SymInt(sizes[oldIndex]) == one) {
        continue;
      }
      if (c10::SymInt(strides[newIndex]) ==
          c10::SymInt(sizes[oldIndex]) * c10::SymInt(strides[oldIndex])) {
        sizes[newIndex] = c10::SymInt(sizes[newIndex]) * c10::SymInt(sizes[oldIndex]);
        strides[newIndex] = c10::SymInt(strides[oldIndex]);
      } else {
        ++newIndex;
        sizes[newIndex] = c10::SymInt(sizes[oldIndex]);
        strides[newIndex] = c10::SymInt(strides[oldIndex]);
      }
    }
    if (oldIndex != dims) {
      ++newIndex;
      sizes[newIndex] = c10::SymInt(sizes[oldIndex]);
      strides[newIndex] = c10::SymInt(strides[oldIndex]);
      remappedExcludedDim = newIndex;
      ++oldIndex;
      stopDim = dims;
    }
  }
  if (newIndex == -1 || (newIndex == 0 && c10::SymInt(sizes[0]) == one)) {
    dims = 1;
    sizes[0] = one;
    strides[0] = one;
    return std::pair<int64_t, int64_t>(0, 1);
  }
  dims = newIndex + 1;
  return std::pair<int64_t, int64_t>(remappedExcludedDim, dims);
}

} // namespace at::cuda::host_trace::ti

namespace at::cuda::host_trace {

// cuda::detail::TensorInfo<T, IndexType> (unsigned for index_select, int32 /
// int64 for triu / tril): the pointer, 25 sizes, 25 strides and the dim
// count. The dim count is structural (the outcome of guarded collapses),
// kept as a plain int beside its slot.
template <class T, class IndexType>
struct Traced<at::cuda::detail::TensorInfo<T, IndexType>> : TracedBase {
  using P = at::cuda::detail::TensorInfo<T, IndexType>;
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::PtrSlot data;
  ti::ArrayOf<ti::IntSlot<IndexType>, sizeof(IndexType), MAX_TENSORINFO_DIMS> sizes;
  ti::ArrayOf<ti::IntSlot<IndexType>, sizeof(IndexType), MAX_TENSORINFO_DIMS> strides;
  ti::IntSlot<int> dims_slot;
  int dims = 0;
  Traced()
      : TracedBase(pod_bytes, sizeof(P)),
        data(this, offsetof(P, data), ti::SlotName{nullptr, "data"}),
        sizes(this, offsetof(P, sizes), ti::SlotName{nullptr, "sizes"}),
        strides(this, offsetof(P, strides), ti::SlotName{nullptr, "strides"}),
        dims_slot(this, offsetof(P, dims), ti::SlotName{nullptr, "dims"}) {}

  void set_dims(int d) {
    dims = d;
    dims_slot = d;
  }
  // IndexUtils.cuh getTensorInfo: the tensor's pointer, sizes and strides
  void fill(const at::TensorBase& t) {
    const int dim = t.dim();
    TORCH_CHECK(dim < MAX_TENSORINFO_DIMS, "CUDA Tensors cannot have more than 25 dimensions");
    for (int i = 0; i < dim; ++i) {
      sizes[i] = t.sym_size(i);
      strides[i] = t.sym_stride(i);
    }
    // inputs (TensorInfo<const T>) through the const accessor, outputs through the
    // mutable one: a copy-on-write input stays lazy, as in getTensorInfo's caller
    if constexpr (std::is_const_v<T>) {
      data = sym_const_data_ptr<std::remove_const_t<T>>(t);
    } else {
      data = sym_mutable_data_ptr<T>(t);
    }
    set_dims(dim);
  }
  // TensorInfo::reduceDim / collapseDims, and Indexing.cu's tensorInfoLegacyIfScalar
  void reduceDim(int dim) {
    TORCH_CHECK(dim < dims && dim >= 0, "expected dim between 0 and dims - 1");
    sizes[dim] = 1;
  }
  int collapseDims(const int excludeDim = -1) {
    auto result = ti::collapse_dims_sym(sizes, strides, dims, excludeDim);
    set_dims(static_cast<int>(result.second));
    return static_cast<int>(result.first);
  }
  void legacy_if_scalar() {
    if (dims == 0) {
      set_dims(1);
      sizes[0] = 1;
      strides[0] = 1;
    }
  }
};

} // namespace at::cuda::host_trace
