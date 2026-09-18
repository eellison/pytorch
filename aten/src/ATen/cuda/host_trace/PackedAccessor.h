// Proxy for a kernel parameter of type at::PackedTensorAccessor64<T, N>
// (GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>, Field.h Traced<P>):
// the pointer, N sizes and N strides in declaration order. The accessor is
// standard-layout with those three members, so the offsets are structural;
// the size static_assert keeps the proxy honest against the header.
#pragma once
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/host_trace/Field.h>
#include <ATen/cuda/host_trace/ti/Slots.h>

namespace at::cuda::host_trace {

// at::GenericPackedTensorAccessor is an alias over the headeronly class with
// two leading helper parameters; the specialization names that class so every
// parameter is deduced.
template <
    class Indexer,
    class Bounds,
    class T,
    size_t N,
    template <class> class PtrTraits,
    class index_t>
struct Traced<torch::headeronly::detail::
                  GenericPackedTensorAccessor<Indexer, Bounds, T, N, PtrTraits, index_t>>
    : TracedBase {
  using P = torch::headeronly::detail::
      GenericPackedTensorAccessor<Indexer, Bounds, T, N, PtrTraits, index_t>;
  using PtrType = typename PtrTraits<T>::PtrType;
  static_assert(
      sizeof(P) == sizeof(PtrType) + 2 * N * sizeof(index_t),
      "PackedTensorAccessor layout is data_, sizes_[N], strides_[N]");
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::PtrSlot data;
  ti::ArrayOf<ti::IntSlot<index_t>, sizeof(index_t), N> sizes;
  ti::ArrayOf<ti::IntSlot<index_t>, sizeof(index_t), N> strides;
  Traced()
      : TracedBase(pod_bytes, sizeof(P)),
        data(this, 0, ti::SlotName{nullptr, "data_"}),
        sizes(this, sizeof(PtrType), ti::SlotName{nullptr, "sizes_"}),
        strides(
            this,
            sizeof(PtrType) + N * sizeof(index_t),
            ti::SlotName{nullptr, "strides_"}) {}

  // Tensor::packed_accessor64<T, N>(): the address the caller read (an
  // input's through sym_const_data_ptr, an output's through
  // sym_mutable_data_ptr), then the tensor's sizes and strides
  void fill(const at::TensorBase& t, const c10::SymInt& address) {
    TORCH_CHECK(
        t.dim() == static_cast<int64_t>(N),
        "TensorAccessor expected ",
        N,
        " dims but tensor has ",
        t.dim());
    data = address;
    for (size_t i = 0; i < N; ++i) {
      sizes[i] = t.sym_size(static_cast<int64_t>(i));
      strides[i] = t.sym_stride(static_cast<int64_t>(i));
    }
  }
};

} // namespace at::cuda::host_trace
