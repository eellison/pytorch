// The multi_tensor_apply host (MultiTensorApply.cuh) re-typed for host
// tracing: the same chunking loops over tensor lists with the numels as
// c10::SymInt, the addresses as sym_*_data_ptr() values, the by-value
// metadata block as a proxy over Slots.h views and the launches through the
// typed helper. The foreach kernels and the fused optimizers are functors
// over this one host, so a functor registers by naming its kernel
// instantiation and its argument roles (the entries at the end of the eager
// hosts' files, where that instantiation lives); the loop below is written
// once. The eager loops are untouched.
//
// What is a field, a guard and a constant of the metadata block:
//   addresses[d][i], state_steps_addresses[i]: pointer slots, one per tensor
//     per list (a traced tensor's address as a value; an input rebinds per
//     call, an allocation follows its root).
//   numel_for_tensor[i]: an i64 slot holding the tensor's sym_numel().
//   block_to_tensor / block_to_chunk / start_tensor_this_launch: structural.
//     They are functions of the tensor order (a constant of the call: the
//     list lengths are argument constants of the tape) and of each tensor's
//     chunk count, which the loop guards (`ceil(numel / kChunkSize)` is read
//     as an int: a recorded guard), so they are written into the POD as the
//     eager loop writes them and never recorded; the launch keeps them as
//     the constant bytes of the image, which the build compares byte for byte
//     with its capture and a two-hint trace compares between the hints.
//   the number of launches: structural as well (the flush points are
//     functions of the chunk counts and the compile-time capacities), so a
//     call with another chunk decomposition misses on the chunk guards.
// After a flush the eager loop leaves the previous launch's entries in the
// block (the next kernel reads only the slots below its own counts). Here the
// used prefix is zeroed and the records dropped after every launch, so each
// launch's image holds its own tensors and zeros: the trace's hint image and
// the build's capture must agree on every byte, and a stale slot would hold a
// placeholder at the trace and a real address at the build.
#pragma once
#include <ATen/cuda/host_trace/ti/Slots.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

namespace at::cuda::host_trace {

namespace mta {

// The members the two metadata layouts share and the loops write.
template <class P, int n>
struct ListMetaBase : TracedBase {
  static constexpr int32_t max_tensors = P::max_tensors_per_launch;
  static constexpr int32_t max_blocks = P::max_blocks_per_launch;
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::ArrayOf<ti::ArrayOf<ti::PtrSlot, sizeof(void*), max_tensors>, sizeof(void*) * max_tensors, n> addresses;
  ti::ArrayOf<ti::IntSlot<int64_t>, sizeof(int64_t), max_tensors> numel_for_tensor;
  ListMetaBase()
      : TracedBase(pod_bytes, sizeof(P)),
        addresses(this, offsetof(P, addresses), ti::SlotName{nullptr, "addresses"}),
        numel_for_tensor(this, offsetof(P, numel_for_tensor), ti::SlotName{nullptr, "numel_for_tensor"}) {}
  // the structural members, written as the eager loop writes them
  P& pod() {
    return *reinterpret_cast<P*>(pod_bytes);
  }
  // after a launch: zero the entries it used and drop their records
  void reset_prefix(int tensors, int blocks) {
    P& p = pod();
    for (int d = 0; d < n; ++d) {
      std::memset(p.addresses[d], 0, sizeof(void*) * tensors);
    }
    std::memset(p.numel_for_tensor, 0, sizeof(int64_t) * tensors);
    std::memset(p.block_to_tensor, 0, sizeof(p.block_to_tensor[0]) * blocks);
    std::memset(p.block_to_chunk, 0, sizeof(int32_t) * blocks);
    clear_records();
  }
};

// The chunk count of one tensor: `numel / kChunkSize + (numel % kChunkSize != 0)`
// of the eager loop is ceil(numel / kChunkSize); the loop bound is read as an
// int, a guard of the trace.
inline int64_t chunks_of(const c10::SymInt& numel) {
  const c10::SymInt chunks = (numel + (at::native::kChunkSize - 1)) / at::native::kChunkSize;
  return chunks.guard_int(__FILE__, __LINE__);
}

inline c10::SymInt address_of(const at::Tensor& t, bool written) {
  return written ? sym_mutable_data_ptr(t) : sym_const_data_ptr(t);
}

} // namespace mta

template <int n>
struct Traced<at::native::TensorListMetadata<n>> : mta::ListMetaBase<at::native::TensorListMetadata<n>, n> {
  static_assert(sizeof(at::native::TensorListMetadata<n>) % alignof(at::native::TensorListMetadata<n>) == 0);
};

template <int n>
struct Traced<at::native::FusedOptimizerTensorListMetadata<n>>
    : mta::ListMetaBase<at::native::FusedOptimizerTensorListMetadata<n>, n> {
  using P = at::native::FusedOptimizerTensorListMetadata<n>;
  using Base = mta::ListMetaBase<P, n>;
  ti::ArrayOf<ti::PtrSlot, sizeof(void*), Base::max_tensors> state_steps_addresses;
  Traced()
      : state_steps_addresses(this, offsetof(P, state_steps_addresses), ti::SlotName{nullptr, "state_steps_addresses"}) {}
  void reset_prefix(int tensors, int blocks) {
    std::memset(this->pod().state_steps_addresses, 0, sizeof(void*) * tensors);
    Base::reset_prefix(tensors, blocks);
  }
};

// MultiTensorApply.cuh's multi_tensor_apply<depth>(tensor_lists, callable, args...)
// with the kernel instantiation named by the caller (its argument types are
// the eager launch's, so the sibling launches the same instantiation) and the
// lists the kernel writes marked (`written[d]`: the address is read through
// the mutable accessor, else the const one, A98).
template <int depth, class T, class... KArgs, class... Args>
void multi_tensor_apply_sym(
    void (*kernel)(at::native::TensorListMetadata<depth>, T, KArgs...),
    const std::array<bool, depth>& written,
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    const T& callable,
    const Args&... args) {
  TORCH_CHECK(tensor_lists.size() == depth, "Number of tensor lists has to match the depth.");
  const size_t n_tensors = tensor_lists[0].size();
  using metadata_t = at::native::TensorListMetadata<depth>;
  Traced<metadata_t> tensorListMeta;
  tensorListMeta.pod().start_tensor_this_launch = 0;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  int processed = 0;

  for (size_t t = 0; t < n_tensors; t++) {
    const c10::SymInt numel = tensor_lists[0][t].sym_numel();
    if (numel == c10::SymInt(0)) {
      continue;
    }
    processed++;
    tensorListMeta.numel_for_tensor[loc_tensor_info] = numel;
    for (int d = 0; d < depth; d++) {
      tensorListMeta.addresses[d][loc_tensor_info] = mta::address_of(tensor_lists[d][t], written[d]);
    }
    loc_tensor_info++;

    const int64_t chunks = mta::chunks_of(numel);
    for (int64_t chunk = 0; chunk < chunks; chunk++) {
      tensorListMeta.pod().block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.pod().block_to_chunk[loc_block_info] = static_cast<int32_t>(chunk);
      loc_block_info++;

      const bool tensors_full = (loc_tensor_info == metadata_t::max_tensors_per_launch && chunk == chunks - 1);
      const bool blocks_full = (loc_block_info == metadata_t::max_blocks_per_launch);

      if (tensors_full || blocks_full) {
        launch(kernel, loc_block_info, at::native::kBlockSize, c10::SymInt(0), stream, tensorListMeta, callable, args...);
        tensorListMeta.reset_prefix(loc_tensor_info, loc_block_info);
        loc_block_info = 0;
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
          tensorListMeta.pod().start_tensor_this_launch = processed;
        } else {
          tensorListMeta.numel_for_tensor[0] = numel;
          for (int d = 0; d < depth; d++) {
            tensorListMeta.addresses[d][0] = mta::address_of(tensor_lists[d][t], written[d]);
          }
          loc_tensor_info = 1;
          tensorListMeta.pod().start_tensor_this_launch = processed - 1;
        }
      }
    }
  }

  if (loc_block_info != 0) {
    launch(kernel, loc_block_info, at::native::kBlockSize, c10::SymInt(0), stream, tensorListMeta, callable, args...);
  }
}

// MultiTensorApply.cuh's multi_tensor_apply_for_fused_optimizer<depth>: the
// same loop with the step counters' addresses beside each tensor's.
template <int depth, class T, class... KArgs, class... Args>
void multi_tensor_apply_for_fused_optimizer_sym(
    void (*kernel)(at::native::FusedOptimizerTensorListMetadata<depth>, T, KArgs...),
    const std::array<bool, depth>& written,
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    const T& callable,
    const Args&... args) {
  TORCH_CHECK(tensor_lists.size() == depth, "Number of tensor lists has to match the depth");
  const auto num_tensors = tensor_lists[0].size();
  using metadata_t = at::native::FusedOptimizerTensorListMetadata<depth>;
  Traced<metadata_t> tensorListMeta;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (const auto& tensor_index : c10::irange(num_tensors)) {
    const c10::SymInt numel = tensor_lists[0][tensor_index].sym_numel();
    if (numel == c10::SymInt(0)) {
      continue;
    }
    tensorListMeta.state_steps_addresses[loc_tensor_info] = sym_const_data_ptr(state_steps[tensor_index]);
    tensorListMeta.numel_for_tensor[loc_tensor_info] = numel;
    for (const auto& d : c10::irange(depth)) {
      tensorListMeta.addresses[d][loc_tensor_info] = mta::address_of(tensor_lists[d][tensor_index], written[d]);
    }
    loc_tensor_info++;

    const int64_t chunks = mta::chunks_of(numel);
    for (const auto& chunk : c10::irange(chunks)) {
      tensorListMeta.pod().block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.pod().block_to_chunk[loc_block_info] = static_cast<int32_t>(chunk);
      loc_block_info++;

      const auto tensor_full = (loc_tensor_info == metadata_t::max_tensors_per_launch && chunk == chunks - 1);
      const auto blocks_full = loc_block_info == metadata_t::max_blocks_per_launch;

      if (tensor_full || blocks_full) {
        launch(kernel, loc_block_info, at::native::kBlockSize, c10::SymInt(0), stream, tensorListMeta, callable, args...);
        tensorListMeta.reset_prefix(loc_tensor_info, loc_block_info);
        loc_block_info = 0;
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
        } else {
          tensorListMeta.numel_for_tensor[0] = numel;
          tensorListMeta.state_steps_addresses[0] = sym_const_data_ptr(state_steps[tensor_index]);
          for (const auto& d : c10::irange(depth)) {
            tensorListMeta.addresses[d][0] = mta::address_of(tensor_lists[d][tensor_index], written[d]);
          }
          loc_tensor_info = 1;
        }
      }
    }
  }

  if (loc_block_info != 0) {
    launch(kernel, loc_block_info, at::native::kBlockSize, c10::SymInt(0), stream, tensorListMeta, callable, args...);
  }
}

// ForeachUtils.h's check_fast_path_restrictions over symbolic sizes and
// strides: the same three walks in the same order, every size and stride
// comparison a guard (the dense check goes through the tensor's own query,
// which the trace answers shape-generically), the dtype and device tests
// constants of the call. False sends the caller to the op's slow path, as in
// eager.
inline bool fast_path_restrictions_sym(
    c10::ArrayRef<at::TensorList> lists,
    c10::ArrayRef<at::Scalar> scalars = {},
    bool promotes_integer_inputs_to_float = false,
    bool skip_cross_list_dtype_check = false) {
  const auto expected_dtype = lists[0][0].dtype();
  const auto expected_device = lists[0][0].device();
  for (const at::TensorList& list : lists) {
    if (list.empty()) {
      continue;
    }
    const auto list_dtype = list[0].dtype();
    for (const at::Tensor& t : list) {
      if (!(t.device() == expected_device && t.layout() == at::kStrided && t.is_non_overlapping_and_dense() &&
            t.dtype() == list_dtype && (skip_cross_list_dtype_check || t.dtype() == expected_dtype))) {
        return false;
      }
    }
  }
  for (const auto i : c10::irange(size_t{1}, lists.size())) {
    for (const auto j : c10::irange(lists[0].size())) {
      const at::Tensor& a = lists[0][j];
      const at::Tensor& b = lists[i][j];
      if (a.sym_sizes() != b.sym_sizes()) {
        return false;
      }
      const auto sizes = a.sym_sizes();
      const auto left = a.sym_strides();
      const auto right = b.sym_strides();
      for (const auto dim : c10::irange(sizes.size())) {
        if (sizes[dim] == c10::SymInt(1)) {
          continue;
        }
        if (left[dim] != right[dim]) {
          return false;
        }
      }
    }
  }
  return at::native::_check_tensors_do_type_promotion_with_scalars(lists[0], scalars, promotes_integer_inputs_to_float);
}

} // namespace at::cuda::host_trace
