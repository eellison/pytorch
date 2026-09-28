#include <ATen/cuda/host_trace/TensorIteratorSym.h>

#include <ATen/ExpandUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif
#include <c10/util/irange.h>

#include <limits>
#include <numeric>

namespace at::cuda::host_trace {

TensorIteratorSym TensorIteratorSym::unary_op(Recorder& rec, const TensorBase& a) {
  TensorIteratorSym iter;
  iter.operands_.resize(2);
  iter.operands_[1].tensor = a;
  iter.build(rec);
  return iter;
}

TensorIteratorSym TensorIteratorSym::binary_op(Recorder& rec, const TensorBase& a, const TensorBase& b) {
  TensorIteratorSym iter;
  iter.operands_.resize(3);
  iter.operands_[1].tensor = a;
  iter.operands_[2].tensor = b;
  iter.build(rec);
  return iter;
}

void TensorIteratorSym::compute_types() {
  for (const auto i : c10::irange(1, ntensors())) {
    const TensorBase& t = operands_[i].tensor;
    if (!t.is_cuda()) {
      decline(c10::str("an operand on ", t.device()));
    }
    if (common_dtype_ == ScalarType::Undefined) {
      common_dtype_ = t.scalar_type();
      common_device_ = t.device();
    } else if (t.scalar_type() != common_dtype_) {
      decline(c10::str("operands of ", common_dtype_, " and ", t.scalar_type()));
    } else if (t.device() != common_device_) {
      decline("operands on different devices");
    }
  }
}

void TensorIteratorSym::compute_shape() {
  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (const auto i : c10::irange(1, ntensors())) {
    auto shape = operands_[i].tensor.sym_sizes();
    if (shape.empty()) {
      has_scalars = true;
    } else {
      has_tensors = true;
    }
    if (has_scalars && has_tensors) {
      all_ops_same_shape_ = false;
    }
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!shape.equals(shape_)) {
      all_ops_same_shape_ = false;
      shape_ = infer_size_symdimvector(shape_, shape);
    }
  }
}

void TensorIteratorSym::compute_strides() {
  for (auto& op : operands_) {
    if (!op.tensor.defined()) {
      continue;
    }
    auto original_shape = op.tensor.sym_sizes();
    auto original_stride = op.tensor.sym_strides();
    auto element_size_in_bytes = op.tensor.element_size();
    auto offset = ndim() - original_shape.size();
    op.stride_bytes.resize(ndim(), 0);
    for (const auto i : c10::irange(original_shape.size())) {
      if (original_shape[i] == 1 && shape_[offset + i] != 1) {
        op.stride_bytes[offset + i] = 0;
      } else {
        op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
      }
    }
  }
}

bool TensorIteratorSym::can_use_32bit_indexing() const {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  if (numel() > max_value) {
    return false;
  }
  for (auto& op : operands_) {
    c10::SymInt max_offset = 1;
    for (const auto dim : c10::irange(ndim())) {
      max_offset += (shape_[dim] - 1) * op.stride_bytes[dim];
    }
    if (max_offset > max_value) {
      return false;
    }
  }
  return true;
}

bool TensorIteratorSym::fast_set_up() {
  switch (compute_fast_setup_type()) {
    case FastSetupType::NONE:
      return false;
    case FastSetupType::CONTIGUOUS:
      set_output(shape_, {}, options().memory_format(MemoryFormat::Contiguous));
      break;
    case FastSetupType::CHANNELS_LAST:
      set_output(shape_, {}, options().memory_format(MemoryFormat::ChannelsLast));
      break;
    case FastSetupType::NON_OVERLAPPING_DENSE:
      set_output(shape_, operands_.back().tensor.sym_strides(), options());
      break;
  }
  if (ndim() > 1) {
    has_coalesced_dimensions_ = true;
  }
  if (ndim() >= 1) {
    shape_[0] = numel();
    shape_.resize(1);
  }
  for (auto& op : operands_) {
    op.stride_bytes.resize(ndim());
    if (ndim() > 0) {
      op.stride_bytes[0] = op.tensor.element_size();
    }
  }
  return true;
}

FastSetupType TensorIteratorSym::compute_fast_setup_type() const {
  if (!all_ops_same_shape_) {
    return FastSetupType::NONE;
  }
  bool is_contiguous = true;
  for (const auto i : c10::irange(1, ntensors())) {
    is_contiguous &= operands_[i].tensor.is_contiguous(MemoryFormat::Contiguous);
    if (!is_contiguous) {
      break;
    }
  }
  if (is_contiguous) {
    return FastSetupType::CONTIGUOUS;
  }
  bool is_channels_last = true;
  bool is_non_overlapping_and_dense = true;
  for (const auto i : c10::irange(1, ntensors())) {
    is_channels_last &= operands_[i].tensor.is_contiguous(MemoryFormat::ChannelsLast);
    is_non_overlapping_and_dense &= operands_[i].tensor.is_non_overlapping_and_dense();
  }
  if (is_channels_last) {
    return FastSetupType::CHANNELS_LAST;
  }
  if (is_non_overlapping_and_dense) {
    for (const auto i : c10::irange(2, ntensors())) {
      if (!operands_[1].tensor.sym_strides().equals(operands_[i].tensor.sym_strides())) {
        return FastSetupType::NONE;
      }
    }
    return FastSetupType::NON_OVERLAPPING_DENSE;
  }
  return FastSetupType::NONE;
}

void TensorIteratorSym::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };
  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };
  int prev_dim = 0;
  for (const auto dim : c10::irange(1, ndim())) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }
  shape_.resize(prev_dim + 1);
  for (const auto i : c10::irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}

c10::SymInt TensorIteratorSym::numel() const {
  c10::SymInt numel = 1;
  for (const c10::SymInt& size : shape_) {
    numel *= size;
  }
  return numel;
}

void TensorIteratorSym::reorder_dimensions() {
  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }
  std::iota(perm_.rbegin(), perm_.rend(), 0);
  // the inputs' strides; the output is allocated after
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto arg : c10::irange(1, ntensors())) {
      const c10::SymInt& stride0 = operands_[arg].stride_bytes[dim0];
      const c10::SymInt& stride1 = operands_[arg].stride_bytes[dim1];
      if (stride0 == 0 || stride1 == 0) {
        continue;
      } else if (stride0 < stride1) {
        return -1;
      } else if (stride0 > stride1) {
        return 1;
      } else if (shape_[dim0] > shape_[dim1]) {
        return 1;
      }
    }
    return 0;
  };
  for (const auto i : c10::irange(1, ndim())) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
  permute_dimensions(perm_);
}

StrideVector TensorIteratorSym::compatible_stride(int64_t element_size) const {
  StrideVector stride;
  c10::SymInt next_stride = element_size;
  for (const auto dim : c10::irange(ndim())) {
    stride.push_back(next_stride);
    next_stride *= shape_[dim];
  }
  return stride;
}

c10::SymDimVector TensorIteratorSym::invert_perm(c10::SymIntArrayRef input) const {
  TORCH_INTERNAL_ASSERT(!has_coalesced_dimensions_);
  TORCH_INTERNAL_ASSERT(input.size() == perm_.size());
  c10::SymDimVector res(input.size());
  for (const auto dim : c10::irange(ndim())) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

void TensorIteratorSym::allocate_outputs() {
  bool inverted = true;
  for (const auto j : c10::irange(ndim())) {
    if (perm_[j] != ndim() - j - 1) {
      inverted = false;
      break;
    }
  }
  auto& op = operands_[0];
  const auto element_size = static_cast<int64_t>(elementSize(common_dtype_));
  op.stride_bytes = compatible_stride(element_size);
  auto tensor_shape = invert_perm(shape_);
  if (inverted) {
    set_output(tensor_shape, {}, options());
  } else {
    auto tensor_stride = invert_perm(op.stride_bytes);
    for (const auto dim : c10::irange(ndim())) {
      tensor_stride[dim] /= element_size;
    }
    set_output(tensor_shape, tensor_stride, options());
  }
}

void TensorIteratorSym::permute_dimensions(IntArrayRef perm) {
  TORCH_INTERNAL_ASSERT(perm.size() == static_cast<unsigned>(ndim()));
  auto reorder = [perm](c10::SymIntArrayRef data) {
    c10::SymDimVector res(data.size(), 0);
    for (const auto i : c10::irange(perm.size())) {
      res[i] = data[perm[i]];
    }
    return res;
  };
  shape_ = reorder(shape_);
  for (auto& op : operands_) {
    if (!op.stride_bytes.empty()) {
      auto reordered = reorder(op.stride_bytes);
      op.stride_bytes.assign(reordered.begin(), reordered.end());
    }
  }
}

bool TensorIteratorSym::is_contiguous() const {
  if (numel() == 1) {
    return true;
  }
  if (ndim() != 1) {
    return false;
  }
  return has_contiguous_first_dim();
}

bool TensorIteratorSym::has_contiguous_first_dim() const {
  if (ndim() == 0) {
    return true;
  }
  for (const auto i : c10::irange(ntensors())) {
    if (strides(i)[0] != element_size(i)) {
      return false;
    }
  }
  return true;
}

void TensorIteratorSym::set_output(c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, TensorOptions options) {
  operands_[0].tensor = strides.empty() ? at::empty_symint(sizes, options) : at::empty_strided_symint(sizes, strides, options);
}

void TensorIteratorSym::build(Recorder& rec) {
  compute_types();
  compute_shape();
  if (!fast_set_up()) {
    compute_strides();
    reorder_dimensions();
    allocate_outputs();
    coalesce_dimensions();
  }
  for (auto& op : operands_) {
    op.data = rec.data_ptr(op.tensor);
  }
}

} // namespace at::cuda::host_trace
