// TensorIterator.cpp's elementwise build, re-typed: sizes, strides and the
// data pointer are c10::SymInt (compute_shape, compute_strides,
// reorder_dimensions, allocate_or_resize_outputs, coalesce_dimensions,
// fast_set_up, can_use_32bit_indexing and set_output_raw_strided are the
// functions of the same name there, with sizes() -> sym_sizes(), the
// *_symint allocation entry points and sym_*_data_ptr()). The build's
// comparisons are guards of the trace; nothing here runs on the ordinary path.
#include <ATen/cuda/host_trace/ti/TensorIteratorSym.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/native/TypeProperties.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif
#include <c10/util/irange.h>

#include <limits>
#include <numeric>

namespace at::cuda::host_trace::ti {

TensorIteratorSym TensorIteratorSym::binary_op(const Tensor& out, const Tensor& a, const Tensor& b, TensorIteratorSymConfig config) {
  TensorIteratorSym iter;
  iter.add_output(out);
  iter.add_input(a);
  iter.add_input(b);
  iter.build(config);
  return iter;
}

TensorIteratorSym TensorIteratorSym::binary_op(const Tensor& out, const Tensor& a, const Tensor& b) {
  TensorIteratorSymConfig config;
  config.allow_cpu_scalars_ = true;
  return binary_op(out, a, b, config);
}

TensorIteratorSym TensorIteratorSym::unary_op(const Tensor& out, const Tensor& a, TensorIteratorSymConfig config) {
  TensorIteratorSym iter;
  iter.add_output(out);
  iter.add_input(a);
  iter.build(config);
  return iter;
}

TensorIteratorSym TensorIteratorSym::unary_op(const Tensor& out, const Tensor& a) {
  return unary_op(out, a, TensorIteratorSymConfig());
}

TensorIteratorSym TensorIteratorSym::ternary_op(const Tensor& out, const Tensor& a, const Tensor& b, const Tensor& c, TensorIteratorSymConfig config) {
  TensorIteratorSym iter;
  iter.add_output(out);
  iter.add_input(a);
  iter.add_input(b);
  iter.add_input(c);
  iter.build(config);
  return iter;
}

TensorIteratorSym TensorIteratorSym::ternary_op(const Tensor& out, const Tensor& a, const Tensor& b, const Tensor& c) {
  // the structured ternary metas (addcmul, lerp.Tensor) allow a CPU scalar as
  // binary_op does
  TensorIteratorSymConfig config;
  config.allow_cpu_scalars_ = true;
  return ternary_op(out, a, b, c, config);
}

TensorIteratorSym TensorIteratorSym::comparison_op(const Tensor& out, const Tensor& a, const Tensor& b) {
  TensorIteratorSymConfig config;
  config.allow_cpu_scalars_ = true;
  if (!out.defined()) {
    config.static_dtype_ = kBool;
  }
  TensorIteratorSym iter;
  iter.add_output(out);
  iter.add_input(a);
  iter.add_input(b);
  iter.build(config);
  return iter;
}

TensorIteratorSym TensorIteratorSym::reduce_op(const Tensor& out, const Tensor& a) {
  TORCH_INTERNAL_ASSERT(out.defined());
  // TensorIterator::reduce_op: no output resize, is_reduction; the real
  // config also promotes the input to the common dtype, which v1 declines
  // in compute_types (the output has the input's dtype)
  TensorIteratorSymConfig config;
  config.resize_outputs_ = false;
  config.is_reduction_ = true;
  TensorIteratorSym iter;
  iter.add_output(out);
  iter.add_input(a);
  iter.build(config);
  return iter;
}

int TensorIteratorSym::num_reduce_dims() const {
  int count = 0;
  for (const auto dim : c10::irange(ndim())) {
    if (operands_[0].stride_bytes[dim] == 0) {
      count++;
    }
  }
  return count;
}

c10::SymInt TensorIteratorSym::num_output_elements() const {
  c10::SymInt elem = 1;
  for (const auto dim : c10::irange(ndim())) {
    if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0)  {
      elem *= shape_[dim];
    }
  }
  return elem;
}

void TensorIteratorSym::compute_types(const TensorIteratorSymConfig& config) {
  // a CPU scalar beside CUDA operands (TensorIteratorBase::compute_types,
  // allow_cpu_scalars: not an output, dim 0, at most one) takes no part in
  // the device and dtype rules below; its promotion is checked after them
  int cpu_scalars = 0;
  at::native::ResultTypeState state = {};
  for (auto& op : operands_) {
    if (!op.tensor_base().defined()) {
      if (config.static_dtype_.has_value()) {
        op.target_dtype = *config.static_dtype_;
      }
      continue;
    }
    if (!op.is_output) {
      state = at::native::update_result_type_state(op.tensor(), state);
    }
    if (config.allow_cpu_scalars_ && !op.is_output && op.tensor_base().dim() == 0 && op.tensor_base().is_cpu()) {
      TORCH_CHECK(++cpu_scalars <= 1, "Trying to pass too many CPU scalars to non-CPU kernel!");
      continue;
    }
    if (!op.tensor_base().is_cuda()) {
      decline(c10::str(
          "host_trace: TensorIterator operand on ",
          op.tensor_base().device(),
          " (a CPU scalar or a non-CUDA tensor) is not traced (declined)"));
    }
    if (common_dtype_ == ScalarType::Undefined) {
      common_dtype_ = op.tensor_base().scalar_type();
      common_device_ = op.tensor_base().device();
    }
    if (config.check_all_same_dtype_ && op.tensor_base().scalar_type() != common_dtype_) {
      decline(c10::str(
          "host_trace: TensorIterator operands of different dtypes (",
          common_dtype_,
          ", ",
          op.tensor_base().scalar_type(),
          "): type promotion is not traced (declined)"));
    }
    if (op.tensor_base().device() != common_device_) {
      decline("host_trace: TensorIterator operands on different devices (declined)");
    }
  }
  if (cpu_scalars > 0) {
    if (common_dtype_ == ScalarType::Undefined) {
      decline("host_trace: TensorIterator with no CUDA operand is not traced (declined)");
    }
    // result_type over the inputs as the real promotion computes it (a
    // wrapped number never raises a device tensor's dtype within its
    // category); a CUDA operand the real op would cast declines
    const ScalarType promoted = at::native::result_type(state);
    if (promoted != common_dtype_) {
      decline(c10::str(
          "host_trace: TensorIterator with a CPU scalar promotes its CUDA operand from ",
          common_dtype_,
          " to ",
          promoted,
          ": type promotion is not traced (declined)"));
    }
  }
  for (auto& op : operands_) {
    if (!op.is_type_defined()) {
      op.target_dtype = common_dtype_;
    }
    if (!op.device.has_value()) {
      op.device = common_device_;
    }
  }
}

void TensorIteratorSym::compute_shape(const TensorIteratorSymConfig& config) {
  if (config.static_shape_.has_value()) {
    shape_ = *config.static_shape_;
    return;
  }

  all_ops_same_shape_ = true;
  bool has_scalars = false;
  bool has_tensors = false;
  for (auto& op : operands_) {
    if (!op.tensor_base().defined()) continue;

    // For now, don't include output tensors when we're resizing outputs.
    // These shapes don't participate in shape computation.
    // This preserves the legacy behavior where torch.add(..., out=dst) resizes
    // the destination tensor.  If the output tensor is also an input, we'll
    // pick it up later in the operands.
    if (config.resize_outputs_ && op.is_output) continue;
    auto shape = op.tensor_base().sym_sizes();
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

void TensorIteratorSym::mark_resize_outputs(const TensorIteratorSymConfig& config) {
  if (config.static_shape_.has_value()) {
    return;
  }
  for (const auto i : c10::irange(num_outputs_)) {
    const auto& output = tensor_base(i);
    if (output.defined() && !output.sym_sizes().equals(shape_)) {
      if (config.resize_outputs_ && !operands_[i].is_read_write) {
        // TensorIteratorBase resizes the out= tensor here (resize_output: a
        // deprecation warning when it had elements); a traced tensor keeps
        // its storage, so the call declines
        decline(c10::str(
            "host_trace: out= of shape ", output.sym_sizes(), " does not match the result's shape ",
            c10::SymIntArrayRef(shape_), "; eager resizes it and the trace does not (declined)"));
      }
      TORCH_CHECK(
          is_reduction_,
          "output with shape ",
          output.sym_sizes(),
          " doesn't match the broadcast shape ",
          shape_);
    }
  }
}

void TensorIteratorSym::compute_strides(const TensorIteratorSymConfig& config) {
  for (auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      c10::SymIntArrayRef original_shape = config.static_shape_ ? c10::SymIntArrayRef(shape_) : op.tensor_base().sym_sizes();
      auto original_stride = op.tensor_base().sym_strides();
      auto element_size_in_bytes = op.tensor_base().element_size();
      auto offset = ndim() - original_shape.size();
      if (offset > 0)
          op.stride_bytes.resize(ndim(), 0);
      else
          op.stride_bytes.resize(ndim());
      for (const auto i : c10::irange(original_shape.size())) {
        // see NOTE: [Computing output strides]
        if (original_shape[i] == 1 && shape_[offset + i] !=1) {
          op.stride_bytes[offset + i] = 0;
        } else {
          op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
        }
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

bool TensorIteratorSym::fast_set_up(const TensorIteratorSymConfig& config) {
  // This function tries to do a fast setup to avoid needless reordering of dimensions and tracking output strides
  // Return true if it can do fast setup or false otherwise
  // TODO enable fast handling for reductions
  FastSetupType setup_type = compute_fast_setup_type(config);
  if (setup_type == FastSetupType::NONE) {
    return false;
  }

  // allocate memory for output, memory format depends on setup_type
  switch (setup_type) {
    case FastSetupType::CONTIGUOUS:
      {
        for (const auto i : c10::irange(num_outputs_)) {
          auto& op = operands_[i];
          if (!op.tensor_base().defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
          }
          set_output_raw_strided(i, shape_, {}, op.options().memory_format(MemoryFormat::Contiguous));
        }
        break;
      }
    case FastSetupType::CHANNELS_LAST:
      {
        for (const auto i : c10::irange(num_outputs_)) {
          auto& op = operands_[i];
          if (!op.tensor_base().defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
          }
          set_output_raw_strided(i, shape_, {}, op.options().memory_format(MemoryFormat::ChannelsLast));
        }
        break;
      }
    case FastSetupType::NON_OVERLAPPING_DENSE:
      {
        // find the index of a defined tensor in operands_ start from input tensor
        int i_defined = -1;
        for (i_defined = ntensors() - 1; i_defined >= 0; --i_defined) {
          if (tensor(i_defined).defined()) break;
        }
        TORCH_CHECK(i_defined >= 0, "Can not find a defined tensor when fast allocating memory to outputs");
        for (const auto i : c10::irange(num_outputs_)) {
          auto& op = operands_[i];
          if (!op.tensor_base().defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
          }
          set_output_raw_strided(i, shape_, tensor_base(i_defined).sym_strides(), op.options());
        }
        break;
      }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", std::to_string((int)setup_type));
  }
  //coalescing dimensions consists of collapsing dimensions to 1 (we are limited to contiguous no-broadcast cases here)
  if (ndim() > 1){
    has_coalesced_dimensions_ = true;
  }
  if (ndim() >= 1) {
    shape_[0] = numel();
    shape_.resize(1);
  }
  for (auto& op : operands_ ) {
    auto element_size_in_bytes = op.tensor_base().element_size();
    op.stride_bytes.resize(ndim());
    if (ndim()>0) {
      op.stride_bytes[0] = element_size_in_bytes;
    }
  }
  return true;
}

FastSetupType TensorIteratorSym::compute_fast_setup_type(const TensorIteratorSymConfig& config) {
  if (is_reduction_ || !all_ops_same_shape_) {
    return FastSetupType::NONE;
  }

  // For linear iteration, only contiguous tensors can be coalesced
  // Fast setup of any other format requires changing iteration order
  if (enforce_linear_iteration_) {
    for (const auto& op : operands_) {
      if (op.tensor_base().defined() && !op.will_resize) {
        auto is_contiguous = op.tensor_base().is_contiguous(at::MemoryFormat::Contiguous);
        if (!is_contiguous) {
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::CONTIGUOUS;
  }

  bool is_contiguous = true;
  for (const auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      is_contiguous &= op.tensor_base().is_contiguous(at::MemoryFormat::Contiguous);
      if (!is_contiguous) {
        break;
      }
    }
  }
  // TODO this leads to ambiguous cases (NC11) to be always treated as contiguous
  if (is_contiguous) {
    return FastSetupType::CONTIGUOUS;
  }

  bool is_channels_last = true;
  bool is_non_overlapping_and_dense = true;
  for (const auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      is_channels_last &= op.tensor_base().is_contiguous(at::MemoryFormat::ChannelsLast);
      is_non_overlapping_and_dense &= op.tensor_base().is_non_overlapping_and_dense();
    }
  }
  if (is_channels_last) {
    return FastSetupType::CHANNELS_LAST;
  }
  if (is_non_overlapping_and_dense) {
    int64_t prev = -1;
    // Fast setup is allowed only when all the defined tensors have the same shape and strides,
    // Iterate from back to check input tensors' strides first, then output tensors'.
    for (int64_t i = ntensors() - 1; i >= 0; --i) {
      const auto& op = operands_[i];
      if (op.tensor_base().defined() && !op.will_resize) {
        if (prev < 0) {
          prev = i;
          continue;
        }
        if (!tensor_base(prev).sym_strides().equals(op.tensor_base().sym_strides())) {
          // [Note: stride check for non contiguous tensors in fast setup]
          // We prevent 3 cases doing fast setup here:
          // 1. input tensors have different strides.
          // 2. output tensors won't be resized and have different strides.
          // 3. input tensors have the same strides, but output tensors have different strides with input tensors.
          //    We don't allow re-stride output tensors in this case since it is not compatible with
          //    numpy. The behavior in numpy is that if the output tensor has same shape as the input
          //    tensor but different strides, the strides of output tensor will be preserved, so we do
          //    the same in tensor iterator.
          return FastSetupType::NONE;
        }
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

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[n] * stride[n] == stride[n + 1].
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

  // replace each operands stride at dim0 with its stride at dim1
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
  // Sort the dimensions based on strides in ascending order with reduced dims
  // at the front. NOTE: that this inverts the order of C-contiguous tensors.
  // strides[0] is the fastest moving dimension instead of strides[ndim - 1].
  // See NOTE: [Computing output strides] and inline  comments for more detailed description

  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // Reordering dimensions changes iteration order
  if (enforce_linear_iteration_) {
    permute_dimensions(perm_);
    return;
  }

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto arg : c10::irange(ntensors())) {
      // ignore undefined or incorrectly sized tensors
      if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
        continue;
      }
      c10::SymInt stride0 = operands_[arg].stride_bytes[dim0];
      c10::SymInt stride1 = operands_[arg].stride_bytes[dim1];
      if (is_reduction_ && operands_[arg].is_output) {
        // move reduced dimensions to the front
        // strides of reduced dimensions are always set to 0 by review_reduce_result
        if ((stride0 == 0) != (stride1 == 0)) {
          return stride1 == 0 ? 1 : -1;
        }
      }
      //move on to the next input if one of the dimensions is broadcasted
      if (stride0 == 0 || stride1 == 0) {
        continue;
      // it is important to return here only with strict comparisons, for equal strides we try to break the tie later
      // by comparing corresponding dimensions or if that does not work, moving on to the next tensor
      } else if (stride0 < stride1) {
        return -1;
      } else  if (stride0 > stride1) {
        return 1;
      } else { //equal strides, use dimensions themselves as the tie-breaker.
        //at this point, with zero strides out of the way, we are guaranteed that operand dimensions are equal to shape_
         auto t_dim0 = shape_[dim0];
         auto t_dim1 = shape_[dim1];
         //return only if dimensions should be swapped, otherwise move on to the next tensor
         if (t_dim0 > t_dim1) {
             return 1;
         }
      }
    }
    return 0;
  };

  // insertion sort with support for ambiguous comparisons
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

  // perform re-ordering of shape and strides
  permute_dimensions(perm_);
}

StrideVector TensorIteratorSym::compatible_stride(int64_t element_size) const {
  auto stride = StrideVector();
  c10::SymInt next_stride = element_size;
  for (const auto dim : c10::irange(ndim())) {
    stride.push_back(next_stride);
    next_stride *= shape_[dim];
  }
  return stride;
}

c10::SymDimVector TensorIteratorSym::invert_perm(c10::SymIntArrayRef input) const {
  // Invert the permutation caused by reorder_dimensions. This is not valid
  // after coalesce_dimensions is called.
  TORCH_INTERNAL_ASSERT(!has_coalesced_dimensions_);
  TORCH_INTERNAL_ASSERT(input.size()==perm_.size());
  auto res = c10::SymDimVector(input.size()); //no initialization needed, every value in res should be written to.
  for (const auto dim : c10::irange(ndim())) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

void TensorIteratorSym::allocate_or_resize_outputs() {
  // check if permutation is just an inverted order
  bool inverted = true;
  for (const auto j : c10::irange(ndim())) {
    if (perm_[j] != ndim() - j - 1) {
      inverted = false;
      break;
    }
  }
  for (const auto i : c10::irange(num_outputs_)) {
    auto& op = operands_[i];
    if (!op.tensor_base().defined() || op.will_resize) {
      TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
      auto element_size = elementSize(op.target_dtype);
      op.stride_bytes = compatible_stride(static_cast<int64_t>(element_size));
      auto tensor_shape = invert_perm(shape_);
      if (inverted) {
        // can just return contiguous output
        // it is faster because it avoids allocating 0 size tensor and
        // resizing and restriding it
        set_output_raw_strided(i, tensor_shape, {}, op.options());
      } else {
        auto tensor_stride = invert_perm(op.stride_bytes);
        for (const auto dim : c10::irange(ndim())) {
          tensor_stride[dim] /= static_cast<int64_t>(element_size);
        }
        set_output_raw_strided(i, tensor_shape, tensor_stride, op.options());
      }
      op.current_dtype = op.target_dtype;
    } else if (op.tensor_base().defined()) {
      // Even if we don't resize, we still need to tell set_output about
      // the output, so that we properly set guard
      set_output_raw_strided(i, op.tensor_base().sym_sizes(), {}, op.options());
    }
  }
}

void TensorIteratorSym::permute_dimensions(IntArrayRef perm) {
  TORCH_INTERNAL_ASSERT(perm.size() == static_cast<unsigned>(ndim()));

  auto reorder = [perm](c10::SymIntArrayRef data) {
    auto res = c10::SymDimVector(data.size(), 0);
    for (const auto i : c10::irange(perm.size())) {
      res[i] = data[perm[i]];
    }
    return res;
  };

  // Update shape and strides
  shape_ = reorder(shape_);
  for (auto& op : operands_) {
    if (!op.stride_bytes.empty()) {
      op.stride_bytes = reorder(op.stride_bytes);
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

  int num_tensors = ntensors();
  for (const auto i : c10::irange(num_tensors)) {
    if (strides(i)[0] != element_size(i)) {
      return false;
    }
  }
  return true;
}

void TensorIteratorSym::set_output_raw_strided(int64_t output_idx, c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, TensorOptions options) {
  auto& op = operands_[output_idx];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
  if (!op.tensor_base().defined()) {
      if (strides.empty()) {
        op.tensor(c10::MaybeOwned<TensorBase>::owned(at::empty_symint(sizes, options)));
      } else {
        op.tensor(c10::MaybeOwned<TensorBase>::owned(at::empty_strided_symint(sizes, strides, options)));
      }
      op.current_dtype = op.target_dtype;
  } else if (op.will_resize) {
      at::native::resize_output_symint(op.tensor(), sizes);
      if (!strides.empty()) {
        TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
        op.tensor().as_strided__symint(sizes, strides);
      } else {
        auto const memory_format = options.memory_format_opt();
        if (memory_format.has_value()) {
          op.tensor_base().unsafeGetTensorImpl()->empty_tensor_restride(*memory_format);
        }
      }
  }
}

namespace {

constexpr const char* kInternalOverlap =
    "unsupported operation: more than one element of the written-to tensor "
    "refers to a single memory location. Please clone() the tensor before "
    "performing the operation.";
constexpr const char* kPartialOverlap =
    "unsupported operation: some elements of the input tensor and the "
    "written-to tensor refer to a single memory location. Please clone() the "
    "tensor before performing the operation.";

const RootInfo* root_of(const TensorBase& t) {
  TraceState* s = active();
  if (s == nullptr) {
    return nullptr;
  }
  auto it = s->roots.find(t.unsafeGetTensorImpl());
  return it == s->roots.end() ? nullptr : &it->second;
}

bool is_allocation(const RootInfo* r) {
  return r != nullptr && r->allocation;
}

bool same_strides(const TensorBase& a, const TensorBase& b) {
  const auto sa = a.sym_strides();
  const auto sb = b.sym_strides();
  if (sa.size() != sb.size()) {
    return false;
  }
  for (const auto i : c10::irange(sa.size())) {
    if (!(sa[i] == sb[i])) {
      return false;
    }
  }
  return true;
}

// [begin, end) of the elements in bytes: from the storage offset when the
// two tensors share a root (the root cancels), else from the address
std::pair<c10::SymInt, c10::SymInt> byte_interval(const TensorBase& t, bool from_offset) {
  const auto itemsize = static_cast<int64_t>(t.itemsize());
  c10::SymInt begin = from_offset ? t.sym_storage_offset() * itemsize : sym_const_data_ptr(t);
  return {begin, begin + t.sym_numel() * itemsize};
}

} // namespace

// at::get_overlap_status without its symbolic-sizes bail-out, in its order.
MemOverlapStatus overlap_status_sym(const TensorBase& a, const TensorBase& b) {
  if (a.unsafeGetTensorImpl() == b.unsafeGetTensorImpl()) {
    return MemOverlapStatus::Full;
  }
  if (a.sym_numel() == 0 || b.sym_numel() == 0) {
    return MemOverlapStatus::No;
  }
  if (!a.unsafeGetTensorImpl()->is_non_overlapping_and_dense_or_false() ||
      !b.unsafeGetTensorImpl()->is_non_overlapping_and_dense_or_false()) {
    return MemOverlapStatus::TooHard;
  }
  const RootInfo* ra = root_of(a);
  const RootInfo* rb = root_of(b);
  const bool same_root = ra != nullptr && rb != nullptr && ra->name == rb->name;
  // eager tests storage identity first: two storages never overlap. An
  // allocation made inside the call is its own storage at every replay, and a
  // tensor on another device (a wrapped CPU scalar) shares nothing with a
  // CUDA one; two inputs, or an input beside a real tensor, may share one
  // storage at replay, and their intervals decide (disjoint storages have
  // disjoint intervals, so the interval test alone is eager's answer).
  if (!same_root && (is_allocation(ra) || is_allocation(rb) || a.device() != b.device())) {
    return MemOverlapStatus::No;
  }
  const auto [a_begin, a_end] = byte_interval(a, same_root);
  const auto [b_begin, b_end] = byte_interval(b, same_root);
  if (a_begin == b_begin && a_end == b_end) {
    return same_strides(a, b) ? MemOverlapStatus::Full : MemOverlapStatus::Partial;
  }
  // one question, so a "no" is recorded as eager's own condition (the
  // intervals are disjoint), not as the order the trace's inputs happened to
  // have in memory
  if (a_begin.sym_lt(b_end).sym_and(b_begin.sym_lt(a_end)).guard_bool(__FILE__, __LINE__)) {
    return MemOverlapStatus::Partial;
  }
  return MemOverlapStatus::No;
}

void assert_no_internal_overlap_sym(const TensorBase& t) {
  // has_internal_overlap works from sizes and strides (guards) and needs no address
  if (at::has_internal_overlap(t) == MemOverlap::Yes) {
    decline(c10::str("host_trace: refused as eager refuses (RuntimeError): ", kInternalOverlap, " (declined)"));
  }
}

void assert_no_partial_overlap_sym(const TensorBase& a, const TensorBase& b) {
  if (overlap_status_sym(a, b) == MemOverlapStatus::Partial) {
    decline(c10::str("host_trace: refused as eager refuses (RuntimeError): ", kPartialOverlap, " (declined)"));
  }
}

void assert_no_overlap_sym(const TensorBase& a, const TensorBase& b) {
  const auto lap = overlap_status_sym(a, b);
  if (lap == MemOverlapStatus::Partial || lap == MemOverlapStatus::Full) {
    decline(c10::str("host_trace: refused as eager refuses (RuntimeError): ", kPartialOverlap, " (declined)"));
  }
}

// TensorIteratorBase::compute_mem_overlaps with the predicates above.
void TensorIteratorSym::compute_mem_overlaps(const TensorIteratorSymConfig& config) {
  if (!config.check_mem_overlap_) {
    return;
  }
  for (const auto i : c10::irange(num_outputs_)) {
    const auto& output = tensor_base(i);
    if (!output.defined()) {
      continue;
    }
    assert_no_internal_overlap_sym(output);
    for (const auto j : c10::irange(num_outputs_, ntensors())) {
      const auto& input = tensor_base(j);
      if (!input.is_same(output)) {
        assert_no_partial_overlap_sym(output, input);
      }
    }
  }
}

void TensorIteratorSym::build(TensorIteratorSymConfig& config) {
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;
  mark_outputs();
  compute_mem_overlaps(config);
  compute_shape(config);
  mark_resize_outputs(config);
  compute_types(config);
  if (!fast_set_up(config)) {
    compute_strides(config);
    reorder_dimensions();
    allocate_or_resize_outputs();
    coalesce_dimensions();
  }
  for (auto& op : operands_) {
    TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
    if (op.tensor_base().is_cpu()) {
      // a CPU scalar: read as a value by the entry, never a launch address
      continue;
    }
    // outputs and in-place operands through the mutable form, inputs through
    // the const form (a copy-on-write input stays lazy, as in the real op)
    op.data = op.is_output ? sym_mutable_data_ptr(op.tensor_base())
                           : sym_const_data_ptr(op.tensor_base());
  }
}

bool TensorIteratorSym::is_cpu_scalar(int64_t arg) const {
  const auto& op = operands_[arg];
  if (!op.tensor_base().is_cpu()) {
    return false;
  }
  // TensorIteratorBase::is_scalar on the (literal zero) strides of the CPU operand
  for (const auto dim : c10::irange(ndim())) {
    if (op.stride_bytes[dim] != 0 && shape_[dim] != 1) {
      return false;
    }
  }
  return true;
}

} // namespace at::cuda::host_trace::ti
