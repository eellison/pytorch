#pragma once
// TensorIteratorBase's shape, stride and output-layout computation, templated
// on the integer type. TensorIterator.cpp instantiates int64_t; c10::SymInt,
// where every comparison is a guard, computes the same layout over symbolic
// sizes (aten/src/ATen/test/tensor_iterator_build_test.cpp).
// Parameters carry
// the names of the members they bind. An operand is duck-typed on
// OperandInfo's tensor_base(), stride_bytes, will_resize, is_output,
// target_dtype, current_dtype and is_type_defined(); outputs are allocated
// through set_output(i, sizes, strides, memory_format). The steps
// TensorIterator.cpp forwards to are C10_ALWAYS_INLINE, so its member
// functions cost no extra call over the untemplated originals.
#include <ATen/ExpandUtils.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <type_traits>

namespace at::detail::ti_build {

template <typename T>
constexpr bool is_symbolic = std::is_same_v<T, c10::SymInt>;

template <typename T>
using DimVectorOf = std::conditional_t<is_symbolic<T>, c10::SymDimVector, DimVector>;

template <typename T>
c10::ArrayRef<T> sizes(const TensorBase& t) {
  if constexpr (is_symbolic<T>) {
    return t.sym_sizes();
  } else {
    return t.sizes();
  }
}

template <typename T>
c10::ArrayRef<T> strides(const TensorBase& t) {
  if constexpr (is_symbolic<T>) {
    return t.sym_strides();
  } else {
    return t.strides();
  }
}

template <typename Shape>
int ndim(const Shape& shape_) {
  return static_cast<int>(shape_.size());
}

template <typename Operands>
int ntensors(const Operands& operands_) {
  return static_cast<int>(operands_.size());
}

template <typename Shape, typename Operands>
C10_ALWAYS_INLINE void compute_shape(Shape& shape_, const Operands& operands_, bool resize_outputs, bool& all_ops_same_shape_, bool& all_ops_are_scalars_) {
  using T = typename Shape::value_type;
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
    if (resize_outputs && op.is_output) continue;
    if constexpr (!is_symbolic<T>) {
      TORCH_CHECK(!op.tensor_base().unsafeGetTensorImpl()->has_symbolic_sizes_strides(),
        "TensorIterator does not support symbolic shapes; please implement this operator in torch/_refs "
        "using the elementwise or reduction helpers (look at backtrace to find out what operator this is)");
    }
    auto shape = sizes<T>(op.tensor_base());
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
      if constexpr (is_symbolic<T>) {
        shape_ = infer_size_symdimvector(shape_, shape);
      } else {
        shape_ = infer_size_dimvector(shape_, shape);
      }
    }
  }
  all_ops_are_scalars_ = !has_tensors;
}

template <typename Shape, typename Operands>
C10_ALWAYS_INLINE void compute_strides(const Shape& shape_, Operands& operands_, bool static_shape) {
  using T = typename Shape::value_type;
  for (auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      c10::ArrayRef<T> original_shape = static_shape ? c10::ArrayRef<T>(shape_) : sizes<T>(op.tensor_base());
      auto original_stride = strides<T>(op.tensor_base());
      auto element_size_in_bytes = op.tensor_base().element_size();
      auto offset = ndim(shape_) - original_shape.size();
      if (offset > 0)
          op.stride_bytes.resize(ndim(shape_), 0);
      else
          op.stride_bytes.resize(ndim(shape_));
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

template <typename Shape, typename Operands>
C10_ALWAYS_INLINE void permute_dimensions(Shape& shape_, Operands& operands_, IntArrayRef perm) {
  using T = typename Shape::value_type;
  TORCH_INTERNAL_ASSERT(perm.size() == static_cast<unsigned>(ndim(shape_)));

  auto reorder = [perm](c10::ArrayRef<T> data) {
    auto res = DimVectorOf<T>(data.size(), 0);
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

// See NOTE: [Computing output strides] in TensorIterator.cpp
template <typename Shape, typename Operands>
C10_ALWAYS_INLINE void reorder_dimensions(Shape& shape_, DimVector& perm_, Operands& operands_, bool is_reduction_, bool enforce_linear_iteration_) {
  // Sort the dimensions based on strides in ascending order with reduced dims
  // at the front. NOTE: that this inverts the order of C-contiguous tensors.
  // strides[0] is the fastest moving dimension instead of strides[ndim - 1].
  // See NOTE: [Computing output strides] and inline  comments for more detailed description

  perm_.resize(ndim(shape_));
  if (ndim(shape_) == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // Reordering dimensions changes iteration order
  if (enforce_linear_iteration_) {
    permute_dimensions(shape_, operands_, perm_);
    return;
  }

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto arg : c10::irange(ntensors(operands_))) {
      // ignore undefined or incorrectly sized tensors
      if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
        continue;
      }
      const auto& stride0 = operands_[arg].stride_bytes[dim0];
      const auto& stride1 = operands_[arg].stride_bytes[dim1];
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
         const auto& t_dim0 = shape_[dim0];
         const auto& t_dim1 = shape_[dim1];
         //return only if dimensions should be swapped, otherwise move on to the next tensor
         if (t_dim0 > t_dim1) {
             return 1;
         }
      }
    }
    return 0;
  };

  // insertion sort with support for ambiguous comparisons
  for (const auto i : c10::irange(1, ndim(shape_))) {
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
  permute_dimensions(shape_, operands_, perm_);
}

template <typename Shape>
SmallVector<typename Shape::value_type, 6> compatible_stride(const Shape& shape_, int64_t element_size) {
  auto stride = SmallVector<typename Shape::value_type, 6>();
  typename Shape::value_type next_stride = element_size;
  for (const auto dim : c10::irange(ndim(shape_))) {
    stride.push_back(next_stride);
    next_stride *= shape_[dim];
  }
  return stride;
}

template <typename T>
DimVectorOf<T> invert_perm(IntArrayRef perm_, c10::ArrayRef<T> input, bool has_coalesced_dimensions_) {
  // Invert the permutation caused by reorder_dimensions. This is not valid
  // after coalesce_dimensions is called.
  TORCH_INTERNAL_ASSERT(!has_coalesced_dimensions_);
  TORCH_INTERNAL_ASSERT(input.size()==perm_.size());
  auto res = DimVectorOf<T>(input.size()); //no initialization needed, every value in res should be written to.
  for (const auto dim : c10::irange(perm_.size())) {
    res[perm_[dim]] = input[dim];
  }
  return res;
}

template <typename Shape, typename Operands, typename SetOutput>
C10_ALWAYS_INLINE void allocate_or_resize_outputs(const Shape& shape_, IntArrayRef perm_, Operands& operands_, int num_outputs_, bool has_coalesced_dimensions_, const SetOutput& set_output) {
  using T = typename Shape::value_type;
  // check if permutation is just an inverted order
  bool inverted = true;
  for (const auto j : c10::irange(ndim(shape_))) {
    if (perm_[j] != ndim(shape_) - j - 1) {
      inverted = false;
      break;
    }
  }
  for (const auto i : c10::irange(num_outputs_)) {
    auto& op = operands_[i];
    if (!op.tensor_base().defined() || op.will_resize) {
      TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
      auto element_size = elementSize(op.target_dtype);
      op.stride_bytes = compatible_stride(shape_, static_cast<int64_t>(element_size));
      auto tensor_shape = invert_perm<T>(perm_, shape_, has_coalesced_dimensions_);
      if (inverted) {
        // can just return contiguous output
        // it is faster because it avoids allocating 0 size tensor and
        // resizing and restriding it
        set_output(i, tensor_shape, {}, std::nullopt);
      } else {
        auto tensor_stride = invert_perm<T>(perm_, op.stride_bytes, has_coalesced_dimensions_);
        for (const auto dim : c10::irange(ndim(shape_))) {
          tensor_stride[dim] /= static_cast<int64_t>(element_size);
        }
        set_output(i, tensor_shape, tensor_stride, std::nullopt);
      }
      op.current_dtype = op.target_dtype;
    } else if (op.tensor_base().defined()) {
      // Even if we don't resize, we still need to tell set_output about
      // the output, so that we properly set guard
      set_output(i, sizes<T>(op.tensor_base()), {}, std::nullopt);
    }
  }
}

template <typename Shape, typename Operands>
C10_ALWAYS_INLINE void coalesce_dimensions(Shape& shape_, Operands& operands_, bool& has_coalesced_dimensions_) {
  if (ndim(shape_) <= 1) {
    return;
  }

  // We can coalesce two adjacent dimensions if either dim has size 1 or if:
  // shape[n] * stride[n] == stride[n + 1].
  auto can_coalesce = [&](int dim0, int dim1) {
    const auto& shape0 = shape_[dim0];
    const auto& shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (const auto i : c10::irange(ntensors(operands_))) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  // replace each operands stride at dim0 with its stride at dim1
  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : c10::irange(ntensors(operands_))) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  int prev_dim = 0;
  for (const auto dim : c10::irange(1, ndim(shape_))) {
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
  for (const auto i : c10::irange(ntensors(operands_))) {
    operands_[i].stride_bytes.resize(ndim(shape_));
  }
  has_coalesced_dimensions_ = true;
}

template <typename Shape>
typename Shape::value_type numel(const Shape& shape_) {
  typename Shape::value_type numel = 1;
  for (const auto& size : shape_) {
    numel *= size;
  }
  return numel;
}

template <typename T, typename Operands>
C10_ALWAYS_INLINE FastSetupType compute_fast_setup_type(const Operands& operands_, bool is_reduction_, bool all_ops_same_shape_, bool enforce_linear_iteration_) {
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
    for (int64_t i = ntensors(operands_) - 1; i >= 0; --i) {
      const auto& op = operands_[i];
      if (op.tensor_base().defined() && !op.will_resize) {
        if (prev < 0) {
          prev = i;
          continue;
        }
        if (!strides<T>(operands_[prev].tensor_base()).equals(strides<T>(op.tensor_base()))) {
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

template <typename Shape, typename Operands, typename SetOutput>
C10_ALWAYS_INLINE bool fast_set_up(FastSetupType setup_type, Shape& shape_, Operands& operands_, int num_outputs_, bool& has_coalesced_dimensions_, const SetOutput& set_output) {
  using T = typename Shape::value_type;
  // This function tries to do a fast setup to avoid needless reordering of dimensions and tracking output strides
  // Return true if it can do fast setup or false otherwise
  // TODO enable fast handling for reductions
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
          set_output(i, shape_, {}, MemoryFormat::Contiguous);
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
          set_output(i, shape_, {}, MemoryFormat::ChannelsLast);
        }
        break;
      }
    case FastSetupType::NON_OVERLAPPING_DENSE:
      {
        // find the index of a defined tensor in operands_ start from input tensor
        int i_defined = -1;
        for (i_defined = ntensors(operands_) - 1; i_defined >= 0; --i_defined) {
          if (operands_[i_defined].tensor_base().defined()) break;
        }
        TORCH_CHECK(i_defined >= 0, "Can not find a defined tensor when fast allocating memory to outputs");
        for (const auto i : c10::irange(num_outputs_)) {
          auto& op = operands_[i];
          if (!op.tensor_base().defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
          }
          set_output(i, shape_, strides<T>(operands_[i_defined].tensor_base()), std::nullopt);
        }
        break;
      }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", std::to_string((int)setup_type));
  }
  //coalescing dimensions consists of collapsing dimensions to 1 (we are limited to contiguous no-broadcast cases here)
  if (ndim(shape_) > 1){
    has_coalesced_dimensions_ = true;
  }
  if (ndim(shape_) >= 1) {
    shape_[0] = numel(shape_);
    shape_.resize(1);
  }
  for (auto& op : operands_ ) {
    auto element_size_in_bytes = op.tensor_base().element_size();
    op.stride_bytes.resize(ndim(shape_));
    if (ndim(shape_)>0) {
      op.stride_bytes[0] = element_size_in_bytes;
    }
  }
  return true;
}

} // namespace at::detail::ti_build
