#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/TensorIterator.h>
#undef TORCH_ASSERT_NO_OPERATORS

#include <ATen/core/Tensor.h>

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorIteratorInternal.h>
#include <ATen/detail/TensorIteratorBuild.h>
#include <ATen/TensorIteratorSym.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#endif

#include <c10/util/irange.h>
#include <c10/util/SmallBuffer.h>

#include <array>
#include <cmath>

namespace at {

using DimMask = TensorIteratorBase::DimMask;
using PtrVector = TensorIteratorBase::PtrVector;
using loop2d_t = TensorIteratorBase::loop2d_t;
using StrideVector = TensorIteratorBase::StrideVector;
namespace ti_build = detail::ti_build;

namespace {

inline void get_base_ptrs(char** ptrs, ArrayRef<OperandInfo> operands) {
  std::transform(operands.begin(), operands.end(), ptrs, [](const OperandInfo& op) {
    return static_cast<char*>(op.data);
  });
}

inline void get_strides(int64_t* strides, ArrayRef<OperandInfo> operands, int64_t ndim) {
  for (const auto dim : c10::irange(ndim)) {
    for (const auto& operand : operands) {
      *strides++ = operand.stride_bytes[dim];
    }
  }
  // Always at least 2d strides to support 2d for_each loops
  if (ndim < 2) {
    auto ntensors = operands.size();
    std::fill_n(strides, (2 - ndim) * ntensors, 0);
  }
}

OptionalTensorRef make_otr(const TensorBase &tensor) {
  if (tensor.defined()) {
    return OptionalTensorRef(tensor);
  } else {
    return OptionalTensorRef();
  }
}

}

namespace internal {

OpaqueOptionalTensorRef::OpaqueOptionalTensorRef() {
  static_assert(alignof(OptionalTensorRef) == alignof(TensorBase));
  static_assert(sizeof(OptionalTensorRef) == sizeof(TensorBase));
  new (data_.data()) OptionalTensorRef();
}

OpaqueOptionalTensorRef::~OpaqueOptionalTensorRef() {
  get()->~OptionalTensorRef();
}

const Tensor& OpaqueOptionalTensorRef::getTensor() const {
  return get()->getTensorRef();
}

}

void OperandInfo::tensor(c10::MaybeOwned<TensorBase> &&tensor) {
  tensor_base_ = std::move(tensor);
  *tensor_storage_ = make_otr(*tensor_base_);
}

void OperandInfo::exchange_tensor(c10::MaybeOwned<TensorBase> &&new_tensor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!original_tensor_base_->defined());
  original_tensor_base_ = std::exchange(tensor_base_, std::move(new_tensor));
  *original_tensor_storage_ = std::exchange(*tensor_storage_, make_otr(*tensor_base_));
}

void OperandInfo::restore_original_tensor() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(original_tensor_base_->defined());
  tensor_base_ = std::move(original_tensor_base_);
  *tensor_storage_ = std::exchange(*original_tensor_storage_, OptionalTensorRef{});
}

/// Construction
TensorIteratorConfig& TensorIteratorConfig::add_owned_output(const TensorBase& output) {
  TORCH_INTERNAL_ASSERT(
      num_inputs_ == 0,
      "Keep in mind that you have to add all outputs first before adding any input. "
      "For more details, see https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator.");
  tensors_.push_back(c10::MaybeOwned<TensorBase>::owned(std::in_place, output));
  num_outputs_++;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_input(const TensorBase& input) {
  tensors_.push_back(c10::MaybeOwned<TensorBase>::owned(std::in_place, input));
  num_inputs_++;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_const_input(const TensorBase& input) {
  const_tensor_indices_.push_back(tensors_.size());
  tensors_.push_back(c10::MaybeOwned<TensorBase>::owned(std::in_place, input));
  num_inputs_++;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_borrowed_output(const TensorBase& output) {
  TORCH_INTERNAL_ASSERT(
      num_inputs_ == 0,
      "Keep in mind that you have to add all outputs first before adding any input. "
      "For more details, see https://github.com/pytorch/pytorch/wiki/How-to-use-TensorIterator.");
  tensors_.push_back(c10::MaybeOwned<TensorBase>::borrowed(output));
  num_outputs_++;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_borrowed_input(const TensorBase& input) {
  tensors_.push_back(c10::MaybeOwned<TensorBase>::borrowed(input));
  num_inputs_++;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_borrowed_const_input(const TensorBase& input) {
  const_tensor_indices_.push_back(tensors_.size());
  tensors_.push_back(c10::MaybeOwned<TensorBase>::borrowed(input));
  num_inputs_++;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_dtype_and_device(ScalarType dtype, Device device) {
  TORCH_CHECK(!check_all_same_dtype_, "check_all_same_dtype(false) must be called before declare_static_dtype(...)");
  static_dtype_ = dtype;
  static_device_ = device;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_dtype(ScalarType dtype) {
  TORCH_CHECK(!check_all_same_dtype_, "check_all_same_dtype(false) must be called before declare_static_dtype(...)");
  static_dtype_ = dtype;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_device(Device device) {
  static_device_ = device;
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_shape(IntArrayRef shape) {
  // WARNING:
  //   This will bypass all shape checking in the TensorIterator. Kernels which call this method
  //   are expected to check shapes before calling `add_owned_input` or `add_owned_output`.
  TORCH_CHECK(!resize_outputs_, "resize_outputs() must be called before declare_static_shape(...)")
  static_shape_ = DimVector(shape);
  return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_shape(IntArrayRef shape, IntArrayRef squash_dims) {
  declare_static_shape(shape);
  TORCH_INTERNAL_ASSERT(static_shape_.has_value());
  if (static_shape_->empty()) return *this;
  for (const auto& squash_dim : squash_dims) {
    TORCH_CHECK(squash_dim >= 0 && squash_dim < static_cast<int64_t>(static_shape_->size()),
                "squash_dim ", squash_dim, " must be in [0, ", static_shape_->size(), ").");
    (*static_shape_)[squash_dim] = 1;
  }
  return *this;
}

// NOTE: [Computing output strides]
// We use the following algorithm to compute output strides
// If correctly sized output is provided, we respect its strides and don't change them
// Otherwise, if provided output is of incorrect size or no output is provided,
// we try to recover permutation that was applied to the inputs
// by sorting the strides of the inputs. Precedence is given to the inputs in the order they were added,
// and to permutations involving non-broadcasted dimensions
// 1. we loop over inputs starting from the first
// 2. for all inputs strides of broadcasted dimensions are set to 0, and 0 compares equal to anything. If one
// of the dimensions being compared has a stride of 0, we move on to the next tensor to determine if
// these dimensions need to be swapped.
// 3. strides of dimensions equal to 1 participate in sorting
// 4. if 2 strides are equal and neither is 0, we try to break the tie by looking at the corresponding dimensions
// of the tensor. Dimensions were permuted if, when iterating from the end, dimensions corresponding to the
// same strides are increasing. If dimensions are non-increasing, we move on to the next input to break the tie.
//
// Instead of applying rule 4 for tie breaking, we could move on to the next tensor directly. This would result in possibly
// losing the correct permutation of the first tensor if there are permuted trivial dimensions, but could potentially
// improve traversal order of the second tensor. We chose the former option to better propagate channels last layout
// for example for a tensor with the sizes N1H1
// These rules result in the intuitive behavior that in most cases recovers permutation of either the first argument (if all
// arguments are of the same size) or the argument that is not broadcasted, regardless of its position.
// As a bonus, it also result in reasonably well-behaved traversal order of the inputs and outputs - in the kernels
// output is traversed linearly, and since it closely follows input layouts, inputs are traversed linearly as well
//
// Examples:
// full size tensor + broadcasted tensor with 0 or 1 non-trivial dimensions => strides of output are same
// as strides of full size input regardless of the order
// 2 tensors of same size but different strides => output strides are the same as first argument
//
// We also have fast path for memory-dense inputs with the same strides (or, trivially, single memory-dense input)
// that outputs a tensor with the same strides as inputs. The only difference in result with the algorithm described
// above is for strides for trivial (1) dimensions, where in ambiguous cases for performance reasons we default to
// contiguous strides.
// Example: tensor with sizes NC11 and strides C1CC will produce output with strides C111 (note differences are only
// in the strides of trivial dimensions, so physical layout is unaffected but permutation information is lost)
// We might change this behavior in future once performance considerations are resolved

void TensorIteratorBase::reorder_dimensions() {
  ti_build::reorder_dimensions(shape_, perm_, operands_, is_reduction_, enforce_linear_iteration_);
}

// Computes a common dtype using type promotion
// See the [Common Dtype Computation] note
ScalarType TensorIteratorBase::compute_common_dtype() {
  at::native::ResultTypeState state = {};
  for (const auto& op : operands_) {
    if (op.is_output) {
      continue;
    }

    state = at::native::update_result_type_state(op.tensor(), state);
  }

  common_dtype_ = at::native::result_type(state);
  TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined);

  return common_dtype_;
}

static TensorOptions original_options(const OperandInfo& op) {
  if (op.original_tensor_base().defined()) {
    return op.original_tensor_base().options();
  } else {
    return op.options();
  }
}

static auto output_setter(TensorIteratorBase& iter) {
  return [&iter](int i, IntArrayRef sizes, IntArrayRef strides, std::optional<MemoryFormat> memory_format) C10_ALWAYS_INLINE_ATTRIBUTE {
    iter.set_output_raw_strided(i, sizes, strides, original_options(iter.operand(i)).memory_format(memory_format));
  };
}

// Implements the behavior of the following flags:
//   - check_all_same_dtype_
//   - check_all_same_device_
//   - enforce_safe_casting_to_output_
//   - promote_inputs_to_common_dtype_
//   - cast_common_dtype_to_outputs_
//
// See their descriptions in TensorIterator.h for details.
// NOTE: Checks for more specific behaviors (e.g. the first and second
//   inputs must share a dtype, but the third must have the long dtype)
//   should be implemented directly and outside of TensorIterator.
void TensorIteratorBase::compute_types(const TensorIteratorConfig& config) {
  // Reviews operands (1/2)
  //   - validates that all input tensors are defined
  //   - computes common device
  //   - determines if there are undefined outputs
  //   - determines if there are different dtypes and attempts
  //       to quickly acquire a common dtype
  Device common_device = kCPU;
  common_dtype_ = ScalarType::Undefined;
  // NB: despite output_dtype's generic sounding name, it only is
  // used in a nontrivial way if check_all_same_dtype is true
  ScalarType output_dtype = ScalarType::Undefined;
  bool has_different_input_dtypes = false;
  bool has_different_output_dtypes = false;
  bool has_undefined_outputs = false;

  for (auto& op : operands_) {
    // Validates that all inputs have type information, and that
    //   if an output is missing type information that we can infer
    //   the device it should be allocated on.
    if (!op.is_type_defined()) {
      TORCH_INTERNAL_ASSERT(op.is_output, "Found type undefined input tensor!");

      if (config.static_dtype_.has_value()) {
        op.target_dtype = config.static_dtype_.value();
      } else {
        has_undefined_outputs = true;
      }

      if (config.static_device_.has_value()) {
        op.device = config.static_device_.value();
      } else {
        TORCH_INTERNAL_ASSERT(config.check_all_same_device_);
      }

      if (has_undefined_outputs || !op.device.has_value()) {
        continue;
      }
    }

    // Validates input tensors are defined
    if (!op.tensor_base().defined()) {
      TORCH_INTERNAL_ASSERT(op.is_output, "Found undefined input tensor!");
      continue;
    }

    TORCH_INTERNAL_ASSERT(op.target_dtype == op.current_dtype)

    // Acquires the first non-CPU device (if any) as the common device
    if (common_device == kCPU && !op.tensor_base().is_cpu()) {
      common_device = op.tensor_base().device();
    }

    if (!op.is_output) {
      // Determines if there are varying input dtypes
      // NOTE: the common dtype is set to the first defined input dtype observed
      if (op.target_dtype != common_dtype_) {
        if (common_dtype_ == ScalarType::Undefined) {
          common_dtype_ = op.target_dtype;
        } else {
          has_different_input_dtypes = true;
        }
      }
    } else {  // op.is_output
      // Determines if there are varying output dtypes
      // NOTE: the output dtype is set to the first defined output dtype observed
      if (op.target_dtype != output_dtype) {
        if (output_dtype == ScalarType::Undefined) {
          output_dtype = op.target_dtype;
        } else {
          has_different_output_dtypes = true;
        }
      }
    }
  }

  // Checks that either the computation type is computable or unneeded
  TORCH_INTERNAL_ASSERT(!(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ &&
                        (has_undefined_outputs || config.enforce_safe_casting_to_output_ ||
                        config.cast_common_dtype_to_outputs_)));

  // Checks that all inputs and defined outputs are the same dtype, if requested
  if (config.check_all_same_dtype_ &&
      (has_different_input_dtypes || has_different_output_dtypes ||
      (common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))) {
    // Throws an informative error message
    for (auto& op : operands_) {
      if (!op.tensor_base().defined()) {
        continue;
      }

      TORCH_CHECK(op.target_dtype == common_dtype_,
                  "Found dtype ", op.target_dtype, " but expected ", common_dtype_);
    }
  }

  // Short-circuits if no additional work required
  if (!has_undefined_outputs && !config.check_all_same_device_ &&
      !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
      !config.enforce_safe_casting_to_output_) {
    // Invalidates common_dtype_ if it could not be inferred
    common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
    return;
  }

  // Computes a common dtype, if needed
  if ((has_different_input_dtypes || all_ops_are_scalars_) && config.promote_inputs_to_common_dtype_) {
    common_dtype_ = compute_common_dtype();
  }

  // Promotes common dtype to the default float scalar type, if needed
  if (config.promote_integer_inputs_to_float_ &&
      c10::isIntegralType(common_dtype_, /*includeBool=*/true)) {
    common_dtype_ = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  // Reviews operands (2/2)
  //   - sets metadata for undefined outputs
  //   - checks that all tensors are on the same device, if requested
  //   - checks that the common dtype can safely cast to each output, if requested
  //   - creates temporaries for CPU operations, if needed and requested
  common_device_ = common_device;
  int max_cpu_scalars_on_non_cpu = config.allow_cpu_scalars_ ? 1 : 0;
  int current_cpu_scalars_on_non_cpu = 0;
  for (auto& op : operands_) {
    bool is_type_defined = op.is_type_defined();
    bool is_device_defined = op.is_device_defined();

    if (!is_type_defined) {
      op.target_dtype = common_dtype_;
    }
    if (!is_device_defined) {
      op.device = common_device;
    }

    if (!is_type_defined && !is_device_defined) {
      continue;
    }

    // Skips undefined tensors
    if (!op.tensor_base().defined()) {
      continue;
    }

    // Checks all tensors are on the same device, if requested
    if (config.check_all_same_device_) {
      // Handles CPU scalars on CUDA kernels that support them
      if (!common_device.is_cpu() &&
          config.allow_cpu_scalars_ && !op.is_output && op.tensor_base().dim() == 0 &&
          op.tensor_base().is_cpu()) {
        TORCH_CHECK(current_cpu_scalars_on_non_cpu < max_cpu_scalars_on_non_cpu,
                    "Trying to pass too many CPU scalars to non-CPU kernel!");
        ++current_cpu_scalars_on_non_cpu;
      } else if (op.device != common_device) {
        TORCH_CHECK(false,
                    "Expected all tensors to be on the same device, but "
                    "found at least two devices, ", common_device, " and ", op.device, "!");
      }
    }

    // Checks safe casting, if requested
    if (config.enforce_safe_casting_to_output_ && op.is_output && op.current_dtype != common_dtype_) {
      TORCH_CHECK(canCast(common_dtype_, op.current_dtype),
                  "result type ", common_dtype_, " can't be cast to the "
                  "desired output type ", op.current_dtype);
    }

    // Creates temporaries for CPU operations, if needed and requested
    // TODO: reuse temporaries when possible (e.g. for inplace operations)
    if (common_device == kCPU) {
      // Casts to outputs by creating temporaries of the correct dtype (if needed)
      // NB: we skip this on is_meta_, because the temporary allocation here is
      // unnecessary if we aren't going to actually do the compute
      if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_ && !is_meta_) {
        TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
        // Marker [Output original_tensor is set]
        // NB: do NOT use set_output here, as the temporary is NOT a true output;
        // op.tensor is the true output and it was pre-provided for us.
        // TODO: The logic for cast_outputs will need to be handled by the
        // structured kernels implementation.  What probably should happen
        // is that we pass in the inferred dtype into the out kernel, and
        // then after calling the out kernel, do the conversion (which
        // is cast_outputs here), but integrating this with existing
        // TensorIterator will take a little doing
        op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
            at::empty_like(op.tensor(),
                           op.tensor_base().options().dtype(common_dtype_),
                           LEGACY_CONTIGUOUS_MEMORY_FORMAT)));
        op.current_dtype = common_dtype_;
        op.target_dtype = common_dtype_;
      }

      // Promotes inputs by creating temporaries of the correct dtype
      if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
        op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(op.tensor().to(common_dtype_)));
        op.current_dtype = common_dtype_;
        op.target_dtype = common_dtype_;
      }
    }
  }
}

StrideVector TensorIteratorBase::compatible_stride(int64_t element_size) const {
  return ti_build::compatible_stride(shape_, element_size);
}

DimVector TensorIteratorBase::invert_perm(IntArrayRef input) const {
  return ti_build::invert_perm(perm_, input, has_coalesced_dimensions_);
}

void TensorIteratorBase::allocate_or_resize_outputs() {
  ti_build::allocate_or_resize_outputs(shape_, perm_, operands_, num_outputs_, has_coalesced_dimensions_, output_setter(*this));
}

void TensorIteratorBase::coalesce_dimensions() {
  ti_build::coalesce_dimensions(shape_, operands_, has_coalesced_dimensions_);
}

int64_t TensorIteratorBase::numel() const {
  int64_t numel = 1;
  for (int64_t size : shape_) {
    numel *= size;
  }
  return numel;
}

StrideVector TensorIteratorBase::get_dim_strides(int dim) const {
  auto dims = ndim();
  auto inner_strides = StrideVector();
  for (auto& op : operands_) {
    inner_strides.push_back(dims == 0 ? 0 : op.stride_bytes[dim]);
  }
  return inner_strides;
}

SmallVector<char*, 4> TensorIteratorBase::get_base_ptrs() const {
  auto ptrs = SmallVector<char*, 4>(ntensors());
  at::get_base_ptrs(ptrs.data(), operands_);
  return ptrs;
}

bool TensorIteratorBase::is_dim_reduced(int dim) const {
  for (auto& op : operands_) {
    if (op.is_output && op.stride_bytes[dim] == 0 && shape_[dim] > 1) {
      return true;
    }
  }
  return false;
}

void TensorIteratorBase::permute_dimensions(IntArrayRef perm) {
  ti_build::permute_dimensions(shape_, operands_, perm);
}

int64_t TensorIteratorBase::num_output_elements() const {
  int64_t elem = 1;
  for (const auto dim : c10::irange(ndim())) {
    if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0)  {
      elem *= shape_[dim];
    }
  }
  return elem;
}

int TensorIteratorBase::num_reduce_dims() const {
  int count = 0;
  for (const auto dim : c10::irange(ndim())) {
    if (operands_[0].stride_bytes[dim] == 0) {
      count++;
    }
  }
  return count;
}

void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();
  if (numel == 0) {
    return;
  } else if (numel < grain_size || at::get_num_threads() == 1) {
    serial_for_each(loop, {0, numel});
    return;
  } else {
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}

StrideVector TensorIteratorBase::get_strides() const {
  const auto dim = ndim();
  StrideVector strides(static_cast<size_t>(std::max(dim, 2)) * ntensors());
  at::get_strides(strides.data(), operands_, dim);
  return strides;
}

void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {
    return;
  }

  const auto ntensors = this->ntensors();
  const auto ndim = this->ndim();

  c10::SmallBuffer<char*, 4> ptrs(ntensors);
  c10::SmallBuffer<int64_t, 8> strides(ntensors * static_cast<size_t>(std::max(ndim, 2)));

  at::get_base_ptrs(ptrs.data(), operands_);
  at::get_strides(strides.data(), operands_, ndim);
  at::internal::serial_for_each(
      shape_, strides, ptrs.data(), ptrs.size(), loop, range);
}

bool TensorIteratorBase::is_trivial_1d() const {
  // TODO: check for casting once it's supported
  return ndim() == 1;
}

bool TensorIteratorBase::is_contiguous() const {
  if (numel() == 1) {
    return true;
  }
  if (ndim() != 1) {
    return false;
  }
  return has_contiguous_first_dim();
}


bool TensorIteratorBase::is_scalar(int64_t arg) const {
  const auto& stride = operands_[arg].stride_bytes;
  for (const auto i : c10::irange(ndim())) {
    if (stride[i] != 0 && shape_[i] != 1) {
      return false;
    }
  }
  return true;
}

bool TensorIteratorBase::is_cpu_scalar(int64_t arg) const {
  return is_scalar(arg) && device(arg).is_cpu();
}

void TensorIteratorBase::cast_outputs() {
  for (auto& op : operands_) {
    if (op.is_output && op.original_tensor_base().defined() &&
        op.original_tensor_base().scalar_type() != op.current_dtype) {
      // TODO: Now that set_output resizes both the original_tensor
      // and tensor, this condition should no longer ever be true
      const auto &original_tensor = op.original_tensor();
      const auto &tensor = op.tensor();
      if (original_tensor.sizes() != tensor.sizes()) {
        original_tensor.resize_as_(tensor).as_strided_(tensor.sizes(), tensor.strides());
      }
      original_tensor.copy_(tensor);
      op.restore_original_tensor();
    }
  }
}

void* TensorIteratorBase::data_ptr(int64_t arg) const {
  return operands_[arg].data;
}

void TensorIteratorBase::remove_operand(int64_t arg) {
  operands_.erase(operands_.begin() + arg);
}

void TensorIteratorBase::unsafe_replace_operand(int64_t arg, void* data) {
  operands_[arg].data = data;
}

void TensorIteratorBase::narrow(int dim, int64_t start, int64_t size) {
  TORCH_INTERNAL_ASSERT(dim < ndim() && size >= 1);
  shape_[dim] = size;
  view_offsets_[dim] += start;
  for (auto& op : operands_) {
    op.data = (static_cast<char*>(op.data)) + op.stride_bytes[dim] * start;
  }
  if (size == 1 && !is_reduction_) {
    coalesce_dimensions();
  }
}

void TensorIteratorBase::select_all_keeping_dim(int start_dim, IntArrayRef indices) {
  TORCH_INTERNAL_ASSERT(start_dim <= ndim());
  for (const auto i : c10::irange(start_dim, ndim())) {
    for (auto& op : operands_) {
      op.data = (static_cast<char*>(op.data)) + op.stride_bytes[i] * indices[i - start_dim];
    }
    shape_[i] = 1;
  }
}

#define BINARY_FLOAT_OP_CONFIG()                \
  TensorIteratorConfig()                        \
    .set_check_mem_overlap(true)                \
    .allow_cpu_scalars(true)                    \
    .promote_inputs_to_common_dtype(true)       \
    .cast_common_dtype_to_outputs(true)         \
    .enforce_safe_casting_to_output(true)       \
    .promote_integer_inputs_to_float(true)

// Helper to construct a binary op that promotes integer inputs to float.
void TensorIteratorBase::build_binary_float_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_FLOAT_OP_CONFIG()
        .add_owned_output(out)
        .add_owned_const_input(a)
        .add_owned_const_input(b));
}

void TensorIteratorBase::build_borrowing_binary_float_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_FLOAT_OP_CONFIG()
        .add_output(out)
        .add_const_input(a)
        .add_const_input(b));
}

static void set_up_comparison_op_config(TensorIteratorConfig& config, const TensorBase& out) {
  config.set_check_mem_overlap(true);
  config.allow_cpu_scalars(true);
  config.promote_inputs_to_common_dtype(true);

  // When 'out' isn't defined (e.g. for the functional operator 'a == b'), we
  // want the output to be bool. Otherwise (e.g. 'torch.eq(a, b, out=c)') we
  // don't coerce the output.
  if (!out.defined()) {
    config.declare_static_dtype(kBool);
  }

  // Note [special-case bool outputs]
  // We explicitly don't call `cast_common_dtype_to_outputs` when the output tensor
  // has `bool` dtype. This is a performance optimization: the functional
  // version of all comparison/logical ops uses a bool output tensor, and we'd like to
  // avoid creating a temporary copy of the output.
  // However, note that all kernels using this TensorIterator will need to special-case when
  // the output tensor has bool dtype, and provide a lambda of type (scalar_t, scalar_t -> bool).
  if (out.defined() && out.scalar_type() != kBool) {
    config.cast_common_dtype_to_outputs(true);
  }
}

void TensorIteratorBase::build_comparison_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  TensorIteratorConfig config;
  set_up_comparison_op_config(config, out);

  config.add_owned_output(out);
  config.add_owned_const_input(a);
  config.add_owned_const_input(b);
  build(config);
}

void TensorIteratorBase::build_borrowing_comparison_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  TensorIteratorConfig config;
  set_up_comparison_op_config(config, out);

  config.add_borrowed_output(out);
  config.add_borrowed_const_input(a);
  config.add_borrowed_const_input(b);
  build(config);
}

void TensorIteratorBase::build_borrowing_except_last_argument_comparison_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  TensorIteratorConfig config;
  set_up_comparison_op_config(config, out);

  config.add_borrowed_output(out);
  config.add_borrowed_const_input(a);
  config.add_owned_const_input(b);
  build(config);
}

void TensorIteratorBase::build_ternary_op(
    const TensorBase& out, const TensorBase& a,
    const TensorBase& b, const TensorBase& c) {
  build(TensorIteratorConfig()
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true)
      .add_owned_output(out)
      .add_owned_const_input(a)
      .add_owned_const_input(b)
      .add_owned_const_input(c));
}

// This cannot be a function because TensorIteratorConfig is not
// copyable or movable, so it can't be returned from the function.
#define BINARY_OP_CONFIG()                              \
  TensorIteratorConfig()                                \
    .set_check_mem_overlap(true)                        \
    .allow_cpu_scalars(true)                            \
    .promote_inputs_to_common_dtype(true)               \
    .cast_common_dtype_to_outputs(true)                 \
    .enforce_safe_casting_to_output(true)               \

void TensorIteratorBase::build_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_OP_CONFIG()
      .add_owned_output(out)
      .add_owned_const_input(a)
      .add_owned_const_input(b));
}

void TensorIteratorBase::build_borrowing_binary_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_OP_CONFIG()
      .add_output(out)
      .add_const_input(a)
      .add_const_input(b));
}

// This cannot be a function because TensorIteratorConfig is not
// copyable or movable, so it can't be returned from the function.
#define UNARY_FLOAT_OP_CONFIG()                                         \
  TensorIteratorConfig()                                                \
  .set_check_mem_overlap(true)                                          \
  .promote_inputs_to_common_dtype(true)                                 \
  .cast_common_dtype_to_outputs(true)                                   \
  .enforce_safe_casting_to_output(true)                                 \
  .promote_integer_inputs_to_float(true)

void TensorIteratorBase::build_unary_float_op(const TensorBase& out, const TensorBase& a) {
  build(UNARY_FLOAT_OP_CONFIG()
      .add_owned_output(out)
      .add_owned_const_input(a));
}

void TensorIteratorBase::build_borrowing_unary_float_op(const TensorBase& out, const TensorBase& a) {
  build(UNARY_FLOAT_OP_CONFIG()
      .add_output(out)
      .add_const_input(a));
}

// This cannot be a function because TensorIteratorConfig is not
// copyable or movable, so it can't be returned from the function.
#define UNARY_OP_CONFIG()                                \
  TensorIteratorConfig()                                 \
    .set_check_mem_overlap(true)                         \
    .cast_common_dtype_to_outputs(false)                 \
    .enforce_safe_casting_to_output(false)               \
    .check_all_same_dtype(true)

void TensorIteratorBase::build_unary_op(const TensorBase& out, const TensorBase& a) {
  build(UNARY_OP_CONFIG()
      .add_owned_output(out)
      .add_owned_const_input(a));
}

void TensorIteratorBase::build_borrowing_unary_op(const TensorBase& out, const TensorBase& a) {
  build(UNARY_OP_CONFIG()
      .add_output(out)
      .add_const_input(a));
}

void TensorIteratorBase::build_output_borrowing_argument_owning_unary_op(
    const TensorBase& out,
    const TensorBase& a) {
  build(TensorIteratorConfig()
            .set_check_mem_overlap(true)
            .cast_common_dtype_to_outputs(true)
            .enforce_safe_casting_to_output(true)
            .add_output(out)
            .add_owned_const_input(a));
}

// Helper to construct a unary op that forcibly promotes output to boolean.
// Only be used when the output tensor must have boolean type.
void TensorIteratorBase::build_borrowing_unary_force_boolean_op(const TensorBase& out, const TensorBase& a) {
  build(TensorIteratorConfig()
      .set_check_mem_overlap(true)
      .check_all_same_dtype(false)
      .declare_static_dtype(at::kBool)
      .declare_static_device(a.device())
      .add_output(out)
      .add_const_input(a));
}

TensorIterator TensorIterator::binary_op(TensorBase& out, const TensorBase& a, const TensorBase& b) {
  TensorIterator iter;
  iter.build_binary_op(out, a, b);
  return iter;
}

TensorIterator TensorIterator::borrowing_binary_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  TensorIterator iter;
  iter.build_borrowing_binary_op(out, a, b);
  return iter;
}

TensorIterator TensorIterator::binary_float_op(TensorBase& out, const TensorBase& a, const TensorBase& b) {
  TensorIterator iter;
  iter.build_binary_float_op(out, a, b);
  return iter;
}

TensorIterator TensorIterator::comparison_op(TensorBase& out, const TensorBase& a,
    const TensorBase& b) {
  TensorIterator iter;
  iter.build_comparison_op(out, a, b);
  return iter;
}

TensorIterator TensorIterator::unary_op(TensorBase& out, const TensorBase& a) {
  TensorIterator iter;
  iter.build_unary_op(out, a);
  return iter;
}

TensorIterator TensorIterator::unary_float_op(TensorBase& out, const TensorBase& a) {
  TensorIterator iter;
  iter.build_unary_float_op(out, a);
  return iter;
}

#define NULLARY_OP_CONFIG()                                     \
  TensorIteratorConfig()                                        \
    .set_check_mem_overlap(true)                                \
    .check_all_same_dtype(false)                                \
  /* FIXME: workaround for bug: https://github.com/pytorch/pytorch/issues/20342 */ \
    .resize_outputs(false)

TensorIterator TensorIterator::nullary_op(TensorBase& out) {
  return NULLARY_OP_CONFIG()
    .add_owned_output(out)
    .build();
}

TensorIterator TensorIterator::borrowing_nullary_op(const TensorBase& out) {
  return NULLARY_OP_CONFIG()
    .add_output(out)
    .build();
}

TensorIterator TensorIterator::reduce_op(TensorBase& out, const TensorBase& a) {
  TORCH_INTERNAL_ASSERT(out.defined());
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .add_owned_output(out)
    .add_owned_const_input(a)
    .resize_outputs(false)
    .is_reduction(true)
    // TODO: not supporting casting to outputs is only really necessary for arg{min,max}
    .promote_inputs_to_common_dtype(true)
    .build();
}

TensorIterator TensorIterator::reduce_op(TensorBase& out1, TensorBase& out2, const TensorBase& a) {
  TORCH_INTERNAL_ASSERT(out1.defined());
  TORCH_INTERNAL_ASSERT(out2.defined());
  TORCH_CHECK(a.device() == out1.device() && out1.device() == out2.device(),
      "reduce_op(): expected input and both outputs to be on same device, but input is on ", a.device(),
      ", output1 is on ", out1.device(), " and output2 is on", out2.device());
  TORCH_CHECK(out1.dim() == out2.dim(), "reduce_op(): expected both outputs to have same number of dims, but output1 has ", out1.dim(),
      " and output2 has ", out2.dim());
  TORCH_CHECK(out1.sizes() == out2.sizes(), "reduce_op(): expected both outputs to have same sizes, but output1 has ", out1.sizes(),
      " and output2 has ", out2.sizes());
  TORCH_CHECK(out1.strides() == out2.strides(), "reduce_op(): expected both outputs to have same strides, but output1 has ", out1.strides(),
      " and output2 has ", out2.strides());
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .add_owned_output(out1)
    .add_owned_output(out2)
    .add_owned_const_input(a)
    .resize_outputs(false)
    .is_reduction(true)
    .check_all_same_dtype(false)
    .build();
}

void TensorIteratorBase::populate_operands(TensorIteratorConfig& config) {
  for (auto& tensor : config.tensors_) {
    // If *any* of the arguments is a meta tensor, the overall
    // computation is a meta computation (don't do any work,
    // just compute output information).  This aligns with
    // our multiple dispatch semantics.
    if (tensor->is_meta()) {
      is_meta_ = true;
    }
    operands_.emplace_back(std::move(tensor));
  }
  // const_tensor_indices_ are indices into operands_.
  for (const auto idx : config.const_tensor_indices_) {
    operands_[idx].is_const = true;
  }
  num_outputs_ = config.num_outputs_;
}

void TensorIteratorBase::mark_outputs() {
  // TODO: merge this into populate_operands
  for (const auto i : c10::irange(num_outputs_)) {
    operands_[i].is_output = true;
    const auto& output = tensor(i);
    if (!output.defined()) continue;

    // check if output is also an input
    for (const auto arg : c10::irange(num_outputs_, ntensors())) {
      const auto& input = tensor(arg);
      if (output.is_same(input)) {
        operands_[i].is_read_write = true;
      }
    }
  }
}

void TensorIteratorBase::mark_resize_outputs(const TensorIteratorConfig& config) {
  // Outputs cannot be broadcasted. Check that the shape of the outputs matches
  // the inferred shape. There's an exception for write-only tensors to support
  // our legacy behavior that functions with `out=` arguments resize their
  // outputs.
  if (config.static_shape_.has_value()) {
    return;
  }
  for (const auto i : c10::irange(num_outputs_)) {
    const auto& output = tensor(i);
    if (!output.defined()) {
      operands_[i].will_resize = true;
    }
    if (output.defined() && !output.sizes().equals(shape_)) {
      if (config.resize_outputs_ && !operands_[i].is_read_write) {
        operands_[i].will_resize = true;
        continue;
      }
      // for reduction, output size does not match shape_, as output is reduced size, and shape_ is size of the input
      TORCH_CHECK(is_reduction_,  "output with shape ", output.sizes(), " doesn't match the broadcast shape ",
                 shape_);
    }
  }
}

void TensorIteratorBase::compute_mem_overlaps(const TensorIteratorConfig& config) {
  if (!config.check_mem_overlap_) {
    return;
  }
  for (const auto i : c10::irange(num_outputs_)) {
    const auto& output = tensor_base(i);
    if (!output.defined()) continue;
    assert_no_internal_overlap(output);
    for (const auto j : c10::irange(num_outputs_, ntensors())) {
      const auto& input = tensor_base(j);
      if (!input.is_same(output)) {
        assert_no_partial_overlap(output, input);
      }
    }
  }
}

void TensorIteratorBase::compute_shape(const TensorIteratorConfig& config) {
  if (config.static_shape_.has_value()) {
    shape_ = *config.static_shape_;
    return;
  }
  ti_build::compute_shape(shape_, operands_, config.resize_outputs_, all_ops_same_shape_, all_ops_are_scalars_);
}

void TensorIteratorBase::compute_strides(const TensorIteratorConfig& config) {
  ti_build::compute_strides(shape_, operands_, config.static_shape_.has_value());
}

bool TensorIteratorBase::can_use_32bit_indexing() const {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  if (numel() > max_value) {
    return false;
  }
  for (auto& op : operands_) {
    int64_t max_offset = 1;
    for (const auto dim : c10::irange(ndim())) {
      max_offset += (shape_[dim] - 1) * op.stride_bytes[dim];
    }
    if (max_offset > max_value) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<TensorIterator> TensorIteratorBase::split(int dim) {
  TORCH_INTERNAL_ASSERT(dim >= 0 && dim < ndim() && shape()[dim] >= 2);
  auto copy = std::make_unique<TensorIterator>(*this);

  bool overlaps = is_dim_reduced(dim);
  auto copy_size = shape_[dim] / 2;
  auto this_size = shape_[dim] - copy_size;
  copy->narrow(dim, 0, copy_size);
  copy->final_output_ &= !overlaps;
  this->narrow(dim, copy_size, this_size);
  this->accumulate_ |= overlaps;

  return copy;
}


int TensorIteratorBase::get_dim_to_split() const {
  TORCH_INTERNAL_ASSERT(ndim() >= 1);
  int64_t max_extent = -1;
  int dim_to_split = -1;
  for (int dim = ndim() - 1; dim >= 0; dim--) {
    const int64_t size = shape_[dim];
    if (size == 0) {
      continue;
    }
    for (auto& op : operands_) {
      // std::abs is necessary to handle some special cases where we support negative strides
      // see the CUDA backend of at::flip
      const int64_t extent = (size - 1) * std::abs(op.stride_bytes[dim]);
      if (extent > max_extent) {
        max_extent = extent;
        dim_to_split = dim;
      }
    }
  }
  TORCH_INTERNAL_ASSERT(max_extent >= 0);
  return dim_to_split;
}

bool TensorIteratorBase::fast_set_up(const TensorIteratorConfig& config) {
  return ti_build::fast_set_up(compute_fast_setup_type(config), shape_, operands_, num_outputs_, has_coalesced_dimensions_, output_setter(*this));
}

FastSetupType TensorIteratorBase::compute_fast_setup_type(const TensorIteratorConfig& config) {
  return ti_build::compute_fast_setup_type<int64_t>(operands_, is_reduction_, all_ops_same_shape_, enforce_linear_iteration_);
}

void TensorIteratorBase::build(TensorIteratorConfig& config) {
  if (C10_UNLIKELY(sym_meta_enabled()) && build_sym(config)) {
    return;
  }
  // populate some persistent configuration fields
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // fill in operands_ based on configuration
  populate_operands(config);
  // set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Check that the outputs have no internal overlap
  // and do not share memory with inputs.
  compute_mem_overlaps(config);
  // compute the broadcasted shape
  compute_shape(config);
  // mark outputs for resizing if necessary
  mark_resize_outputs(config);
  // compute the result dtype and device
  compute_types(config);
  // try fast setup output tensor, if failed, fallback to normal setup
  if (!fast_set_up(config)) {
    // compute each tensor's stride after broadcasting
    compute_strides(config);
    // re-order dimensions to improve coalescing
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_or_resize_outputs();
    // coalesce adjacent dimensions when possible
    if (!is_meta_) coalesce_dimensions();
  }

  if (is_meta_) return;

  bool privateuse1_without_storage = false;
  if (common_device_.type() == DeviceType::PrivateUse1) {
    bool has_storage = true;
    for (auto& op : operands_) {
      has_storage &= op.tensor_base().has_storage();
    }
    privateuse1_without_storage = !has_storage;
  }

  // XLA and lazy tensors don't have storage, so they don't have an underlying data pointer.
  // Nothing beyond this point is important for meta functions, so it's fine to exit early here.
  // Extend the condition to MAIA tensors as MAIA tensors also don't have storage.
  if (privateuse1_without_storage  ||
      common_device_.type() == DeviceType::XLA  ||
      common_device_.type() == DeviceType::IPU  ||
      common_device_.type() == DeviceType::Lazy ||
      common_device_.type() == DeviceType::MAIA  ||
      common_device_.type() == DeviceType::HPU) return;

  for (auto& op : operands_) {
    TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
    if (op.is_const) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      op.data = const_cast<void*>(op.tensor_base().const_data_ptr());
    } else {
      op.data = op.tensor_base().mutable_data_ptr();
    }
  }

  // zero out offsets
  // If the tensor is a scalar, we leave room for it
  // So index translations in reduction can access
  // a valid value for the offset
  int64_t ndim_offsets = (ndim() ? ndim() : 1);
  view_offsets_.assign(ndim_offsets, 0);
}

// This is the structured kernels' implementation of set_output.  It is
// NEVER actually called directly; instead, a subclass of TensorIteratorBase
// will override set_output to actually do the operation, and then call
// set_output on the TensorIteratorBase to setup TI's metadata.
// The precondition for this function is that maybe_get_output() now
// unconditionally returns a real Tensor (prior to output setting,
// this function may return an undefined tensor.)
void TensorIteratorBase::set_output_raw_strided(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) {
  auto& op = operands_[output_idx];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
  const auto& t = maybe_get_output(output_idx);
  TORCH_INTERNAL_ASSERT(t.defined());
  if (!op.tensor_base().defined()) {
    op.tensor(c10::MaybeOwned<TensorBase>::borrowed(t));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(op.target_dtype == t.scalar_type());
  } else if (op.will_resize) {
    if (op.original_tensor_base().defined()) {
      // OK, so this is pretty weird.  To understand how we can end up in
      // this situation, first look at Marker [Output original_tensor is set].
      // That is the sole site where original_tensor may be set on an
      // output operand.  Essentially, when we are given an explicit output
      // tensor whose dtype doesn't match the computed common dtype from
      // the input operands, we do a switcheroo: we replace the (incorrectly
      // typed) output tensor with a correctly typed, *temporary* tensor,
      // and remember the original tensor in original_tensor (which will
      // then get written back to when we cast_outputs).
      //
      // Now, what if the given output tensor also happened to be zero
      // size (meaning that we will_resize it)?  Well, at the call site
      // above, we don't necessarily(*) know what the correct shape should
      // be, so we give the temporary tensor the same shape as the original.
      // At the time of set_output is when we DO know what the correct size
      // is, and the subclass's implementation of set_output in structured class
      // responsible for resizing original_tensor.  But we still have this
      // incorrectly sized temporary output which the structured subclass
      // knows nothing about, so we are obligated to also resize it here.
      //
      // This is a slight memory pessimization, because previously
      // original_tensor only got resized at the end of the computation, rather
      // than at the beginning (as happens here).  However, the peak memory
      // usage is the same, since you need to materialize both original tensor
      // and temporary tensor to do the copy.
      //
      // (*) Actually, technically, we probably do know what the shape
      // should be, since we do shape computation before dtype computation.
      // So hypothetically we could figure out what the correct shape is
      // at that point in time and directly allocate the temporary at
      // the right size.
      //
      // But a better solution is to delay allocation of temporaries until
      // after TensorIterator builder, waiting until we actually want
      // to do the computation.  That would also remove the necessity
      // for the is_meta_ test.
      TORCH_INTERNAL_ASSERT(op.original_tensor_base().is_same(t));
      TORCH_INTERNAL_ASSERT(!op.tensor_base().is_same(t));
      OptionalTensorRef tensor(op.tensor());
      at::native::resize_output(*tensor, sizes);
      auto const memory_format_opt = options.memory_format_opt();
      if (!strides.empty()) {
        TORCH_INTERNAL_ASSERT(!memory_format_opt.has_value());
        tensor->as_strided_(sizes, strides);
      } else{
         if (memory_format_opt.has_value()) {
          tensor->unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
        }
      }
    }
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      op.tensor_base().is_same(t) || op.current_dtype == op.tensor_base().scalar_type());
// For simplicity, just always update the cached current_type.
  op.current_dtype = op.tensor_base().scalar_type();
}

// This is the "traditional" implementation of set_output.  On TensorIterator
// instances, it is invoked directly from various call sites in this file.  No
// funny business.
void TensorIterator::set_output_raw_strided(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) {
  // NB: intentionally no superclass call
  auto& op = operands_[output_idx];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(output_idx < num_outputs_);
  if (!op.tensor_base().defined()) {
      if (strides.empty()) {
        op.tensor(c10::MaybeOwned<TensorBase>::owned(at::empty(sizes, options)));
      } else {
        op.tensor(c10::MaybeOwned<TensorBase>::owned(at::empty_strided(sizes, strides, options)));
      }
      op.current_dtype = op.target_dtype;
  } else if (op.will_resize) {
      at::native::resize_output(op.tensor(), sizes);
      if (!strides.empty()) {
        TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
        op.tensor().as_strided_(sizes, strides);
      } else {
        auto const memory_format = options.memory_format_opt();
        if (memory_format.has_value()) {
          op.tensor_base().unsafeGetTensorImpl()->empty_tensor_restride(*memory_format);
        }
      }
  }
}

// Not actually used by anything (TensorIterator subclass calls
// its own implementation of set_output which knows exactly where
// all the outputs are), but we have to provide all pure virtual methods
// for MetaBase
const Tensor& TensorIterator::maybe_get_output(int64_t output_idx) {
  return output(output_idx);
}

SplitUntil32Bit TensorIteratorBase::with_32bit_indexing() const {
  return SplitUntil32Bit(*this);
}

/// SplitUntil32Bit. Recursively splits an iterator into sub-iterators that
/// can use 32-bit indexing.

SplitUntil32Bit::iterator::iterator(const TensorIteratorBase& iter) {
  vec.emplace_back(new TensorIterator(iter));
  vec.emplace_back(nullptr); // ++ first pops the last element
  ++(*this);
}

SplitUntil32Bit::iterator& SplitUntil32Bit::iterator::operator++() {
  vec.pop_back();
  while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
    auto& iter = *vec.back();
    auto split_dim = iter.get_dim_to_split();
    vec.emplace_back(iter.split(split_dim));
  }
  return *this;
}

TensorIterator& SplitUntil32Bit::iterator::operator*() const {
  return *vec.back();
}

SplitUntil32Bit::iterator SplitUntil32Bit::begin() const {
  return SplitUntil32Bit::iterator(iter);
}

SplitUntil32Bit::iterator SplitUntil32Bit::end() const {
  return SplitUntil32Bit::iterator();
}

DimCounter::DimCounter(IntArrayRef shape, Range range)
  : shape(shape)
  , range(range)
  , values(shape.size())
  , offset(range.begin) {
  std::fill(values.begin(), values.end(), 0);
  if (range.begin == 0) {
    return;
  }

  int64_t linear_offset = range.begin;
  auto ndim = values.size();
  for (const auto dim : c10::irange(ndim)) {
    int64_t size = shape[dim];
    if (size > 0) {
      values[dim] = linear_offset % size;
      linear_offset /= size;
    }
  }
  TORCH_INTERNAL_ASSERT(linear_offset == 0);
}

bool DimCounter::is_done() const {
  return offset >= range.end;
}

void DimCounter::increment(const std::array<int64_t, 2>& step) {
  offset += step[0] * step[1];
  auto ndim = values.size();
  int64_t overflow = step[0];
  size_t i = 0;
  if (step[1] != 1) {
    TORCH_INTERNAL_ASSERT(step[0] == shape[0] && values[0] == 0);
    i = 1;
    overflow = step[1];
  }
  for (; i < ndim && overflow > 0; i++) {
    auto size = shape[i];
    auto prev = values[i];
    auto value = prev + overflow;
    if (value >= size) {
      overflow = 1;
      value -= size;
      TORCH_INTERNAL_ASSERT(value < size);
    } else {
      overflow = 0;
    }
    values[i] = static_cast<int64_t>(value);
  }
  TORCH_INTERNAL_ASSERT(overflow == 0 || overflow == 1);
}

std::array<int64_t, 2> DimCounter::max_2d_step() const {
  int64_t step0 = std::min(shape[0] - values[0], range.end - offset);
  int64_t step1 = 1;
  if (!shape.empty() && step0 == shape[0]) {
    step1 = std::min(shape[1] - values[1], (range.end - offset) / shape[0]);
  }
  return {step0, step1};
}

}  // namespace at
