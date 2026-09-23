// A SymInt-typed sibling of at::TensorIteratorBase for the trace mode. An op
// that opts in builds this iterator instead of the real one when
// its inputs are traced tensors, and launches the same elementwise kernel
// instantiations the real op launches (LoopsSym.cuh). The real iterator, the
// CUDA loop headers and the ordinary path are untouched: this class exists
// only under a trace, where sizes and strides are values over the ShapeEnv.
//
// The member function bodies in TensorIteratorSym.cpp are TensorIterator.cpp's,
// with int64_t sizes and strides as c10::SymInt and the data pointer as the
// SymInt of sym_*_data_ptr(); every comparison the build makes is a guard. v1
// covers one-output elementwise ops on CUDA operands of one dtype, plus one
// CPU scalar operand where the real binary_op config allows it (the entry
// reads its value and removes it before the launch, as the real kernel hosts
// do): type promotion of a CUDA operand and the 64-bit indexing split decline.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/SymInt.h>
#include <c10/util/SmallVector.h>

#include <optional>
#include <vector>

namespace at::cuda::host_trace::ti {

using StrideVector = c10::SmallVector<c10::SymInt, 6>;

// ATen/MemoryOverlap.h's predicates over traced tensors. get_overlap_status
// answers TooHard for any tensor with symbolic sizes, which under a trace is
// every traced tensor, so eager's refusal was neither reproduced nor guarded.
// These are eager's rules on the trace's values: a comparison of sizes,
// strides or offsets is a guard; between two roots the byte-interval test is
// structural where root identity decides it (an allocation made inside the
// call shares no storage with anything else alive) and an address guard
// between two inputs (one storage may arrive twice at replay). Where eager
// raises, the traced call declines by name with eager's text.
TORCH_CUDA_CPP_API MemOverlapStatus overlap_status_sym(const TensorBase& a, const TensorBase& b);
TORCH_CUDA_CPP_API void assert_no_internal_overlap_sym(const TensorBase& t);
TORCH_CUDA_CPP_API void assert_no_partial_overlap_sym(const TensorBase& a, const TensorBase& b);
TORCH_CUDA_CPP_API void assert_no_overlap_sym(const TensorBase& a, const TensorBase& b);
enum class FastSetupType : uint8_t { NONE, CONTIGUOUS, CHANNELS_LAST, NON_OVERLAPPING_DENSE };

struct OperandInfo {
  OperandInfo() = default;
  explicit OperandInfo(c10::MaybeOwned<TensorBase>&& t) : tensor_base_(std::move(t)) {
    if (tensor_base_->defined()) {
      device = tensor_base_->device();
      target_dtype = tensor_base_->scalar_type();
      current_dtype = target_dtype;
    }
  }
  StrideVector stride_bytes;
  ScalarType target_dtype = ScalarType::Undefined;
  ScalarType current_dtype = ScalarType::Undefined;
  std::optional<Device> device;
  c10::SymInt data{0}; // the byte address, from sym_const_data_ptr() / sym_mutable_data_ptr()
  bool is_output = false;
  bool will_resize = false;
  bool is_read_write = false;
  bool is_type_defined() const {
    return target_dtype != ScalarType::Undefined;
  }
  TensorOptions options() const {
    return TensorOptions(target_dtype).device(device);
  }
  const TensorBase& tensor_base() const {
    return *tensor_base_;
  }
  const Tensor& tensor() const {
    return static_cast<const Tensor&>(*tensor_base_);
  }
  void tensor(c10::MaybeOwned<TensorBase>&& t) {
    tensor_base_ = std::move(t);
  }

 private:
  c10::MaybeOwned<TensorBase> tensor_base_ = c10::MaybeOwned<TensorBase>::owned(std::in_place);
};

struct TensorIteratorSymConfig {
  bool resize_outputs_ = true;
  bool check_mem_overlap_ = true;
  // TensorIteratorConfig::check_all_same_dtype: off for copy_, whose
  // operands may differ in dtype (the kernel casts); on elsewhere, where a
  // dtype difference is type promotion, which v1 declines
  bool check_all_same_dtype_ = true;
  // TensorIteratorConfig::allow_cpu_scalars: on for binary_op, so a 0-dim CPU
  // tensor (a wrapped Python number) may join a CUDA computation
  bool allow_cpu_scalars_ = false;
  // TensorIteratorConfig::promote_inputs_to_common_dtype, as the real
  // iterator applies it on CUDA: the computation dtype is result_type over
  // the inputs, each input keeps its own dtype and the kernel casts it on
  // load (gpu_kernel's dynamic-cast route); takes precedence over
  // check_all_same_dtype_ (reduce_op, and the binary configs)
  bool promote_inputs_to_common_dtype_ = false;
  // TensorIteratorConfig::cast_common_dtype_to_outputs with
  // enforce_safe_casting_to_output (BINARY_OP_CONFIG): a defined output of
  // another dtype than the common one passes eager's canCast check and then
  // declines, since an entry dispatching on the output's dtype would not
  // launch the kernel eager launches for the common one
  bool cast_common_dtype_to_outputs_ = false;
  bool is_reduction_ = false;
  bool enforce_linear_iteration_ = false;
  std::optional<c10::SymDimVector> static_shape_;
  // TensorIteratorConfig::declare_static_dtype: the dtype of an output the
  // iterator allocates (bool for a comparison), apart from the inputs' dtype
  std::optional<ScalarType> static_dtype_;
};

struct TORCH_CUDA_CPP_API TensorIteratorSym {
  // TensorIterator::binary_op / unary_op with an undefined `out` allocates the
  // output; copy passes the destination and keeps its strides.
  static TensorIteratorSym binary_op(const Tensor& out, const Tensor& a, const Tensor& b);
  static TensorIteratorSym unary_op(const Tensor& out, const Tensor& a);
  static TensorIteratorSym ternary_op(const Tensor& out, const Tensor& a, const Tensor& b, const Tensor& c);
  // the same builds under a caller's config (the generated siblings: an
  // operand of a fixed dtype beside the scalar_t ones)
  static TensorIteratorSym unary_op(const Tensor& out, const Tensor& a, TensorIteratorSymConfig config);
  static TensorIteratorSym binary_op(const Tensor& out, const Tensor& a, const Tensor& b, TensorIteratorSymConfig config);
  static TensorIteratorSym ternary_op(const Tensor& out, const Tensor& a, const Tensor& b, const Tensor& c, TensorIteratorSymConfig config);
  // TensorIterator::comparison_op: a CPU scalar allowed, the output bool
  // when the iterator allocates it (TensorIterator.cpp
  // set_up_comparison_op_config)
  static TensorIteratorSym comparison_op(const Tensor& out, const Tensor& a, const Tensor& b);

  // TensorIterator::reduce_op: `out` is the result viewed with a size-1,
  // stride-0 dim at every reduced position (ReduceOpsUtils.h
  // review_reduce_result); the reduced dims come first after reordering.
  static TensorIteratorSym reduce_op(const Tensor& out, const Tensor& a);

  c10::SymDimVector shape_;
  DimVector perm_;
  bool has_coalesced_dimensions_ = false;
  bool enforce_linear_iteration_ = false;
  c10::SmallVector<OperandInfo, 4> operands_;
  int num_outputs_ = 0;
  bool all_ops_same_shape_ = false;
  bool is_reduction_ = false;
  ScalarType common_dtype_ = ScalarType::Undefined;
  Device common_device_ = kCPU;

  int ndim() const {
    return static_cast<int>(shape_.size());
  }
  c10::SymIntArrayRef shape() const {
    return shape_;
  }
  int ntensors() const {
    return static_cast<int>(operands_.size());
  }
  int noutputs() const {
    return num_outputs_;
  }
  int ninputs() const {
    return ntensors() - noutputs();
  }
  const TensorBase& tensor_base(int64_t arg) const {
    return operands_[arg].tensor_base();
  }
  const Tensor& tensor(int64_t arg) const {
    return operands_[arg].tensor();
  }
  const Tensor& output(int64_t arg = 0) const {
    return operands_[arg].tensor();
  }
  Device device(int64_t arg) const {
    return operands_[arg].tensor_base().device();
  }
  c10::SymIntArrayRef strides(int64_t arg) const {
    return operands_[arg].stride_bytes;
  }
  c10::SymInt data_ptr(int64_t arg) const {
    return operands_[arg].data;
  }
  ScalarType dtype(int64_t arg = 0) const {
    return operands_[arg].current_dtype;
  }
  ScalarType common_dtype() const {
    return common_dtype_;
  }
  int64_t element_size(int64_t arg) const {
    return static_cast<int64_t>(elementSize(dtype(arg)));
  }
  std::vector<TensorIteratorSym> with_32bit_indexing() const {
    decline("host_trace: TensorIterator's 64-bit indexing split is not traced (declined)");
  }
  // TensorIteratorBase::is_cpu_scalar / scalar_value / remove_operand: the
  // CPU scalar's value is read from its (real) storage as a trace-time
  // constant; the device is tested first so no stride of a CUDA operand is
  // guarded by the question
  bool is_cpu_scalar(int64_t arg) const;
  template <typename T>
  T scalar_value(int64_t arg) const {
    const auto& t = operands_[arg].tensor_base();
    return c10::fetch_and_cast<T>(t.scalar_type(), t.const_data_ptr());
  }
  void remove_operand(int64_t arg) {
    operands_.erase(operands_.begin() + arg);
  }

  c10::SymInt numel() const;
  // the reduction accessors of TensorIteratorBase: reduced dims are those
  // where the output's stride is 0
  int num_reduce_dims() const;
  c10::SymInt num_output_elements() const;
  ScalarType input_dtype(int64_t arg = 0) const {
    return operands_[num_outputs_ + arg].current_dtype;
  }
  bool is_contiguous() const;
  bool has_contiguous_first_dim() const;
  bool can_use_32bit_indexing() const;
  StrideVector compatible_stride(int64_t element_size) const;
  c10::SymDimVector invert_perm(c10::SymIntArrayRef input) const;
  void compute_shape(const TensorIteratorSymConfig& config);
  void compute_strides(const TensorIteratorSymConfig& config);
  void reorder_dimensions();
  void permute_dimensions(IntArrayRef perm);
  void allocate_or_resize_outputs();
  void coalesce_dimensions();
  bool fast_set_up(const TensorIteratorSymConfig& config);
  FastSetupType compute_fast_setup_type(const TensorIteratorSymConfig& config);
  void set_output_raw_strided(int64_t output_idx, c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, TensorOptions options);
  void build(TensorIteratorSymConfig& config);

  void add_output(const Tensor& t) {
    operands_.emplace_back(t.defined() ? c10::MaybeOwned<TensorBase>::borrowed(t) : c10::MaybeOwned<TensorBase>::owned(std::in_place));
    num_outputs_++;
  }
  void add_input(const Tensor& t) {
    operands_.emplace_back(c10::MaybeOwned<TensorBase>::borrowed(t));
  }
  // TensorIteratorBase::mark_outputs: an output that is also an input (an
  // in-place op) is read-write and is never resized
  void mark_outputs() {
    for (int i = 0; i < num_outputs_; i++) {
      operands_[i].is_output = true;
      const auto& output = tensor_base(i);
      if (!output.defined()) {
        continue;
      }
      for (int arg = num_outputs_; arg < ntensors(); arg++) {
        if (output.is_same(tensor_base(arg))) {
          operands_[i].is_read_write = true;
        }
      }
    }
  }
  void mark_resize_outputs(const TensorIteratorSymConfig& config);
  void compute_mem_overlaps(const TensorIteratorSymConfig& config);
  // v1: every defined operand is a CUDA tensor of one dtype (plus at most one
  // CPU scalar under allow_cpu_scalars_, whose promotion must leave that dtype
  // unchanged), so type promotion and the cast path of gpu_kernel never
  // arise; anything else declines.
  void compute_types(const TensorIteratorSymConfig& config);
};

} // namespace at::cuda::host_trace::ti
