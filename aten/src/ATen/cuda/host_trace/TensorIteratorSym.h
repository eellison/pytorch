// TensorIteratorBase's build (ATen/detail/TensorIteratorBuild.h) on c10::SymInt
// for a traced host; configurations it does not reproduce decline().
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/SmallVector.h>

#include <initializer_list>
#include <utility>

namespace at::cuda::host_trace {

using SymStrideVector = c10::SmallVector<c10::SymInt, 6>;

// the at::OperandInfo fields the build reads; the trace never resizes an output
struct SymOperandInfo {
  explicit SymOperandInfo(TensorBase t) : tensor(std::move(t)) {
    if (tensor.defined()) {
      target_dtype = current_dtype = tensor.scalar_type();
    }
  }
  const TensorBase& tensor_base() const {
    return tensor;
  }
  bool is_type_defined() const {
    return target_dtype != ScalarType::Undefined;
  }

  TensorBase tensor;
  SymStrideVector stride_bytes;
  c10::SymInt data{0};
  ScalarType target_dtype = ScalarType::Undefined;
  ScalarType current_dtype = ScalarType::Undefined;
  bool is_output = false;
  static constexpr bool will_resize = false;
};

struct TORCH_CUDA_CPP_API TensorIteratorSym {
  static TensorIteratorSym unary_op(Recorder& rec, const TensorBase& a);
  static TensorIteratorSym binary_op(Recorder& rec, const TensorBase& a, const TensorBase& b);
  // bool cond, a and b of one dtype, the output's
  static TensorIteratorSym where_op(Recorder& rec, const TensorBase& cond, const TensorBase& a, const TensorBase& b);
  // copy_impl's: dst is the output, src broadcasts to it
  static TensorIteratorSym copy_op(Recorder& rec, const TensorBase& dst, const TensorBase& src);
  // out is make_reduction's view of the result
  static TensorIteratorSym reduce_op(Recorder& rec, const TensorBase& out, const TensorBase& a);

  int ndim() const {
    return static_cast<int>(shape_.size());
  }
  int ntensors() const {
    return static_cast<int>(operands_.size());
  }
  int ninputs() const {
    return ntensors() - 1;
  }
  static constexpr int noutputs() {
    return 1;
  }
  const TensorBase& output() const {
    return operands_[0].tensor;
  }
  c10::SymIntArrayRef shape() const {
    return shape_;
  }
  c10::SymIntArrayRef strides(int arg) const {
    return operands_[arg].stride_bytes;
  }
  const c10::SymInt& data_ptr(int arg) const {
    return operands_[arg].data;
  }
  ScalarType common_dtype() const {
    return common_dtype_;
  }
  Device device() const {
    return common_device_;
  }
  ScalarType dtype(int arg) const {
    return operands_[arg].tensor.scalar_type();
  }
  int64_t element_size(int arg) const {
    return static_cast<int64_t>(operands_[arg].tensor.element_size());
  }

  c10::SymInt numel() const;
  int num_reduce_dims() const;
  c10::SymInt num_output_elements() const;
  bool is_contiguous() const;
  bool can_use_32bit_indexing() const;

 private:
  explicit TensorIteratorSym(std::initializer_list<TensorBase> operands);
  void build(Recorder& rec);
  void compute_types();

  c10::SymDimVector shape_;
  DimVector perm_;
  bool has_coalesced_dimensions_ = false;
  // the output first, undefined until allocated unless given
  c10::SmallVector<SymOperandInfo, 4> operands_;
  bool all_ops_same_shape_ = false;
  bool check_all_same_dtype_ = true;
  bool is_reduction_ = false;
  ScalarType common_dtype_ = ScalarType::Undefined;
  Device common_device_ = kCPU;
};

// meta::make_reduction for a result of self's dtype (shape_from_dim_mask and
// review_reduce_result on SymInt): allocates result, then iterates its view
TORCH_CUDA_CPP_API TensorIteratorSym make_reduction(Recorder& rec, TensorBase& result, const TensorBase& self, IntArrayRef dims, bool keepdim);

// gpu_reduce_kernel's global reduction buffer and its semaphores, zeroed; here
// because the Reduce*.cu TUs are TORCH_ASSERT_NO_OPERATORS
TORCH_CUDA_CPP_API std::pair<TensorBase, TensorBase> reduce_buffers(const c10::SymInt& buffer_bytes, const c10::SymInt& semaphore_bytes, Device device);

} // namespace at::cuda::host_trace
