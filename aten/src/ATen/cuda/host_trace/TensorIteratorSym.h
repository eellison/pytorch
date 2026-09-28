// TensorIteratorBase's build on c10::SymInt for a traced host (Recorder.h):
// the shape, stride and output-layout computation is TensorIterator.cpp's own
// (ATen/detail/TensorIteratorBuild.h), so sizes, strides and data addresses are
// SymInts over the trace's symbols, every comparison the build makes is a
// guard, and the output is allocated through at::empty_symint /
// at::empty_strided_symint, which the trace records. Covered: a functional op
// on CUDA operands of one dtype, where's condition, copy_'s given output of any
// dtype, and a reduction into a result of its input's dtype; a CPU scalar, type
// promotion and 64-bit indexing decline.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/SmallVector.h>

#include <initializer_list>
#include <utility>

namespace at::cuda::host_trace {

using StrideVector = c10::SmallVector<c10::SymInt, 6>;

// the at::OperandInfo fields the build reads; the trace never resizes an output
struct OperandInfo {
  explicit OperandInfo(TensorBase t) : tensor(std::move(t)) {
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
  StrideVector stride_bytes;
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

  c10::SymDimVector shape_;
  DimVector perm_;
  bool has_coalesced_dimensions_ = false;
  // the output first, undefined until allocated unless given
  c10::SmallVector<OperandInfo, 4> operands_;
  bool all_ops_same_shape_ = false;
  bool check_all_same_dtype_ = true;
  bool is_reduction_ = false;
  ScalarType common_dtype_ = ScalarType::Undefined;
  Device common_device_ = kCPU;

  int ndim() const {
    return static_cast<int>(shape_.size());
  }
  int ntensors() const {
    return static_cast<int>(operands_.size());
  }
  int ninputs() const {
    return ntensors() - 1;
  }
  int noutputs() const {
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
};

// meta::make_reduction for a result of self's dtype: allocates result, then
// the iterator over its view
TORCH_CUDA_CPP_API TensorIteratorSym make_reduction(Recorder& rec, TensorBase& result, const TensorBase& self, IntArrayRef dims, bool keepdim);

// gpu_reduce_kernel's global reduction buffer and its semaphores, zeroed
TORCH_CUDA_CPP_API std::pair<TensorBase, TensorBase> reduce_buffers(const c10::SymInt& buffer_bytes, const c10::SymInt& semaphore_bytes, Device device);

} // namespace at::cuda::host_trace
