// A c10::SymInt sibling of TensorIteratorBase's elementwise build for a traced
// host (Recorder.h): sizes, strides and data addresses are SymInts over the
// trace's symbols, every comparison the build makes is a guard, and the output
// is allocated through at::empty_symint / at::empty_strided_symint, which the
// trace records. The member functions are TensorIterator.cpp's of the same
// name. Covered: a functional op on CUDA operands of one dtype; a CPU scalar,
// type promotion and 64-bit indexing decline.
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/SmallVector.h>

namespace at::cuda::host_trace {

using StrideVector = c10::SmallVector<c10::SymInt, 6>;

enum class FastSetupType : uint8_t { NONE, CONTIGUOUS, CHANNELS_LAST, NON_OVERLAPPING_DENSE };

struct OperandInfo {
  TensorBase tensor;
  StrideVector stride_bytes;
  c10::SymInt data{0};
};

struct TORCH_CUDA_CPP_API TensorIteratorSym {
  static TensorIteratorSym unary_op(Recorder& rec, const TensorBase& a);
  static TensorIteratorSym binary_op(Recorder& rec, const TensorBase& a, const TensorBase& b);

  c10::SymDimVector shape_;
  DimVector perm_;
  bool has_coalesced_dimensions_ = false;
  // the output first, undefined until allocated
  c10::SmallVector<OperandInfo, 4> operands_;
  bool all_ops_same_shape_ = false;
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
  int64_t element_size(int /*arg*/) const {
    return static_cast<int64_t>(elementSize(common_dtype_));
  }

  c10::SymInt numel() const;
  bool is_contiguous() const;
  bool has_contiguous_first_dim() const;
  bool can_use_32bit_indexing() const;

 private:
  TensorOptions options() const {
    return TensorOptions(common_dtype_).device(common_device_);
  }
  void build(Recorder& rec);
  void compute_types();
  void compute_shape();
  void compute_strides();
  bool fast_set_up();
  FastSetupType compute_fast_setup_type() const;
  void reorder_dimensions();
  void permute_dimensions(IntArrayRef perm);
  StrideVector compatible_stride(int64_t element_size) const;
  c10::SymDimVector invert_perm(c10::SymIntArrayRef input) const;
  void allocate_outputs();
  void coalesce_dimensions();
  void set_output(c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, TensorOptions options);
};

} // namespace at::cuda::host_trace
