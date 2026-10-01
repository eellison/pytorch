#pragma once
// TensorIterator on c10::SymInt sizes and strides, built by the templates
// TensorIterator itself instantiates for int64_t
// (ATen/detail/TensorIteratorBuild.h). dtype, device and overlap checks are
// TensorIteratorBase's own.
//
// The constructor stops where TensorIteratorBase::build stops for meta tensors:
// outputs are allocated, dimensions are not coalesced. Output metadata is
// final here. coalesce_dimensions() is available separately; it does not
// change output metadata.
//
// Supports configs whose outputs are undefined, without declare_static_shape,
// and not reductions.
//
// With a sink, outputs are created by sink->set_output_raw_strided_symint and
// borrowed from sink->maybe_get_output; this is how a Meta structured kernel
// reaches it (TensorIteratorBase::build_sym).
#include <ATen/TensorIterator.h>

namespace at {

// While set, TensorIteratorBase::build on operands with symbolic sizes builds a
// TensorIteratorSym and hands its outputs to the structured kernel through
// set_output_raw_strided_symint. Only the Meta functional kernels implement
// that; torch._C._ti_meta sets this around calling one.
TORCH_API bool sym_meta_enabled();

struct TORCH_API SymMetaGuard {
  SymMetaGuard();
  ~SymMetaGuard();
  SymMetaGuard(const SymMetaGuard&) = delete;
  SymMetaGuard& operator=(const SymMetaGuard&) = delete;

 private:
  bool prev_;
};

struct TORCH_API TensorIteratorSym final : private TensorIteratorBase {
  // OperandInfo with SymInt stride_bytes, in the shape the templates expect.
  struct Operand {
    explicit Operand(OperandInfo& info)
        : info(&info),
          is_output(info.is_output),
          will_resize(info.will_resize),
          target_dtype(info.target_dtype),
          current_dtype(info.current_dtype) {}
    const TensorBase& tensor_base() const {
      return info->tensor_base();
    }
    bool is_type_defined() const {
      return info->is_type_defined();
    }

    OperandInfo* info;
    SmallVector<c10::SymInt, 6> stride_bytes;
    const bool& is_output;
    bool& will_resize;
    ScalarType& target_dtype;
    ScalarType& current_dtype;
  };

  explicit TensorIteratorSym(
      TensorIteratorConfig& config,
      impl::MetaBase* sink = nullptr);
  TensorIteratorSym(const TensorIteratorSym&) = delete;
  TensorIteratorSym& operator=(const TensorIteratorSym&) = delete;

  void coalesce_dimensions();

  using TensorIteratorBase::common_dtype;
  using TensorIteratorBase::dtype;
  using TensorIteratorBase::noutputs;
  using TensorIteratorBase::ntensors;
  using TensorIteratorBase::output;

  int ndim() const {
    return static_cast<int>(sym_shape_.size());
  }
  c10::SymIntArrayRef shape() const {
    return sym_shape_;
  }
  c10::SymIntArrayRef strides(int arg) const {
    return sym_operands_[arg].stride_bytes;
  }

  const Tensor& maybe_get_output(int64_t output_idx) override;

 private:
  friend struct TensorIteratorBase;
  SymDimVector sym_shape_;
  SmallVector<Operand, 4> sym_operands_;
};

} // namespace at
