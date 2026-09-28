#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/TensorIteratorSym.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/detail/TensorIteratorBuild.h>
#include <ATen/native/ReduceOpsUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/as_strided.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif
#include <c10/util/irange.h>

namespace at::cuda::host_trace {

namespace ti_build = at::detail::ti_build;

TensorIteratorSym::TensorIteratorSym(std::initializer_list<TensorBase> operands) {
  for (const TensorBase& t : operands) {
    operands_.emplace_back(t);
  }
  operands_[0].is_output = true;
}

TensorIteratorSym TensorIteratorSym::unary_op(Recorder& rec, const TensorBase& a) {
  TensorIteratorSym iter({TensorBase(), a});
  iter.build(rec);
  return iter;
}

TensorIteratorSym TensorIteratorSym::binary_op(Recorder& rec, const TensorBase& a, const TensorBase& b) {
  TensorIteratorSym iter({TensorBase(), a, b});
  iter.build(rec);
  return iter;
}

TensorIteratorSym TensorIteratorSym::where_op(Recorder& rec, const TensorBase& cond, const TensorBase& a, const TensorBase& b) {
  if (cond.scalar_type() != kBool) {
    decline(c10::str("a ", cond.scalar_type(), " condition"));
  }
  if (a.scalar_type() != b.scalar_type()) {
    decline(c10::str("where of ", a.scalar_type(), " and ", b.scalar_type()));
  }
  TensorIteratorSym iter({TensorBase(), cond, a, b});
  iter.check_all_same_dtype_ = false;
  iter.operands_[0].target_dtype = iter.operands_[0].current_dtype = a.scalar_type();
  iter.build(rec);
  return iter;
}

TensorIteratorSym TensorIteratorSym::copy_op(Recorder& rec, const TensorBase& dst, const TensorBase& src) {
  if (at::has_internal_overlap(dst) == MemOverlap::Yes) {
    decline("a copy into a tensor with internal overlap");
  }
  TensorIteratorSym iter({dst, src});
  iter.check_all_same_dtype_ = false;
  iter.build(rec);
  return iter;
}

TensorIteratorSym TensorIteratorSym::reduce_op(Recorder& rec, const TensorBase& out, const TensorBase& a) {
  TensorIteratorSym iter({out, a});
  iter.is_reduction_ = true;
  iter.build(rec);
  return iter;
}

TensorIteratorSym make_reduction(Recorder& rec, TensorBase& result, const TensorBase& self, IntArrayRef dims, bool keepdim) {
  const int64_t ndim = self.dim();
  const auto mask = at::native::make_dim_mask(dims, ndim);
  c10::SymDimVector shape(self.sym_sizes().begin(), self.sym_sizes().end());
  for (int64_t dim = ndim - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  result = at::empty_symint(shape, self.options());
  if (keepdim) {
    return TensorIteratorSym::reduce_op(rec, result, self);
  }
  c10::SymDimVector viewed_shape(result.sym_sizes().begin(), result.sym_sizes().end());
  c10::SymDimVector viewed_stride(result.sym_strides().begin(), result.sym_strides().end());
  for (const auto dim : c10::irange(ndim)) {
    if (mask[dim]) {
      viewed_shape.insert(viewed_shape.begin() + dim, 1);
      viewed_stride.insert(viewed_stride.begin() + dim, 0);
    }
  }
  return TensorIteratorSym::reduce_op(rec, at::as_strided_symint(at::Tensor(result), viewed_shape, viewed_stride), self);
}

std::pair<TensorBase, TensorBase> reduce_buffers(const c10::SymInt& buffer_bytes, const c10::SymInt& semaphore_bytes, Device device) {
  const auto options = TensorOptions(kByte).device(device);
  at::Tensor buffer = at::empty_symint({buffer_bytes}, options);
  at::Tensor semaphores = at::empty_symint({semaphore_bytes}, options);
  semaphores.zero_();
  return {buffer, semaphores};
}

void TensorIteratorSym::compute_types() {
  for (const auto i : c10::irange(ntensors())) {
    const TensorBase& t = operands_[i].tensor;
    if (!t.defined()) {
      continue;
    }
    if (!t.is_cuda()) {
      decline(c10::str("an operand on ", t.device()));
    }
    if (common_device_ == kCPU) {
      common_device_ = t.device();
    } else if (t.device() != common_device_) {
      decline("operands on different devices");
    }
    if (i == 0) {
      continue;
    }
    if (common_dtype_ == ScalarType::Undefined) {
      common_dtype_ = t.scalar_type();
    } else if (check_all_same_dtype_ && t.scalar_type() != common_dtype_) {
      decline(c10::str("operands of ", common_dtype_, " and ", t.scalar_type()));
    }
  }
  auto& out = operands_[0];
  if (!out.is_type_defined()) {
    out.target_dtype = out.current_dtype = common_dtype_;
  }
}

c10::SymInt TensorIteratorSym::numel() const {
  return ti_build::numel(shape_);
}

int TensorIteratorSym::num_reduce_dims() const {
  return ti_build::num_reduce_dims(shape_, operands_);
}

c10::SymInt TensorIteratorSym::num_output_elements() const {
  return ti_build::num_output_elements(shape_, operands_);
}

bool TensorIteratorSym::is_contiguous() const {
  return ti_build::is_contiguous(shape_, operands_);
}

bool TensorIteratorSym::can_use_32bit_indexing() const {
  return ti_build::can_use_32bit_indexing(shape_, operands_);
}

void TensorIteratorSym::build(Recorder& rec) {
  compute_types();
  bool all_ops_are_scalars = false;
  ti_build::compute_shape(shape_, operands_, /*resize_outputs=*/false, all_ops_same_shape_, all_ops_are_scalars);
  if (!is_reduction_ && output().defined() && !output().sym_sizes().equals(shape_)) {
    decline("an output of a shape other than the broadcast shape");
  }
  auto set_output = [this](int /*i*/, c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, std::optional<MemoryFormat> memory_format) {
    auto& out = operands_[0];
    if (out.tensor.defined()) {
      return;
    }
    const auto options = TensorOptions(out.target_dtype).device(common_device_).memory_format(memory_format);
    out.tensor = strides.empty() ? at::empty_symint(sizes, options) : at::empty_strided_symint(sizes, strides, options);
  };
  const auto setup_type = ti_build::compute_fast_setup_type<c10::SymInt>(operands_, is_reduction_, all_ops_same_shape_, false);
  if (!ti_build::fast_set_up(setup_type, shape_, operands_, noutputs(), has_coalesced_dimensions_, set_output)) {
    ti_build::compute_strides(shape_, operands_, false);
    ti_build::reorder_dimensions(shape_, perm_, operands_, is_reduction_, false);
    ti_build::allocate_or_resize_outputs(shape_, perm_, operands_, noutputs(), has_coalesced_dimensions_, set_output);
    ti_build::coalesce_dimensions(shape_, operands_, has_coalesced_dimensions_);
  }
  for (auto& op : operands_) {
    op.data = rec.data_ptr(op.tensor);
  }
}

} // namespace at::cuda::host_trace
#endif
