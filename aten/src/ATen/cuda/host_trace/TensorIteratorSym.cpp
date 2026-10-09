#if !defined(USE_ROCM)
#include <ATen/cuda/host_trace/TensorIteratorSym.h>
#include <ATen/cuda/host_trace/Ops.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/TensorUtils.h>
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

#include <algorithm>

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

TensorIteratorSym TensorIteratorSym::gather_op(Recorder& rec, const TensorBase& out, const TensorBase& src, const TensorBase& index) {
  TensorIteratorSym iter({out, src, index});
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

TensorIteratorSym TensorIteratorSym::reduce_op(Recorder& rec, const TensorBase& out1, const TensorBase& out2, const TensorBase& a) {
  TensorIteratorSym iter({out1, out2, a});
  iter.operands_[1].is_output = true;
  iter.noutputs_ = 2;
  iter.is_reduction_ = true;
  iter.check_all_same_dtype_ = false;
  iter.build(rec);
  return iter;
}

TensorIteratorSym TensorIteratorSym::pointwise_op(Recorder& rec, c10::ArrayRef<TensorBase> outs, c10::ArrayRef<ScalarType> out_dtypes, c10::ArrayRef<TensorBase> inputs) {
  TORCH_CHECK(!outs.empty() && outs.size() == out_dtypes.size(), "a pointwise op of ", outs.size(), " outputs and ", out_dtypes.size(), " output dtypes");
  TensorIteratorSym iter({outs[0]});
  for (const auto i : c10::irange(outs.size())) {
    if (outs[i].defined() && at::has_internal_overlap(outs[i]) == MemOverlap::Yes) {
      decline("a pointwise op into a tensor with internal overlap");
    }
    if (i > 0) {
      iter.operands_.emplace_back(outs[i]).is_output = true;
    }
    iter.operands_[i].target_dtype = iter.operands_[i].current_dtype = out_dtypes[i];
  }
  iter.noutputs_ = static_cast<int>(outs.size());
  for (const TensorBase& t : inputs) {
    iter.operands_.emplace_back(t);
  }
  iter.check_all_same_dtype_ = false;
  iter.allow_cpu_scalars_ = true;
  iter.build(rec);
  auto& ops = iter.operands_;
  ops.erase(std::remove_if(ops.begin() + iter.noutputs_, ops.end(), [](const SymOperandInfo& op) { return !op.tensor.is_cuda(); }), ops.end());
  if (iter.ninputs() == 0 && !outs[0].defined()) {
    decline("a pointwise op of no operand on the device");
  }
  return iter;
}

namespace {

// shape_from_dim_mask
c10::SymDimVector reduction_shape(const TensorBase& self, const at::native::DimMask& mask, bool keepdim) {
  c10::SymDimVector shape(self.sym_sizes().begin(), self.sym_sizes().end());
  for (int64_t dim = self.dim() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  return shape;
}

// review_reduce_result
TensorBase review_reduce_result(const TensorBase& result, int64_t ndim, const at::native::DimMask& mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  c10::SymDimVector viewed_shape(result.sym_sizes().begin(), result.sym_sizes().end());
  c10::SymDimVector viewed_stride(result.sym_strides().begin(), result.sym_strides().end());
  for (const auto dim : c10::irange(ndim)) {
    if (mask[dim]) {
      viewed_shape.insert(viewed_shape.begin() + dim, 1);
      viewed_stride.insert(viewed_stride.begin() + dim, 0);
    }
  }
  return at::as_strided_symint(at::Tensor(result), viewed_shape, viewed_stride);
}

} // namespace

TensorIteratorSym make_reduction(Recorder& rec, TensorBase& result, const TensorBase& self, IntArrayRef dims, bool keepdim, std::optional<ScalarType> dtype) {
  const auto mask = at::native::make_dim_mask(dims, self.dim());
  if (!result.defined()) {
    result = at::empty_symint(reduction_shape(self, mask, keepdim), self.options().dtype(dtype.value_or(self.scalar_type())));
  }
  return TensorIteratorSym::reduce_op(rec, review_reduce_result(result, self.dim(), mask, keepdim), self);
}

TensorIteratorSym make_reduction(Recorder& rec, const TensorBase& values, const TensorBase& indices, const TensorBase& self, int64_t dim, bool keepdim) {
  const auto mask = at::native::make_dim_mask(dim, self.dim());
  return TensorIteratorSym::reduce_op(rec, review_reduce_result(values, self.dim(), mask, keepdim), review_reduce_result(indices, self.dim(), mask, keepdim), self);
}

std::optional<TensorIteratorSym> make_minmax_reduction(Recorder& rec, TensorBase& values, TensorBase& indices, const TensorBase& self, int64_t dim, bool keepdim) {
  dim = c10::maybe_wrap_dim(dim, self.dim());
  const auto shape = reduction_shape(self, at::native::make_dim_mask(dim, self.dim()), keepdim);
  values = at::empty_symint(shape, self.options());
  indices = at::empty_symint(shape, self.options().dtype(kLong));
  if (self.sym_numel() == 0) {
    return std::nullopt;
  }
  if (self.dim() == 0) {
    // values.fill_(self), a copy of a 0-dim tensor
    copy_(rec, values, self);
    fill_(rec, indices, 0);
    return std::nullopt;
  }
  return make_reduction(rec, values, indices, self, dim, keepdim);
}

std::optional<TensorIteratorSym> make_arg_reduction(Recorder& rec, TensorBase& result, const TensorBase& self, std::optional<int64_t> dim, bool keepdim) {
  if (!dim) {
    // self.reshape({-1}): a view where the geometry allows, else a view of a contiguous copy
    const c10::SymDimVector shape{self.sym_numel()};
    const auto stride = at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape);
    const TensorBase in = stride ? self : copy_(rec, at::empty_symint(self.sym_sizes(), self.options()), self);
    const c10::SymDimVector flat_stride = stride ? *stride : c10::SymDimVector{1};
    const at::Tensor flat = at::as_strided_symint(at::Tensor(in), shape, flat_stride, in.sym_storage_offset());
    // the meta result keeps self's dims under keepdim; the iterator is make_reduction's over flat either way
    result = at::empty_symint(c10::SymDimVector(keepdim ? self.dim() : 0, c10::SymInt(1)), self.options().dtype(kLong));
    auto iter = TensorIteratorSym::reduce_op(rec, at::as_strided_symint(at::Tensor(result), {1}, {0}), flat);
    return iter.numel() == 0 ? std::nullopt : std::optional(std::move(iter));
  }
  if (self.dim() == 0) {
    decline("an arg reduction of a 0-dim tensor over a dim");
  }
  const int64_t d = c10::maybe_wrap_dim(*dim, self.dim());
  if (self.sym_size(d) == 1) {
    result = at::empty_symint(reduction_shape(self, at::native::make_dim_mask(d, self.dim()), keepdim), self.options().dtype(kLong));
    fill_(rec, result, 0);
    return std::nullopt;
  }
  auto iter = make_reduction(rec, result, self, d, keepdim, kLong);
  return iter.numel() == 0 ? std::nullopt : std::optional(std::move(iter));
}

TensorBase contiguous(Recorder& rec, const TensorBase& t) {
  return t.is_contiguous() ? t : copy_(rec, at::empty_symint(t.sym_sizes(), t.options()), t);
}

std::pair<TensorBase, TensorBase> reduce_buffers(const c10::SymInt& buffer_bytes, const c10::SymInt& semaphore_bytes, Device device) {
  const auto options = TensorOptions(kByte).device(device);
  at::Tensor buffer = at::empty_symint({buffer_bytes}, options);
  at::Tensor semaphores = at::empty_symint({semaphore_bytes}, options);
  semaphores.zero_();
  return {buffer, semaphores};
}

void TensorIteratorSym::compute_types() {
  int cpu_scalars = 0;
  for (const auto i : c10::irange(ntensors())) {
    const TensorBase& t = operands_[i].tensor;
    if (!t.defined()) {
      continue;
    }
    if (allow_cpu_scalars_ && i >= noutputs_ && t.dim() == 0 && t.is_cpu()) {
      TORCH_CHECK(++cpu_scalars <= 1, "Trying to pass too many CPU scalars to non-CPU kernel!");
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
    if (i < noutputs_) {
      continue;
    }
    if (common_dtype_ == ScalarType::Undefined) {
      common_dtype_ = t.scalar_type();
    } else if (check_all_same_dtype_ && t.scalar_type() != common_dtype_) {
      decline(c10::str("operands of ", common_dtype_, " and ", t.scalar_type()));
    }
  }
  if (common_device_ == kCPU) {
    decline("no operand on the device");
  }
  for (const auto i : c10::irange(noutputs_)) {
    auto& out = operands_[i];
    if (!out.is_type_defined()) {
      out.target_dtype = out.current_dtype = common_dtype_;
    }
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
  for (const auto i : c10::irange(noutputs_)) {
    if (!is_reduction_ && output(i).defined() && !output(i).sym_sizes().equals(shape_)) {
      decline("an output of a shape other than the broadcast shape");
    }
  }
  auto set_output = [this](int i, c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, std::optional<MemoryFormat> memory_format) {
    auto& out = operands_[i];
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
    if (op.tensor.is_cuda()) {
      op.data = rec.data_ptr(op.tensor);
    }
  }
}

} // namespace at::cuda::host_trace
#endif
