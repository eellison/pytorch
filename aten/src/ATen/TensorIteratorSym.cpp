#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIteratorSym.h>
#include <ATen/detail/TensorIteratorBuild.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif

namespace at {

namespace ti_build = detail::ti_build;

// Mirrors TensorIteratorBase::build up to its is_meta_ early exit.
TensorIteratorSym::TensorIteratorSym(TensorIteratorConfig& config) {
  populate_operands(config);
  for (const auto i : c10::irange(num_outputs_)) {
    TORCH_CHECK(!operands_[i].tensor_base().defined(), "TensorIteratorSym: outputs must be undefined");
  }
  mark_outputs();
  compute_mem_overlaps(config);
  for (auto& op : operands_) {
    sym_operands_.emplace_back(op);
  }
  ti_build::compute_shape(sym_shape_, sym_operands_, /*resize_outputs=*/true, all_ops_same_shape_, all_ops_are_scalars_);
  mark_resize_outputs(config);
  compute_types(config);

  auto set_output = [this](int i, c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides, std::optional<MemoryFormat> memory_format) {
    auto& op = operands_[i];
    auto options = op.options().memory_format(memory_format);
    op.tensor(c10::MaybeOwned<TensorBase>::owned(strides.empty() ? at::empty_symint(sizes, options) : at::empty_strided_symint(sizes, strides, options)));
    op.current_dtype = op.target_dtype;
  };
  auto setup_type = ti_build::compute_fast_setup_type<c10::SymInt>(sym_operands_, is_reduction_, all_ops_same_shape_, enforce_linear_iteration_);
  if (!ti_build::fast_set_up(setup_type, sym_shape_, sym_operands_, num_outputs_, has_coalesced_dimensions_, set_output)) {
    ti_build::compute_strides(sym_shape_, sym_operands_, /*static_shape=*/false);
    ti_build::reorder_dimensions(sym_shape_, perm_, sym_operands_, is_reduction_, enforce_linear_iteration_);
    ti_build::allocate_or_resize_outputs(sym_shape_, perm_, sym_operands_, num_outputs_, has_coalesced_dimensions_, set_output);
  }
}

void TensorIteratorSym::coalesce_dimensions() {
  ti_build::coalesce_dimensions(sym_shape_, sym_operands_, has_coalesced_dimensions_);
}

const Tensor& TensorIteratorSym::maybe_get_output(int64_t output_idx) {
  return output(output_idx);
}

} // namespace at
