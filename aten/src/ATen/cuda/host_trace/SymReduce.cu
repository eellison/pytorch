// gpu_reduce_kernel's launch (Reduce.cuh) over a symbolic TensorIterator, one
// instantiation for every ReduceOp: set_reduce_config over c10::SymInt, then
// the ReduceOp eager builds, laid out from its facts, and its reduce_kernel's
// launch recorded.
#if !defined(USE_ROCM)
#include <ATen/core/Tensor.h>
#include <ATen/cuda/host_trace/SymImage.h>
#include <ATen/cuda/host_trace/SymReduce.h>
#include <ATen/cuda/host_trace/SymReduceConfig.h>
#include <ATen/ops/empty.h>

#include <cstddef>

namespace at::cuda::host_trace {

namespace {

struct PendingOpsField {
  const void* ops;
  size_t offset;
  size_t width;
  c10::SymInt bits;
};
thread_local std::vector<PendingOpsField> pending_ops_fields;

// The iterator as set_reduce_config reads it, over its symbolic sizes and strides
struct SymReduceIter {
  const TensorIteratorBase& iter;

  c10::SymInt numel() const {
    return iter.sym_numel();
  }
  c10::SymInt num_output_elements() const {
    return iter.sym_num_output_elements();
  }
  int ntensors() const {
    return iter.ntensors();
  }
  int noutputs() const {
    return iter.noutputs();
  }
  int ndim() const {
    return iter.ndim();
  }
  int num_reduce_dims() const {
    return iter.num_reduce_dims();
  }
  c10::SymIntArrayRef shape() const {
    return iter.sym_shape();
  }
  c10::SymIntArrayRef strides(int arg) const {
    return iter.sym_strides(arg);
  }
};

// Reduce.cuh's get_output_vec_size, found by set_reduce_config through ADL:
// eager halves the width while (n / scalar_size) % width != 0, the same test as
// n % (width * scalar_size) for the multiples of scalar_size an address and a
// byte stride are, exact on SymInt
int get_output_vec_size(const SymReduceIter& v, int64_t scalar_size) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](const c10::SymInt& n, int64_t unit) {
    while (vec_size > 1 && n % (vec_size * unit) != 0) {
      vec_size /= 2;
    }
  };
  update_vec_size(operand_address(v.iter, v.noutputs()), scalar_size);
  const int output_index = v.num_reduce_dims();
  update_vec_size(v.shape()[output_index], 1);
  int j = 0;
  for (const auto& i : v.strides(v.noutputs())) {
    if (j != output_index) {
      update_vec_size(i, scalar_size);
    }
    j++;
  }
  return vec_size;
}

// make_output_calculator / make_input_calculator<uint32_t>
SymOffsets reduce_offsets(const TensorIteratorBase& iter, bool output) {
  const int num_reduce_dims = iter.num_reduce_dims();
  const int input_index = iter.ntensors() - 1;
  const auto shape = iter.sym_shape();
  SymOffsets c;
  const int first = output ? num_reduce_dims : 0;
  c.dims = output ? iter.ndim() - num_reduce_dims : num_reduce_dims;
  c.sizes.assign(shape.begin() + first, shape.begin() + first + c.dims);
  const int args[] = {0, input_index};
  for (const int arg : c10::ArrayRef<int>(output ? args : args + 1, output ? 2 : 1)) {
    const auto st = iter.sym_strides(arg);
    c.strides.emplace_back(st.begin() + first, st.begin() + first + c.dims);
  }
  return c;
}

} // namespace

OpsFieldScope::OpsFieldScope(const void* ops, const void* member, size_t width, c10::SymInt bits) {
  const auto offset = static_cast<size_t>(static_cast<const char*>(member) - static_cast<const char*>(ops));
  pending_ops_fields.push_back({ops, offset, width, std::move(bits)});
}

OpsFieldScope::~OpsFieldScope() {
  pending_ops_fields.pop_back();
}

void sym_gpu_reduce_kernel(const TensorIteratorBase& iter, const ReduceFacts& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() - iter.noutputs() == 1 && iter.noutputs() >= 1);
  // eager splits the iterator, accumulating through a buffer across the parts
  TORCH_CHECK_NOT_IMPLEMENTED(iter.can_use_32bit_indexing(), "host_trace: a reduction beyond 32-bit indexing");
  Recorder& rec = recorder_for("a reduction");
  const ReduceConfigSymMath math{rec};
  const ReduceConfig config = at::native::set_reduce_config(
      SymReduceIter{iter}, math, at::native::ReduceSizes{f.arg_size, f.scalar_size, f.input_vec_size, f.max_threads});
  TensorBase buffer, semaphores;
  if (config.should_global_reduce()) {
    // eager allocates both from the caching allocator and zeroes the semaphores
    const auto options = TensorOptions(kByte).device(iter.device());
    buffer = at::empty_symint({config.global_memory_size()}, options);
    at::Tensor s = at::empty_symint({config.semaphore_size()}, options);
    s.zero_();
    semaphores = s;
  }

  // ReduceOp{ops, config, input_calc, output_calc, src, dst0, dst1, acc_buf
  // (none: 32-bit indexing), cta_buf, semaphores, ident, noutputs, base_idx 0}
  using EagerConfig = at::native::ReduceConfig;
  ParamImage op(f.size, f.align);
  if (f.ops_size > 0) {
    op.put(f.ops, f.ops_bytes, f.ops_size);
    for (const auto& p : pending_ops_fields) {
      if (p.ops == f.ops_bytes) {
        op.set(f.ops + p.offset, p.width, p.bits);
      }
    }
  }
  op.put(f.ident, f.ident_bytes, f.ident_size);
  auto set_int = [&](size_t offset, const c10::SymInt& v) { op.set(f.config + offset, sizeof(int), v); };
  set_int(offsetof(EagerConfig, element_size_bytes), config.element_size_bytes);
  set_int(offsetof(EagerConfig, num_inputs), config.num_inputs);
  set_int(offsetof(EagerConfig, num_outputs), config.num_outputs);
  set_int(offsetof(EagerConfig, step_input), config.step_input);
  set_int(offsetof(EagerConfig, step_output), config.step_output);
  set_int(offsetof(EagerConfig, ctas_per_output), config.ctas_per_output);
  for (const auto i : c10::irange(3)) {
    set_int(offsetof(EagerConfig, input_mult) + i * sizeof(int), config.input_mult[i]);
  }
  for (const auto i : c10::irange(2)) {
    set_int(offsetof(EagerConfig, output_mult) + i * sizeof(int), config.output_mult[i]);
  }
  set_int(offsetof(EagerConfig, block_width), config.block_width);
  set_int(offsetof(EagerConfig, block_height), config.block_height);
  set_int(offsetof(EagerConfig, num_threads), config.num_threads);
  op.set(f.config + offsetof(EagerConfig, vectorize_input), sizeof(bool), math.select(config.vectorize_input, 1, 0));
  set_int(offsetof(EagerConfig, output_vec_size), config.output_vec_size);
  put_offset_calculator(op, f.input_calc, 1, reduce_offsets(iter, /*output=*/false));
  put_offset_calculator(op, f.output_calc, 2, reduce_offsets(iter, /*output=*/true));
  op.set(f.src, sizeof(void*), operand_address(iter, iter.ntensors() - 1), true);
  op.set(f.dst, sizeof(char*), operand_address(iter, 0), true);
  if (iter.noutputs() > 1) {
    op.set(f.dst + sizeof(char*), sizeof(char*), operand_address(iter, 1), true);
  }
  if (config.should_global_reduce()) {
    op.set(f.cta_buf, sizeof(void*), rec.data_ptr(buffer), true);
    op.set(f.semaphores, sizeof(int*), rec.data_ptr(semaphores), true);
  }
  const bool accumulate = iter.should_accumulate();
  const bool final_output = iter.is_final_output();
  op.put(f.accumulate, &accumulate, sizeof(bool));
  op.put(f.final_output, &final_output, sizeof(bool));
  const int noutputs = iter.noutputs();
  op.put(f.noutputs, &noutputs, sizeof(int));

  // launch_reduce_kernel
  std::vector<ParamImage> params;
  params.push_back(std::move(op));
  const c10::SymInt smem = at::native::shared_memory_size(math, config);
  if (f.jit_function != nullptr) {
    // launch_jitted_reduce_kernel
    auto function = static_cast<cudaFunction_t>(f.jit_function(f.jit, config.output_vec_size));
    return record_images(function, std::move(params), config.grid(), config.block(), smem);
  }
  const int k = config.output_vec_size == 4 ? 0 : config.output_vec_size == 2 ? 1 : 2;
  record_images(f.kernels[k], std::move(params), config.grid(), config.block(), smem);
}

} // namespace at::cuda::host_trace
#endif
