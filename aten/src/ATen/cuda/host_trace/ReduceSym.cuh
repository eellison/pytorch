// gpu_reduce_kernel for a traced host over a TensorIteratorSym, recording the
// launch of the reduce_kernel eager launches. The config computation is
// Reduce.cuh's setReduceConfig over ReduceConfigSymMath; the config's fields,
// its plain accessors, get_output_vec_size and the launch mirror Reduce.cuh
// (keep in sync). The config is size expressions (block, shared bytes, grid and
// the ReduceOp's config are patched per call); only choices of the kernel
// instantiation or of global reduction's buffers are guarded.
#pragma once
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/cuda/host_trace/SymReduceConfig.h>
#include <ATen/native/cuda/Reduce.cuh>

#include <algorithm>
#include <tuple>

namespace at::cuda::host_trace {

// make_output_calculator<uint32_t>(iter)
template <class K>
void set_output_calculator(Recorder& rec, Param<K>& p, ::OffsetCalculator<2>& calc, const TensorIteratorSym& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  int input_index = iter.ntensors() - 1;
  int output_index = 0;
  std::array<const c10::SymInt*, 2> strides = {
      iter.strides(output_index).data() + num_reduce_dims,
      iter.strides(input_index).data() + num_reduce_dims,
  };
  auto shape = iter.shape().data() + num_reduce_dims;
  set_offset_calculator(rec, p, calc, num_output_dims, shape, strides.data());
}

// make_input_calculator<uint32_t>(iter)
template <class K>
void set_input_calculator(Recorder& rec, Param<K>& p, ::OffsetCalculator<1>& calc, const TensorIteratorSym& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int input_index = iter.ntensors() - 1;
  std::array<const c10::SymInt*, 1> strides = {
      iter.strides(input_index).data(),
  };
  set_offset_calculator(rec, p, calc, num_reduce_dims, iter.shape().data(), strides.data());
}

// Reduce.cuh's, found by the shared setReduceConfig through ADL. Eager tests
// (n / sizeof(scalar_t)) % vec_size; n % (vec_size * sizeof(scalar_t)) is the
// same test for an address aligned to sizeof(scalar_t), and exact on SymInt,
// where a data pointer is only tested for its alignment
inline int get_output_vec_size(const TensorIteratorSym& iter, int64_t scalar_size) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](const c10::SymInt& n, int64_t unit) {
    while (vec_size > 1 && n % (vec_size * unit) != 0) {
      vec_size /= 2;
    }
  };

  update_vec_size(iter.data_ptr(iter.noutputs()), scalar_size);

  const int output_index = iter.num_reduce_dims();
  update_vec_size(iter.shape()[output_index], 1);

  int j = 0;
  for (const auto& i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_vec_size(i, scalar_size);
    }
    j++;
  }
  return vec_size;
}

template <int max_threads, typename R>
void launch_reduce_kernel(Recorder& rec, const ReduceConfig& config, const Param<R>& reduction) {
  SymDim3 block = config.block();
  SymDim3 grid = config.grid();
  c10::SymInt shared_memory = at::native::shared_memory_size(ReduceConfigSymMath{rec}, config);
  switch (config.output_vec_size) {
    case 4:
      return launch(rec, &at::native::reduce_kernel<max_threads / 4, 4, R>, grid, block, shared_memory, reduction);
    case 2:
      return launch(rec, &at::native::reduce_kernel<max_threads / 2, 2, R>, grid, block, shared_memory, reduction);
    default:
      return launch(rec, &at::native::reduce_kernel<max_threads / 1, 1, R>, grid, block, shared_memory, reduction);
  }
}

// no accumulation buffer: 64-bit indexing declines
template <typename scalar_t, typename out_scalar_t, int vt0 = 4, int input_vec_size = vt0, typename ops_t, typename ident_t = double>
void gpu_reduce_kernel(Recorder& rec, const TensorIteratorSym& iter, const Param<ops_t>& ops, ident_t ident = 0) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() - iter.noutputs() == 1);
  using traits = function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename traits::template arg<0>::type;

  if (!iter.can_use_32bit_indexing()) {
    decline("a reduction beyond 32-bit indexing");
  }

  const ReduceConfigSymMath math{rec};
  ReduceConfig config = at::native::setReduceConfig<arg_t, scalar_t, vt0, input_vec_size>(iter, math);
  TensorBase buffer;
  TensorBase semaphores;
  if (config.should_global_reduce()) {
    std::tie(buffer, semaphores) = reduce_buffers(config.global_memory_size(), config.semaphore_size(), iter.device());
  }

  using reduce_op_t = at::native::ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, vt0, input_vec_size>;
  Param<reduce_op_t> reduce;
  reduce_op_t& r = reduce.value();
  reduce.set(r.ops, ops);
  r.ident = ident;
  r.config.element_size_bytes = config.element_size_bytes;
  reduce.set(r.config.num_inputs, config.num_inputs);
  reduce.set(r.config.num_outputs, config.num_outputs);
  reduce.set(r.config.step_input, config.step_input);
  reduce.set(r.config.step_output, config.step_output);
  reduce.set(r.config.ctas_per_output, config.ctas_per_output);
  for (const auto i : c10::irange(3)) {
    reduce.set(r.config.input_mult[i], config.input_mult[i]);
  }
  for (const auto i : c10::irange(2)) {
    reduce.set(r.config.output_mult[i], config.output_mult[i]);
  }
  reduce.set(r.config.block_width, config.block_width);
  reduce.set(r.config.block_height, config.block_height);
  reduce.set(r.config.num_threads, config.num_threads);
  reduce.set(r.config.vectorize_input, math.select(config.vectorize_input, 1, 0));
  r.config.output_vec_size = config.output_vec_size;
  set_input_calculator(rec, reduce, r.input_calc, iter);
  set_output_calculator(rec, reduce, r.output_calc, iter);
  reduce.set(r.src, iter.data_ptr(iter.ntensors() - 1));
  reduce.set(r.dst[0], iter.data_ptr(0));
  if (iter.noutputs() > 1) {
    reduce.set(r.dst[1], iter.data_ptr(1));
  }
  if (config.should_global_reduce()) {
    reduce.set(r.cta_buf, rec.data_ptr(buffer));
    reduce.set(r.semaphores, rec.data_ptr(semaphores));
  }
  r.noutputs = iter.noutputs();
  r.final_output = true;

  launch_reduce_kernel<at::native::mnt_wrapper<scalar_t>::MAX_NUM_THREADS>(rec, config, reduce);
}

} // namespace at::cuda::host_trace
