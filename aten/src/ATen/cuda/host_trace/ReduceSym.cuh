// gpu_reduce_kernel for a traced host: Reduce.cuh's ReduceConfig,
// setReduceConfig and launch over a TensorIteratorSym, recording the launch of
// the reduce_kernel eager launches. Block dimensions are guarded to their
// power-of-two brackets, so block and shared memory are per-variant constants.
#pragma once
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/native/cuda/Reduce.cuh>

#include <algorithm>
#include <tuple>

namespace at::cuda::host_trace {

inline c10::SymInt div_up(const c10::SymInt& a, const c10::SymInt& b) {
  return (a + b - 1) / b;
}

// `dim < cap ? last_pow2(dim) : cap` for a power-of-two cap
inline int last_pow2(const c10::SymInt& dim, int cap) {
  while (cap > 1 && dim < cap) {
    cap /= 2;
  }
  return cap;
}

struct ReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  ReduceConfig(int element_size_bytes, c10::SymInt num_outputs, c10::SymInt num_inputs)
      : element_size_bytes(element_size_bytes), num_inputs(std::move(num_inputs)), num_outputs(std::move(num_outputs)) {}
  int element_size_bytes;
  c10::SymInt num_inputs;
  c10::SymInt num_outputs;
  c10::SymInt step_input = 1;
  c10::SymInt step_output = 1;
  c10::SymInt ctas_per_output = 1;
  c10::SymInt input_mult[3] = {0, 0, 0};
  c10::SymInt output_mult[2] = {0, 0};

  int block_width = 0;
  int block_height = 0;
  int num_threads = 0;

  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void set_block_dimension(const c10::SymInt& dim0, const c10::SymInt& dim1) {
    const int max_num_threads = at::native::mnt_wrapper<T>::MAX_NUM_THREADS / output_vec_size;
    int dim0_pow2 = last_pow2(dim0, max_num_threads);
    int dim1_pow2 = last_pow2(dim1, max_num_threads);
    block_width = std::min(dim0_pow2, int(at::cuda::warp_size()));
    block_height = std::min(dim1_pow2, int(max_num_threads / block_width));
    block_width = std::min(dim0_pow2, int(max_num_threads / block_height));
    num_threads = block_width * block_height;
  }

  c10::SymInt split_input(const c10::SymInt& parallelism) {
    c10::SymInt step = step_input;
    step_input *= parallelism;
    return step;
  }

  c10::SymInt split_output(const c10::SymInt& parallelism) {
    c10::SymInt step = step_output;
    step_output *= parallelism;
    return step;
  }

  dim3 block() const {
    return dim3(block_width, block_height);
  }

  SymDim3 grid() const {
    return SymDim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
  }

  bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() && (!should_block_x_reduce() || block_width <= at::cuda::warp_size())) {
      return 0;
    }
    return element_size_bytes * num_threads * output_vec_size;
  }

  c10::SymInt global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block_width * output_vec_size;
    }
    return size;
  }

  c10::SymInt semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return int64_t(sizeof(int)) * grid().x;
  }

  c10::SymInt values_per_thread() const {
    return div_up(num_inputs, step_input);
  }
};

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

// eager tests (n / sizeof(scalar_t)) % vec_size; n % (vec_size * sizeof(scalar_t))
// is the same test, exact on SymInt
template <typename scalar_t>
int get_output_vec_size(const TensorIteratorSym& iter) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](const c10::SymInt& n, int64_t unit) {
    while (vec_size > 1 && n % (vec_size * unit) != 0) {
      vec_size /= 2;
    }
  };

  update_vec_size(iter.data_ptr(iter.noutputs()), sizeof(scalar_t));

  const int output_index = iter.num_reduce_dims();
  update_vec_size(iter.shape()[output_index], 1);

  int j = 0;
  for (const auto& i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_vec_size(i, sizeof(scalar_t));
    }
    j++;
  }
  return vec_size;
}

template <typename arg_t, typename scalar_t, int vt0, int input_vec_size = vt0>
ReduceConfig setReduceConfig(const TensorIteratorSym& iter) {
  c10::SymInt num_outputs = iter.num_output_elements();
  c10::SymInt inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

  c10::SymInt dim0;
  c10::SymInt dim1;
  c10::SymInt fastest_moving_stride;
  bool reduction_on_fastest_striding_dimension;

  if (iter.ndim() > 0) {
    reduction_on_fastest_striding_dimension = (iter.num_reduce_dims() == iter.ndim()) ||
        (iter.strides(input_index)[0] < iter.strides(input_index)[iter.num_reduce_dims()]);
    if (reduction_on_fastest_striding_dimension) {
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(input_index)[0];
    } else {
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride = iter.strides(input_index)[iter.num_reduce_dims()];
    }
  } else {
    reduction_on_fastest_striding_dimension = true;
    fastest_moving_stride = int64_t(sizeof(scalar_t));
    dim0 = 1;
    dim1 = 1;
  }

  if (fastest_moving_stride == int64_t(sizeof(scalar_t))) {
    if (reduction_on_fastest_striding_dimension && dim0 >= 128 && iter.num_reduce_dims() == 1) {
      config.vectorize_input = true;
      dim0 /= input_vec_size;
    } else if (!reduction_on_fastest_striding_dimension) {
      config.output_vec_size = get_output_vec_size<scalar_t>(iter);
      dim0 /= config.output_vec_size;
    }
  }

  config.set_block_dimension<scalar_t>(dim0, dim1);

  int block_width = config.block_width;
  int block_height = config.block_height;

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    config.input_mult[0] = config.split_input(block_width);
  } else {
    config.output_mult[0] = config.split_output(block_width);
  }

  constexpr int min_values_per_thread = 16;
  constexpr int max_values_per_thread = 256;

  const int warp_split_threshold = std::min<int>(block_height * 16, max_values_per_thread);
  bool split_across_warps = config.values_per_thread() >= warp_split_threshold;
  const int num_mp = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (split_across_warps) {
    config.input_mult[1] = config.split_input(block_height);
  } else {
    config.output_mult[1] = config.split_output(block_height);
  }

  int max_threads_per_mp = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  const int blocks_per_sm = max_threads_per_mp / config.num_threads;
  const int target_grid_size = num_mp * blocks_per_sm;
  c10::SymInt grid = config.grid().x;
  if (config.input_mult[1] != 0 && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
    c10::SymInt ctas_per_output1 = div_up(target_grid_size, grid);
    c10::SymInt ctas_per_output2 = div_up(config.values_per_thread(), min_values_per_thread);
    c10::SymInt ctas_per_output3 = div_up(config.values_per_thread(), max_values_per_thread);
    // std::clamp(ctas_per_output1, ctas_per_output3, ctas_per_output2), as ctas_per_output3 <= ctas_per_output2
    config.ctas_per_output = ctas_per_output3.max(ctas_per_output1.min(ctas_per_output2));
    if (config.ctas_per_output > 1) {
      config.input_mult[2] = config.split_input(config.ctas_per_output);
    }
  }
  return config;
}

template <int max_threads, typename R>
void launch_reduce_kernel(Recorder& rec, const ReduceConfig& config, const Param<R>& reduction) {
  dim3 block = config.block();
  SymDim3 grid = config.grid();
  int shared_memory = config.shared_memory_size();
  switch (config.output_vec_size) {
    case 4:
      return launch(rec, &at::native::reduce_kernel<max_threads / 4, 4, R>, grid, block, shared_memory, reduction);
    case 2:
      return launch(rec, &at::native::reduce_kernel<max_threads / 2, 2, R>, grid, block, shared_memory, reduction);
    default:
      return launch(rec, &at::native::reduce_kernel<max_threads / 1, 1, R>, grid, block, shared_memory, reduction);
  }
}

// one output, no accumulation buffer: 64-bit indexing declines
template <typename scalar_t, typename out_scalar_t, int vt0 = 4, int input_vec_size = vt0, typename ops_t, typename ident_t = double>
void gpu_reduce_kernel(Recorder& rec, const TensorIteratorSym& iter, const Param<ops_t>& ops, ident_t ident = 0) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() - iter.noutputs() == 1);
  using traits = function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename traits::template arg<0>::type;

  if (!iter.can_use_32bit_indexing()) {
    decline("a reduction beyond 32-bit indexing");
  }

  ReduceConfig config = setReduceConfig<arg_t, scalar_t, vt0, input_vec_size>(iter);
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
  r.config.block_width = config.block_width;
  r.config.block_height = config.block_height;
  r.config.num_threads = config.num_threads;
  r.config.vectorize_input = config.vectorize_input;
  r.config.output_vec_size = config.output_vec_size;
  set_input_calculator(rec, reduce, r.input_calc, iter);
  set_output_calculator(rec, reduce, r.output_calc, iter);
  reduce.set(r.src, iter.data_ptr(iter.ntensors() - 1));
  reduce.set(r.dst[0], iter.data_ptr(0));
  if (config.should_global_reduce()) {
    reduce.set(r.cta_buf, rec.data_ptr(buffer));
    reduce.set(r.semaphores, rec.data_ptr(semaphores));
  }
  r.noutputs = iter.noutputs();
  r.final_output = true;

  launch_reduce_kernel<at::native::mnt_wrapper<scalar_t>::MAX_NUM_THREADS>(rec, config, reduce);
}

} // namespace at::cuda::host_trace
