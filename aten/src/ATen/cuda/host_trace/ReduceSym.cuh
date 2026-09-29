// gpu_reduce_kernel for a traced host over a TensorIteratorSym, recording the
// launch of the reduce_kernel eager launches. The config computation is
// Reduce.cuh's setReduceConfig over ReduceConfigSymMath; the config's fields,
// its plain accessors, get_output_vec_size and the launch mirror Reduce.cuh
// (keep in sync). The config is size expressions (block, shared bytes, grid and
// the ReduceOp's config are patched per call); only choices of the kernel
// instantiation or of global reduction's buffers are guarded.
#pragma once
#include <ATen/cuda/host_trace/LoopsSym.cuh>
#include <ATen/native/cuda/Reduce.cuh>

#include <algorithm>
#include <tuple>

namespace at::cuda::host_trace {

inline c10::SymInt div_up(const c10::SymInt& a, const c10::SymInt& b) {
  return (a + b - 1) / b;
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

  c10::SymInt block_width = 0;
  c10::SymInt block_height = 0;
  c10::SymInt num_threads = 0;

  c10::SymBool vectorize_input = false;
  int output_vec_size = 1;

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

  SymDim3 block() const {
    return SymDim3(block_width, block_height);
  }

  SymDim3 grid() const {
    return SymDim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
  }

  bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  c10::SymBool should_block_y_reduce() const {
    return input_mult[BLOCK_Y].sym_ne(0);
  }

  bool should_global_reduce() const {
    return input_mult[CTA] != 0;
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

// Reduce.cuh's ReduceConfigMath over SymInt: a size choice is a select row
struct ReduceConfigSymMath {
  using Config = ReduceConfig;
  using Int = c10::SymInt;
  using Index = c10::SymInt;
  using Bool = c10::SymBool;
  Recorder& rec;

  c10::SymInt select(const c10::SymBool& c, const c10::SymInt& a, const c10::SymInt& b) const {
    if (auto v = c.maybe_as_bool()) {
      return *v ? a : b;
    }
    return rec.select(c, a, b);
  }
  c10::SymInt last_pow2(const c10::SymInt& n) const {
    if (auto c = n.maybe_as_int()) {
      return at::native::last_pow2(static_cast<int>(*c));
    }
    return rec.pow2((rec.bit_length(n) - 1).max(0));
  }
  static c10::SymInt div_up(const c10::SymInt& a, const c10::SymInt& b) {
    return host_trace::div_up(a, b);
  }
  static c10::SymInt min(const c10::SymInt& a, const c10::SymInt& b) {
    return a.min(b);
  }
  static c10::SymInt max(const c10::SymInt& a, const c10::SymInt& b) {
    return a.max(b);
  }
  // std::clamp's value, as its callers have lo <= hi
  static c10::SymInt clamp(const c10::SymInt& v, const c10::SymInt& lo, const c10::SymInt& hi) {
    return lo.max(v.min(hi));
  }
  static c10::SymBool lt(const c10::SymInt& a, const c10::SymInt& b) {
    return a.sym_lt(b);
  }
  static c10::SymBool le(const c10::SymInt& a, const c10::SymInt& b) {
    return a.sym_le(b);
  }
  static c10::SymBool ge(const c10::SymInt& a, const c10::SymInt& b) {
    return a.sym_ge(b);
  }
  static c10::SymBool ne(const c10::SymInt& a, const c10::SymInt& b) {
    return a.sym_ne(b);
  }
  static c10::SymBool logical_not(const c10::SymBool& a) {
    return ~a;
  }
  static c10::SymBool logical_and(const c10::SymBool& a, const c10::SymBool& b) {
    return a & b;
  }
  static c10::SymBool logical_or(const c10::SymBool& a, const c10::SymBool& b) {
    return a | b;
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

// Reduce.cuh's, found by the shared setReduceConfig through ADL. Eager tests
// (n / sizeof(scalar_t)) % vec_size; n % (vec_size * sizeof(scalar_t)) is the
// same test for an address aligned to sizeof(scalar_t), and exact on SymInt,
// where a data pointer is only tested for its alignment
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
