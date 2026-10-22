// Reduce.cuh's ReduceConfig over c10::SymInt and the Math its shared
// set_reduce_config computes it with: a choice that only picks a launch size
// is a select row, a choice of the kernel instantiation or of global
// reduction's buffers a guard. The fields and plain accessors mirror
// Reduce.cuh's ReduceConfig (keep in sync).
#pragma once
#include <ATen/cuda/host_trace/Recorder.h>
#include <ATen/native/cuda/Reduce.cuh>

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

} // namespace at::cuda::host_trace
