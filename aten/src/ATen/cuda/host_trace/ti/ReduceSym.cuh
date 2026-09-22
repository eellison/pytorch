// Reduce.cuh's host for the traced sibling iterator: setReduceConfig,
// gpu_reduce_kernel and launch_reduce_kernel with the sizes as c10::SymInt,
// the buffers as traced allocations, the semaphore memset as a memset record
// and the launch through the typed helper. The kernel is ATen's
// reduce_kernel<nt, output_vec_size, ReduceOp<...>> from Reduce.cuh,
// instantiated here for the same ReduceOp the real op instantiates (the ops
// functor is a named struct where the real op uses a lambda), so the device
// side is shared and the real launch path is never executed under a trace.
//
// Every decision the real host takes on a value is a guard; every derived
// integer stays an expression. Two things are opaque rebinds of their inputs
// rather than expressions: IntDivider's magic and shift (LoopsSym.cuh) and
// last_pow2, whose bit smearing has no SymInt form and whose interval guards
// would otherwise pin every size to its power-of-two bracket. The block
// dimensions, the shared memory and the grid are therefore expressions of the
// rebound values, and a size sweep across powers of two keeps serving.
#pragma once
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/Slots.h>
#include <ATen/cuda/host_trace/ti/TensorIteratorSym.h>

#include <ATen/WrapDimUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <c10/util/irange.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <cstring>
#include <new>
#include <vector>

namespace at::cuda::host_trace::ti {

namespace detail {
// at::native::last_pow2 for n >= 1 (std::max(1, ...) for n <= 1)
inline int64_t last_pow2_impl(const int64_t* a, size_t) {
  int64_t p = 1;
  while (p * 2 <= a[0]) {
    p *= 2;
  }
  return p;
}
} // namespace detail

inline c10::SymInt div_up_sym(const c10::SymInt& a, const c10::SymInt& b) {
  return (a + b - 1) / b;
}

// last_pow2 as a value re-evaluated per call: its result feeds the block
// dimensions (expressions on the launch record) and the decisions taken on
// them (guards on the rebound value). The result is at least 1 for every
// input (last_pow2_impl starts at 1 and only doubles), declared so the
// divisions by the block dimensions below carry no domain guard.
inline c10::SymInt last_pow2_sym(const c10::SymInt& n) {
  return opaque("last_pow2", {n}, &detail::last_pow2_impl, "rebind", "positive");
}

// ---- Reduce.cuh: ReduceConfig with SymInt members. The multipliers that the
// real config tests against zero are tracked by the branches that set them.
struct ReduceConfigSym {
  ReduceConfigSym(int element_size_bytes, c10::SymInt num_outputs, c10::SymInt num_inputs)
    : element_size_bytes(element_size_bytes)
    , num_inputs(std::move(num_inputs))
    , num_outputs(std::move(num_outputs)) {}
  int element_size_bytes;
  c10::SymInt num_inputs;
  c10::SymInt num_outputs;
  c10::SymInt step_input{1};
  c10::SymInt step_output{1};
  c10::SymInt ctas_per_output{1};
  c10::SymInt input_mult[3] = {0, 0, 0};
  c10::SymInt output_mult[2] = {0, 0};
  bool block_x_reduce = false;
  bool block_y_reduce = false;
  bool global_reduce = false;

  c10::SymInt block_width{1};
  c10::SymInt block_height{1};
  c10::SymInt num_threads{1};

  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void set_block_dimension(const c10::SymInt& dim0, const c10::SymInt& dim1) {
    const int max_num_threads = mnt_wrapper<T>::MAX_NUM_THREADS / output_vec_size;
    // `dim < max ? last_pow2(dim) : max` is min(last_pow2(dim), max) for a
    // power-of-two max: the branch becomes an expression
    c10::SymInt dim0_pow2 = last_pow2_sym(dim0).min(c10::SymInt(max_num_threads));
    c10::SymInt dim1_pow2 = last_pow2_sym(dim1).min(c10::SymInt(max_num_threads));
    block_width = dim0_pow2.min(c10::SymInt(at::cuda::warp_size()));
    block_height = dim1_pow2.min(c10::SymInt(max_num_threads) / block_width);
    block_width = dim0_pow2.min(c10::SymInt(max_num_threads) / block_height);
    num_threads = block_width * block_height;
  }

  c10::SymInt split_input(const c10::SymInt& parallelism) {
    c10::SymInt step = step_input;
    step_input = step_input * parallelism;
    return step;
  }

  c10::SymInt split_output(const c10::SymInt& parallelism) {
    c10::SymInt step = step_output;
    step_output = step_output * parallelism;
    return step;
  }

  Block block() const {
    return Block(block_width, block_height, 1);
  }

  Grid grid() const {
    return Grid(div_up_sym(num_outputs / output_vec_size, step_output), ctas_per_output, 1);
  }

  c10::SymInt shared_memory_size() const {
    if (!block_y_reduce &&
        (!block_x_reduce ||
         block_width <= at::cuda::warp_size())) {
      return 0;
    }
    return c10::SymInt(element_size_bytes) * num_threads * output_vec_size;
  }

  c10::SymInt global_memory_size() const {
    if (!global_reduce) {
      return 0;
    }
    c10::SymInt size = c10::SymInt(element_size_bytes) * num_outputs * ctas_per_output;
    if (!block_x_reduce) {
      size = size * block_width * output_vec_size;
    }
    return size;
  }

  c10::SymInt semaphore_size() const {
    if (!global_reduce) {
      return 0;
    }
    return c10::SymInt(static_cast<int64_t>(sizeof(int))) * grid().x;
  }

  c10::SymInt values_per_thread() const {
    return div_up_sym(num_inputs, step_input);
  }
};

// make_output_calculator / make_input_calculator on the sibling iterator
inline OCT<2> make_output_calculator(const TensorIteratorSym& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  int input_index = iter.ntensors() - 1;
  int output_index = 0;
  std::array<const c10::SymInt*, 2> strides = {
    iter.strides(output_index).data() + num_reduce_dims,
    iter.strides(input_index).data() + num_reduce_dims,
  };
  auto shape = iter.shape().data() + num_reduce_dims;
  return OCT<2>(num_output_dims, shape, strides.data());
}

inline OCT<1> make_input_calculator(const TensorIteratorSym& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int input_index = iter.ntensors() - 1;
  std::array<const c10::SymInt*, 1> strides = {
    iter.strides(input_index).data(),
  };
  return OCT<1>(num_reduce_dims, iter.shape().data(), strides.data());
}

// get_output_vec_size: the input's base alignment, the output extent and the
// input's other strides must each divide the vector size; each test a guard
template <typename scalar_t>
int get_output_vec_size(const TensorIteratorSym& iter) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](const c10::SymInt& n) {
    while (vec_size > 1 && !((n % c10::SymInt(vec_size)) == 0)) {
      vec_size /= 2;
    }
  };

  const c10::SymInt base_address = iter.data_ptr(iter.noutputs());
  while (vec_size > 1 && !aligned(base_address, vec_size * static_cast<int64_t>(sizeof(scalar_t))).guard_bool(__FILE__, __LINE__)) {
    vec_size /= 2;
  }

  const int output_index = iter.num_reduce_dims();
  update_vec_size(iter.shape()[output_index]);

  int j = 0;
  for (const auto& i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_vec_size(i / static_cast<int64_t>(sizeof(scalar_t)));
    }
    j++;
  }
  return vec_size;
}

template<typename arg_t, typename scalar_t, int vt0, int input_vec_size=vt0>
ReduceConfigSym setReduceConfig(const TensorIteratorSym& iter){
  // Start by assuming that each thread handles a single output and all
  // the inputs for that output.
  c10::SymInt num_outputs = iter.num_output_elements();
  c10::SymInt inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config = ReduceConfigSym(sizeof(arg_t), num_outputs, inputs_per_output);

  c10::SymInt dim0;
  c10::SymInt dim1;
  c10::SymInt fastest_moving_stride;
  bool reduction_on_fastest_striding_dimension;

  if (iter.ndim() > 0) {
    // Adjust block size to map block width to fastest changing dimension of input
    // tensor. This grants the best possible memory accessing pattern, given that
    // for non-contiguous tensor with space in between, we cannot have perfect
    // memory coalescing.
    reduction_on_fastest_striding_dimension =
        (iter.num_reduce_dims() == iter.ndim()) ||
        (iter.strides(/*arg=*/input_index)[0] <
        iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()]);
    // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
    // dim0 & dim1 are more like the upper bound of the block dimension. The
    // actual launch config and reduction scheme is determined by setting values
    // to `config.input_mult` and `config.output_mult`.
    // We try to max out dim1 so that we have enough threads per CTA to deliver
    // performance for larger problem size.
    if (reduction_on_fastest_striding_dimension) {
      // Map block.x to the fastest reducing dimension. It implies:
      //   1. block_x_reduce is required.
      //   2. block.y now max out to num_outputs.
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[0];
    } else {
      // Map block.x to the fastest non reducing dimension. It implies:
      //   1. block_x_reduce is turned off.
      //   2. block.y now max out to inputs_per_output.
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()];
    }
  } else {
    reduction_on_fastest_striding_dimension = true;
    fastest_moving_stride = static_cast<int64_t>(sizeof(scalar_t));
    dim0 = 1;
    dim1 = 1;
  }

  // We do vectorization to gain better memory access, there are two cases which we call
  // "vectorize along input" and "vectorize along output". Note that the "input/output"
  // here does not mean we are vectorizing load/store instructions. We always only vectorize
  // load instructions.
  //
  // Case 1: "vectorize along input"
  // This case happens when we are reducing along fastest moving dimension. In such case, threads
  // with the same threadIdx.y works on the same reduction cooperatively and will produce results
  // for the same output. In such case, values in each loaded vector always correspond to the same output.
  //
  // Case 2: "vectorize along output"
  // This case happens when the fastest moving dimension is not the dimension of reduction. In such case,
  // threads with different threadIdx.x are independent and will produce results for different outputs.
  // In such case, values in each loaded vector always correspond to different outputs.
  if (fastest_moving_stride == static_cast<int64_t>(sizeof(scalar_t))) {
    if (reduction_on_fastest_striding_dimension && dim0 >= 128 && iter.num_reduce_dims() == 1) {
      // Case 1: "vectorize along input"
      // Note that if vt0 < ReduceConfig::vec_size, then this means the register pressure could be high, in such case,
      // we should avoid vectorization.
      config.vectorize_input = true;
      dim0 = dim0 / input_vec_size;
    } else if (!reduction_on_fastest_striding_dimension) {
      // Case 2: "vectorize along output"
      config.output_vec_size = get_output_vec_size<scalar_t>(iter);
      dim0 = dim0 / config.output_vec_size;
    }
  }

  // Adjust block_width and block_height
  config.set_block_dimension<scalar_t>(dim0, dim1);

  c10::SymInt block_width = config.block_width;
  c10::SymInt block_height = config.block_height;

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions and shared memory (if block_width > C10_WARP_SIZE).
    config.input_mult[0] = config.split_input(block_width);
    config.block_x_reduce = true;
  } else {
    // Otherwise split the output across lanes in a warp.
    config.output_mult[0] = config.split_output(block_width);
  }

#ifdef USE_ROCM
  constexpr int min_values_per_thread = 128;
#else
  constexpr int min_values_per_thread = 16;
#endif
  constexpr int max_values_per_thread = 256;

  const c10::SymInt warp_split_threshold =
      (block_height * 16).min(c10::SymInt(max_values_per_thread));
  bool split_across_warps = config.values_per_thread() >= warp_split_threshold;
  const int num_mp =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (split_across_warps) {
    // Divide the input across warps in a thread-block, if that leaves at least
    // 16 elements to be summed by each thread. This will require inter-warp
    // reduction using shared memory.
    config.input_mult[1] = config.split_input(block_height);
    config.block_y_reduce = true;
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(block_height);
  }

  int max_threads_per_mp =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor;
  const c10::SymInt blocks_per_sm = c10::SymInt(max_threads_per_mp) / config.num_threads;
  const c10::SymInt target_grid_size = c10::SymInt(num_mp) * blocks_per_sm;
  c10::SymInt grid = config.grid().x;
  if (config.block_y_reduce && config.values_per_thread() >= max_values_per_thread && grid <= target_grid_size) {
    // Divide the input across thread-blocks if the amount of work per-thread
    // is large enough and the size of the output is small enough. This will
    // require a reduction using global memory.
    // If we decide to split input across blocks, as long as we can get enough
    // number of blocks (`target_grid_size`) to balance SM, we should still
    // make the number of values per thread large for best performance.
    c10::SymInt ctas_per_output1 = div_up_sym(target_grid_size, grid);
    c10::SymInt ctas_per_output2 = div_up_sym(config.values_per_thread(), min_values_per_thread);
    c10::SymInt ctas_per_output3 = div_up_sym(config.values_per_thread(), max_values_per_thread);
    // We want the minimum of ctas_per_output1 and ctas_per_output2, so that each thread can have
    // a large number of values to deal with. But we don't want values_per_thread to be larger than
    // max_values_per_thread
    // std::clamp<int>(c1, c3, c2) with c3 <= c2, as an expression
    config.ctas_per_output = ctas_per_output3.max(ctas_per_output1.min(ctas_per_output2));
    if (config.ctas_per_output > 1) {
#ifdef USE_ROCM
      // Set min ctas value as 64. Having more reductions (i.e less values_per_thread) seems to improve perf.
      config.ctas_per_output = config.ctas_per_output.max(c10::SymInt(64));
#endif
      config.input_mult[2] = config.split_input(config.ctas_per_output);
      config.global_reduce = true;
    }
  }
  return config;
}

// ---- the proxy over ReduceOp: views over the real type's layout (offsetof
// on the ATen struct), one hand-written template for every instantiation
struct ReduceConfigView {
  using C = at::native::ReduceConfig;
  TracedBase* o;
  size_t base;
  SlotName nm;
  IntSlot<int> element_size_bytes;
  IntSlot<int> num_inputs;
  IntSlot<int> num_outputs;
  IntSlot<int> step_input;
  IntSlot<int> step_output;
  IntSlot<int> ctas_per_output;
  ArrayOf<IntSlot<int>, sizeof(int), 3> input_mult;
  ArrayOf<IntSlot<int>, sizeof(int), 2> output_mult;
  IntSlot<int> block_width;
  IntSlot<int> block_height;
  IntSlot<int> num_threads;
  IntSlot<bool> vectorize_input;
  IntSlot<int> output_vec_size;
  ReduceConfigView(TracedBase* o, size_t base, SlotName nm)
      : o(o), base(base), nm(nm),
        element_size_bytes(o, base + offsetof(C, element_size_bytes), SlotName{&this->nm, "element_size_bytes"}),
        num_inputs(o, base + offsetof(C, num_inputs), SlotName{&this->nm, "num_inputs"}),
        num_outputs(o, base + offsetof(C, num_outputs), SlotName{&this->nm, "num_outputs"}),
        step_input(o, base + offsetof(C, step_input), SlotName{&this->nm, "step_input"}),
        step_output(o, base + offsetof(C, step_output), SlotName{&this->nm, "step_output"}),
        ctas_per_output(o, base + offsetof(C, ctas_per_output), SlotName{&this->nm, "ctas_per_output"}),
        input_mult(o, base + offsetof(C, input_mult), SlotName{&this->nm, "input_mult"}),
        output_mult(o, base + offsetof(C, output_mult), SlotName{&this->nm, "output_mult"}),
        block_width(o, base + offsetof(C, block_width), SlotName{&this->nm, "block_width"}),
        block_height(o, base + offsetof(C, block_height), SlotName{&this->nm, "block_height"}),
        num_threads(o, base + offsetof(C, num_threads), SlotName{&this->nm, "num_threads"}),
        vectorize_input(o, base + offsetof(C, vectorize_input), SlotName{&this->nm, "vectorize_input"}),
        output_vec_size(o, base + offsetof(C, output_vec_size), SlotName{&this->nm, "output_vec_size"}) {}
  ReduceConfigView(const ReduceConfigView&) = delete;
  ReduceConfigView& operator=(const ReduceConfigSym& s) {
    element_size_bytes = s.element_size_bytes;
    num_inputs = s.num_inputs;
    num_outputs = s.num_outputs;
    step_input = s.step_input;
    step_output = s.step_output;
    ctas_per_output = s.ctas_per_output;
    for (int i = 0; i < 3; ++i) {
      input_mult[i] = s.input_mult[i];
    }
    for (int i = 0; i < 2; ++i) {
      output_mult[i] = s.output_mult[i];
    }
    block_width = s.block_width;
    block_height = s.block_height;
    num_threads = s.num_threads;
    vectorize_input = s.vectorize_input;
    output_vec_size = s.output_vec_size;
    return *this;
  }
};

// The ops functor inside the proxy. A functor without a runtime member is a
// constant of the variant (copied, not recorded); MeanOps carries its factor.
template <class Ops>
struct OpsView {
  TracedBase* o;
  size_t base;
  SlotName nm;
  OpsView(TracedBase* o, size_t base, SlotName nm) : o(o), base(base), nm(nm) {}
  OpsView(const OpsView&) = delete;
  OpsView& operator=(const Ops& s) {
    new (static_cast<char*>(o->pod) + base) Ops(s);
    return *this;
  }
};

// MeanOps::factor is written as the bit pattern of the float the real host
// computes (an opaque rebind of the two counts), so the bytes are exact.
template <class scalar_t, class acc_t, class factor_t, class out_t>
struct OpsView<at::native::MeanOps<scalar_t, acc_t, factor_t, out_t>> {
  using Ops = at::native::MeanOps<scalar_t, acc_t, factor_t, out_t>;
  using bits_t = std::conditional_t<sizeof(factor_t) == 8, int64_t, int32_t>;
  static_assert(sizeof(factor_t) == 4 || sizeof(factor_t) == 8, "MeanOps factor width");
  TracedBase* o;
  size_t base;
  SlotName nm;
  IntSlot<bits_t> factor;
  OpsView(TracedBase* o, size_t base, SlotName nm)
      : o(o), base(base), nm(nm), factor(o, base + offsetof(Ops, factor), SlotName{&this->nm, "factor"}) {}
  OpsView(const OpsView&) = delete;
  OpsView& operator=(const Ops& s) {
    new (static_cast<char*>(o->pod) + base) Ops(s);
    return *this;
  }
};

} // namespace at::cuda::host_trace::ti

namespace at::cuda::host_trace {

template <class scalar_t, class ops_t, class out_t, int vt0, int input_vec_size>
struct Traced<at::native::ReduceOp<scalar_t, ops_t, uint32_t, out_t, vt0, input_vec_size>> : TracedBase {
  using P = at::native::ReduceOp<scalar_t, ops_t, uint32_t, out_t, vt0, input_vec_size>;
  using arg_t = typename P::arg_t;
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::OpsView<ops_t> ops;
  ti::ReduceConfigView config;
  ti::OffsetCalculatorView<1> input_calc;
  ti::OffsetCalculatorView<2> output_calc;
  ti::PtrSlot src;
  ti::ArrayOf<ti::PtrSlot, sizeof(char*), 2> dst;
  ti::PtrSlot acc_buf;
  ti::PtrSlot cta_buf;
  ti::PtrSlot semaphores;
  ti::IntSlot<int64_t> base_idx;
  ti::IntSlot<bool> accumulate;
  ti::IntSlot<bool> final_output;
  ti::IntSlot<int> noutputs;
  Traced()
      : TracedBase(pod_bytes, sizeof(P)),
        ops(this, offsetof(P, ops), ti::SlotName{nullptr, "ops"}),
        config(this, offsetof(P, config), ti::SlotName{nullptr, "config"}),
        input_calc(this, offsetof(P, input_calc), ti::SlotName{nullptr, "input_calc"}),
        output_calc(this, offsetof(P, output_calc), ti::SlotName{nullptr, "output_calc"}),
        src(this, offsetof(P, src), ti::SlotName{nullptr, "src"}),
        dst(this, offsetof(P, dst), ti::SlotName{nullptr, "dst"}),
        acc_buf(this, offsetof(P, acc_buf), ti::SlotName{nullptr, "acc_buf"}),
        cta_buf(this, offsetof(P, cta_buf), ti::SlotName{nullptr, "cta_buf"}),
        semaphores(this, offsetof(P, semaphores), ti::SlotName{nullptr, "semaphores"}),
        base_idx(this, offsetof(P, base_idx), ti::SlotName{nullptr, "base_idx"}),
        accumulate(this, offsetof(P, accumulate), ti::SlotName{nullptr, "accumulate"}),
        final_output(this, offsetof(P, final_output), ti::SlotName{nullptr, "final_output"}),
        noutputs(this, offsetof(P, noutputs), ti::SlotName{nullptr, "noutputs"}) {}
  // the identity is a constant of the variant: its bytes, no record
  void set_ident(arg_t v) {
    std::memcpy(pod_bytes + offsetof(P, ident), &v, sizeof(arg_t));
  }
};

} // namespace at::cuda::host_trace

namespace at::cuda::host_trace::ti {

template<int max_threads, typename R>
void launch_reduce_kernel(const ReduceConfigSym& config, const Traced<R>& reduction) {
  Block block = config.block();
  Grid grid = config.grid();

  auto stream = at::cuda::getCurrentCUDAStream();
  c10::SymInt shared_memory = config.shared_memory_size();

  switch(config.output_vec_size) {
  case 4:
    launch(at::native::reduce_kernel<max_threads / 4, 4, R>, grid, block, shared_memory, stream, reduction);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  case 2:
    launch(at::native::reduce_kernel<max_threads / 2, 2, R>, grid, block, shared_memory, stream, reduction);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    break;
  default:
    launch(at::native::reduce_kernel<max_threads / 1, 1, R>, grid, block, shared_memory, stream, reduction);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

struct NoFill {
  template <class T>
  void operator()(T&) const {}
};

// Reduce.cuh's gpu_reduce_kernel for one output and one input of the same
// dtype under 32-bit indexing (the split declines). `fill_ops` records the
// ops functor's runtime members on the proxy (MeanOps::factor).
template <typename scalar_t, typename out_scalar_t, int vt0=4, int input_vec_size=vt0, typename ops_t, typename ident_t=double, typename FillOps = NoFill>
inline void gpu_reduce_kernel(TensorIteratorSym& iter, const ops_t& ops, ident_t ident=0, const FillOps& fill_ops = FillOps()) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() - iter.noutputs() == 1 && iter.noutputs() == 1);
  // the result is allocated (make_reduction): the index width, the block and
  // split configuration, its scratch and the launch are the kernel choice
  KernelChoice choice;

  using traits = ::function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename traits::template arg<0>::type;

  if (!iter.can_use_32bit_indexing()) {
    iter.with_32bit_indexing(); // declines
  }
  // acc_buf: only for a reduction that cannot accumulate in its output AND
  // needs the 64-bit split; the split declines above, so never here

  const c10::SymInt in_data = iter.data_ptr(iter.ntensors() - 1);
  const c10::SymInt out_data = iter.data_ptr(0);

  ReduceConfigSym config = setReduceConfig<arg_t, scalar_t, vt0, input_vec_size>(iter);
  Tensor buffer;
  Tensor semaphores;
  c10::SymInt buffer_ptr(0);
  c10::SymInt semaphores_ptr(0);
  if (config.global_reduce) {
    // the real host takes two caching-allocator blocks; here they are byte
    // tensors, so the trace records them as allocations the replay binds
    const auto options = iter.tensor(0).options().dtype(kByte);
    buffer = at::empty_symint({config.global_memory_size()}, options);
    semaphores = at::empty_symint({config.semaphore_size()}, options);
    buffer_ptr = sym_mutable_data_ptr(buffer);
    semaphores_ptr = sym_mutable_data_ptr(semaphores);
    auto stream = at::cuda::getCurrentCUDAStream();
    memset_async(semaphores_ptr, 0, config.semaphore_size(), stream);
  }

  auto output_calc = make_output_calculator(iter);
  auto input_calc = make_input_calculator(iter);
  using R = at::native::ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t, vt0, input_vec_size>;
  Traced<R> reduce;
  reduce.ops = ops;
  fill_ops(reduce);
  reduce.set_ident(static_cast<arg_t>(ident));
  reduce.config = config;
  reduce.input_calc = input_calc;
  reduce.output_calc = output_calc;
  reduce.src = in_data;
  reduce.dst[0] = out_data;
  reduce.dst[1] = nullptr;
  reduce.acc_buf = nullptr;
  reduce.cta_buf = buffer_ptr;
  reduce.semaphores = semaphores_ptr;
  reduce.base_idx = 0;
  reduce.accumulate = false; // iter.should_accumulate() of the top-level iterator
  reduce.final_output = true; // iter.is_final_output()
  reduce.noutputs = 1;

  launch_reduce_kernel<mnt_wrapper<scalar_t>::MAX_NUM_THREADS>(config, reduce);
}

// ReduceOps.cpp's make_reduction on the sibling: the result allocated for the
// reduced shape (an allocation the trace records) or, for the out= form, the
// caller's tensor of that shape and dtype (a mismatch declines: the real op
// would resize or cast, which the sibling does not trace), viewed with a
// size-1, stride-0 dim at every reduced position (review_reduce_result), and
// the reduce_op iterator over it. dims=[] reduces every dim, as dim=None does.
struct ReductionSym {
  Tensor result;
  TensorIteratorSym iter;
};

inline ReductionSym make_reduction(const Tensor& self, IntArrayRef dims_in, bool keepdim, ScalarType out_dtype, const std::optional<Tensor>& out = std::nullopt) {
  const int64_t ndim = self.dim();
  c10::DimVector dims(dims_in.begin(), dims_in.end());
  at::maybe_wrap_dims(dims, ndim);
  // a 0-d input accepts dim 0 and -1 (both wrap to 0, as dim_list_to_bitset's
  // 64-bit mask does); the mask keeps one slot for it, the shape loops below
  // run over ndim and see none of it
  std::vector<bool> mask(static_cast<size_t>(ndim == 0 ? 1 : ndim), dims.empty());
  for (const int64_t d : dims) {
    TORCH_CHECK(!mask[d], "dim ", d, " appears multiple times in the list of dims");
    mask[d] = true;
  }
  const auto sizes = self.sym_sizes();
  c10::SymDimVector shape;
  for (const auto i : c10::irange(ndim)) {
    if (!mask[i]) {
      shape.push_back(sizes[i]);
    } else if (keepdim) {
      shape.emplace_back(1);
    }
  }
  Tensor result;
  if (out.has_value()) {
    if (!(out->sym_sizes() == c10::SymIntArrayRef(shape)) || out->scalar_type() != out_dtype || out->device() != self.device()) {
      decline(c10::str("host_trace: the out= tensor of a reduction has shape ", out->sym_sizes(), " and dtype ", out->scalar_type(), ", the result has ", c10::SymIntArrayRef(shape), " and ", out_dtype, " (declined)"));
    }
    result = *out;
  } else {
    result = at::empty_symint(shape, self.options().dtype(out_dtype));
  }
  Tensor viewed = result;
  if (!keepdim) {
    c10::SymDimVector vshape;
    c10::SymDimVector vstride;
    const auto rsizes = result.sym_sizes();
    const auto rstrides = result.sym_strides();
    size_t k = 0;
    for (const auto i : c10::irange(ndim)) {
      if (mask[i]) {
        vshape.emplace_back(1);
        vstride.emplace_back(0);
      } else {
        vshape.push_back(rsizes[k]);
        vstride.push_back(rstrides[k]);
        ++k;
      }
    }
    viewed = result.as_strided_symint(vshape, vstride);
  }
  return ReductionSym{result, TensorIteratorSym::reduce_op(viewed, self)};
}

} // namespace at::cuda::host_trace::ti
