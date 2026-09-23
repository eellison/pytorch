// The traced sibling of DistributionTemplates.h's distribution_nullary_kernel:
// the same host (calc_execution_policy's grid and counter offset, the default
// generator's philox state, the trivial-1d or strided store) on the
// SymInt-typed sibling iterator, launching eager's kernel instantiation
// through the typed helper with the store's proxy (out_data, stride0 or the
// offset calculator as slots) and the philox state as an `rng` field. The
// philox increment is declared with rng_increment, so the tape carries it as
// an expression of the element count and a replay advances the generator as
// eager does at every shape. Eager's per-distribution hosts
// (templates::cuda) are templates over the iterator type and reach the
// overload below by argument-dependent lookup; each entry (the host-tracing
// section of its Distribution*.cu) builds the sibling iterator and calls
// eager's host with it.
#pragma once
#include <ATen/cuda/host_trace/Launch.h>
#include <ATen/cuda/host_trace/Philox.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>
#include <ATen/cuda/host_trace/ti/TensorIteratorSym.h>
#include <ATen/native/cuda/DistributionTemplates.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstddef>
#include <mutex>
#include <optional>

namespace at::cuda::host_trace {

// The store functors of distribution_nullary_kernel: the output pointer and
// its stride (or offset calculator) are slots, the distribution's own
// transform a constant of the variant (its members are the op's numbers, as
// eager bakes them into the closure).
template <class scalar_t, class accscalar_t, class F>
struct Traced<at::native::DistributionTrivialStore<scalar_t, accscalar_t, F>> : TracedBase {
  using P = at::native::DistributionTrivialStore<scalar_t, accscalar_t, F>;
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::PtrSlot out_data;
  ti::IntSlot<int> stride0;
  ti::FunctorView<F> transform_func;
  Traced()
      : TracedBase(pod_bytes, sizeof(P)),
        out_data(this, offsetof(P, out_data), ti::SlotName{nullptr, "out_data"}),
        stride0(this, offsetof(P, stride0), ti::SlotName{nullptr, "stride0"}),
        transform_func(this, offsetof(P, transform_func), ti::SlotName{nullptr, "transform_func"}) {}
};

template <class scalar_t, class accscalar_t, class F>
struct Traced<at::native::DistributionStridedStore<scalar_t, accscalar_t, F>> : TracedBase {
  using P = at::native::DistributionStridedStore<scalar_t, accscalar_t, F>;
  alignas(P) unsigned char pod_bytes[sizeof(P)] = {};
  ti::OffsetCalculatorView<1> offset_calc;
  ti::PtrSlot out_data;
  ti::FunctorView<F> transform_func;
  Traced()
      : TracedBase(pod_bytes, sizeof(P)),
        offset_calc(this, offsetof(P, offset_calc), ti::SlotName{nullptr, "offset_calc"}),
        out_data(this, offsetof(P, out_data), ti::SlotName{nullptr, "out_data"}),
        transform_func(this, offsetof(P, transform_func), ti::SlotName{nullptr, "transform_func"}) {}
};

} // namespace at::cuda::host_trace

namespace at::cuda::host_trace::ti {

// TensorIterator::borrowing_nullary_op on the sibling: one output, no dtype
// check, no resize, the memory-overlap check on (an expanded self is refused
// with eager's text).
inline TensorIteratorSym nullary_iterator(const Tensor& self) {
  TensorIteratorSymConfig config;
  config.check_all_same_dtype_ = false;
  config.resize_outputs_ = false;
  TensorIteratorSym iter;
  iter.add_output(self);
  iter.build(config);
  return iter;
}

// The generator a distribution draws from: the default CUDA generator of
// self's device, whose philox offset the tape's rng slot advances. Another
// generator declines by name: its offset is no value of the tape.
inline at::CUDAGeneratorImpl* distribution_generator(const char* op, const Tensor& self, const std::optional<at::Generator>& gen) {
  auto* dflt = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator(self.device().index()));
  if (gen.has_value() && gen->defined() && gen->unsafeGetGeneratorImpl() != dflt) {
    decline(c10::str("host_trace: ", op, " with a generator other than the default CUDA generator is not traced (declined)"));
  }
  return dflt;
}

// CHECK_EMPTY_AND_RETURN of ATen/native/DistributionTemplates.h: a guard
inline bool is_empty(const Tensor& self) {
  return self.sym_numel().sym_eq(0).guard_bool(__FILE__, __LINE__);
}

// distribution_nullary_kernel (ATen/native/cuda/DistributionTemplates.h) on
// the sibling iterator: the entries return before this on an empty tensor,
// as eager's impl_ functions do; the 64-bit indexing split declines.
template<typename scalar_t,
         typename accscalar_t,
         typename dist_func_return_t,
         typename RNG,
         typename dist_t,
         typename transform_t>
void distribution_nullary_kernel(TensorIteratorSym& iter,
                                 RNG gen,
                                 const dist_t& dist_func,
                                 const transform_t transform_func) {
  constexpr int unroll_factor = sizeof(dist_func_return_t) / sizeof(accscalar_t);
  static_assert(unroll_factor >= 1, "unroll_factor must be >= 1.");
  // everything from here on picks the launch configuration
  KernelChoice choice;
  const c10::SymInt numel = iter.numel();
  if (!iter.can_use_32bit_indexing()) {
    iter.with_32bit_indexing(); // declines
  }
  // calc_execution_policy on the symbolic count
  const int64_t block_size = at::native::block_size_bound;
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const int64_t blocks_per_sm = props->maxThreadsPerMultiProcessor / block_size;
  const c10::SymInt grid_cap(static_cast<int64_t>(props->multiProcessorCount) * blocks_per_sm);
  const c10::SymInt grid_x = ((numel + (block_size - 1)) / block_size).min(grid_cap);
  const c10::SymInt counter_offset = ((numel - 1) / (block_size * grid_x * unroll_factor) + 1) * static_cast<int64_t>(at::native::max_generator_offsets_per_curand_call);
  const int64_t increment = rng_increment(counter_offset);
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(increment);
  }
  const c10::SymInt out_data = iter.data_ptr(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  Grid grid(grid_x);
  const Block block{c10::SymInt(block_size)};
  TracedPhilox philox(rng_engine_inputs);
  if (iter.ndim() == 1) { // is_trivial_1d
    using store_t = at::native::DistributionTrivialStore<scalar_t, accscalar_t, transform_t>;
    Traced<store_t> store;
    store.out_data = out_data;
    store.stride0 = iter.strides(0)[0];
    store.transform_func = transform_func;
    launch(at::native::distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor, dist_t, store_t>,
           grid, block, 0, stream, numel, philox, dist_func, store);
  } else {
    using store_t = at::native::DistributionStridedStore<scalar_t, accscalar_t, transform_t>;
    Traced<store_t> store;
    store.offset_calc = make_offset_calculator<1>(iter);
    store.out_data = out_data;
    store.transform_func = transform_func;
    launch(at::native::distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor, dist_t, store_t>,
           grid, block, 0, stream, numel, philox, dist_func, store);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace at::cuda::host_trace::ti
