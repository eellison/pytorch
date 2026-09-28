// gpu_kernel for a traced host: CUDALoops.cuh's same-dtype route
// (gpu_kernel_impl_nocast) over a TensorIteratorSym, recording the launch of
// the kernel eager launches. Each host decision is eager's, a guard where it
// reads a size or an address.
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <ATen/cuda/host_trace/TensorIteratorSym.h>
#include <ATen/native/cuda/Loops.cuh>

#include <limits>
#include <utility>

namespace at::cuda::host_trace {

template <typename scalar_t>
int can_vectorize_up_to(const c10::SymInt& address) {
  using at::native::memory::aligned_vector;
  if (address % int64_t(alignof(aligned_vector<scalar_t, 8>)) == 0) {
    return 8;
  } else if (address % int64_t(alignof(aligned_vector<scalar_t, 4>)) == 0) {
    return 4;
  } else if (address % int64_t(alignof(aligned_vector<scalar_t, 2>)) == 0) {
    return 2;
  }
  return 1;
}

template <typename func_t, size_t N>
int can_vectorize_up_to(const std::array<c10::SymInt, N>& data) {
  using traits = function_traits<func_t>;
  int result = can_vectorize_up_to<typename traits::result_type>(data[0]);
  [&]<size_t... I>(std::index_sequence<I...>) {
    ((result = std::min(result, can_vectorize_up_to<typename traits::template arg<I>::type>(data[I + 1]))), ...);
  }(std::make_index_sequence<traits::arity>{});
  return result;
}

// IntDivider<unsigned int>(d): shift = ceil(log2(d)), and m1 as a signed int32,
// the width the launch packs a 4-byte field at
template <class K>
void set_divider(Recorder& rec, Param<K>& p, detail::IntDivider<unsigned int>& dst, const c10::SymInt& d) {
  const c10::SymInt shift = rec.bit_length(d - 1);
  const int64_t two32 = int64_t(1) << 32;
  const c10::SymInt m1 = two32 * (rec.pow2(shift) - d) / d + 1;
  p.set(dst.divisor, d);
  p.set(dst.m1, m1 - m1 / (int64_t(1) << 31) * two32);
  p.set(dst.shift, shift);
}

template <typename func_t, size_t N>
void launch_vectorized_kernel(Recorder& rec, const c10::SymInt& n, const func_t& f, const std::array<c10::SymInt, N>& data) {
  using traits = function_traits<func_t>;
  using cpp_type = typename traits::result_type;
  using array_t = std::array<char*, N>;
  const cudaDeviceProp* p = at::cuda::getCurrentDeviceProperties();
  if (p->major == 10 && p->minor == 7) {
    decline("the sm_107 work size");
  }
  int vec_size = std::min<int>(16 / sizeof(cpp_type), can_vectorize_up_to<func_t>(data));
  if (p->major != 9 && p->major != 10) {
    vec_size = std::min(vec_size, 4);
  }
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min(vec_size, 4);
  }
#endif
  Param<int> numel = scalar<int>(n);
  Param<func_t> fn(f);
  Param<array_t> ptrs;
  for (const auto i : c10::irange(N)) {
    ptrs.set(ptrs.pod()[i], data[i]);
  }
  constexpr int64_t threads = num_threads();
  constexpr int64_t bws = at::native::elems_per_thread<at::native::calc_io_size<func_t>()>() * threads;
  const c10::SymInt grid = (n + bws - 1) / bws;
  switch (vec_size) {
    case 8:
      return launch(rec, &at::native::vectorized_elementwise_kernel<8, func_t, array_t>, grid, threads, 0, numel, fn, ptrs);
    case 4:
      return launch(rec, &at::native::vectorized_elementwise_kernel<4, func_t, array_t>, grid, threads, 0, numel, fn, ptrs);
    case 2:
      return launch(rec, &at::native::vectorized_elementwise_kernel<2, func_t, array_t>, grid, threads, 0, numel, fn, ptrs);
    default: {
      using ic_t = TrivialOffsetCalculator<traits::arity>;
      using oc_t = TrivialOffsetCalculator<1>;
      using at::native::memory::LoadWithoutCast;
      using at::native::memory::StoreWithoutCast;
      constexpr int64_t ubws = at::native::elementwise_block_work_size();
      auto kernel = &at::native::unrolled_elementwise_kernel<func_t, array_t, at::native::elementwise_thread_work_size(), ic_t, oc_t, LoadWithoutCast, StoreWithoutCast>;
      return launch(
          rec, kernel, (n + ubws - 1) / ubws, threads, 0, numel, fn, ptrs,
          Param<ic_t>(ic_t()), Param<oc_t>(oc_t()), Param<LoadWithoutCast>(LoadWithoutCast()), Param<StoreWithoutCast>(StoreWithoutCast()));
    }
  }
}

template <typename func_t>
void gpu_kernel(Recorder& rec, const TensorIteratorSym& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  if (iter.numel() == 0) {
    return;
  }
  if (!iter.can_use_32bit_indexing()) {
    decline("an iterator beyond 32-bit indexing");
  }
  std::array<c10::SymInt, ntensors> data;
  for (const auto i : c10::irange(ntensors)) {
    data[i] = iter.data_ptr(i);
  }
  const c10::SymInt numel = iter.numel();
  if (iter.is_contiguous()) {
    return launch_vectorized_kernel(rec, numel, f, data);
  }
  TORCH_CHECK(iter.ndim() <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
  using op_t = at::native::StridedOp<func_t, ntensors>;
  Param<op_t> op;
  op_t& body = op.pod();
  for (const auto i : c10::irange(ntensors)) {
    op.set(body.data[i], data[i]);
  }
  op.set(body.offset_calc.dims, iter.ndim());
  for (const auto dim : c10::irange(iter.ndim())) {
    set_divider(rec, op, body.offset_calc.sizes_[dim], iter.shape()[dim]);
    for (const auto arg : c10::irange(ntensors)) {
      op.set(body.offset_calc.strides_[dim][arg], iter.strides(arg)[dim]);
    }
  }
  new (&body.f) func_t(f);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  constexpr int64_t work = 128 * unroll_factor;
  launch(rec, &at::native::elementwise_kernel<128, unroll_factor, op_t>, (numel + work - 1) / work, 128, 0, scalar<int>(numel), op);
}

} // namespace at::cuda::host_trace
