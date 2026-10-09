// gpu_kernel and gpu_kernel_nocast for a traced host: CUDALoops.cuh's routes
// over a TensorIteratorSym, recording the launch of the kernel eager launches.
#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/host_trace/Recorder.h>
#include <ATen/cuda/host_trace/TensorIteratorSym.h>
#include <ATen/native/cuda/Loops.cuh>

#include <limits>
#include <optional>
#include <utility>

namespace at::cuda::host_trace {

// memory::can_vectorize_up_to on a symbolic address
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

// IntDivider<unsigned int>(d): shift = ceil(log2(d)); m1 is written as the
// int32 with m1's bits, since the launch packs 4-byte fields signed
template <class K>
void set_divider(Recorder& rec, Param<K>& p, detail::IntDivider<unsigned int>& dst, const c10::SymInt& d) {
  const c10::SymInt shift = rec.bit_length(d - 1);
  const int64_t two32 = int64_t(1) << 32;
  const c10::SymInt m1 = two32 * (rec.pow2(shift) - d) / d + 1;
  p.set(dst.divisor, d);
  p.set(dst.m1, m1 - m1 / (int64_t(1) << 31) * two32);
  p.set(dst.shift, shift);
}

template <typename func_t, size_t N, typename loader_t, typename storer_t>
void launch_unrolled_kernel(Recorder& rec, const c10::SymInt& n, const func_t& f, const std::array<c10::SymInt, N>& data, const Param<loader_t>& loader, const Param<storer_t>& storer) {
  using array_t = std::array<char*, N>;
  using ic_t = TrivialOffsetCalculator<N - 1>;
  using oc_t = TrivialOffsetCalculator<1>;
  Param<array_t> ptrs;
  for (const auto i : c10::irange(N)) {
    ptrs.set(ptrs.value()[i], data[i]);
  }
  constexpr int64_t bws = at::native::elementwise_block_work_size();
  auto kernel = &at::native::unrolled_elementwise_kernel<func_t, array_t, at::native::elementwise_thread_work_size(), ic_t, oc_t, loader_t, storer_t>;
  launch(rec, kernel, (n + bws - 1) / bws, num_threads(), 0, scalar_param<int>(n), Param<func_t>(f), ptrs, Param<ic_t>(ic_t()), Param<oc_t>(oc_t()), loader, storer);
}

// Keep in sync with launch_vectorized_kernel in CUDALoops.cuh.
template <typename func_t, size_t N>
void launch_vectorized_kernel(Recorder& rec, const c10::SymInt& n, const func_t& f, const std::array<c10::SymInt, N>& data) {
  using traits = function_traits<func_t>;
  using cpp_type = typename traits::result_type;
  using array_t = std::array<char*, N>;
  const cudaDeviceProp* p = at::cuda::getDeviceProperties(at::cuda::getCurrentCUDAStream().device().index());
  constexpr int64_t min_sm107_io_size = 16 * 1024 * 1024;
  if (p->major == 10 && p->minor == 7 && n * int64_t(at::native::calc_io_size<func_t>()) >= min_sm107_io_size) {
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
  Param<int> numel = scalar_param<int>(n);
  Param<func_t> fn(f);
  Param<array_t> ptrs;
  for (const auto i : c10::irange(N)) {
    ptrs.set(ptrs.value()[i], data[i]);
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
      using at::native::memory::LoadWithoutCast;
      using at::native::memory::StoreWithoutCast;
      return launch_unrolled_kernel(rec, n, f, data, Param<LoadWithoutCast>(LoadWithoutCast()), Param<StoreWithoutCast>(StoreWithoutCast()));
    }
  }
}

// OffsetCalculator<N>(dims, sizes, strides), byte strides
template <class K, int N>
void set_offset_calculator(Recorder& rec, Param<K>& p, ::OffsetCalculator<N>& calc, int dims, const c10::SymInt* sizes, const c10::SymInt* const* strides) {
  TORCH_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
  p.set(calc.dims, dims);
  for (const auto i : c10::irange(dims)) {
    set_divider(rec, p, calc.sizes_[i], sizes[i]);
    for (const auto arg : c10::irange(N)) {
      p.set(calc.strides_[i][arg], strides[arg][i]);
    }
  }
}

// make_offset_calculator<N>(iter)
template <class K, int N>
void set_offset_calculator(Recorder& rec, Param<K>& p, ::OffsetCalculator<N>& calc, const TensorIteratorSym& iter) {
  std::array<const c10::SymInt*, N> strides;
  for (const auto i : c10::irange(N)) {
    strides[i] = iter.strides(i).data();
  }
  set_offset_calculator(rec, p, calc, iter.ndim(), iter.shape().data(), strides.data());
}

// needs_dynamic_casting<func_t>::check(iter) on a TensorIteratorSym
template <typename func_t>
bool needs_dynamic_casting(const TensorIteratorSym& iter) {
  using traits = function_traits<func_t>;
  bool cast = iter.dtype(0) != c10::CppTypeToScalarType<typename traits::result_type>::value;
  [&]<size_t... I>(std::index_sequence<I...>) {
    ((cast |= iter.dtype(I + 1) != c10::CppTypeToScalarType<typename traits::template arg<I>::type>::value), ...);
  }(std::make_index_sequence<traits::arity>{});
  return cast;
}

// Loops.cuh's gpu_kernel checks, then gpu_kernel_impl's data array; nullopt
// for an empty iterator
template <typename func_t>
std::optional<std::array<c10::SymInt, function_traits<func_t>::arity + 1>> kernel_data(const TensorIteratorSym& iter) {
  TORCH_INTERNAL_ASSERT(iter.ninputs() == function_traits<func_t>::arity);
  if (iter.numel() == 0) {
    return std::nullopt;
  }
  if (!iter.can_use_32bit_indexing()) {
    decline("an iterator beyond 32-bit indexing");
  }
  std::array<c10::SymInt, function_traits<func_t>::arity + 1> data;
  for (const auto i : c10::irange(data.size())) {
    data[i] = iter.data_ptr(i);
  }
  return data;
}

template <typename func_t>
void gpu_kernel_nocast(Recorder& rec, const TensorIteratorSym& iter, const func_t& f) {
  using arg0_t = typename function_traits<func_t>::result_type;
  constexpr int ntensors = function_traits<func_t>::arity + 1;
  const auto data = kernel_data<func_t>(iter);
  if (!data) {
    return;
  }
  const c10::SymInt numel = iter.numel();
  if (iter.is_contiguous()) {
    return launch_vectorized_kernel(rec, numel, f, *data);
  }
  using op_t = at::native::StridedOp<func_t, ntensors>;
  Param<op_t> op;
  op_t& body = op.value();
  for (const auto i : c10::irange(ntensors)) {
    op.set(body.data[i], (*data)[i]);
  }
  set_offset_calculator(rec, op, body.offset_calc, iter);
  body.f = f;
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  constexpr int64_t work = 128 * unroll_factor;
  launch(rec, &at::native::elementwise_kernel<128, unroll_factor, op_t>, (numel + work - 1) / work, 128, 0, scalar_param<int>(numel), op);
}

// gpu_kernel_impl's dynamic-cast route where an operand's dtype is not f's
template <typename func_t>
void gpu_kernel(Recorder& rec, const TensorIteratorSym& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>(iter)) {
    return gpu_kernel_nocast(rec, iter, f);
  }
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;
  const auto data = kernel_data<func_t>(iter);
  if (!data) {
    return;
  }
  const c10::SymInt numel = iter.numel();
  if (iter.is_contiguous()) {
    using loader_t = at::native::memory::LoadWithCast<traits::arity>;
    using storer_t = at::native::memory::StoreWithCast<1>;
    Param<loader_t> loader;
    for (const auto i : c10::irange(ntensors - 1)) {
      loader.value().dtypes[i] = iter.dtype(i + 1);
      loader.value().element_sizes[i] = iter.element_size(i + 1);
    }
    Param<storer_t> storer;
    storer.value().dtypes[0] = iter.dtype(0);
    storer.value().element_sizes[0] = iter.element_size(0);
    return launch_unrolled_kernel(rec, numel, f, *data, loader, storer);
  }
  using op_t = at::native::StridedCastOp<func_t, ntensors>;
  Param<op_t> op;
  op_t& body = op.value();
  for (const auto i : c10::irange(ntensors)) {
    op.set(body.data[i], (*data)[i]);
    body.dtypes[i] = iter.dtype(i);
  }
  set_offset_calculator(rec, op, body.offset_calc, iter);
  body.f = f;
  constexpr int64_t work = 128 * 4;
  launch(rec, &at::native::elementwise_kernel<128, 4, op_t>, (numel + work - 1) / work, 128, 0, scalar_param<int>(numel), op);
}

} // namespace at::cuda::host_trace
