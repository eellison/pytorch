#pragma once
// CUDALoops.cuh's no-cast elementwise path (gpu_kernel, gpu_kernel_impl_nocast,
// launch_vectorized_kernel, launch_legacy_kernel) written once over the host
// policy H (ATen/native/HostPolicy.h). Iter is TensorIterator on EagerHost and
// TensorIteratorSym on SymHost.
//
// All of it runs inside the op's kernel context: the 32-bit check, the
// contiguity test, the vectorization width and the sm107 threshold are kernel
// choice. On SymHost each SymInt comparison is a guard on the hint, recorded
// as a kernel-choice guard for this op. On EagerHost the SymInt overloads are
// not instantiated and the code is CUDALoops.cuh's.
//
// Differences from CUDALoops.cuh: the strided route launches a named StridedOp
// instead of a lambda, because the trace has to name the functor's layout;
// SymHost does not split 64-bit iterators; ROCm and the dynamic-cast route are
// left out.
//
// Preview only: not in any build target and not tested.
#include <ATen/TensorIteratorSym.h>
#include <ATen/native/HostPolicy.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/HostLaunch.cuh>

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

namespace at::native::host {

template <class H>
constexpr bool is_sym_v = std::is_same_v<H, ht::SymHost>;

// TensorIteratorBase's queries. EagerHost calls the iterator's own; the
// TensorIteratorSym overloads are their SymInt twins.
inline int64_t numel(const TensorIteratorBase& iter) {
  return iter.numel();
}
inline c10::SymInt numel(const TensorIteratorSym& iter) {
  c10::SymInt n = 1;
  for (const auto& s : iter.shape()) {
    n *= s;
  }
  return n;
}

inline bool is_contiguous(const TensorIteratorBase& iter) {
  return iter.is_contiguous();
}
inline bool is_contiguous(const TensorIteratorSym& iter) {
  if (numel(iter) == 1) {
    return true;
  }
  if (iter.ndim() != 1) {
    return false;
  }
  for (int i = 0; i < iter.ntensors(); i++) {
    if (iter.strides(i)[0] != static_cast<int64_t>(c10::elementSize(iter.dtype(i)))) {
      return false;
    }
  }
  return true;
}

inline bool can_use_32bit_indexing(const TensorIteratorBase& iter) {
  return iter.can_use_32bit_indexing();
}
inline bool can_use_32bit_indexing(const TensorIteratorSym& iter) {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  if (numel(iter) > max_value) {
    return false;
  }
  for (int i = 0; i < iter.ntensors(); i++) {
    c10::SymInt max_offset = 1;
    for (int dim = 0; dim < iter.ndim(); dim++) {
      max_offset += (iter.shape()[dim] - 1) * iter.strides(i)[dim];
    }
    if (max_offset > max_value) {
      return false;
    }
  }
  return true;
}

inline char* data_ptr(TensorIteratorBase& iter, int arg) {
  return static_cast<char*>(iter.data_ptr(arg));
}
inline c10::SymInt data_ptr(TensorIteratorSym& iter, int arg) {
  return ht::data_ptr(iter.tensor_base(arg));
}

// OffsetCalculator with SymInt sizes and strides. The recorder writes it into
// OffsetCalculator's image; IntDivider's magic number and shift are functions
// of the size evaluated at replay, not guards.
template <int NARGS>
struct OffsetCalculatorSym {
  int dims;
  std::array<c10::SymInt, MAX_DIMS> sizes;
  std::array<std::array<c10::SymInt, std::max<int>(NARGS, 1)>, MAX_DIMS> strides;
};

template <int N>
OffsetCalculator<N> make_offset_calculator(const TensorIteratorBase& iter) {
  return ::make_offset_calculator<N>(iter);
}
template <int N>
OffsetCalculatorSym<N> make_offset_calculator(const TensorIteratorSym& iter) {
  TORCH_CHECK(iter.ndim() <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
  OffsetCalculatorSym<N> oc{iter.ndim(), {}, {}};
  for (int i = 0; i < iter.ndim(); i++) {
    oc.sizes[i] = iter.shape()[i];
    for (int arg = 0; arg < N; arg++) {
      oc.strides[i][arg] = iter.strides(arg)[i];
    }
  }
  return oc;
}

// The lambda gpu_kernel_impl_nocast launches on the strided route, named.
template <typename func_t, int NTENSORS>
struct StridedOp {
  std::array<char*, NTENSORS> data;
  OffsetCalculator<NTENSORS> offset_calc;
  func_t f;
  __device__ void operator()(int idx) const {
    using arg0_t = typename function_traits<func_t>::result_type;
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data[1], &offsets[1], 1);
  }
};

// StridedOp's SymHost twin. The functor's bytes are a constant of the trace.
template <typename func_t, int NTENSORS>
struct StridedOpSym {
  std::array<c10::SymInt, NTENSORS> data;
  OffsetCalculatorSym<NTENSORS> offset_calc;
  func_t f;
};

template <typename func_t, size_t N>
int can_vectorize_up_to(const std::array<char*, N>& data) {
  return memory::can_vectorize_up_to<func_t>(data);
}

// memory::can_vectorize_up_to on SymInt addresses: each alignment test is a
// guard.
template <typename scalar_t>
int can_vectorize_up_to(const c10::SymInt& address) {
  using memory::aligned_vector;
  if (address % std::alignment_of_v<aligned_vector<scalar_t, 8>> == 0) {
    return 8;
  } else if (address % std::alignment_of_v<aligned_vector<scalar_t, 4>> == 0) {
    return 4;
  } else if (address % std::alignment_of_v<aligned_vector<scalar_t, 2>> == 0) {
    return 2;
  }
  return 1;
}
template <typename func_t, size_t N, size_t... I>
int can_vectorize_up_to(const std::array<c10::SymInt, N>& data, std::index_sequence<I...>) {
  using traits = function_traits<func_t>;
  int result = can_vectorize_up_to<typename traits::result_type>(data[0]);
  ((result = std::min(result, can_vectorize_up_to<typename traits::template arg<I>::type>(data[I + 1]))), ...);
  return result;
}
template <typename func_t, size_t N>
int can_vectorize_up_to(const std::array<c10::SymInt, N>& data) {
  return can_vectorize_up_to<func_t>(data, std::make_index_sequence<N - 1>{});
}

// array_t is the kernels' parameter type on both policies; data is
// std::array<char*, N> on EagerHost and std::array<SymInt, N> on SymHost.
template <typename array_t, class H, class Int, typename func_t, typename data_t>
void launch_vectorized_kernel(H& h, Int N, const func_t& f, const data_t& data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  constexpr auto io_size = calc_io_size<func_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  using cpp_type = typename function_traits<func_t>::result_type;
  const uint16_t max_vec_size = can_vectorize_up_to<func_t>(data);
  uint16_t vec_size = 16 / static_cast<uint16_t>(sizeof(cpp_type));
  cudaDeviceProp* p = at::cuda::getDeviceProperties(stream.device().index());
  constexpr int64_t min_sm107_io_size = 16 * 1024 * 1024;
  const bool use_sm107_optimizations =
      p->major == 10 && p->minor == 7 && N * io_size >= min_sm107_io_size;
  if (use_sm107_optimizations) {
    using args_t = typename function_traits<func_t>::ArgsTuple;
    constexpr auto max_input_size = max_of_sizes(args_t{}, std::make_index_sequence<std::tuple_size_v<args_t>>{});
    vec_size = 32 / static_cast<uint16_t>(std::max(sizeof(cpp_type), max_input_size));
  }
  vec_size = std::min<uint16_t>(vec_size, max_vec_size);
  if (p->major != 9 && p->major != 10) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  if constexpr (sizeof(cpp_type) < 2) {
    vec_size = std::min<uint16_t>(vec_size, 4);
  }
#endif
  int tws = elems_per_thread<io_size>();
  constexpr auto input_size = io_size - sizeof(cpp_type);
  constexpr auto tws_128b = elems_per_thread_128b<input_size>(sizeof(cpp_type));
  if (tws_128b >= 8 && use_sm107_optimizations && vec_size == 8) {
    tws = tws_128b;
  }
  int bws = tws * num_threads();
  Int grid = (N + bws - 1) / bws;
  switch (vec_size) {
    case 8:
      if (use_sm107_optimizations) {
        ht::launch<&vectorized_elementwise_kernel<8, func_t, array_t, (tws_128b >= 8)>>(h, grid, num_threads(), 0, N, f, data);
      } else {
        ht::launch<&vectorized_elementwise_kernel<8, func_t, array_t>>(h, grid, num_threads(), 0, N, f, data);
      }
      break;
    case 4:
      ht::launch<&vectorized_elementwise_kernel<4, func_t, array_t>>(h, grid, num_threads(), 0, N, f, data);
      break;
    case 2:
      ht::launch<&vectorized_elementwise_kernel<2, func_t, array_t>>(h, grid, num_threads(), 0, N, f, data);
      break;
    case 1: {
      using input_calc_t = TrivialOffsetCalculator<traits::arity>;
      using output_calc_t = TrivialOffsetCalculator<1>;
      using loader_t = memory::LoadWithoutCast;
      using storer_t = memory::StoreWithoutCast;
      Int grid_unrolled = (N + elementwise_block_work_size() - 1) / elementwise_block_work_size();
      ht::launch<&unrolled_elementwise_kernel<func_t, array_t, elementwise_thread_work_size(), input_calc_t, output_calc_t, loader_t, storer_t>>(
          h, grid_unrolled, num_threads(), 0, N, f, data, input_calc_t(), output_calc_t(), loader_t(), storer_t());
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

// kernel_func_t names the kernel; f is that functor on EagerHost and its twin
// on SymHost.
template <int nt, int vt, typename kernel_func_t, class H, class Int, typename arg_t>
void launch_legacy_kernel(H& h, Int N, const arg_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  Int grid = (N + block.x * vt - 1) / (block.x * vt);
  ht::launch<&elementwise_kernel<nt, vt, kernel_func_t>>(h, grid, block, 0, N, f);
}

template <class H, class Iter, typename func_t>
void gpu_kernel_impl_nocast(H& h, Iter& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;
  using array_t = std::array<char*, ntensors>;
  using ptr_t = std::conditional_t<is_sym_v<H>, c10::SymInt, char*>;

  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  std::array<ptr_t, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = data_ptr(iter, i);
  }
  auto N = numel(iter);
  if (is_contiguous(iter)) {
    return launch_vectorized_kernel<array_t>(h, N, f, data);
  }
  auto offset_calc = make_offset_calculator<ntensors>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  using op_t = std::conditional_t<is_sym_v<H>, StridedOpSym<func_t, ntensors>, StridedOp<func_t, ntensors>>;
  launch_legacy_kernel<128, unroll_factor, StridedOp<func_t, ntensors>>(h, N, op_t{data, offset_calc, f});
}

// Loops.cuh's gpu_kernel for one output and no dynamic cast. Call it inside
// the op's kernel context.
template <class H, class Iter, typename func_t>
void gpu_kernel(H& h, Iter& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_cuda(), "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }
  if (numel(iter) == 0) {
    return;
  }
  if (!can_use_32bit_indexing(iter)) {
    if constexpr (is_sym_v<H>) {
      TORCH_CHECK(false, "host trace: the 64-bit split is not part of this preview");
    } else {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        gpu_kernel(h, sub_iter, f);
      }
      return;
    }
  }
  gpu_kernel_impl_nocast(h, iter, f);
}

} // namespace at::native::host
