#pragma once

#include <ATen/OpMathType.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <c10/core/Scalar.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/native/cuda/thread_constants.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <c10/util/C++17.h>
#include <tuple>



namespace at::native {

template<int N>
static OffsetCalculator<N> make_input_offset_calculator(const TensorIteratorBase& iter) {
  // array size can not be 0, this happens when N == 0
  constexpr int array_size = std::max<int>(N, 1);
  TORCH_INTERNAL_ASSERT(N == iter.ntensors() - iter.noutputs());
  std::array<const int64_t*, array_size> strides;
  int64_t element_sizes[array_size];
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i + iter.noutputs()).data();
    element_sizes[i] = iter.element_size(i + iter.noutputs());
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <int num_outputs = 1>
static OffsetCalculator<num_outputs> make_output_offset_calculator(const TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(num_outputs == iter.noutputs());
  std::array<const int64_t*, num_outputs> strides;
  int64_t element_sizes[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<num_outputs>(iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <bool reverted_idx = false, typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;
  constexpr int elems_per_thread = policy_t::tws;

  int idx = blockIdx.x;
  if constexpr (reverted_idx)
    idx = gridDim.x - blockIdx.x - 1;

  return_t results[elems_per_thread];
  args_t args[elems_per_thread];

  // load
  policy.load(args, idx);

  // compute
  #pragma unroll
  for (int i = 0; i < elems_per_thread; i++) {
    if (policy.check_inbounds(i)) {
#if defined(__HIP__)
      results[i] = c10::guts::apply(f, args[i]);
#else
      results[i] = std::apply(f, args[i]);
#endif
    }
  }

  // store
  policy.store(results, idx);
}

}  // namespace at::native

#include <ATen/native/cuda/CUDALoops.cuh>

namespace at:: native {

// The entry points below run gpu_kernel's routing (CUDALoops.cuh) over
// StaticFacts<func_t> for eager, or for a symbolic iterator (a CUDA graph host
// trace) its one host_trace::SymHost instantiation over the functor's
// run-time facts (make_sym_facts), which records the launch. A symbolic
// iterator beyond 32-bit indexing raises NotImplementedError from
// with_32bit_indexing().
template <typename func_t>
void gpu_kernel_nocast(TensorIteratorBase& iter, const func_t& f, bool check_cast = true);

template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f);

template <class H, bool nocast, class Facts, typename functor_t>
void gpu_kernel_body(TensorIteratorBase& iter, const Facts& facts, const functor_t& f, bool check_cast) {

  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
      iter.device(arg).is_cuda(),
      "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
  }

  if (H::numel(iter) == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    if constexpr (H::symbolic) {
      // raises: no symbolic split
      (void)iter.with_32bit_indexing();
    } else {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        if constexpr (nocast) {
          gpu_kernel_nocast(sub_iter, f, check_cast);
        } else {
          gpu_kernel(sub_iter, f);
        }
      }
      return;
    }
  }

  if constexpr (nocast) {
    if (check_cast) {
      TORCH_INTERNAL_ASSERT(!facts.needs_dynamic_casting(iter));
    }
    gpu_kernel_impl_nocast<H>(iter, facts, f);
  } else {
    gpu_kernel_impl<H>(iter, facts, f);
  }
}

template <typename func_t>
void gpu_kernel_nocast(TensorIteratorBase& iter, const func_t& f, bool check_cast) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return at::cuda::host_trace::sym_gpu_kernel(iter, make_sym_facts<func_t, /*nocast=*/true>(f), /*nocast=*/true, check_cast);
  }
#endif
  gpu_kernel_body<EagerHost, /*nocast=*/true>(iter, StaticFacts<func_t>{}, f, check_cast);
}

template <typename func_t>
void gpu_kernel_opaque(TensorIteratorBase& iter, const func_t& f) {
  gpu_kernel_nocast(iter, f, false);
}

template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return at::cuda::host_trace::sym_gpu_kernel(iter, make_sym_facts<func_t, /*nocast=*/false>(f), /*nocast=*/false, /*check_cast=*/false);
  }
#endif
  gpu_kernel_body<EagerHost, /*nocast=*/false>(iter, StaticFacts<func_t>{}, f, /*check_cast=*/false);
}

// gpu_kernel as host kind H
template <class H, typename func_t>
void gpu_kernel_as(TensorIteratorBase& iter, const func_t& f) {
#ifndef USE_ROCM
  if constexpr (H::symbolic) {
    at::cuda::host_trace::sym_gpu_kernel(iter, make_sym_facts<func_t, /*nocast=*/false>(f), /*nocast=*/false, /*check_cast=*/false);
  } else
#endif
  {
    gpu_kernel_body<EagerHost, /*nocast=*/false>(iter, StaticFacts<func_t>{}, f, /*check_cast=*/false);
  }
}

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  __device__ return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  // NB: scalar is stored in higher precision!
  AUnaryFunctor(func_t f_, opmath_arg1_t a_): f(f_), a(a_) {}
#ifndef USE_ROCM
  auto host_trace_fields() const {
    return std::tie(f, a);
  }
  auto host_trace_scalars() const {
    return std::tie(a);
  }
#endif
  private:
    func_t f;
    opmath_arg1_t a;
};

template<typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  __device__ return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  // NB: scalar is stored in higher precision!
  BUnaryFunctor(func_t f_, opmath_arg2_t b_): f(f_), b(b_) {}
#ifndef USE_ROCM
  auto host_trace_fields() const {
    return std::tie(f, b);
  }
  auto host_trace_scalars() const {
    return std::tie(b);
  }
#endif
  private:
    func_t f;
    opmath_arg2_t b;
};

// Though seemingly noop, this inserts casts from arg1_t to func_t's type
// (which may be higher precision), as well as casts to return_t
template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  __device__ return_t operator()(arg1_t a, arg2_t b) const {
    return f(a, b);
  }
  BinaryFunctor(func_t f_): f(f_) {}
#ifndef USE_ROCM
  auto host_trace_fields() const {
    return std::tie(f);
  }
#endif
  private:
    func_t f;
};

// Unlike gpu_kernel_with_scalars, this allows you to pass a func_t which
// accepts inputs at higher precision (typically opmath_t), but then
// ensure that we load from memory at the correct precision (scalar_t)
// to avoid expensive loads.  For the whole sordid story see
// https://dev-discuss.pytorch.org/t/cuda-loops-case-study-code-generation-vs-templates/302
// A CPU scalar operand's value as the functor holds it. For SymHost a host
// buffer's (a host trace's stand-in) is a placeholder that the scope returned
// by scalar_member marks for each replay to read from the buffer.
template <class H, typename T>
T cpu_scalar_value(TensorIteratorBase& iter, int arg) {
#ifndef USE_ROCM
  if constexpr (H::symbolic) {
    return H::template scalar_value<T>(iter, arg);
  }
#endif
  return iter.scalar_value<T>(arg);
}

template <class H, typename F>
auto cpu_scalar_member(const F& functor, const TensorIteratorBase& iter, int arg) {
#ifndef USE_ROCM
  if constexpr (H::symbolic) {
    return H::scalar_member(functor, std::get<0>(functor.host_trace_scalars()), iter, arg);
  } else
#endif
  {
    return 0;
  }
}

// For a kernel stub that reads a CPU scalar operand into its functor itself:
// host_scalar reads it (for a symbolic iterator's host-buffer operand, a host
// trace's stand-in, a placeholder), and host_scalar_member marks the functor
// member holding it, untransformed, for each replay to read from the buffer
// (`reciprocal`: the member holds 1 / the value in double precision). Only
// for a value that reaches the member as read; a stub that branches on it or
// transforms it otherwise stops at the stand-in's read.
template <typename T>
T host_scalar(TensorIteratorBase& iter, int arg) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return at::cuda::host_trace::SymHost::scalar_value<T>(iter, arg);
  }
#endif
  return iter.scalar_value<T>(arg);
}

// src: the CPU scalar operand, once removed from iter (TensorIteratorBase::remove_operand)
template <typename F, typename M>
auto host_scalar_member(const F& functor, const M& member, const TensorIteratorBase& iter, const TensorBase& src, bool reciprocal = false) {
#ifndef USE_ROCM
  std::unique_ptr<at::cuda::host_trace::ScalarScope> scope;
  if (C10_UNLIKELY(iter.is_symbolic()) && at::cuda::host_trace::is_host_buffer(src)) {
    scope = at::cuda::host_trace::SymHost::scalar_member_of(functor, member, src, reciprocal);
  }
  return scope;
#else
  return 0;
#endif
}

template <typename F, typename M>
auto host_scalar_member(const F& functor, const M& member, const TensorIteratorBase& iter, int arg, bool reciprocal = false) {
  return host_scalar_member(functor, member, iter, iter.tensor_base(arg), reciprocal);
}

// For a kernel stub that reads a Scalar argument into its functor: scalar_arg
// reads it (for a symbolic iterator's symbolic Scalar, a host trace's, a
// placeholder), and scalar_arg_member marks the functor member holding it, as
// Scalar::to converts it, for each replay to compute from its symbol. Only for
// a value that reaches the member as read, as host_scalar_member
template <typename T>
T scalar_arg(const TensorIteratorBase& iter, const Scalar& s) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(s.isSymbolic() && iter.is_symbolic())) {
    return T{};
  }
#endif
  return s.template to<T>();
}

template <typename F, typename M>
auto scalar_arg_member(const F& functor, const M& member, const TensorIteratorBase& iter, const Scalar& s) {
#ifndef USE_ROCM
  std::unique_ptr<at::cuda::host_trace::ScalarScope> scope;
  if (C10_UNLIKELY(s.isSymbolic() && iter.is_symbolic())) {
    TORCH_CHECK_NOT_IMPLEMENTED(s.isIntegral(/*includeBool=*/false), "host_trace: a symbolic ", s.type(), " Scalar argument in a functor member");
    scope = at::cuda::host_trace::SymHost::symbol_member_of(functor, member, s.toSymInt());
  }
  return scope;
#else
  return 0;
#endif
}

// the CPU scalar operand arg, for host_scalar_member once it is removed: an
// undefined tensor for an eager iterator
inline TensorBase host_scalar_source(const TensorIteratorBase& iter, int arg) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return iter.tensor_base(arg);
  }
#endif
  return TensorBase();
}

template <class H, typename arg1_t, typename arg2_t, typename return_t, typename func_t>
void opmath_gpu_kernel_with_scalars_body(TensorIteratorBase& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(f, cpu_scalar_value<H, opmath_arg1_t>(iter, 1));
    [[maybe_unused]] auto scalar = cpu_scalar_member<H>(af, iter, 1);
    iter.remove_operand(1);
    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    const OptionalDeviceGuard device_guard(iter.device(1));
    gpu_kernel_as<H>(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(f, cpu_scalar_value<H, opmath_arg2_t>(iter, 2));
    [[maybe_unused]] auto scalar = cpu_scalar_member<H>(bf, iter, 2);
    iter.remove_operand(2);
    gpu_kernel_as<H>(iter, bf);
  } else {
    gpu_kernel_as<H>(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
  }
}

template <typename arg1_t, typename arg2_t = arg1_t, typename return_t = arg1_t, typename func_t>
void opmath_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return opmath_gpu_kernel_with_scalars_body<at::cuda::host_trace::SymHost, arg1_t, arg2_t, return_t>(iter, f);
  }
#endif
  return opmath_gpu_kernel_with_scalars_body<EagerHost, arg1_t, arg2_t, return_t>(iter, f);
}

template <class H, typename scalar_t, typename return_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars_body(TensorIteratorBase& iter, const func_t& f) {
  // Use symmetric property of the functor to reduce number of kernels,
  // requires f(a, b) == f(b, a)
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  static_assert(std::is_same_v<opmath_arg_t, typename traits::template arg<1>::type>,
                "f is not symmetric");

  OptionalDeviceGuard device_guard;
  opmath_arg_t scalar_val{};
  // SymHost: the CPU scalar operand removed below, for its ScalarScope
  [[maybe_unused]] std::conditional_t<H::symbolic, TensorBase, int> scalar_src{};

  if (iter.is_cpu_scalar(1)) {
    scalar_val = cpu_scalar_value<H, opmath_arg_t>(iter, 1);
    if constexpr (H::symbolic) {
      scalar_src = iter.tensor_base(1);
    }
    iter.remove_operand(1);

    // TODO: When all kernels that use gpu_kernel_with_scalars are
    // ported to structured, this device guard can be deleted.  This
    // works around incorrect device guard generation for pre-structured
    // kernels device guards, but structured kernels do it right and
    // we can assume the device is already set correctly
    device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = cpu_scalar_value<H, opmath_arg_t>(iter, 2);
    if constexpr (H::symbolic) {
      scalar_src = iter.tensor_base(2);
    }
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    gpu_kernel_as<H>(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
#ifndef USE_ROCM
    std::unique_ptr<at::cuda::host_trace::ScalarScope> scalar;
    if constexpr (H::symbolic) {
      if (scalar_src.defined() && at::cuda::host_trace::is_host_buffer(scalar_src)) {
        scalar = H::scalar_member_of(unary_f, std::get<0>(unary_f.host_trace_scalars()), scalar_src);
      }
    }
#endif
    gpu_kernel_as<H>(iter, unary_f);
  }
}

template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return opmath_symmetric_gpu_kernel_with_scalars_body<at::cuda::host_trace::SymHost, scalar_t, return_t>(iter, f);
  }
#endif
  return opmath_symmetric_gpu_kernel_with_scalars_body<EagerHost, scalar_t, return_t>(iter, f);
}

// Legacy variant that assumes that func_t has the correct types
// that we expect to load from memory
template <typename func_t>
void gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t>(iter, f);
}

namespace { // functions for `gpu_kernel_multiple_outputs`.

// check the return type is `thrust::tuple`, not `std::tuple`.
template <typename T> struct is_tuple: std::false_type {};

template <typename ...T> struct is_tuple<thrust::tuple<T...>>: std::true_type {};

template <int num_outputs, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel_for_multi_outputs(int N, func_t f, array_t data, inp_calc_t ic, out_calc_t oc) {
  int remaining = N - block_work_size() * blockIdx.x;
  elementwise_kernel_helper(f, memory::policies::multi_outputs_unroll<array_t, inp_calc_t, out_calc_t, num_outputs>(data, remaining, ic, oc));
}

template <int num_outputs, class H = EagerHost, typename func_t, typename data_t, typename inp_calc_t, typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(typename H::int_t N, const func_t& f, data_t data, inp_calc_t ic, out_calc_t oc) {
  using array_t = std::array<char*, std::tuple_size_v<data_t>>;
#ifndef USE_ROCM
  if constexpr (H::symbolic) {
    typename H::int_t grid = (N + block_work_size() - 1) / block_work_size();
    using ic_t = typename H::template offset_calculator_t<inp_calc_t>;
    using oc_t = typename H::template offset_calculator_t<out_calc_t>;
    return H::launch(unrolled_elementwise_kernel_for_multi_outputs<num_outputs, func_t, array_t, ic_t, oc_t>, grid, num_threads(), 0, N, f, data, ic, oc);
  } else
#endif
  {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  report_launch_if_harvesting(unrolled_elementwise_kernel_for_multi_outputs<num_outputs, func_t, array_t, inp_calc_t, out_calc_t>, grid, num_threads(), N, f, data, ic, oc);
  unrolled_elementwise_kernel_for_multi_outputs<num_outputs, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <class H, typename func_t>
void gpu_kernel_multiple_outputs_impl(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  static_assert(is_tuple<output_t>::value, "f's return type must be `thrust::tuple`");
  constexpr int num_outputs = thrust::tuple_size<output_t>::value;
  constexpr int num_inputs = traits::arity;
  constexpr int ntensors = num_outputs + num_inputs;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  auto data = H::template data<ntensors>(iter);

  typename H::int_t numel = H::numel(iter);

  if (iter.is_contiguous()) {
    auto input_calc = TrivialOffsetCalculator<num_inputs>();
    auto output_calc = TrivialOffsetCalculator<num_outputs>();
    launch_unrolled_kernel_for_multi_outputs<num_outputs, H>(numel, f, data, input_calc, output_calc);
  } else {
#ifndef USE_ROCM
    if constexpr (H::symbolic) {
      // make_input/output_offset_calculator: element strides
      auto input_calc = H::template offset_calculator<num_inputs>(iter, num_outputs, /*element_strides=*/true);
      auto output_calc = H::template offset_calculator<num_outputs>(iter, 0, /*element_strides=*/true);
      launch_unrolled_kernel_for_multi_outputs<num_outputs, H>(numel, f, data, input_calc, output_calc);
      return;
    } else
#endif
    {
    auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    auto output_calc = make_output_offset_calculator<num_outputs>(iter);
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(numel, f, data, input_calc, output_calc);
    }
  }
}
} // namespace

template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f);

template <class H, typename func_t>
void gpu_kernel_multiple_outputs_body(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_cuda());
  }

  if (H::numel(iter) == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_multiple_outputs(sub_iter, f);
    }
    return;
  }

  gpu_kernel_multiple_outputs_impl<H>(iter, f);
}

template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
#ifndef USE_ROCM
  if (C10_UNLIKELY(iter.is_symbolic())) {
    return gpu_kernel_multiple_outputs_body<at::cuda::host_trace::SymHost>(iter, f);
  }
#endif
  gpu_kernel_multiple_outputs_body<EagerHost>(iter, f);
}

} //namespace at::native
