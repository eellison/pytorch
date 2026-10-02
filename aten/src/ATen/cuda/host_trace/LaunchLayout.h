// What eager's pointwise launch sites (ATen/native/cuda/CUDALoops.cuh, Loops.cuh,
// CUDAJitLoops.cuh) report while the host-trace harvest captures an op
// (c10::cuda::isHostTraceHarvesting): the kernel, its launch configuration and
// each parameter's bytes with a class for each byte, so that the harvest reads
// the captured node's bytes by their types.
#pragma once

#include <ATen/core/TensorBase.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Export.h>
#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <new>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace at::cuda::host_trace {

// a parameter byte's class: 'k' a member the host sets or eager computes from
// the guarded dtypes (count, addresses, offset calculators, dtypes, element
// sizes), 's' a functor's scalar member (an op argument, a constant) or a
// jitted kernel's scalar and extra arguments, 'p' a functor's pointer member,
// 'z' a functor's member eager computes from the operands' sizes, '.' none:
// padding, an empty class. A functor's member eager reads from a CPU scalar
// operand (TensorIteratorBase::scalar_value) as ScalarType t: 'A' + t, or
// '0' + t for its reciprocal (1 / the value in double precision, cast to t)
struct LaunchLayout {
  cudaFunction_t function;
  dim3 grid;
  dim3 block;
  std::vector<std::string> classes;
  std::vector<std::string> bytes;
};

// c10::cuda::isHostTraceHarvesting, its first test inlined at each launch site:
// GCC leaves that function out of line in the large pointwise units, a PLT call
// per eager launch
C10_ALWAYS_INLINE bool harvesting() {
  return C10_UNLIKELY(c10::cuda::detail::host_trace_harvesting_threads.load(std::memory_order_relaxed) != 0) &&
      c10::cuda::detail::isHostTraceHarvestingThread();
}

// starts (into) or stops (nullptr) this thread's record; returns the previous sink
TORCH_CUDA_CPP_API std::vector<LaunchLayout>* record_launch_layouts(std::vector<LaunchLayout>* into);
TORCH_CUDA_CPP_API void report_launch_layout(LaunchLayout layout);

// the bytes of the member a CPU scalar class ('A' + type or '0' + type) marks,
// read from src as eager reads it; false for another class
TORCH_CUDA_CPP_API bool cpu_scalar_bytes(const at::TensorBase& src, char cls, void* out);

// v's bytes of class cls; its padding '.'
template <class T>
void value_bytes(const T& v, char* c, char cls) {
  alignas(T) unsigned char a[sizeof(T)];
  alignas(T) unsigned char b[sizeof(T)];
  std::memset(a, 0, sizeof(T));
  std::memset(b, 0xff, sizeof(T));
  if constexpr (std::is_empty_v<T>) {
#if defined(__GNUC__) && !defined(__clang__)
  } else if constexpr (std::is_trivially_copyable_v<T>) {
    __builtin_clear_padding(reinterpret_cast<T*>(b));
    std::memset(a, 0xff, sizeof(T));
#endif
  } else {
    // a copy writes the members, not the padding; the barriers keep the
    // fills and the reads from GCC's lifetime DSE
    asm volatile("" : : "r"(a), "r"(b) : "memory");
    T* x = new (a) T(v);
    T* y = new (b) T(v);
    asm volatile("" : : "r"(a), "r"(b) : "memory");
    for (size_t i = 0; i < sizeof(T); i++) {
      c[i] = a[i] == b[i] ? cls : '.';
    }
    x->~T();
    y->~T();
    return;
  }
  for (size_t i = 0; i < sizeof(T); i++) {
    c[i] = a[i] == b[i] ? cls : '.';
  }
}

template <class T, class M>
size_t member_offset(const T& v, const M& m) {
  return reinterpret_cast<const char*>(&m) - reinterpret_cast<const char*>(&v);
}

// A functor passed to the pointwise launch sites is empty or declares its
// members, `auto host_trace_fields() const { return std::tie(a, b); }`, so the
// harvest knows each of its bytes; `static constexpr bool host_trace_sizes =
// true` marks one whose members are of the operands' sizes. Of its fields,
// `host_trace_scalars()` ties those it takes from a CPU scalar operand and
// `host_trace_reciprocals()` those it takes as its reciprocal
template <class T, class = void>
struct has_fields : std::false_type {};
template <class T>
struct has_fields<T, std::void_t<decltype(std::declval<const T&>().host_trace_fields())>> : std::true_type {};

template <class T, class = void>
struct of_sizes : std::false_type {};
template <class T>
struct of_sizes<T, std::enable_if_t<T::host_trace_sizes>> : std::true_type {};

template <class T, class = void>
struct has_scalars : std::false_type {};
template <class T>
struct has_scalars<T, std::void_t<decltype(std::declval<const T&>().host_trace_scalars())>> : std::true_type {};

template <class T, class = void>
struct has_reciprocals : std::false_type {};
template <class T>
struct has_reciprocals<T, std::void_t<decltype(std::declval<const T&>().host_trace_reciprocals())>> : std::true_type {};

template <class T, class = void>
struct is_scalar_value : std::bool_constant<std::is_arithmetic_v<T> || std::is_enum_v<T>> {};
template <class T>
struct is_scalar_value<T, std::void_t<decltype(c10::CppTypeToScalarType<T>::value)>> : std::true_type {};

template <class T, class = void>
struct has_scalar_type : std::false_type {};
template <class T>
struct has_scalar_type<T, std::void_t<decltype(c10::CppTypeToScalarType<T>::value)>> : std::true_type {};

// a CPU scalar member's class; 's' for a type of no class, which the harvest
// then takes as a constant
template <class T>
constexpr char cpu_scalar_class(bool reciprocal) {
  if constexpr (has_scalar_type<T>::value) {
    constexpr int t = static_cast<int>(c10::CppTypeToScalarType<T>::value);
    if (reciprocal) {
      return std::is_same_v<T, float> || std::is_same_v<T, double> ? '0' + t : 's';
    }
    return t < 26 ? 'A' + t : 's';
  } else {
    return 's';
  }
}

template <class T>
void functor_bytes(const T& v, char* c);

template <class T, size_t N>
void functor_bytes(const std::array<T, N>& v, char* c) {
  for (size_t i = 0; i < N; i++) {
    functor_bytes(v[i], c + i * sizeof(T));
  }
}

template <class T>
void functor_bytes(const T& v, char* c) {
  if constexpr (std::is_empty_v<T>) {
  } else if constexpr (std::is_pointer_v<T>) {
    std::memset(c, 'p', sizeof(T));
  } else if constexpr (is_scalar_value<T>::value) {
    value_bytes(v, c, 's');
  } else if constexpr (of_sizes<T>::value) {
    value_bytes(v, c, 'z');
  } else if constexpr (has_fields<T>::value) {
    std::apply([&](const auto&... m) { (functor_bytes(m, c + member_offset(v, m)), ...); }, v.host_trace_fields());
    if constexpr (has_scalars<T>::value) {
      std::apply([&](const auto&... m) { (value_bytes(m, c + member_offset(v, m), cpu_scalar_class<std::decay_t<decltype(m)>>(false)), ...); }, v.host_trace_scalars());
    }
    if constexpr (has_reciprocals<T>::value) {
      std::apply([&](const auto&... m) { (value_bytes(m, c + member_offset(v, m), cpu_scalar_class<std::decay_t<decltype(m)>>(true)), ...); }, v.host_trace_reciprocals());
    }
  } else {
    static_assert(has_fields<T>::value, "a functor passed to a pointwise launch is empty or declares its members (host_trace_fields)");
  }
}

} // namespace at::cuda::host_trace
