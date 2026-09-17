// Support for the generated siblings (HostTraceSibling_*.cu, from
// torchgen/dest/ufunc.py over ti/siblings.yaml and native_functions.yaml's
// ufunc_inner_loop): the field, slot and symbolic types of a functor's
// constant member by its opmath type.
#pragma once
#include <ATen/cuda/host_trace/Field.h>
#include <ATen/cuda/host_trace/ti/LoopsSym.cuh>

#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>

#include <cstdint>
#include <type_traits>

namespace at::cuda::host_trace::ti::gen {

// an int field for an integral opmath type, a float field otherwise
template <class T, size_t Off>
using field_t = std::conditional_t<std::is_integral_v<T>, IntField<T, Off>, FloatField<T, Off>>;
template <class T>
using slot_t = std::conditional_t<std::is_integral_v<T>, IntSlot<T>, FloatSlot<T>>;
template <class T>
using sym_t = std::conditional_t<std::is_integral_v<T>, c10::SymInt, c10::SymFloat>;

// a host value (a CPU scalar operand, a Scalar argument converted with the
// checked to<opmath_t>()) into a field, as the value the field types take
template <class T>
sym_t<T> sibling_value(T v) {
  if constexpr (std::is_integral_v<T>) {
    return c10::SymInt(static_cast<int64_t>(v));
  } else {
    return c10::SymFloat(static_cast<double>(v));
  }
}

} // namespace at::cuda::host_trace::ti::gen
