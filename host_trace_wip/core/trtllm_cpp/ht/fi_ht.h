// The type aliases the patched trtllm-gen launcher (land/core/trtllm_cpp/src) is written over. In FlashInfer's
// ordinary build they are the original types and functions, so the patched source compiles to the stock code
// (tests/test_drift.py checks the disassembly). With FI_HT_TRACED (the host-trace build, ht_traced.cu) they are
// c10::SymInt and the recording shim in ht_trace.h.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#ifndef FI_HT_TRACED
namespace fi_ht {
// a size or count that is symbolic under a trace
using sint = int;
using s32 = int32_t;
using s64 = int64_t;
using u64 = uint64_t;
using sz = size_t;
// a device address that is symbolic under a trace
template <class T>
using ptr = T*;

template <class P, class T>
inline P ptr_cast(T* p) {
  return (P)p;
}
inline void* const_cast_void(void const* p) { return const_cast<void*>(p); }
inline void const* byte_offset(void const* p, int64_t n) {
  return reinterpret_cast<void const*>(reinterpret_cast<char const*>(p) + n);
}
template <class T>
inline bool aligned(T* p, uint64_t alignment) {
  return reinterpret_cast<uint64_t>(p) % alignment == 0;
}
// a size read symbolically (t.size(i) here)
template <class Tensor>
inline auto sym_size(Tensor const& t, int64_t i) {
  return t.size(i);
}
// min/max of sizes: std::min/max themselves (an expression, not a guard, in the traced build)
using std::max;
using std::min;
using std::to_string;
template <class Tensor>
inline auto sym_numel(Tensor const& t) {
  return t.numel();
}
// a cubin about to be loaded (a trace refuses where a capture holds)
inline void loading(char const*) {}
}  // namespace fi_ht
#else
#include "ht_trace.h"
#endif
