// Kernel parameters laid out by their caller (a ReduceOp, SymReduce.cu; the
// index kernel's functor, IndexKernel.cu): bytes and symbolic fields, recorded
// by SymHost.cpp's recorder.
#pragma once
#include <ATen/cuda/host_trace/SymHost.h>

#include <array>
#include <cstring>
#include <vector>

namespace at::cuda::host_trace {

struct ParamImage {
  std::vector<uint8_t> bytes;
  std::vector<Field> fields;
  size_t align = 1;

  ParamImage() = default;
  ParamImage(size_t size, size_t align) : bytes(size, 0), align(align) {}

  void put(size_t at, const void* src, size_t size) {
    TORCH_INTERNAL_ASSERT(at + size <= bytes.size());
    std::memcpy(bytes.data() + at, src, size);
  }
  // a member of `width` bytes at `at`: a constant written, a symbol recorded
  void set(size_t at, size_t width, const c10::SymInt& v, bool pointer = false) {
    if (auto c = v.maybe_as_int()) {
      const int64_t value = *c;
      put(at, &value, width);  // little-endian: the low bytes
    } else {
      fields.push_back({0, at, width, v, pointer});
    }
  }
};

// a launch parameter of type T: a constant's bytes, or a symbolic value's field
template <typename T>
ParamImage param_of(const c10::SymInt& v) {
  ParamImage p(sizeof(T), alignof(T));
  p.set(0, sizeof(T), v);
  return p;
}

// a launch parameter of type T holding v's bytes
template <typename T>
ParamImage param_bytes(const T& v) {
  ParamImage p(sizeof(T), alignof(T));
  p.put(0, &v, sizeof(T));
  return p;
}

// a pointer parameter to t's data
inline ParamImage param_pointer(Recorder& rec, const TensorBase& t) {
  ParamImage p(sizeof(void*), alignof(void*));
  p.set(0, sizeof(void*), rec.data_ptr(t), true);
  return p;
}

// a PackedTensorAccessor32<T, N> (Acc) parameter over t: its symbolic address,
// sizes and strides, at the members' offsets in Acc (found once from an
// accessor of known values; its members are protected)
template <class Acc, size_t N>
ParamImage param_accessor32(Recorder& rec, const TensorBase& t) {
  static const std::array<size_t, 2 * N + 1> at = [] {
    std::array<int64_t, N> sizes{}, strides{};
    for (size_t i = 0; i < N; i++) {
      sizes[i] = 0x51000 + static_cast<int64_t>(i);
      strides[i] = 0x52000 + static_cast<int64_t>(i);
    }
    const uintptr_t data = 0x0123456789abcdefULL;
    const Acc a(reinterpret_cast<typename Acc::PtrType>(data), sizes.data(), strides.data());
    unsigned char bytes[sizeof(Acc)];
    std::memcpy(bytes, &a, sizeof(Acc));
    auto find = [&bytes](const void* v, size_t n) {
      for (size_t k = 0; k + n <= sizeof(Acc); k++) {
        if (std::memcmp(bytes + k, v, n) == 0) {
          return k;
        }
      }
      TORCH_INTERNAL_ASSERT(false, "host_trace: a PackedTensorAccessor's layout");
    };
    std::array<size_t, 2 * N + 1> r{};
    r[0] = find(&data, sizeof(data));
    for (size_t i = 0; i < N; i++) {
      const int32_t s = static_cast<int32_t>(sizes[i]), st = static_cast<int32_t>(strides[i]);
      r[1 + i] = find(&s, sizeof(s));
      r[1 + N + i] = find(&st, sizeof(st));
    }
    return r;
  }();
  TORCH_CHECK(static_cast<size_t>(t.dim()) == N);
  ParamImage p(sizeof(Acc), alignof(Acc));
  p.set(at[0], sizeof(void*), rec.data_ptr(t), true);
  for (size_t i = 0; i < N; i++) {
    p.set(at[1 + i], sizeof(int32_t), t.sym_size(static_cast<int64_t>(i)));
    p.set(at[1 + N + i], sizeof(int32_t), t.sym_stride(static_cast<int64_t>(i)));
  }
  return p;
}

// std::min / std::max of a launch configuration, as the program's select rows
inline c10::SymInt sym_min(Recorder& rec, const c10::SymInt& a, const c10::SymInt& b) {
  return rec.select(a.sym_lt(b), a, b);
}

inline c10::SymInt sym_max(Recorder& rec, const c10::SymInt& a, const c10::SymInt& b) {
  return rec.select(a.sym_lt(b), b, a);
}

// OffsetCalculator<n> (uint32_t indices, byte strides) over c, at `at` of image
TORCH_CUDA_CPP_API void put_offset_calculator(ParamImage& image, size_t at, int n, const SymOffsets& c);

// a launch of kernel (or of a jitted function) with these parameters, each at its alignment
TORCH_CUDA_CPP_API void record_images(const void* kernel, std::vector<ParamImage> params, SymDim3 grid, SymDim3 block, c10::SymInt smem);
TORCH_CUDA_CPP_API void record_images(cudaFunction_t function, std::vector<ParamImage> params, SymDim3 grid, SymDim3 block, c10::SymInt smem);

} // namespace at::cuda::host_trace
